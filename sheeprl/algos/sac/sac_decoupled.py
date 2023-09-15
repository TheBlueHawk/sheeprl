import os
import time
import warnings
from datetime import datetime, timedelta
from math import prod

import gymnasium as gym
import hydra
import numpy as np
import torch
from lightning.fabric import Fabric
from lightning.fabric.fabric import _is_using_cli
from lightning.fabric.loggers import TensorBoardLogger
from lightning.fabric.plugins.collectives import TorchCollective
from lightning.fabric.plugins.collectives.collective import CollectibleGroup
from lightning.fabric.strategies import DDPStrategy
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict, make_tensordict
from tensordict.tensordict import TensorDictBase
from torch.utils.data.sampler import BatchSampler
from torchmetrics import MeanMetric, SumMetric

from sheeprl.algos.sac.agent import SACActor, SACAgent, SACCritic
from sheeprl.algos.sac.sac import train
from sheeprl.algos.sac.utils import test
from sheeprl.data.buffers import ReplayBuffer
from sheeprl.utils.callback import CheckpointCallback
from sheeprl.utils.env import make_env
from sheeprl.utils.metric import MetricAggregator
from sheeprl.utils.registry import register_algorithm
from sheeprl.utils.timer import timer
from sheeprl.utils.utils import print_config


@torch.no_grad()
def player(cfg: DictConfig, world_collective: TorchCollective, player_trainer_collective: TorchCollective):
    print_config(cfg)

    # Initialize logger
    root_dir = (
        os.path.join("logs", "runs", cfg.root_dir)
        if cfg.root_dir is not None
        else os.path.join("logs", "runs", "sac_decoupled", datetime.today().strftime("%Y-%m-%d_%H-%M-%S"))
    )
    run_name = (
        cfg.run_name if cfg.run_name is not None else f"{cfg.env.id}_{cfg.exp_name}_{cfg.seed}_{int(time.time())}"
    )
    logger = TensorBoardLogger(root_dir=root_dir, name=run_name)
    logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    # Initialize Fabric
    fabric = Fabric(loggers=logger, callbacks=[CheckpointCallback()])
    if not _is_using_cli():
        fabric.launch()
    rank = fabric.global_rank
    device = fabric.device
    fabric.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic

    # Environment setup
    vectorized_env = gym.vector.SyncVectorEnv if cfg.env.sync_env else gym.vector.AsyncVectorEnv
    envs = vectorized_env(
        [
            make_env(
                cfg.env.id,
                cfg.seed + rank * cfg.env.num_envs + i,
                rank,
                cfg.env.capture_video,
                logger.log_dir,
                "train",
                mask_velocities=False,
                vector_env_idx=i,
            )
            for i in range(cfg.env.num_envs)
        ]
    )
    if not isinstance(envs.single_action_space, gym.spaces.Box):
        raise ValueError("Only continuous action space is supported for the SAC agent")
    if len(envs.single_observation_space.shape) > 1:
        raise ValueError(
            "Only environments with vector-only observations are supported by the SAC agent. "
            f"Provided environment: {cfg.env.id}"
        )

    # Send (possibly updated, by the make_dict_env method for example) cfg to the trainers
    world_collective.broadcast_object_list([cfg], src=0)

    # Define the agent and the optimizer and setup them with Fabric
    act_dim = prod(envs.single_action_space.shape)
    obs_dim = prod(envs.single_observation_space.shape)
    actor = SACActor(
        observation_dim=obs_dim,
        action_dim=act_dim,
        hidden_size=cfg.algo.actor.hidden_size,
        action_low=envs.single_action_space.low,
        action_high=envs.single_action_space.high,
    ).to(device)
    flattened_parameters = torch.empty_like(
        torch.nn.utils.convert_parameters.parameters_to_vector(actor.parameters()), device=device
    )

    # Receive the first weights from the rank-1, a.k.a. the first of the trainers
    # In this way we are sure that before the first iteration everyone starts with the same parameters
    player_trainer_collective.broadcast(flattened_parameters, src=1)
    torch.nn.utils.convert_parameters.vector_to_parameters(flattened_parameters, actor.parameters())

    # Metrics
    aggregator = MetricAggregator(
        {"Rewards/rew_avg": MeanMetric(sync_on_compute=False), "Game/ep_len_avg": MeanMetric(sync_on_compute=False)}
    ).to(device)

    # Local data
    buffer_size = cfg.buffer.size // cfg.env.num_envs if not cfg.dry_run else 1
    rb = ReplayBuffer(
        buffer_size,
        cfg.env.num_envs,
        device=device,
        memmap=cfg.buffer.memmap,
        memmap_dir=os.path.join(logger.log_dir, "memmap_buffer", f"rank_{fabric.global_rank}"),
    )
    step_data = TensorDict({}, batch_size=[cfg.env.num_envs], device=device)

    # Global variables
    last_log = 0
    policy_step = 0
    last_checkpoint = 0
    first_info_sent = False
    policy_steps_per_update = int(cfg.env.num_envs)
    num_updates = int(cfg.total_steps // policy_steps_per_update) if not cfg.dry_run else 1
    learning_starts = cfg.algo.learning_starts // policy_steps_per_update if not cfg.dry_run else 0

    # Warning for log and checkpoint every
    if cfg.metric.log_every % policy_steps_per_update != 0:
        warnings.warn(
            f"The metric.log_every parameter ({cfg.metric.log_every}) is not a multiple of the "
            f"policy_steps_per_update value ({policy_steps_per_update}), so "
            "the metrics will be logged at the nearest greater multiple of the "
            "policy_steps_per_update value."
        )
    if cfg.checkpoint.every % policy_steps_per_update != 0:
        warnings.warn(
            f"The checkpoint.every parameter ({cfg.checkpoint.every}) is not a multiple of the "
            f"policy_steps_per_update value ({policy_steps_per_update}), so "
            "the checkpoint will be saved at the nearest greater multiple of the "
            "policy_steps_per_update value."
        )

    with device:
        # Get the first environment observation and start the optimization
        obs = torch.tensor(envs.reset(seed=cfg.seed)[0], dtype=torch.float32)  # [N_envs, N_obs]

    for update in range(1, num_updates + 1):
        policy_step += cfg.env.num_envs

        # Measure environment interaction time: this considers both the model forward
        # to get the action given the observation and the time taken into the environment
        with timer("Time/env_interaction_time", SumMetric(sync_on_compute=False)):
            if update <= learning_starts:
                actions = envs.action_space.sample()
            else:
                # Sample an action given the observation received by the environment
                with torch.no_grad():
                    actions, _ = actor(obs)
                    actions = actions.cpu().numpy()
            next_obs, rewards, dones, truncated, infos = envs.step(actions)
            dones = np.logical_or(dones, truncated)

        if "final_info" in infos:
            for i, agent_ep_info in enumerate(infos["final_info"]):
                if agent_ep_info is not None:
                    ep_rew = agent_ep_info["episode"]["r"]
                    ep_len = agent_ep_info["episode"]["l"]
                    aggregator.update("Rewards/rew_avg", ep_rew)
                    aggregator.update("Game/ep_len_avg", ep_len)
                    fabric.print(f"Rank-0: policy_step={policy_step}, reward_env_{i}={ep_rew[-1]}")

        # Save the real next observation
        real_next_obs = next_obs.copy()
        if "final_observation" in infos:
            for idx, final_obs in enumerate(infos["final_observation"]):
                if final_obs is not None:
                    real_next_obs[idx] = final_obs

        with device:
            next_obs = torch.tensor(real_next_obs, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.float32).view(cfg.env.num_envs, -1)
            rewards = torch.tensor(rewards, dtype=torch.float32).view(cfg.env.num_envs, -1)  # [N_envs, 1]
            dones = torch.tensor(dones, dtype=torch.float32).view(cfg.env.num_envs, -1)

        step_data["dones"] = dones
        step_data["actions"] = actions
        step_data["observations"] = obs
        if not cfg.buffer.sample_next_obs:
            step_data["next_observations"] = real_next_obs
        step_data["rewards"] = rewards
        rb.add(step_data.unsqueeze(0))

        # next_obs becomes the new obs
        obs = next_obs

        # Send data to the training agents
        if update >= learning_starts:
            # Send local info to the trainers
            if not first_info_sent:
                world_collective.broadcast_object_list(
                    [{"update": update, "last_log": last_log, "last_checkpoint": last_checkpoint}], src=0
                )
                first_info_sent = True

            # Sample data to be sent to the trainers
            training_steps = learning_starts if update == learning_starts else 1
            chunks = rb.sample(
                training_steps * cfg.algo.per_rank_gradient_steps * cfg.per_rank_batch_size * (fabric.world_size - 1),
                sample_next_obs=cfg.buffer.sample_next_obs,
            ).split(training_steps * cfg.algo.per_rank_gradient_steps * cfg.per_rank_batch_size)
            world_collective.scatter_object_list([None], [None] + chunks, src=0)

            # Wait the trainers to finish
            player_trainer_collective.broadcast(flattened_parameters, src=1)

            # Convert back the parameters
            torch.nn.utils.convert_parameters.vector_to_parameters(flattened_parameters, actor.parameters())

            # Logs trainers-only metrics
            if policy_step - last_log >= cfg.metric.log_every or cfg.dry_run:
                # Gather metrics from the trainers
                metrics = [None]
                player_trainer_collective.broadcast_object_list(metrics, src=1)

                # Log metrics
                fabric.log_dict(metrics[0], policy_step)

        # Logs player-only metrics
        if policy_step - last_log >= cfg.metric.log_every or cfg.dry_run:
            fabric.log_dict(aggregator.compute(), policy_step)
            aggregator.reset()

            # Sync timers
            timer_metrics = timer.compute()
            fabric.log(
                "Time/sps_env_interaction",
                ((policy_step - last_log) * cfg.env.action_repeat) / timer_metrics["Time/env_interaction_time"],
                policy_step,
            )
            timer.reset()

            # Reset counters
            last_log = policy_step

        # Checkpoint model
        if (
            update >= learning_starts  # otherwise the processes end up deadlocked
            and cfg.checkpoint.every > 0
            and policy_step - last_checkpoint >= cfg.checkpoint.every
        ) or cfg.dry_run:
            last_checkpoint = policy_step
            ckpt_path = fabric.logger.log_dir + f"/checkpoint/ckpt_{policy_step}_{fabric.global_rank}.ckpt"
            fabric.call(
                "on_checkpoint_player",
                fabric=fabric,
                player_trainer_collective=player_trainer_collective,
                ckpt_path=ckpt_path,
                replay_buffer=rb if cfg.buffer.checkpoint else None,
            )

    world_collective.scatter_object_list([None], [None] + [-1] * (world_collective.world_size - 1), src=0)

    # Last Checkpoint
    ckpt_path = fabric.logger.log_dir + f"/checkpoint/ckpt_{policy_step}_{fabric.global_rank}.ckpt"
    fabric.call(
        "on_checkpoint_player",
        fabric=fabric,
        player_trainer_collective=player_trainer_collective,
        ckpt_path=ckpt_path,
        replay_buffer=rb if cfg.buffer.checkpoint else None,
    )

    envs.close()
    if fabric.is_global_zero:
        test_env = make_env(
            cfg.env.id,
            None,
            0,
            cfg.env.capture_video,
            fabric.logger.log_dir,
            "test",
            mask_velocities=False,
            vector_env_idx=0,
        )()
        test(actor, test_env, fabric, cfg)


def trainer(
    world_collective: TorchCollective,
    player_trainer_collective: TorchCollective,
    optimization_pg: CollectibleGroup,
):
    global_rank = world_collective.rank
    group_world_size = world_collective.world_size - 1

    # Receive (possibly updated, by the make_dict_env method for example) cfg from the player
    data = [None]
    world_collective.broadcast_object_list(data, src=0)
    cfg: DictConfig = data[0]

    # Initialize Fabric
    fabric = Fabric(strategy=DDPStrategy(process_group=optimization_pg), callbacks=[CheckpointCallback()])
    if not _is_using_cli():
        fabric.launch()
    device = fabric.device
    fabric.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic

    # Environment setup
    vectorized_env = gym.vector.SyncVectorEnv if cfg.env.sync_env else gym.vector.AsyncVectorEnv
    envs = vectorized_env([make_env(cfg.env.id, 0, 0, False, None, mask_velocities=False)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # Define the agent and the optimizer and setup them with Fabric
    act_dim = prod(envs.single_action_space.shape)
    obs_dim = prod(envs.single_observation_space.shape)
    actor = fabric.setup_module(
        SACActor(
            observation_dim=obs_dim,
            action_dim=act_dim,
            hidden_size=cfg.algo.actor.hidden_size,
            action_low=envs.single_action_space.low,
            action_high=envs.single_action_space.high,
        )
    )
    critics = [
        fabric.setup_module(
            SACCritic(observation_dim=obs_dim + act_dim, hidden_size=cfg.algo.critic.hidden_size, num_critics=1)
        )
        for _ in range(cfg.algo.critic.n)
    ]
    target_entropy = -act_dim
    agent = SACAgent(actor, critics, target_entropy, alpha=cfg.algo.alpha.alpha, tau=cfg.algo.tau, device=fabric.device)

    # Optimizers
    qf_optimizer, actor_optimizer, alpha_optimizer = fabric.setup_optimizers(
        hydra.utils.instantiate(cfg.algo.critic.optimizer, params=agent.qfs.parameters()),
        hydra.utils.instantiate(cfg.algo.actor.optimizer, params=agent.actor.parameters()),
        hydra.utils.instantiate(cfg.algo.alpha.optimizer, params=[agent.log_alpha]),
    )

    # Send weights to rank-0, a.k.a. the player
    if global_rank == 1:
        player_trainer_collective.broadcast(
            torch.nn.utils.convert_parameters.parameters_to_vector(actor.parameters()), src=1
        )

    # Metrics
    aggregator = MetricAggregator(
        {
            "Loss/value_loss": MeanMetric(sync_on_compute=cfg.metric.sync_on_compute, process_group=optimization_pg),
            "Loss/policy_loss": MeanMetric(sync_on_compute=cfg.metric.sync_on_compute, process_group=optimization_pg),
            "Loss/alpha_loss": MeanMetric(sync_on_compute=cfg.metric.sync_on_compute, process_group=optimization_pg),
        }
    ).to(device)

    # Receive data from player reagrding the:
    # * update
    # * last_log
    # * last_checkpoint
    data = [None]
    world_collective.broadcast_object_list(data, src=0)
    update = data[0]["update"]
    last_log = data[0]["last_log"]
    last_checkpoint = data[0]["last_checkpoint"]

    # Start training
    train_step = 0
    last_train = 0
    policy_steps_per_update = cfg.env.num_envs
    policy_step = update * policy_steps_per_update
    while True:
        # Wait for data
        data = [None]
        world_collective.scatter_object_list(data, [None for _ in range(world_collective.world_size)], src=0)
        data = data[0]
        if not isinstance(data, TensorDictBase) and data == -1:
            # Last Checkpoint
            if global_rank == 1:
                state = {
                    "agent": agent.state_dict(),
                    "qf_optimizer": qf_optimizer.state_dict(),
                    "actor_optimizer": actor_optimizer.state_dict(),
                    "alpha_optimizer": alpha_optimizer.state_dict(),
                    "update": update,
                }
                fabric.call("on_checkpoint_trainer", player_trainer_collective=player_trainer_collective, state=state)
            return
        data = make_tensordict(data, device=device)
        sampler = BatchSampler(range(len(data)), batch_size=cfg.per_rank_batch_size, drop_last=False)

        # Start training
        with timer(
            "Time/train_time", SumMetric(sync_on_compute=cfg.metric.sync_on_compute, process_group=optimization_pg)
        ):
            for batch_idxes in sampler:
                train(
                    fabric,
                    agent,
                    actor_optimizer,
                    qf_optimizer,
                    alpha_optimizer,
                    data[batch_idxes],
                    aggregator,
                    update,
                    cfg,
                    policy_steps_per_update,
                    group=optimization_pg,
                )
            train_step += group_world_size

        if global_rank == 1:
            player_trainer_collective.broadcast(
                torch.nn.utils.convert_parameters.parameters_to_vector(actor.parameters()), src=1
            )

        if policy_step - last_log >= cfg.metric.log_every or cfg.dry_run:
            # Sync distributed metrics
            metrics = aggregator.compute()
            aggregator.reset()

            # Sync distributed timers
            timers = timer.compute()
            metrics.update({"Time/sps_train": (train_step - last_train) / timers["Time/train_time"]})
            timer.reset()

            if global_rank == 1:
                player_trainer_collective.broadcast_object_list(
                    [metrics], src=1
                )  # Broadcast metrics: fake send with object list between rank-0 and rank-1

            # Reset counters
            last_log = policy_step
            last_train = train_step

        # Checkpoint model on rank-0: send it everything
        if (cfg.checkpoint.every > 0 and policy_step - last_checkpoint >= cfg.checkpoint.every) or cfg.dry_run:
            last_checkpoint = policy_step
            if global_rank == 1:
                state = {
                    "agent": agent.state_dict(),
                    "qf_optimizer": qf_optimizer.state_dict(),
                    "actor_optimizer": actor_optimizer.state_dict(),
                    "alpha_optimizer": alpha_optimizer.state_dict(),
                    "update": update,
                }
                fabric.call("on_checkpoint_trainer", player_trainer_collective=player_trainer_collective, state=state)

        # Update counters
        update += 1
        policy_step += policy_steps_per_update


@register_algorithm(decoupled=True)
@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    world_collective = TorchCollective()
    player_trainer_collective = TorchCollective()
    world_collective.setup(
        backend="nccl" if os.environ.get("LT_ACCELERATOR", None) in ("gpu", "cuda") else "gloo",
        timeout=timedelta(days=1),
    )

    # Create a global group, assigning it to the collective: used by the player to exchange
    # collected experiences with the trainers
    world_collective.create_group(timeout=timedelta(days=1))
    global_rank = world_collective.rank

    if world_collective.world_size == 1:
        raise RuntimeError(
            "Please run the script with the number of devices greater than 1: "
            "`lightning run model --devices=2 sheeprl.py ...`"
        )

    # Create a group between rank-0 (player) and rank-1 (trainer), assigning it to the collective:
    # used by rank-1 to send metrics to be tracked by the rank-0 at the end of a training episode
    player_trainer_collective.create_group(ranks=[0, 1], timeout=timedelta(days=1))

    # Create a new group, without assigning it to the collective: in this way the trainers can
    # still communicate with the player through the global group, but they can optimize the agent
    # between themselves
    optimization_pg = world_collective.new_group(
        ranks=list(range(1, world_collective.world_size)), timeout=timedelta(days=1)
    )
    if global_rank == 0:
        player(cfg, world_collective, player_trainer_collective)
    else:
        trainer(world_collective, player_trainer_collective, optimization_pg)


if __name__ == "__main__":
    main()
