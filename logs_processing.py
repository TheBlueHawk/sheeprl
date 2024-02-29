import os
import yaml
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import subprocess
import sys

def create_experiment_df(tensorboard_path, config_path):
    # Load TensorBoard data
    ea = event_accumulator.EventAccumulator(tensorboard_path)
    ea.Reload()

    # Load config data
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Extract parameters from config
    def get_ae_keys(mlp_keys: dict, cnn_keys: dict):
        if mlp_keys['encoder'] == []:
            return 'rgb'
        elif cnn_keys['encoder'] == []:
            return 'obj'
        elif mlp_keys['decoder'] == ["objects_position"] and cnn_keys['decoder'] == ["rgb"]:
            return 'rgb + obj'
        elif mlp_keys['decoder'] == ["objects_position"] and cnn_keys['decoder'] == []:
            return 'obj + rgb_encoding_only'
        elif mlp_keys['decoder'] == [] and cnn_keys['decoder'] == ["rgb"]:
            return 'rgb + obj_encoding_only'
        else:
            return 'Not found'
    

    params = {
        'seed': config.get('seed', 'Not found'),
        'ae_keys': get_ae_keys(config.get('algo', {}).get('mlp_keys', 'Not found'), config.get('algo', {}).get('cnn_keys', 'Not found')),
        'train_every': config.get('algo', {}).get('train_every', 'Not found'),
        'run_name': config.get('run_name', 'Not found'),
        'buffer_size': config.get('buffer', {}).get('size', 'Not found'),
        'obs_loss_regularizer': config.get('algo', {}).get('world_model', {}).get('obs_loss_regularizer', 1),
    }
    # Extract metrics and populate DataFrame
    list_of_entries = []
    for key in ea.scalars.Keys():
        # skip hp_metric
        if key == 'hp_metric':
            continue
        for event in ea.Scalars(key):
            list_of_entries.append({
                'Metric': key,
                'Step': event.step,
                'Value': event.value,
                'Seed': params['seed'],
                'AE_Keys': params['ae_keys'],
                'Train_Every': params['train_every'],
                'Run_ID': params['run_name'],
                'Buffer_Size': params['buffer_size'],
                'Obs_Loss_Regularizer': params['obs_loss_regularizer'],
            })
    df = pd.DataFrame.from_records(list_of_entries, columns=['Metric', 'Step', 'Value', 'Seed', 'AE_Keys', 'Train_Every', 'Run_ID', 'Buffer_Size', 'Obs_Loss_Regularizer'])
    return df


def run_tests(logs_base_dir, gpu_id=0):
    print(f"Running tests ...")
    checkpoint_pattern = 'ckpt_100000'
    config_filename = 'config.yaml'

    # Iterate over each subdirectory in the logs directory
    for root, dirs, files in os.walk(logs_base_dir):
        for file in files:
            if file.startswith(checkpoint_pattern):
                # Get checkpoint file path
                checkpoint_path = os.path.join(root, file)
                # Get parent directory
                parent_dir = os.path.dirname(root) # version_0

                # check if version_N for N = 1,2,3,4,5 exists at the same level as version_0
                # if not add to the list of seeds to run

                exp_dir = os.path.dirname(parent_dir)
                seeds = []
                for i in range(1, 6):
                    version_dir = os.path.join(exp_dir, f'version_{i}')
                    if not os.path.exists(version_dir):
                        seeds.append(i)
                if len(seeds) == 0:
                    continue

                print(f"Running test for {parent_dir}")
                # get config file path
                config_path = os.path.join(parent_dir, config_filename)
                # Check if config file exists   
                if not os.path.exists(config_path):
                    raise FileNotFoundError(f"Config file not found for {parent_dir}")
                config = yaml.safe_load(open(config_path, 'r'))
                env_id = config['env']['id']
                run_name = config['run_name']
                algo_mlp_encoder = config['algo']['mlp_keys']['encoder']
                algo_mlp_keys_decoder = config['algo']['mlp_keys']['decoder']
                algo_cnn_keys_encoder = config['algo']['cnn_keys']['encoder']
                algo_cnn_keys_decoder = config['algo']['cnn_keys']['decoder']
                algo_train_every = config['algo']['train_every']
                obs_loss_regularizer = config['algo']['world_model'].get('obs_loss_regularizer', 1)
                print(run_name, env_id, algo_train_every, algo_mlp_encoder, algo_mlp_keys_decoder, algo_cnn_keys_encoder, algo_cnn_keys_decoder, obs_loss_regularizer)
                for seed in seeds:
                    # execute python command to run the test
                    command = [
                        "python", "sheeprl.py",
                        "exp=dreamer_v3_100k_ms_pacman_oc",
                        f"+resume_from={checkpoint_path}",
                        f"env.id={env_id}",
                        f"algo.train_every={algo_train_every}",
                        f"fabric.devices=[{gpu_id}]",
                        f"seed={seed}",
                        f"algo.mlp_keys.encoder={algo_mlp_encoder}",
                        f"algo.mlp_keys.decoder={algo_mlp_keys_decoder}",
                        f"algo.cnn_keys.encoder={algo_cnn_keys_encoder}",
                        f"algo.cnn_keys.decoder={algo_cnn_keys_decoder}",
                        "algo.total_steps=0",
                        f"algo.world_model.obs_loss_regularizer={obs_loss_regularizer}",
                        f"run_name={run_name}"
                    ]
                    subprocess.run(command, check=True)


def process_all_experiments(logs_base_dir):
    # Pattern for TensorFlow event files and config files
    tf_event_pattern = 'events.out.tfevents'
    config_filename = 'config.yaml'

    # List to hold DataFrames from each experiment
    experiments_dfs = []

    # Iterate over each subdirectory in the logs directory
    for root, dirs, files in os.walk(logs_base_dir):
        for file in files:
            if file.startswith(tf_event_pattern):
                # TensorFlow event file path
                tf_event_path = os.path.join(root, file)
                # Corresponding config file path
                config_path = os.path.join(root, config_filename)
                # This is a completed either a completed training run or a test run 
                test_path = os.path.join(root, 'test_videos')
                if os.path.exists(test_path):
                    pass
                else:
                    continue
        
                # Check if config file exists
                if os.path.exists(config_path):
                    experiment_df = create_experiment_df(tf_event_path, config_path)
                    experiments_dfs.append(experiment_df)

    # Concatenate all DataFrames into a single one
    all_experiments_df = pd.concat(experiments_dfs, ignore_index=True)
    print(f"Total number of experiments processed: {len(experiments_dfs)}")
    return all_experiments_df


def rew_barplot(df: pd.DataFrame, env_id: str, train_every: int = 2):
    sns.set_theme(style="whitegrid")
    # Plot bar plot of Test/cumulative_reward grouped by MLP_Keys, show mean and confidence interval
    plt.figure(figsize=(10, 6))
    filtered_df = df[(df['Metric']=='Test/cumulative_reward') & (df['Train_Every'] == train_every) & (df['Buffer_Size'] == 100000) & (df['Obs_Loss_Regularizer'] == 1)]
    print(filtered_df.groupby('AE_Keys')['Run_ID'].nunique())
    filtered_df = filtered_df.drop(columns=['Run_ID', 'Metric', 'Train_Every', 'Seed', 'Step'])
    sns.set_style("whitegrid")

    # Create a bar plot showing the mean score grouped by MLP_Keys
    # Seaborn automatically calculates the confidence interval (95% by default)
    # and adds it as error bars
    plt.figure(figsize=(10, 6))
    sns.barplot(x="AE_Keys", y="Value", data=filtered_df, errorbar="sd", palette="muted", hue="AE_Keys")

    plt.title(f'Test Cumulative Reward on {env_id.replace("NoFrameskip-v4", "")} trained every {train_every}, grouped by AE_Keys (95% CI)')
    plt.xlabel('AE_Keys')
    plt.ylabel('Mean Test Reward')
    # position y-label on the top left of the plot
    plt.gca().yaxis.set_label_coords(-0.1, 0.5)

    plt.tight_layout()  # Adjust layout to make room for the rotated x-labels
    plt.savefig(f'viz/{env_id}/{train_every}_Test_cumulative_reward.png')
    plt.close()
    

def create_viz(df, env_id):
    metrics = df['Metric'].unique()
    train_every_list = df['Train_Every'].unique()
    print(f"Unique train_every: {train_every_list}")
    for train_every in train_every_list:
        if train_every not in [2, 8]:
            continue
        print(f"Plotting for train_every: {train_every}")
        for metric in metrics:
            if metric == 'hp_metric' or metric == 'Params/exploration_amount':
                continue
            print(f"Plotting for metric: {metric}")
            if metric == 'Test/cumulative_reward':
                rew_barplot(df, env_id, train_every)
                continue
            # Group by Step and Metric, then calculate mean and std
            filtered_df = df[(df['Metric']==metric) & (df['Train_Every'] == train_every) & (df['Buffer_Size'] == 100000) & (df['Obs_Loss_Regularizer'] == 1)]
            # get number of unique run ids by group of AE_Keys
            print(filtered_df.groupby('AE_Keys')['Run_ID'].nunique())
            grouped = filtered_df.drop(columns=['Run_ID', 'Metric', 'Buffer_Size', 'Train_Every', 'Seed']).groupby(['Step', 'AE_Keys'])
            mean_std_df = grouped['Value'].agg(['mean', 'std']).reset_index()
            # smoothing the plot
            mean_std_df['mean'] = mean_std_df['mean'].rolling(window=5).mean()
            #mean_std_df['std'] = mean_std_df['std'].rolling(window=3).mean()


            # Plotting
            plt.figure(figsize=(10, 6))

            # Unique AE_Keys
            ae_keys_unique = filtered_df['AE_Keys'].unique()

            for ae_key in sorted(ae_keys_unique):
                subset = mean_std_df[mean_std_df['AE_Keys'] == ae_key]
                
                # Plot mean
                plt.plot(subset['Step'], subset['mean'], label=f'AE Keys: {ae_key}')
                
                # Fill between mean ± std
                plt.fill_between(subset['Step'], subset['mean'] - subset['std'], subset['mean'] + subset['std'], alpha=0.3)

            plt.title(f'{metric} on {env_id.replace("NoFrameskip-v4", "")}, trained every {train_every}, grouped by AE_Keys')
            plt.xlabel('Step')
            plt.ylabel(f'Average {metric}')
            # log scale for y-axis if metric contains "loss" or "grad"    
            if "Loss" in metric or "Grads" in metric:
                plt.yscale('log')
            plt.legend()
            plt.savefig(f'viz/{env_id}/{train_every}_{metric.replace("/", "_")}.png')
            plt.close()

def update_viz(path, gpu_id):
    run_tests(path, gpu_id)
    all_experiments_df = process_all_experiments(path)
    env_id = path.split('/')[3]
    # make sure viz/ENV_ID directory exists before saving 
    if not os.path.exists(f'viz/{env_id}'):
        os.makedirs(f'viz/{env_id}')
    create_viz(all_experiments_df, env_id)


def main():
    # arguments are game and gpu_id
    game = sys.argv[1]
    gpu_id = sys.argv[2]
    print(f"Processing {game}...")
    base_dir = f'logs/runs/dreamer_v3/{game}NoFrameskip-v4'
    update_viz(base_dir, gpu_id)

if __name__ == "__main__":
    main()
