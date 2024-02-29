#!/bin/bash



for seed in 0 1
do
    python sheeprl.py exp=dreamer_v3_100k_ms_pacman_oc fabric.devices="[2]" seed=$seed algo.train_every=2 algo.world_model.obs_loss_regularizer=100.0
    python sheeprl.py exp=dreamer_v3_100k_ms_pacman_oc fabric.devices="[2]" seed=$seed algo.train_every=2 algo.world_model.obs_loss_regularizer=300.0
done
