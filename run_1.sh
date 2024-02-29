#!/bin/bash

DEVICE=1

for seed in 0 1
do
    python sheeprl.py exp=dreamer_v3_100k_ms_pacman_oc fabric.devices="[$DEVICE]" seed=$seed algo.train_every=2 
    python sheeprl.py exp=dreamer_v3_100k_ms_pacman_oc fabric.devices="[$DEVICE]" seed=$seed algo.train_every=2 algo.world_model.obs_loss_regularizer=10.0
done