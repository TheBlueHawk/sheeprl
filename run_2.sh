#!/bin/bash

DEVICE=2

for seed in 0 1 2
do
    python sheeprl.py exp=dreamer_v3_100k_ms_pacman_oc env.id=CarnivalNoFrameskip-v4 algo.train_every=2 fabric.devices="[$DEVICE]" seed=$seed algo.mlp_keys.decoder=[] 
done
