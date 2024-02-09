#!/bin/bash


for seed in 0 1 2
do
    python sheeprl.py exp=dreamer_v3_100k_ms_pacman_oc env.id=CarnivalNoFrameskip-v4 fabric.devices="[1]" seed=$seed
    python sheeprl.py exp=dreamer_v3_100k_ms_pacman_oc env.id=CarnivalNoFrameskip-v4 fabric.devices="[1]" seed=$seed algo.mlp_keys.decoder="[]"
    python sheeprl.py exp=dreamer_v3_100k_ms_pacman_oc env.id=CarnivalNoFrameskip-v4 fabric.devices="[1]" seed=$seed algo.mlp_keys.decoder="[]" algo.mlp_keys.encoder="[]"
    python sheeprl.py exp=dreamer_v3_100k_ms_pacman_oc env.id=CarnivalNoFrameskip-v4 fabric.devices="[1]" seed=$seed algo.cnn_keys.decoder="[]" algo.cnn_keys.encoder="[]"
done
