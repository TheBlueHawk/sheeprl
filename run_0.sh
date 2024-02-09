#!/bin/bash

# for loop over the seeds
for seed in 0 1 2
do
    python sheeprl.py exp=dreamer_v3_100k_ms_pacman_oc env.id=AssaultNoFrameskip-v4 fabric.devices="[0]" seed=$seed
    python sheeprl.py exp=dreamer_v3_100k_ms_pacman_oc env.id=AssaultNoFrameskip-v4 fabric.devices="[0]" seed=$seed algo.mlp_keys.decoder="[]"
    python sheeprl.py exp=dreamer_v3_100k_ms_pacman_oc env.id=AssaultNoFrameskip-v4 fabric.devices="[0]" seed=$seed algo.mlp_keys.decoder="[]" algo.mlp_keys.encoder="[]"
    python sheeprl.py exp=dreamer_v3_100k_ms_pacman_oc env.id=AssaultNoFrameskip-v4 fabric.devices="[0]" seed=$seed algo.cnn_keys.decoder="[]" algo.cnn_keys.encoder="[]"
done