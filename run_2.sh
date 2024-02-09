#!/bin/bash

python sheeprl.py exp=dreamer_v3_100k_ms_pacman_oc env.id=MsPacmanNoFrameskip-v4 fabric.devices="[3]" seed=0 algo.cnn_keys.decoder="[]" algo.cnn_keys.encoder="[]"
for seed in 1 2
do
    python sheeprl.py exp=dreamer_v3_100k_ms_pacman_oc env.id=BreakoutNoFrameskip-v4 fabric.devices="[3]" seed=$seed algo.cnn_keys.decoder="[]" algo.cnn_keys.encoder="[]"
    python sheeprl.py exp=dreamer_v3_100k_ms_pacman_oc env.id=MsPacmanNoFrameskip-v4 fabric.devices="[3]" seed=$seed algo.cnn_keys.decoder="[]" algo.cnn_keys.encoder="[]"
done