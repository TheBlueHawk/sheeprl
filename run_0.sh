#!/bin/bash



for seed in 0 1 2
do
    python sheeprl.py exp=dreamer_v3_100k_ms_pacman_oc env.id=AssaultNoFrameskip-v4 fabric.devices="[0]" seed=$seed algo.train_every=2 algo.mlp_keys.decoder="[]"
done

python sheeprl.py exp=dreamer_v3_100k_ms_pacman_oc fabric.devices="[0]" seed=0 algo.train_every=8 algo.cnn_keys.decoder="[]" algo.cnn_keys.encoder="[]"
python sheeprl.py exp=dreamer_v3_100k_ms_pacman_oc fabric.devices="[0]" seed=0 algo.train_every=8 algo.mlp_keys.decoder="[]"