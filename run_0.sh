#!/bin/bash
DEVICE=0

for seed in 0 1 2
do
    python sheeprl.py exp=dreamer_v3_100k_ms_pacman_oc algo.train_every=2 algo.cnn_keys.decoder=[] fabric.devices="[$DEVICE]" seed=$seed 
    python sheeprl.py exp=dreamer_v3_100k_ms_pacman_oc algo.train_every=2 algo.cnn_keys.decoder=[] env.id=AssaultNoFrameskip-v4 fabric.devices="[$DEVICE]" seed=$seed
done