#!/bin/bash

DEVICE=1

for seed in 3 4
do
    python sheeprl.py exp=dreamer_v3_100k_ms_pacman_oc env.id=AssaultNoFrameskip-v4 algo.train_every=2 fabric.devices="[$DEVICE]" seed=$seed 
    python sheeprl.py exp=dreamer_v3_100k_ms_pacman_oc env.id=AssaultNoFrameskip-v4 algo.train_every=2 fabric.devices="[$DEVICE]" seed=$seed algo.cnn_keys.decoder=[]
    python sheeprl.py exp=dreamer_v3_100k_ms_pacman_oc env.id=AssaultNoFrameskip-v4 algo.train_every=2 fabric.devices="[$DEVICE]" seed=$seed algo.mlp_keys.decoder=[]
    python sheeprl.py exp=dreamer_v3_100k_ms_pacman_oc env.id=AssaultNoFrameskip-v4 algo.train_every=2 fabric.devices="[$DEVICE]" seed=$seed algo.cnn_keys.decoder=[] algo.cnn_keys.encoder=[]
    python sheeprl.py exp=dreamer_v3_100k_ms_pacman_oc env.id=AssaultNoFrameskip-v4 algo.train_every=2 fabric.devices="[$DEVICE]" seed=$seed algo.mlp_keys.decoder=[] algo.mlp_keys.encoder=[]
done


