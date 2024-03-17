#!/bin/bash

DEVICE=1

for seed in 1
do
    python sheeprl.py exp=dreamer_v3_100k_ms_pacman_oc algo.train_every=2 fabric.devices="[$DEVICE]" seed=$seed 
    python sheeprl.py exp=dreamer_v3_100k_ms_pacman_oc algo.train_every=2 fabric.devices="[$DEVICE]" seed=$seed algo.cnn_keys.decoder=[]
    python sheeprl.py exp=dreamer_v3_100k_ms_pacman_oc algo.train_every=2 fabric.devices="[$DEVICE]" seed=$seed algo.mlp_keys.decoder=[]
    python sheeprl.py exp=dreamer_v3_100k_ms_pacman_oc algo.train_every=2 fabric.devices="[$DEVICE]" seed=$seed algo.cnn_keys.decoder=[] algo.cnn_keys.encoder=[]
    python sheeprl.py exp=dreamer_v3_100k_ms_pacman_oc algo.train_every=2 fabric.devices="[$DEVICE]" seed=$seed algo.mlp_keys.decoder=[] algo.mlp_keys.encoder=[]
done


