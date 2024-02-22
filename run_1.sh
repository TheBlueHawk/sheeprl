#!/bin/bash


for seed in 1
do
    python sheeprl.py exp=dreamer_v3_100k_ms_pacman_oc fabric.devices="[1]" seed=$seed algo.train_every=2 buffer.size=10000
done