#!/bin/bash

DEVICE=1

for seed in 0 1 2
do
    python sheeprl.py exp=dreamer_v3_100k_ms_pacman_oc env.id=AssaultNoFrameskip-v4 algo.train_every=2 fabric.devices="[$DEVICE]" seed=$seed env.wrapper.env.perturbation=1
    python sheeprl.py exp=dreamer_v3_100k_ms_pacman_oc env.id=AssaultNoFrameskip-v4 algo.train_every=2 fabric.devices="[$DEVICE]" seed=$seed env.wrapper.env.perturbation=2
done


