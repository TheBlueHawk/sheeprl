#!/bin/bash

DEVICE=1

for seed in 1
do
    python sheeprl.py exp=dreamer_v3_100k_ms_pacman_oc algo.train_every=2 fabric.devices="[$DEVICE]" seed=$seed env.wrapper.env.perturbation=1
    python sheeprl.py exp=dreamer_v3_100k_ms_pacman_oc algo.train_every=2 fabric.devices="[$DEVICE]" seed=$seed env.wrapper.env.perturbation=3
    python sheeprl.py exp=dreamer_v3_100k_ms_pacman_oc algo.train_every=2 fabric.devices="[$DEVICE]" seed=$seed env.wrapper.env.perturbation=2
    python sheeprl.py exp=dreamer_v3_100k_ms_pacman_oc algo.train_every=2 fabric.devices="[$DEVICE]" seed=$seed env.wrapper.env.perturbation=7
done 




