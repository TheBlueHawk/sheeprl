#!/bin/bash

python sheeprl.py exp=dreamer_v3_100k_ms_pacman fabric=gpu  fabric.devices="[2]" seed=0

python sheeprl.py exp=dreamer_v3_100k_ms_pacman fabric=gpu  fabric.devices="[2]" seed=1

python sheeprl.py exp=dreamer_v3_100k_ms_pacman fabric=gpu  fabric.devices="[2]" seed=0 algo.train_every=4