#!/bin/bash

python sheeprl.py exp=dreamer_v3_100k_ms_pacman_oc fabric.devices="[2]" seed=1 algo.mlp_keys.decoder="[]" algo.mlp_keys.encoder="[]"
python sheeprl.py exp=dreamer_v3_100k_ms_pacman_oc fabric.devices="[2]" seed=1
python sheeprl.py exp=dreamer_v3_100k_ms_pacman_oc fabric.devices="[2]" seed=2
python sheeprl.py exp=dreamer_v3_100k_ms_pacman_oc fabric.devices="[2]" seed=3
