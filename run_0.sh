#!/bin/bash

python sheeprl.py exp=dreamer_v3_100k_ms_pacman_oc fabric.devices="[0]" seed=0 algo.mlp_keys.decoder="[]"
python sheeprl.py exp=dreamer_v3_100k_ms_pacman_oc fabric.devices="[0]" seed=1 algo.mlp_keys.decoder="[]"
python sheeprl.py exp=dreamer_v3_100k_ms_pacman_oc fabric.devices="[0]" seed=4 algo.mlp_keys.decoder="[]"
python sheeprl.py exp=dreamer_v3_100k_ms_pacman_oc fabric.devices="[0]" seed=0 algo.mlp_keys.decoder="[]" algo.mlp_keys.encoder="[]"