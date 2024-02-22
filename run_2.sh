#!/bin/bash


# python sheeprl.py exp=dreamer_v3_100k_ms_pacman_oc fabric.devices="[2]" seed=2 algo.train_every=8 algo.cnn_keys.decoder="[]" algo.cnn_keys.encoder="[]"
python sheeprl.py exp=dreamer_v3_100k_ms_pacman_oc fabric.devices="[2]" seed=2 algo.train_every=8 algo.mlp_keys.decoder="[]"
python sheeprl.py exp=dreamer_v3_100k_ms_pacman_oc fabric.devices="[2]" seed=2 algo.train_every=2 buffer.size=10000