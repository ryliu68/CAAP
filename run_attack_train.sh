#!/usr/bin/env bash
set -e

# ===== User settings =====
DATASET="tongji"              # tongji / iitd / aisec
NET="compnet"                 # e.g. compnet / ccnet / co3net / resnet18 / vgg16 / mobilenetv2 / shufflenetv2
MODEL_CKPT="ckpt/aug_0.5/${DATASET}/${NET}.pth"

ATTACK_MODE="untargeted"      # untargeted / targeted



# ===== Run =====
python -m attack.main \
  --dataset "${DATASET}" \
  --net "${NET}" \
  --model_ckpt "${MODEL_CKPT}" \
  --attack_mode "${ATTACK_MODE}" 