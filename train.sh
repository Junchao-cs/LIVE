#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" \
  main.py -b configs/live_re10k.yaml --logdir logs/live-re10k "$@"
