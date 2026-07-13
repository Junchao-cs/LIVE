#!/usr/bin/env bash
set -euo pipefail

python inference_evaluate.py --max_videos 1 --save_videos "$@"
