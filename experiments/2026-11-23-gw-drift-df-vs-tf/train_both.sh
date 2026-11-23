#!/usr/bin/env bash
set -e
cd /home/eschmitt/projects/research/world-models
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for arm in df tf; do
  vendor/.venv-mg2/bin/python -u src/wm/train_df.py --out /home/eschmitt/projects/research/world-models/experiments/2026-11-23-gw-drift-df-vs-tf/results/$arm --mode $arm --hours 4.0 --eval_every 2000 --seed 0
done
