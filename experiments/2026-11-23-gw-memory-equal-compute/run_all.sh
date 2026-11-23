#!/usr/bin/env bash
set -e
cd /home/eschmitt/projects/research/world-models
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for arm in mem win; do
  vendor/.venv-mg2/bin/python -u src/wm/train_mem.py --out /home/eschmitt/projects/research/world-models/experiments/2026-11-23-gw-memory-equal-compute/results/$arm --arm $arm --hours 1.75
done
vendor/.venv-mg2/bin/python -u src/wm/train_mem.py --eval /home/eschmitt/projects/research/world-models/experiments/2026-11-23-gw-memory-equal-compute/results/mem /home/eschmitt/projects/research/world-models/experiments/2026-11-23-gw-memory-equal-compute/results/win /home/eschmitt/projects/research/world-models/experiments/2026-11-23-gw-drift-df-vs-tf/results/df --gaps 8 24 48 96 --n 24 --save /home/eschmitt/projects/research/world-models/experiments/2026-11-23-gw-memory-equal-compute/results/eval.json
