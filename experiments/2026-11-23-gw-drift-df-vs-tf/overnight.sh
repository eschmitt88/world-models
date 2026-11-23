#!/usr/bin/env bash
# Overnight sequence (one attributed job, strictly sequential; nothing else on the GPU):
#   1) df arm 3.5 h, 2) tf arm 3.5 h, 3) long-rollout drift eval,
#   4) memory arms (mem, win) 1.75 h each, 5) memory eval.
set -e
cd /home/eschmitt/projects/research/world-models
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=vendor/.venv-mg2/bin/python
for arm in df tf; do
  $PY -u src/wm/train_df.py --out /home/eschmitt/projects/research/world-models/experiments/2026-11-23-gw-drift-df-vs-tf/results/$arm --mode $arm --hours 3.5 --eval_every 2000 --seed 0
done
$PY -u /home/eschmitt/projects/research/world-models/experiments/2026-11-23-gw-drift-df-vs-tf/eval_long.py --n 24 --horizon 384 > /home/eschmitt/projects/research/world-models/experiments/2026-11-23-gw-drift-df-vs-tf/results/long_eval.log 2>&1
/home/eschmitt/projects/research/world-models/experiments/2026-11-23-gw-memory-equal-compute/run_all.sh > /home/eschmitt/projects/research/world-models/experiments/2026-11-23-gw-memory-equal-compute/results/run.log 2>&1
