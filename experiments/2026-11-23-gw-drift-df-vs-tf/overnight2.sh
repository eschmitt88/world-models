#!/usr/bin/env bash
# Relaunch after the tf arm OOM'd at step 0 (doubled sequence at batch 32). Sequential; nothing else on the GPU.
cd /home/eschmitt/projects/research/world-models
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=vendor/.venv-mg2/bin/python
$PY -u src/wm/train_df.py --out /home/eschmitt/projects/research/world-models/experiments/2026-11-23-gw-drift-df-vs-tf/results/tf --mode tf --hours 3.5 --accum 4 --eval_every 2000 --seed 0 || echo "tf arm failed"
$PY -u /home/eschmitt/projects/research/world-models/experiments/2026-11-23-gw-drift-df-vs-tf/eval_long.py --n 16 --horizon 384 > /home/eschmitt/projects/research/world-models/experiments/2026-11-23-gw-drift-df-vs-tf/results/long_eval.log 2>&1 || echo "long eval failed"
# climate eval: DF context-noise optimum picked from the long eval (walk script, frames 224-384, nonzero levels)
BEST=$($PY -c "
import json,numpy as np
r=json.load(open('/home/eschmitt/projects/research/world-models/experiments/2026-11-23-gw-drift-df-vs-tf/results/long_eval.json'))
best=max((float(np.mean(v['acc_aligned'][224:])),float(k.split('|')[2])) for k,v in r.items() if k.startswith('df|walk|') and float(k.split('|')[2])>0)
print(best[1])" 2>/dev/null || echo 0.1)
echo "climate eval: df best nonzero ctx noise = $BEST" >> /home/eschmitt/projects/research/world-models/experiments/2026-11-24-gw-weather-climate-eval/log.md
$PY -u src/wm/climate_eval.py --exp /home/eschmitt/projects/research/world-models/experiments/2026-11-24-gw-weather-climate-eval --train_args /home/eschmitt/projects/research/world-models/experiments/2026-11-23-gw-drift-df-vs-tf/results/df/args.json --section contraction \
   --ckpts /home/eschmitt/projects/research/world-models/experiments/2026-11-23-gw-drift-df-vs-tf/results/df/ckpt.pt $(ls /home/eschmitt/projects/research/world-models/experiments/2026-11-23-gw-drift-df-vs-tf/results/tf/snapshots/*.pt 2>/dev/null) /home/eschmitt/projects/research/world-models/experiments/2026-11-23-gw-drift-df-vs-tf/results/tf/ckpt.pt > /home/eschmitt/projects/research/world-models/experiments/2026-11-24-gw-weather-climate-eval/results/contraction.log 2>&1 || echo "contraction failed"
$PY -u src/wm/climate_eval.py --exp /home/eschmitt/projects/research/world-models/experiments/2026-11-24-gw-weather-climate-eval --train_args /home/eschmitt/projects/research/world-models/experiments/2026-11-23-gw-drift-df-vs-tf/results/df/args.json --section ensemble \
   --arm df_cn0 /home/eschmitt/projects/research/world-models/experiments/2026-11-23-gw-drift-df-vs-tf/results/df/ckpt.pt 0.0 0 --arm df_cnbest /home/eschmitt/projects/research/world-models/experiments/2026-11-23-gw-drift-df-vs-tf/results/df/ckpt.pt $BEST 0 --arm tf_cn0 /home/eschmitt/projects/research/world-models/experiments/2026-11-23-gw-drift-df-vs-tf/results/tf/ckpt.pt 0.0 0 \
   --arm df_cn0_snap8 /home/eschmitt/projects/research/world-models/experiments/2026-11-23-gw-drift-df-vs-tf/results/df/ckpt.pt 0.0 8 --arm tf_cn0_snap8 /home/eschmitt/projects/research/world-models/experiments/2026-11-23-gw-drift-df-vs-tf/results/tf/ckpt.pt 0.0 8 > /home/eschmitt/projects/research/world-models/experiments/2026-11-24-gw-weather-climate-eval/results/ensemble.log 2>&1 || echo "ensemble failed"
/home/eschmitt/projects/research/world-models/experiments/2026-11-23-gw-memory-equal-compute/run_all.sh > /home/eschmitt/projects/research/world-models/experiments/2026-11-23-gw-memory-equal-compute/results/run.log 2>&1 || echo "memory experiment failed"
