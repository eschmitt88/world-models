# log — gw-memory-equal-compute
2026-11-23 chained run_all.sh (mem, win 1.75 h each, then eval) to start after the df-vs-tf long eval
2026-11-23 19:21 now launched from ../2026-11-23-gw-drift-df-vs-tf/overnight.sh after the drift eval (single sequential job)
2026-11-24 05:12 first launch OOM'd in the mem arm at batch 32 (36-frame sequences); added --accum (default 2) to train_mem.py and relaunched run_all.sh (outputs of the failed launch in results/_failed_oom/)
