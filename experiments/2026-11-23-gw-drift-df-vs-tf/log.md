# log — gw-drift-df-vs-tf
2026-11-23 19:07 launched train_both.sh (df 4 h then tf 4 h), attributed job wm-gw-df-tf
2026-11-23 chained eval_long.py (n 24, horizon 384) to start after training
2026-11-23 19:11 ~5 min of concurrent GPU smoke tests (train_mem) overlapped the df arm
2026-11-23 19:21 FIRST LAUNCH FAILED: my concurrent train_mem smoke tests (3.8 GB) OOM'd the df arm at step 0; the chained long eval then ran on the step-0 checkpoint. Outputs moved to results/_failed_launch1/. Relaunched as one sequential job (overnight.sh): df 3.5 h, tf 3.5 h, long eval, then the memory experiment. Rule: no GPU tests while a run is active.
2026-11-23 22:58 tf arm OOM'd at step 0 (doubled 4096-token sequence at batch 32; outputs in results/_failed_launch1/tf_oom). set -e then skipped the rest. Relaunched overnight2.sh: tf with --accum 4 (same effective batch 32, ~half the samples/hour of df: matched wall clock, compare at matched one-step accuracy), long eval n 16, then climate eval, then the memory experiment. No set -e now, so one failure cannot skip later steps.
