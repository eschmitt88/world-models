# log — gw-drift-df-vs-tf
2026-11-23 19:07 launched train_both.sh (df 4 h then tf 4 h), attributed job wm-gw-df-tf
2026-11-23 chained eval_long.py (n 24, horizon 384) to start after training
2026-11-23 19:11 ~5 min of concurrent GPU smoke tests (train_mem) overlapped the df arm
2026-11-23 19:21 FIRST LAUNCH FAILED: my concurrent train_mem smoke tests (3.8 GB) OOM'd the df arm at step 0; the chained long eval then ran on the step-0 checkpoint. Outputs moved to results/_failed_launch1/. Relaunched as one sequential job (overnight.sh): df 3.5 h, tf 3.5 h, long eval, then the memory experiment. Rule: no GPU tests while a run is active.
