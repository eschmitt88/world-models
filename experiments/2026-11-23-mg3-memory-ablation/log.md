# log — mg3-memory-ablation
2026-11-23 launched run.py; MG3 on 16 GB needed: T5 on CPU, int8 for all large block linears (6.61 GB DiT), token-chunked FFN, VAE encoder parked on CPU, last 10 blocks streamed from pinned CPU memory. Smoke: 1.8 fps at 704x1280, peak 11.9 GB (16 offloaded).
2026-11-23 STOPPED first launch after 3 rollouts: int8 (W8A8) on action_model linears erased the mouse signal (camera did not turn; return LPIPS 0.003). Excluded action_model from int8; results moved to raw_invalid_int8_action.jsonl. Relaunching.
