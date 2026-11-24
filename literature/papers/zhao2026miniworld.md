---
kind: paper
title: "MiniWorld: Democratizing the Training of Video World Models from Scratch"
authors: ["Yian Zhao", "Ruochong Zheng", "Hongcan Guo", "Yu Yan", "Jian Zhang", "Jie Chen"]
institutions: ["Peking University"]
year: 2026
venue: "arXiv v2 (tech report)"
peer_reviewed: false
url: https://arxiv.org/abs/2608.01127
code_url: https://github.com/zhao-yian/MiniWorld
citations:
source: "raw/papers/zhao2026miniworld.pdf"
added: "2026-11-24"
relevance: 4
credibility: 3
status: skimmed
related_experiments: ["2026-11-23-gw-drift-df-vs-tf"]
related_concepts: ["Diffusion Forcing", "Exposure Bias", "Bidirectional vs Causal Video Diffusion"]
tags: [diffusion-forcing, block-causal, kv-cache, from-scratch, rectified-flow, droid, re10k, open-weights]
---

# MiniWorld: Democratizing the Training of Video World Models from Scratch

## TL;DR

An open recipe for training an action- or pose-conditioned streaming video
world model from scratch: a block-causal video DiT with rectified flow in a
pretrained Wan2.2 VAE latent space, chunk-wise diffusion forcing with a
non-decreasing noise schedule, two-stage short-then-long training, and
inference with a rolling KV cache and pipelined denoising. It is the closest
published recipe to ours (block-causal DiT, DF, rectified flow, rolling
context), at 0.12B to 3B parameters on real video instead of a 2D gridworld.

## Claims

- A from-scratch causal recipe avoids the mismatch of adapting a bidirectional
  pretrained generator, and is simple enough to train in "several days on a
  single 8-GPU server".
- Against a bidirectional 29-frame sliding-window baseline, it improves most
  metrics on 253-frame DROID rollouts (e.g. LPIPS +216%, depth accuracy +238%
  as ratios to the baseline) and on RE10K (PSNR +34%, SSIM +19%).
- A non-decreasing chunk noise schedule (CoPP, from AR-Diffusion) stabilises
  optimisation.
- An 8-chunk in-flight window with a rolling KV cache keeps quality and gives
  2.2x throughput (3.31 to 7.29 FPS) and 15x lower first-chunk latency.
- Quality rises monotonically from 0.5B to 3B on RE10K.

## Methods

- Rectified flow in the Wan2.2 VAE latent (4x temporal, 16x spatial, 48
  channels); frames 240x320. Chunk size 2 latent frames; bidirectional
  attention inside a chunk, causal across chunks.
- Action conditioning via AdaLN with an AdaLN-LoRA shared modulation; 10%
  action dropout for CFG. DROID: 7-d actions; RE10K: per-pixel ray maps.
- Diffusion forcing with a per-chunk timestep, constrained tau_1 <= ... <=
  tau_M, sampled by CoPP (anchor chunk, timesteps propagated outward).
  Loss only on the prediction chunks; the first frame or first chunk is clean.
- Training: 21 then 46 frames (batch 64, lr 1e-4, Muon), then 125 then 253
  frames (batch 8, lr 2e-5) with timestep shifting. Eight GPUs, bf16; GPU
  model and hours are not stated in the paper.
- Inference: committed chunks go to a FIFO KV cache with a persistent sink
  chunk and RoPE re-shifting; up to 8 chunks denoise concurrently.

## Results

- Main comparison is MiniWorld-1B vs a bidirectional sliding-window baseline,
  50 held-out videos, 253 frames, reported only as ratios (baseline = 1) in
  bar charts.
- 253 to 381 frames: appearance and dynamics keep about 94% and 96%; geometry,
  VLM and fidelity degrade more, so drift remains.
- KV cache 24 to 12 to 6 chunks: nearly flat at T = 64 latent frames. Sink
  frames: roughly neutral.
- Scaling (Fig. 4a) normalised to 0.5B: 3B gains 12 to 22% on four metrics.

## Critique / open questions

- No teacher-forcing arm and no ablation of the DF schedule, monotonic
  constraint or two-stage training. The only comparison is against a
  different paradigm (bidirectional sliding window), so the paper does not
  show that DF is what reduces drift, and it does not isolate
  [[Exposure Bias]].
- Results are relative scores with no absolute numbers, seeds or error bars
  in the text; the long-rollout evidence is two lengths (253, 381 frames).
- The authors say errors "still accumulate" and list self-forcing-style
  train-test alignment as future work, so exposure bias is open here too.
- Compute is stated only as "several days" on 8 unnamed GPUs, which is not a
  reproducible budget.

## Trust signals

- **Credibility:** 3, a Peking University tech report with no peer review and
  no citation count I could establish, but code (training, inference,
  throughput) and the 0.5B and 1B DROID and RE10K checkpoints are public,
  which makes the recipe checkable.

## Follow-up

**Relevance:** 4, near-exact prior art for our recipe (block-causal DiT, DF,
rectified flow, rolling context) and a ready citation for the drift study; it
shifts no concept, but its lack of a TF ablation leaves our DF-vs-TF
experiment as a missing piece.
- **Code and artifacts are released**, not promised: the GitHub repo
  (zhao-yian/MiniWorld, Apache-2.0 weights card, pushed 2026-08-04) has
  training and inference scripts, and Hugging Face has `MiniWorld_0_5b_*` and
  `MiniWorld_1b_*` for DROID and RE10K. The 3B, 0.12B and 0.39B variants in
  Table 1 have no released checkpoint. README says 8 GPUs are recommended and
  describes a single-GPU mode with smaller batches.
- **Design differences from ours** (`src/wm/df.py`): we work in pixel space on
  64x64 frames with 8x8 patches (64 tokens/frame), full attention within a
  frame (chunk of one frame) and independent per-frame t, with no monotonic
  constraint; they use a VAE latent, chunks of two latent frames (8 RGB
  frames) and a monotone schedule. We recompute a sliding window; they use a
  KV cache with a sink and RoPE re-shifting. We have a TF arm via a doubled
  sequence; they have none.
- **Compute differs by orders of magnitude**: 0.55B to 3B parameters,
  multi-day on 8 GPUs, against a 16 GB RTX 5080 for us.
- **Cheap to borrow**: the monotone CoPP schedule as an extra DF arm, and the
  rolling cache with sink once rollouts outgrow the window.
- Cite in the drift write-up as the nearest from-scratch DF recipe.
