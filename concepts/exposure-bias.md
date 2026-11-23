---
kind: concept
name: "Exposure Bias"
status: seedling
added: "2026-11-22"
sources: [welch2026fei-fei-li-atlas, oasis-model-github-io-oasis, guo2025mineworld, zhang2025matrix, yin2024from, bruce2024genie, chen2024diffusion, huang2025self]
related_concepts: ["Diffusion Forcing", "Bidirectional vs Causal Video Diffusion"]
related_experiments: []
tags: [autoregressive, drift, train-test-mismatch, rollout]
---

# Exposure Bias

## Definition

The train/test mismatch of teacher forcing. A model trained to predict step
*t+1* from **ground-truth** history is run at inference on **its own**
predictions, so its errors feed back into the context and compound.

## Why it matters here

For continuous outputs (RGB, latents) nothing quantizes small errors away the
way a discrete token vocabulary does, so autoregressive video and world-model
rollouts drift into blur or noise ([[welch2026fei-fei-li-atlas]]). Drift over
rollout length is the main thing to measure when this project trials open
world models.

## What the primary papers add

- The diffusion-forcing paper reports that teacher-forced and full-sequence
  baselines diverge on Minecraft/DMLab rollouts while DF stays stable to about
  1000 frames ([[chen2024diffusion]]). The video presents this as the fix;
  [[huang2025self]] disputes that it fully resolves the mismatch (see
  [[Diffusion Forcing]]).
- Measured impact: CausVid shows visible error accumulation in long rollouts
  that Self Forcing reduces at the same 1.3B size ([[huang2025self]]).

- Matrix-Game 2.0 adopts Self Forcing's DMD-on-own-rollouts and says it
  mitigates the error accumulation of Teacher Forcing and Diffusion Forcing
  ([[zhang2025matrix]]; author claim).
- Genie sidesteps continuous-latent drift by predicting discrete VQ tokens with
  MaskGIT, but keeps only 16 frames of memory ([[bruce2024genie]]); whether
  discreteness actually hides exposure bias is untested here.

## Connections

- [[Diffusion Forcing]]: the noise-augmented-context fix
- [[huang2025self]]: Self Forcing trains on the model's own rollouts with KV
  caching; it states that Teacher Forcing and Diffusion Forcing both leave the
  context off the inference distribution. CausVid is the earlier
  DF-based version, which Self Forcing says matches the wrong distribution
- Scheduled sampling: the older sequence-modeling fix
