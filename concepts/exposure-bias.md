---
kind: concept
name: "Exposure Bias"
status: seedling
added: "2026-11-22"
sources: [welch2026fei-fei-li-atlas, oasis-model-github-io-oasis, guo2025mineworld, zhang2025matrix, yin2024from, bruce2024genie, chen2024diffusion, huang2025self, alonso2024diffusion, li2025smallworlds, zhao2026miniworld, po2025bagger]
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
- DIAMOND shows the sampler matters too: with the same teacher-forced
  training, an EDM-parametrised diffusion world model at 3 denoising steps
  stays stable to t = 1000 on Atari, while DDPM at low step counts drifts out
  of distribution. This is qualitative, on one game
  ([[alonso2024diffusion]]; see [[Diffusion World Model]]).
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

- MiniWorld claims DF training reduces drift versus a bidirectional
  sliding-window baseline, but errors still accumulate past 253 frames
  (geometry and fidelity fall first) and it lists self-forcing-style
  alignment as future work; no TF arm, so the DF-vs-TF effect is untested
  there ([[zhao2026miniworld]]).
- BAgger fixes it without a teacher: fine-tune the DF model on time-reversed
  copies of its own drifted rollouts (DAgger-style, standard flow-matching
  loss). Three rounds at 1.3B cut aesthetic/imaging drift on 50 s videos from
  5.1 / 7.3 (DF) to 3.3 / 3.6, but round 1 alone can be worse than the seed
  model, and the reversal assumes the data is time-reversible
  ([[po2025bagger]]; preprint, no code yet).
