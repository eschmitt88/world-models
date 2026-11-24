---
kind: concept
name: "Exposure Bias"
status: seedling
added: "2026-11-22"
sources: [welch2026fei-fei-li-atlas, oasis-model-github-io-oasis, guo2025mineworld, zhang2025matrix, yin2024from, bruce2024genie, chen2024diffusion, huang2025self, alonso2024diffusion, li2025smallworlds, zhao2026miniworld, po2025bagger, pedersen2025thermalizer, lippe2023pderefiner, zhao2026frechet, li2026freqforcing, wang2026worldprobe]
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
- Thermalizer treats the same drift in PDE emulators as an inference-time
  fix: a separate diffusion prior of the stationary distribution denoises
  off-attractor states during rollout, leaving the emulator untouched. It
  restores the attractor, not the trajectory
  ([[pedersen2025thermalizer]]; peer-reviewed, 2D turbulence only).
- PDE-Refiner reports the pushforward trick (training on own predictions without
  gradient through the input) gave no gain on KS, since it cannot learn the
  low-amplitude content; its fix is a denoising refinement of each step instead
  ([[lippe2023pderefiner]]; PDEs, not video).
- Fréchet Forcing fine-tunes MiniWorld-1B on its own free rollouts with Fréchet
  distances to real-data feature statistics, with no teacher; rollouts stay stable
  beyond 4,000 frames on DROID and Sekai against failure shortly after the 253-frame
  pretraining horizon. Flow matching on self-generated history alone (FM-only)
  barely helped, and DMD configurations tried gave no gain
  ([[zhao2026frechet]]; preprint, one 1B model; see [[Distributional Rollout Training]]).
- FreqForcing keeps Self Forcing's weights and fixes the same drift at
  inference: attention sink plus a low-pass fusion with the attention output of
  six early anchor frames. It reports that the drift appears as lost DC and
  low-band energy over a 60 s rollout, and takes Self Forcing from 5 s to 120 s
  without training ([[li2026freqforcing]]; preprint, text-to-video, no actions;
  see [[Spectral Rollout Diagnostics]]).
- WorldProbe measures the downstream picture across ten public interactive
  models: per-rollout quality slopes are negative for nine of ten (visual) and
  all ten (content) over up to 180 s, with abrupt collapse in the older models
  ([[wang2026worldprobe]]; VLM- and IQA-scored, survivors only).
