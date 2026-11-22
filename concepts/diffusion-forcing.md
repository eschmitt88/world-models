---
kind: concept
name: "Diffusion Forcing"
status: seedling
added: "2026-11-22"
sources: [welch2026fei-fei-li-atlas]
related_concepts: ["Exposure Bias", "Bidirectional vs Causal Video Diffusion", "Novel View Prediction"]
related_experiments: []
tags: [autoregressive-diffusion, training-objective, noise-schedule, rollout-stability]
---

# Diffusion Forcing

## Definition

Training a causal sequence diffusion model with an **independent noise level
per token or frame**, sampled separately and passed to the model as input. The
two usual schemes are special cases: all-equal noise is full-sequence
(bidirectional-style) diffusion, and clean-past/noisy-next is teacher forcing.
At inference the per-frame noise schedule is a free control knob (Chen et al.,
MIT 2024; primary paper not yet ingested).

## Why it matters here

It is the reported fix for [[Exposure Bias]]. The model trains on imperfect,
partly noised context, so at rollout its own outputs look in-distribution.
That lets rollouts stay stable well past the training horizon. World Labs
says it generalizes the idea to per-*modality* noise levels ("latent forcing"
/ "modality forcing"). One Atlas model then does text-to-image, NVS, 360°
panoramas and interactive exploration, depending on which inputs are clean
and which are noised ([[welch2026fei-fei-li-atlas]]; company claim,
unverified).

## Connections

- [[Exposure Bias]]: the failure it addresses
- [[Bidirectional vs Causal Video Diffusion]]: sits between the two
- [[Novel View Prediction]]: Atlas's target, trained this way
- Testable on 16 GB: drift vs rollout length, and vs the noise level assigned
  to context frames at inference
