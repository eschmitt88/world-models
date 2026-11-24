---
kind: concept
name: "Diffusion Forcing"
status: seedling
added: "2026-11-22"
sources: [welch2026fei-fei-li-atlas, oasis-model-github-io-oasis, worldlabs-ai-atlas, yin2024from, chen2024diffusion, song2025history, huang2025self, xiao2025worldmem, li2025smallworlds, zhao2026miniworld, po2025bagger]
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
At inference the per-frame noise schedule is a free control knob (Chen et al., MIT 2024, [[chen2024diffusion]]). The original paper's
video model is a small convolutional RNN on Minecraft and DMLab, and the
authors list scaling as untested. "Noising as masking" does not require
causal attention: [[song2025history]] applies it to a *non-causal* transformer
(DFoT) and gets flexible-history conditioning plus 862-frame rollouts.

## Why it matters here

It is the reported fix for [[Exposure Bias]]. The model trains on imperfect,
partly noised context, so at rollout its own outputs look in-distribution.
That lets rollouts stay stable well past the training horizon. World Labs
says it generalizes the idea to per-*modality* noise levels ("latent forcing"
/ "modality forcing"). One Atlas model then does text-to-image, NVS, 360°
panoramas and interactive exploration, depending on which inputs are clean
and which are noised ([[welch2026fei-fei-li-atlas]]; company claim,
unverified).

## What the primary papers add

- Stabilised rollout comes from feeding the previous frame back at a *small*
  noise level, not clean ([[chen2024diffusion]], Sec. 3.3). Context noise at
  inference is the knob to sweep.
- It is *not* a complete cure for [[Exposure Bias]]. [[huang2025self]] argues
  DF still conditions on noised ground truth rather than the model's own
  outputs, and that the noisy-context trick trades away temporal consistency.
- Diffusion forcing and causality are separable: DFoT is bidirectional within
  its window ([[song2025history]]).

- CausVid and the DF-based few-step models inherit the scheme, and Self Forcing
  flags a training/inference mismatch in the CausVid DMD loss that comes from it
  ([[yin2024from]], [[huang2025self]]).

- Oasis is the clearest deployed instance: a DiT trained with diffusion forcing, plus
  "dynamic noising" at inference (noise injected in early passes, removed later) to limit
  error accumulation ([[oasis-model-github-io-oasis]]; company blog, weights public).
- The Atlas launch post names an autoregressive diffusion transformer but, per the
  summary I could fetch, does not mention diffusion forcing or per-modality noise; the
  "modality forcing" attribution rests on the video alone ([[worldlabs-ai-atlas]]).

## Connections

- [[Exposure Bias]]: the failure it addresses
- [[Bidirectional vs Causal Video Diffusion]]: sits between the two
- [[Novel View Prediction]]: Atlas's target, trained this way
- Testable on 16 GB: drift vs rollout length, and vs the noise level assigned
  to context frames at inference

- MiniWorld is a from-scratch open recipe: block-causal DiT, chunk-wise DF with
  a *non-decreasing* noise schedule (CoPP), rolling KV cache; checkpoints at
  0.5B and 1B are public. Its drift evidence is against a bidirectional
  sliding window only, with no teacher-forcing or schedule ablation
  ([[zhao2026miniworld]]).
- BAgger starts from a DF model and adds aggregated reversed self-rollouts to
  the training set, showing DF with or without context noise still drifts
  ([[po2025bagger]]).
