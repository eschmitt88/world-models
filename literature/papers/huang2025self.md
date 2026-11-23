---
kind: paper
title: "Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion"
authors: ["Xun Huang", "Zhengqi Li", "Guande He", "Mingyuan Zhou", "Eli Shechtman"]
institutions: ["Adobe Research", "The University of Texas at Austin"]
year: 2025
venue: "NeurIPS 2025"
peer_reviewed: true
url: https://arxiv.org/abs/2506.08009
code_url: https://github.com/guandeh17/Self-Forcing
citations:
source: "raw/papers/huang2025self.pdf"
added: "2026-11-23"
relevance: 5
credibility: 5
status: skimmed
related_experiments: []
related_concepts: ["Exposure Bias", "Diffusion Forcing", "Bidirectional vs Causal Video Diffusion"]
tags: [self-forcing, exposure-bias, causal-video-diffusion, kv-cache, real-time]
---

# Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion

## TL;DR

Train an autoregressive video diffusion model by unrolling its **own**
generation, with KV caching, during training, and apply a video-level
distribution-matching loss. This attacks [[Exposure Bias]] directly and gives
real-time streaming on one GPU while matching or beating slower
bidirectional models.

## Claims

- Teacher Forcing and Diffusion Forcing both denoise frames conditioned on
  context that does not come from the model's inference distribution.
  Noisy-context tricks "do not fundamentally resolve" exposure bias.
- Self Forcing conditions each frame on previously self-generated frames,
  so the training loss sees the inference distribution.
- CausVid's DMD loss matches the wrong distribution because its training
  outputs come from the DF scheme.

## Methods

- Wan2.1-T2V-1.3B initialisation, 4-step diffusion, frame-wise and
  chunk-wise (3 latent frames) variants; stochastic gradient truncation to
  keep the unrolled training affordable; rolling KV cache for extrapolation.
- Holistic losses (SiD, DMD, GAN/R3GAN) on the whole generated video.

## Results

- Chunk-wise model: 17.0 FPS, 0.69 s latency on a single H100, with VBench total 84.31
  (CausVid 81.20, Wan2.1 84.26 at 0.78 FPS). Same 1.3B parameters.
- User study: preferred over Wan2.1 (62.7% vs 37.3%) and the other baselines.
- CausVid shows error accumulation in long rollouts that Self Forcing reduces.

## Critique / open questions

- It is text-to-video; there is no action or camera conditioning in the paper,
  so its use as a world model needs an action pathway (cf. Matrix-Game 2.0).
- Throughput is measured on an H100. The repo README says "Nvidia GPU with at
  least 24 GB memory (RTX 4090, A100, and H100 are tested)", so 16 GB needs
  offloading or quantisation and is untested.
- Extrapolation past the training length relies on a rolling KV cache and
  is not a proof of unbounded stability.

## Trust signals

- **Credibility:** 5, NeurIPS 2025, Adobe Research, Apache-2.0 code and weights public.

## Follow-up

**Relevance:** 5, the main counter-position to [[Diffusion Forcing]] on
[[Exposure Bias]], and the post-training route in
[[Bidirectional vs Causal Video Diffusion]].
- Run the released 1.3B checkpoint and time it on the 16 GB card.
