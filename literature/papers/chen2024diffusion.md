---
kind: paper
title: "Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion"
authors: ["Boyuan Chen", "Diego Marti Monso", "Yilun Du", "Max Simchowitz", "Russ Tedrake", "Vincent Sitzmann"]
institutions: ["MIT CSAIL", "Technical University of Munich"]
year: 2024
venue: "NeurIPS 2024"
peer_reviewed: true
url: https://arxiv.org/abs/2407.01392
code_url: https://github.com/buoyancy99/diffusion-forcing
citations:
source: "raw/papers/chen2024diffusion.pdf"
added: "2026-11-23"
relevance: 5
credibility: 5
status: skimmed
related_experiments: []
related_concepts: ["Diffusion Forcing", "Exposure Bias", "Bidirectional vs Causal Video Diffusion"]
tags: [diffusion-forcing, autoregressive-diffusion, noise-schedule, rollout-stability]
---

# Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion

## TL;DR

Train a sequence model to denoise tokens that each carry an **independent
noise level**, so noise acts as partial masking. One model covers teacher
forcing (clean past) and full-sequence diffusion (equal noise), and at
sampling time the per-frame noise schedule is a control knob. The paper's
video results use a small convolutional RNN, not a transformer.

## Claims

- Independent per-token noise unifies next-token prediction with
  full-sequence diffusion; the objective is proven to optimize a variational
  lower bound on the likelihoods of all subsequences.
- Causal Diffusion Forcing rolls out video far past the training horizon
  (about 1000 frames), where teacher-forcing and full-sequence baselines
  diverge.
- Stabilisation recipe: when updating the recurrent latent during rollout,
  use the previous frame at a **small** noise level (0 < k << K) rather than clean.
- The causal structure also gives variable-horizon guidance and planning
  (Monte Carlo Guidance) for decision-making tasks.

## Methods

- Causal implementation: a vanilla RNN (a convolutional RNN for video) whose latent
  summarises past noisy tokens; trained with the standard denoising loss
  with noise levels sampled independently per token (Algorithm 1).
- Appendix: transformers work too. A non-causal masked transformer can
  recover causality by labelling future tokens as fully noised.
- Baselines share the architecture: teacher-forced next-frame diffusion and
  causal full-sequence diffusion.

## Results

- Minecraft and DMLab video: DF rolls out stably (about 1000 frames); teacher forcing
  diverges, and full-sequence diffusion shows frame-to-frame jumps.
- Planning and robot-manipulation tasks: gains over Diffuser-style baselines.
- Competitive on multivariate time-series forecasting (Appendix E).

## Critique / open questions

- Authors' own limitation: the causal implementation is an RNN, scaling to
  high-resolution video or internet-scale data is untested. Large-scale use
  is the follow-up work ([[song2025history]], and Atlas per the video).
- [[huang2025self]] argues the noisy-context trick "does not fundamentally
  resolve" [[Exposure Bias]]: DF still conditions on noised ground truth, not on
  the model's own outputs.

## Trust signals

- **Credibility:** 5, NeurIPS 2024 from MIT CSAIL, public code, widely
  built on (Self Forcing, CausVid, DFoT all cite it).

## Follow-up

**Relevance:** 5, primary evidence for [[Diffusion Forcing]], and a basis for
[[Exposure Bias]] and [[Bidirectional vs Causal Video Diffusion]].
- Reproduce the small-noise-context rollout stabilisation and measure drift
  against rollout length on a 16 GB GPU.
