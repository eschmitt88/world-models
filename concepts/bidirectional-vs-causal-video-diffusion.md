---
kind: concept
name: "Bidirectional vs Causal Video Diffusion"
status: seedling
added: "2026-11-22"
sources: [welch2026fei-fei-li-atlas, huang2025self, song2025history]
related_concepts: ["Diffusion Forcing", "Exposure Bias", "Novel View Prediction"]
related_experiments: []
tags: [video-diffusion, attention, interactivity, dit]
---

# Bidirectional vs Causal Video Diffusion

## Definition

A video diffusion transformer is **bidirectional** when every space-time
patch attends to every other, past and future, and all frames share one
denoising timestep (Sora, Wan). It is **causal** when attention to future
frames is masked, so frames can be produced, and conditioned on actions or
poses, one step at a time.

## Why it matters here

Bidirectional models can't be interactive: frame *t* depends on frame *t+k*,
so the full action or camera trajectory has to be supplied upfront. Wan's
attention heads visibly use this, with some following static background
across all frames and others looking ahead to where a moving object will be
([[welch2026fei-fei-li-atlas]]). Going causal buys interactivity but invites
[[Exposure Bias]]. There are two routes: **post-train** a bidirectional model
into a causal one (CausVid, Self Forcing) or **train causal from scratch**
(Atlas). This choice decides which open checkpoints are usable as
interactive world models.

## Connections

- [[Diffusion Forcing]]: the training scheme that makes causal rollouts stable
- [[Exposure Bias]]: the cost of naive causality
- Wan 2.1 1.3B: an open bidirectional base that fits 16 GB

## What the primary papers add

- Self Forcing is the post-train route made concrete: Wan2.1-1.3B becomes a
  chunk-wise causal model at 17 FPS and 0.69 s latency on an H100, with
  VBench 84.31 vs 84.26 for the bidirectional Wan2.1 at 0.78 FPS
  ([[huang2025self]]). Its README asks for 24 GB.
- Correction to "diffusion forcing = causal": DFoT keeps a non-causal
  transformer and recovers flexible conditioning purely from per-frame noise
  ([[song2025history]]). Causality and per-frame noise are separate choices.
