---
kind: concept
name: "Bidirectional vs Causal Video Diffusion"
status: seedling
added: "2026-11-22"
sources: [welch2026fei-fei-li-atlas]
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
