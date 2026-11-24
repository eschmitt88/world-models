---
kind: concept
name: "Diffusion World Model"
status: seedling
added: "2026-11-24"
sources: [alonso2024diffusion, savov2025statespacediffuser]
related_concepts: ["Exposure Bias", "Diffusion Forcing", "Few-Step Distillation for Real-Time Rollout", "Spatial Memory"]
related_experiments: []
tags: [diffusion, edm, pixel-space, small-scale, frame-stacking]
---

# Diffusion World Model

## Definition

An action-conditioned model that predicts the next observation by running a
reverse diffusion process in pixel (or image-latent) space, conditioned on
recent frames and actions, and is rolled out autoregressively as a simulator.

## Why it matters here

The project trains its own causal DiT this way on one 16 GB GPU, so the
design choices that make the rollout stable are the ones to copy or test.
DIAMOND is the small-scale reference: an EDM-parametrised U-Net needs only 3
denoising steps per frame and stays stable to 1000 steps, where a DDPM
parametrisation at the same step count drifts ([[alonso2024diffusion]]).
Its only memory is a stack of the last few frames, which the authors list
as a limitation. [[savov2025statespacediffuser]] uses DIAMOND as its generative
branch and adds long context to it; see [[Spatial Memory]].

## Connections

- [[Exposure Bias]]: stability here comes from the sampler and
  parametrisation, with teacher-forced training unchanged.
- [[Diffusion Forcing]]: the training-side alternative.
