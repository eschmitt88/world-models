---
kind: concept
name: "Exposure Bias"
status: seedling
added: "2026-11-22"
sources: [welch2026fei-fei-li-atlas]
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

## Connections

- [[Diffusion Forcing]]: the noise-augmented-context fix
- Self Forcing / CausVid: train on the model's own rollouts (to ingest)
- Scheduled sampling: the older sequence-modeling fix
