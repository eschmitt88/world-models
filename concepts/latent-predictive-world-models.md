---
kind: concept
name: "Latent-Predictive World Models"
status: seedling
added: "2026-11-23"
sources: [assran2025vjepa]
related_concepts: ["Latent Action Model", "Real-to-Sim-to-Real", "Exposure Bias"]
related_experiments: []
tags: [jepa, v-jepa, representation-space, planning]
---

# Latent-Predictive World Models

## Definition

World models that predict the next **representation** produced by a frozen
or jointly trained encoder, not pixels. V-JEPA 2 is trained with a mask-denoising
objective in latent space on over 1M hours of video, then an
action-conditioned predictor (V-JEPA 2-AC) is post-trained on frozen features
([[assran2025vjepa]]).

## Why it matters here

They give up viewable rollouts for cheaper planning: the paper reports
16 s per action against 4 min for an action-conditioned Cosmos diffusion
baseline on the same robot tasks, plus zero-shot Franka pick-and-place from
image goals. The cost is that nothing renders, so the project cannot
eyeball drift as it can for pixel models, and the paper reports sensitivity
to camera placement. A fair project question is how embedding-space rollouts
drift with horizon, i.e. whether [[Exposure Bias]] applies to them.

## Connections

- [[Latent Action Model]]: how actions get attached cheaply
- [[Real-to-Sim-to-Real]]: the robotics setting
- Dreamer-style latent dynamics belong to `../ai-robotics`
