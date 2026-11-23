---
kind: concept
name: "Real-to-Sim-to-Real"
status: seedling
added: "2026-11-22"
sources: [welch2026fei-fei-li-atlas, worldlabs-ai-atlas, assran2025vjepa]
related_concepts: ["Latent-Predictive World Models", "Novel View Prediction"]
related_experiments: []
tags: [robotics, simulation, reconstruction, sim-to-real]
---

# Real-to-Sim-to-Real

## Definition

Reconstruct a real scene from a few casual captures, lift it into a fast
physics simulator, train a policy there on many counterfactual rollouts, then
deploy it back on the real robot.

## Why it matters here

It is the robotics payoff a spatial world model is pitched at. World Labs'
July 2026 demo used three iPhone videos → Atlas reconstruction → physics sim
→ RL → real hardware ([[welch2026fei-fei-li-atlas]]; company demo,
unverified). It shows the explicit-vs-implicit tension around
[[Novel View Prediction]]: the simulator needs explicit geometry even when
the generative model doesn't.

## Connections

- [[Novel View Prediction]]: supplies the reconstruction
- `../ai-robotics`: sim-to-real gap, GPU-parallel simulation, and the
  Dreamer-style "world model" (latent dynamics for RL, a different sense of
  the term)

## What the primary papers add

- A different real-to-sim loop from Atlas's reconstruct-then-simulate: V-JEPA 2-AC
  plans in embedding space, post-trained on under 62 hours of unlabelled robot
  video, and runs zero-shot on Franka arms in two labs with image goals. It needs
  16 s per action against 4 min for an action-conditioned Cosmos baseline, and is
  sensitive to camera placement ([[assran2025vjepa]]; see
  [[Latent-Predictive World Models]]).
