---
kind: concept
name: "Spatial Memory"
status: seedling
added: "2026-11-24"
sources: [savov2025statespacediffuser, alonso2024diffusion]
related_concepts: ["Diffusion World Model", "Exposure Bias", "Novel View Prediction", "Bidirectional vs Causal Video Diffusion"]
related_experiments: ["2026-11-23-gw-memory-equal-compute", "2026-11-23-mg2-memory-horizon", "2026-11-23-mg3-memory-ablation"]
tags: [memory, revisit, long-context, consistency, state-space-model]
---

# Spatial Memory

## Definition

A world model's ability to reproduce what it has already shown or seen of a
place when the agent returns to it after a long gap, beyond its short
context window.

## Why it matters here

Most diffusion world models condition on a few recent frames, so a revisit
after more than the window length is generated afresh. Our earlier trials found
this for Matrix-Game 2.0 (about 1 s window) and a retrieval memory partly
fixing it in Matrix-Game 3 (see the experiments above). Two published
designs bracket the options:

- Frame stacking only: DIAMOND's whole memory is the last few frames, which
  its authors list as a limitation ([[alonso2024diffusion]]).
- A recurrent state: [[savov2025statespacediffuser]] adds a Mamba state
  over the full history and recalls a marker colour across 50 to 150 steps on
  MiniGrid, but with ground-truth history, not free rollout.

The project's `gw-memory-equal-compute` experiment tests retrieval against
plain context on revisits in our own gridworld, where the true state is
known.

## Connections

- [[Diffusion World Model]]: the short-window baseline this extends.
- [[Exposure Bias]]: recall measured under free rollout also needs the
  rollout to stay stable; the MiniGrid protocol does not test that.
- [[Novel View Prediction]]: persistent-scene models (Atlas, Marble) take the
  explicit-geometry route to the same problem.
