---
name: index
description: Entry-point index for this project's knowledge graph.
---

# Index

Orientation for the project knowledge graph. Updated by `/wrap`, `/ingest`,
and `/new-experiment`.

## Maps of Content

(promote a cluster of ≥5 related concepts into `mocs/<theme>.md`)

- *Candidate:* **autoregressive world-model architecture**: Diffusion Forcing,
  Exposure Bias, Bidirectional vs Causal Video Diffusion, Novel View
  Prediction (+ Real-to-Sim-to-Real). Five concepts, all from one secondary
  source; promote once the primary papers are in.

## Active experiments

- `experiments/2026-11-23-mg2-memory-horizon/`: how long MG2 remembers a view
- `experiments/2026-11-23-mg2-context-noise-drift/`: context-noise knob vs long-rollout drift

## Open questions

- Explicit vs implicit 3D: does NVS-style implicit geometry hold up when a
  downstream simulator needs explicit geometry?
- Which open checkpoints are genuinely causal/interactive vs bidirectional,
  and how fast does each drift on 16 GB?
