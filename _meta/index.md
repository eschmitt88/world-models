---
name: index
description: Entry-point index for this project's knowledge graph.
---

# Index

Orientation for the project knowledge graph. Updated by `/wrap`, `/ingest`,
and `/new-experiment`.

## Maps of Content

(promote a cluster of ≥5 related concepts into `mocs/<theme>.md`)

- [[mocs/interactive-world-models|Interactive World Models]]: causal vs
  bidirectional, diffusion forcing, exposure bias, few-step distillation,
  latent actions (5 concepts).
- Not yet ripe: a **weather-climate evaluation** cluster (Long-Rollout Evaluation, Conservation by Construction), 2 concepts.
- Not yet ripe: a **spatial / representation** cluster (Novel View Prediction,
  Latent-Predictive World Models, Real-to-Sim-to-Real, Spatial Memory), 4 concepts.

## Active experiments

- `experiments/2026-11-23-gw-drift-df-vs-tf/`: our own 2D world, diffusion forcing vs teacher forcing, drift against true state
- `experiments/2026-11-23-gw-memory-equal-compute/`: pose retrieval vs plain context at equal tokens; imagined vs seen recall

## Finished experiments

- `experiments/2026-11-23-mg2-memory-horizon/`: MG2 forgets a view once the round trip exceeds its ~1 s window
- `experiments/2026-11-23-mg3-memory-ablation/`: MG3 retrieval memory recognisably returns after 180° (7/8 vs 0/8 off)
- `experiments/2026-11-23-mg2-context-noise-drift/`: context noise picks the long-rollout attractor (dark / saturated / murky)

## Reports

- `reports/2026-11-23-first-trials/`: "Look Away, Look Back" (published artifact)

## Open questions

- Explicit vs implicit 3D: does NVS-style implicit geometry hold up when a
  downstream simulator needs explicit geometry?
- Which open checkpoints are genuinely causal/interactive vs bidirectional,
  and how fast does each drift on 16 GB? (MG2: collapses within ~15 s at default settings.)
- Does pose-indexed retrieval memory (MG3) survive translation and parallax, not just rotation?
