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
- [[mocs/long-rollout-stability|Long-Rollout Stability]]: weather vs climate
  evaluation, ensemble calibration, spectral diagnostics, conservation by
  construction, distributional rollout training (5 concepts; ADR 0002's literature).
- Not yet ripe: a **spatial / representation** cluster (Novel View Prediction,
  Latent-Predictive World Models, Real-to-Sim-to-Real, Spatial Memory), 4 concepts.

## Active experiments

(none)

## Finished experiments

- `experiments/2026-11-23-gw-drift-df-vs-tf/`: tf wins short, collapses long; df exact for 384 frames at context noise 0.1
- `experiments/2026-11-24-gw-weather-climate-eval/`: two failure modes (off-manifold vs on-manifold climate collapse); Gaussian contraction refuted
- `experiments/2026-11-23-gw-memory-equal-compute/`: 4 retrieved frames hold recall at 0.98 vs 0.39 plain; imagination claim unresolved

- `experiments/2026-11-23-mg2-memory-horizon/`: MG2 forgets a view once the round trip exceeds its ~1 s window
- `experiments/2026-11-23-mg3-memory-ablation/`: MG3 retrieval memory recognisably returns after 180° (7/8 vs 0/8 off)
- `experiments/2026-11-23-mg2-context-noise-drift/`: context noise picks the long-rollout attractor (dark / saturated / murky)

## Reports

- `reports/2026-11-24-small-world/`: "Small World, Long Rollouts" (published artifact)

- `reports/2026-11-23-first-trials/`: "Look Away, Look Back" (published artifact)

## Open questions

- Does on-manifold perturbation recovery predict long-horizon climate survival across ≥6 world models? (ADR 0002)
- Do imagined regions get remembered worse than seen ones at matched gap?

- Explicit vs implicit 3D: does NVS-style implicit geometry hold up when a
  downstream simulator needs explicit geometry?
- Which open checkpoints are genuinely causal/interactive vs bidirectional,
  and how fast does each drift on 16 GB? (MG2: collapses within ~15 s at default settings.)
- Does pose-indexed retrieval memory (MG3) survive translation and parallax, not just rotation?
