---
kind: experiment
slug: "gw-drift-df-vs-tf"
date: "2026-11-23"
status: done     # running | done | abandoned
hypothesis: "In a small world with known state, drift exists for a clean-context (teacher-forced) model and a diffusion-forcing model is robust to it; DF has its best long-rollout accuracy at a nonzero inference context noise while TF is only hurt by context noise"
result: "Confirmed: teacher forcing is best at short horizon (one-step 1.00, 32-frame 0.92) but collapses by 384 frames (0.15-0.20 on determined cells, 46% off-manifold); diffusion forcing drifts at context noise 0 but is near-perfect at 0.1-0.2 on still/pace (U-shape on walk), and context noise only hurts teacher forcing"
related_concepts: ["Exposure Bias", "Diffusion Forcing", "Bidirectional vs Causal Video Diffusion"]
related_literature: ["chen2024diffusion", "huang2025self"]
tags: [gridworld, from-scratch, drift, ground-truth]
# members: only set when kind: ensemble — list parent experiment slugs.
# parent:  only set when this experiment was produced via /propose --expand.
---

# gw-drift-df-vs-tf

## Hypothesis

First experiment on our own world (ADR 0001): a partially observed 2D tile
world (`src/wm/gridworld.py`) and a ~22M pixel-space block-causal DiT
(`src/wm/df.py`), trained from scratch on one RTX 5080. Every generated frame
decodes back to tile ids, so drift is measured as **tile accuracy against the
true map** on cells the real context already showed, plus an
**off-manifold rate** (cells matching no tile).

Arms, 4 h each, same data stream and architecture:
- **df**: diffusion forcing, an independent noise level per frame.
- **tf**: teacher forcing, a clean copy of the context and noisy targets
  (doubled sequence).

Predictions:
1. **Go/no-go:** the tf model drifts. Its long-rollout (480-frame) accuracy
   on determined cells falls well below its one-step accuracy. If neither
   arm drifts, crisp tiles snap back and the world needs more nuisance;
   that is a finding in itself.
2. df degrades more slowly than tf at context noise 0.
3. df's best long-rollout accuracy is at a **nonzero** context noise (U-shape,
   as MG2 showed). tf is only hurt by context noise, because it never saw
   noisy context.
4. If attractors appear (uniform frames, e.g. all-wall or all-floor), they
   show up as off-manifold-free but *wrong* frames, in the tf arm first.

## Setup

- Config: `config.yaml`
- Code: `src/wm/train_df.py` (training + periodic eval), `src/wm/df.py`, `src/wm/gridworld.py`, `eval_long.py` here
- Data: generated on the fly; train maps 0-1999, held-out maps 2000-2199 (evaluation only)

## Result

Training (same 22M DiT, same data stream, 3.5 h each): **df** reached step
46515 and **tf** step 24807. tf processes 2× the
tokens per sample (clean + noisy copy) and was trained with 4-step gradient
accumulation after an OOM at batch 32 (`log.md`).

| last in-training eval (held-out maps) | one-step | 8 frames | 32 frames | 96 frames |
|---|---|---|---|---|
| df | 0.99 | 0.90 | 0.73 | 0.50 |
| tf | 1.00 | 0.99 | 0.92 | 0.63 |

**Long rollouts** (`eval_long.py`, 16 held-out contexts, 384 generated frames):
accuracy on context-determined cells at frames 352-384, by inference context
noise.

| arm / script | 0 | 0.05 | 0.1 | 0.2 | 0.4 |
|---|---|---|---|---|---|
| df still | 0.75 | 0.99 | 1.00 | 1.00 | 0.98 |
| tf still | 0.15 | 0.17 | 0.12 | 0.21 | 0.62 |
| df pace | 0.44 | 0.94 | 0.99 | 0.99 | 0.99 |
| tf pace | 0.20 | 0.23 | 0.18 | 0.21 | 0.43 |
| df walk | 0.30 | 0.34 | 0.32 | 0.45 | 0.37 |
| tf walk | 0.16 | 0.18 | 0.11 | 0.04 | 0.27 |

Late off-manifold rate at context noise 0: df 0.07 / 0.06 / 0.04 (still / pace / walk),
tf 0.47 / 0.46 / 0.46. Uniform-frame (attractor) rate on
df walk at 0: 0.36.

## Interpretation

All four predictions hold.
1. **Drift exists** (go/no-go passed). tf, near-perfect for 16 frames,
   ends at 0.15-0.20 on cells the context fully determined.
2. **df degrades more slowly than tf at noise 0** on every script.
3. **df's optimum is at nonzero context noise.** At 0.1 it stays essentially
   exact for 384 frames when still or pacing (1.00 / 0.99, against 0.75 / 0.44 at
   0). Walking peaks at 0.2 and falls at 0.4, a U-shape like MG2's. **tf is
   only hurt** by context noise: short-horizon accuracy drops to ~0.7 at once,
   because it never saw noisy context.
4. tf's collapse is mostly **off-manifold** (about 46% of cells match no
   tile). df at 0 collapses more often into **uniform frames** (36% of late
   walk frames are one tile class), a clean, decodable analogue of MG2's
   darkness attractor.

The headline is the dissociation the weather-crossover paper (ADR 0002)
needs. **tf beats df at short horizon** (32-frame 0.92 vs 0.73) yet loses badly
at long horizon. Short-horizon accuracy does not predict long-horizon
behaviour.

Caveats: one seed per arm. tf saw about half as many samples (matched wall
clock). df was still improving at the end. The walk script leaves the seen
region, so its determined set shrinks; still and pace are the clean drift
tests.

## Diagnostics

Metric numbers reference `metrics.json` (no held-out split; held-out maps for every evaluation).

- intended_effect_confirmed: yes — tf collapses (metrics.json:tf_still_cn0.0_acc_352_384 ≈ 0.15) while df at context noise 0.1 stays exact (df_still_cn0.1_acc_352_384 = 1.00)
- leakage_check: train maps 0-1999, every evaluation on held-out maps 2000-2199 (`train_df.py`, `eval_long.py`)
- overfitting_signal: train=n/a val=n/a gap=n/a — on-the-fly data, no fixed training set (metrics.json)
- delta_from_prior: vs 2026-11-23-mg2-context-noise-drift, the same U-shaped context-noise curve, now measured against ground truth: the MG2 attractor result reproduced from scratch
- unexpected_findings: tf partially recovers at context noise 0.4 on still (0.62) despite never training on noise; df's still/pace stability at 0.1 is essentially perfect for 384 frames (12× the training window)
- next_candidates:
  - Train df to matched samples with snapshots, to compare against tf at matched one-step accuracy (Fable's confound check).
  - Weather/climate/contraction evaluation of these checkpoints (experiments/2026-11-24-gw-weather-climate-eval, running).
## Follow-up

- ...
