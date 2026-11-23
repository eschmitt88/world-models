---
kind: experiment
slug: "gw-drift-df-vs-tf"
date: "2026-11-23"
status: running     # running | done | abandoned
hypothesis: "In a small world with known state, drift exists for a clean-context (teacher-forced) model and a diffusion-forcing model is robust to it; DF has its best long-rollout accuracy at a nonzero inference context noise while TF is only hurt by context noise"
result: ""
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

Fill in after the run. Point at `metrics.json` (validation split — this
is the search signal and the file every other skill reads). A separate
`final_metrics.json` holds held-out test-split numbers and is written
only by the `dvc repro final_eval` pass at chain end. See
`~/claude-system/claude/rules/evaluation.md`.

## Interpretation

What did you actually learn? What surprised you?

## Diagnostics

Fill in after the run. One line per field; leave `n/a` rather than
blank. `next_candidates` must list ≥2 concrete one-sentence proposals.
Every concrete claim below needs a **citation anchor** — a code
reference like `train.py:42-58`, a metrics file path like
`metrics.json:val_acc`, or a wikilink into `literature/`. Unanchored
assertions are flagged by `/lint` (Kosmos, arXiv 2511.02824).

Unless otherwise noted, metric numbers here reference `metrics.json`
(validation split). Cite `final_metrics.json` only if this experiment
is itself the final-scoring pass.

- intended_effect_confirmed: <yes | no | partial> — <one-line evidence with anchor>
- leakage_check: <method used> — <finding>
- overfitting_signal: train=<x> val=<y> gap=<z> — <interpretation> (from metrics.json)
- delta_from_prior: vs <related_prior_slug>, <metric_delta> attributed to <cause> (metrics.json)
- unexpected_findings: <one or two sentences, or "none">
- next_candidates:
  - <one-sentence proposal 1>
  - <one-sentence proposal 2>

## Follow-up

- ...
