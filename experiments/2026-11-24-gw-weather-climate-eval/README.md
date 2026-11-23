---
kind: experiment
slug: "gw-weather-climate-eval"
date: "2026-11-24"
status: running     # running | done | abandoned
hypothesis: "Short-horizon accuracy does not determine long-horizon climate (DF vs TF or snap vs no-snap dissociate on weather vs climate), and one-step contraction gain predicts rollout survival across checkpoints"
result: ""
related_concepts: ["Exposure Bias", "Diffusion Forcing"]
related_literature: ["chen2024diffusion", "huang2025self"]
tags: [gridworld, weather-crossover, calibration, climate, contraction]
# members: only set when kind: ensemble — list parent experiment slugs.
# parent:  only set when this experiment was produced via /propose --expand.
---

# gw-weather-climate-eval

## Hypothesis

Pre-registered from ADR 0002 (weather-ML crossover). It evaluates tonight's
DF and TF checkpoints (`2026-11-23-gw-drift-df-vs-tf`) with no new training.

Predictions:
1. **Dissociation:** at least one pair of arms (DF vs TF, context noise 0 vs
   the DF optimum, snap vs no snap) orders differently on weather (skill
   horizon) than on climate (JS distance of imagined-cell statistics from real).
2. **Climate drifts before collapse:** climate distance rises above the
   real-sample noise floor several frames *before* determined-cell skill
   reaches zero, so it is an early warning.
3. **Calibration:** per-cell ensemble disagreement on determined cells
   correlates positively with error.
4. **Contraction predicts survival:** across TF snapshots, a lower one-step
   gain goes with longer survival (later off-manifold or uniform-frame entry).
5. **Snap** improves survival and the off-manifold rate but does not fix
   accuracy, because the dominant failure is legal frames in the wrong place.

The kill criteria are in ADR 0002. Any one drops the framing.

## Setup

- Config: `config.yaml`
- Code: `src/wm/climate_eval.py`
- Data: held-out maps 2000-2199, generated on the fly

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
