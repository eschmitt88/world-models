---
kind: experiment
slug: "mg2-context-noise-drift"
date: "2026-11-23"
status: running     # running | done | abandoned
hypothesis: "Re-noising cached context (context_noise > 0) trades per-frame sharpness for slower long-horizon drift in MG2"
result: ""
related_concepts: ["Exposure Bias", "Diffusion Forcing"]
related_literature: []
tags: [matrix-game-2, open-model-trial, rollout]
# members: only set when kind: ensemble — list parent experiment slugs.
# parent:  only set when this experiment was produced via /propose --expand.
---

# mg2-context-noise-drift

## Hypothesis

MG2 caches each generated block back into its KV cache by re-running the
model at timestep `context_noise` (`pipeline/causal_inference.py`, step 3.3).
The default is 0, which conditions on the model's own outputs as if they were
clean. That is the setup where [[Exposure Bias]] predicts compounding drift.
Diffusion forcing and Self Forcing argue that conditioning on slightly noised
context makes the model robust to its own errors.

**Prediction.** Over a long rollout (~2,400 frames, ~100 s, which is far past
the 150-latent demos), `context_noise = 0` drifts: sharpness collapses or
saturation and contrast blow up, and the scene loses structure. Moderate
values (~50-150 out of 1000) slow the drift at some cost in fine detail.
Large values (≥ 400) lose temporal coherence. If MG2 was distilled only with
clean context, any nonzero value may simply hurt. That would also be
informative, because it says the robustness has to be trained in, not dialled
in at inference.

## Setup

- Config: `config.yaml`
- Code: `src/wm/mg2.py` (loads MG2 once, runs scripted action episodes), `run.py` in this folder
- Model: `Skywork/Matrix-Game-2.0` base_distilled_model, upstream commit 51913b2 (vendor/, gitignored)
- Data: start images from MG2 `demo_images/universal/` (no holdout; not an HCE project)

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
