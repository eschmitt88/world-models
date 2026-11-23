---
kind: experiment
slug: "mg2-memory-horizon"
date: "2026-11-23"
status: running     # running | done | abandoned
hypothesis: "MG2's spatial memory is bounded by its KV attention window: after an out-and-back camera pan, revisited views match the original only when the time gap is within the window"
result: ""
related_concepts: ["Exposure Bias", "Bidirectional vs Causal Video Diffusion", "Novel View Prediction"]
related_literature: []
tags: [matrix-game-2, open-model-trial, rollout]
# members: only set when kind: ensemble — list parent experiment slugs.
# parent:  only set when this experiment was produced via /propose --expand.
---

# mg2-memory-horizon

## Hypothesis

Matrix-Game 2.0 (Skywork, 1.8B causal Wan-based, distilled to 3 steps) attends
over a sliding window of `local_attn_size = 6` latent frames (~21-24 video
frames, about 1 s at 25 fps; `configs/distilled_model/universal/config.json`).
Its only longer-range memory is the CLIP embedding of the start image,
injected by cross-attention (`inference.py`, `visual_context`).

**Prediction.** In an out-and-back pan (camera right for T frames, then left for
T frames), the frames at `t_peak - k` and `t_peak + k` should show the same view.
Their perceptual distance (LPIPS) should stay low while the gap `2k` is under
about 24 frames, then rise steeply toward the *unrelated-scene* baseline (LPIPS
between different seeds at the same timestep). The return frame (k = T) should
match frame 0 only for small T. A CLIP-level "same place, different
details" plateau above the noise floor would mean the image embedding carries
coarse memory past the window.

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
