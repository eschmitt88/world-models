---
kind: experiment
slug: "mg3-memory-ablation"
date: "2026-11-23"
status: running     # running | done | abandoned
hypothesis: "MG3's camera-FOV retrieval memory keeps revisited views consistent long after the attention window; with retrieval disabled the same model forgets like MG2"
result: ""
related_concepts: ["Novel View Prediction", "Exposure Bias", "Bidirectional vs Causal Video Diffusion"]
related_literature: []
tags: [matrix-game-3, open-model-trial, memory, ablation]
# members: only set when kind: ensemble — list parent experiment slugs.
# parent:  only set when this experiment was produced via /propose --expand.
---

# mg3-memory-ablation

## Hypothesis

Matrix-Game 3.0 (Skywork, ~6.5B, Wan2.2-based, distilled to 3 steps) generates
in 56-frame windows. Each window also attends to 5 **memory latents** retrieved
from the whole history by camera field-of-view overlap with the current pose
(`select_memory_idx_fov`; the last slot is pinned to the first frame), with
Plücker-ray embeddings of their relative pose. MG2 has no such mechanism and,
in `2026-11-23-mg2-memory-horizon`, forgot a view completely once the round
trip exceeded ~4 s (LPIPS at the unrelated-scene level).

**Prediction.** Same out-and-back protocol (yaw right for T frames, then back):
with memory ON, the return-to-start LPIPS stays well below the
unrelated-scene and seed-spread reference levels at every T up to a 180° turn.
With memory OFF (x_memory=None, the path upstream uses for its CFG null
branch), return LPIPS climbs to the reference level as in MG2. If ON ≈ OFF,
then the retrieval memory isn't what keeps MG3 consistent (or doesn't work at
this length).

## Setup

- Config: `config.yaml`
- Code: `src/wm/mg3.py` (T5 on CPU, int8 q/k/v/o, memory flag), `run.py`, `analyze.py`
- Model: `Skywork/Matrix-Game-3.0` base_distilled_model, upstream commit 51913b2
- Data: MG3 `demo_images/00{1,4,7}` and `010` with their prompts (no holdout; not HCE)

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
