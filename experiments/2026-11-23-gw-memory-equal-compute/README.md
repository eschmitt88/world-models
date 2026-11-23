---
kind: experiment
slug: "gw-memory-equal-compute"
date: "2026-11-23"
status: running     # running | done | abandoned
hypothesis: "At equal tokens, 4 pose-retrieved memory frames beat 36 frames of plain context on revisit accuracy once the gap exceeds the window, and the model keeps regions it imagined less well than regions it saw"
result: ""
related_concepts: ["Exposure Bias", "Novel View Prediction"]
related_literature: ["zhang2025matrix", "song2025history"]
tags: [gridworld, from-scratch, memory, imagination, ground-truth]
# members: only set when kind: ensemble — list parent experiment slugs.
# parent:  only set when this experiment was produced via /propose --expand.
---

# gw-memory-equal-compute

## Hypothesis

Same world and DiT as `2026-11-23-gw-drift-df-vs-tf`, trained on
*revisit walks*: walk out, retrace exactly, carry on. Two arms with the same
tokens per sequence (36 frames) and the same wall clock (1.75 h):
**win**, 36 frames of plain context; **mem**, 32 frames plus 4 frames
retrieved from older history by oracle pose (MG3's mechanism in miniature).
Both have a relative-pose embedding.

Eval (`python -m wm.train_mem --eval`): 31 real frames, then the model walks
G more steps out and retraces all the way back.
- **real_acc_end:** on the final 16 return frames, tile accuracy on cells the
  real context showed (memory scored against truth).
- **imag_cons:** on earlier return frames, agreement with the model's *own*
  first decoded tile for cells it only ever saw in frames it generated.

Predictions:
1. With small gaps (G = 8) both arms recall the context, since it is still
   in the window.
2. With G ≥ 24 (round trip > 36 frames), **win** falls to its map-prior level
   and **mem** stays high (MG2 vs MG3, reproduced with ground truth). This is
   the harness check, not a novel finding.
3. **imag_cons < real_acc for mem at equal gap:** imagined regions are
   recalled worse than seen ones (Fable's experiment 3; no prior work found).
   If they are equal, retrieval treats the model's own frames as reliably as
   real ones.

## Setup

- Config: `config.yaml`
- Code: `src/wm/train_mem.py` (training and `--eval`), `src/wm/df.py` (`rollout_mem`, `retrieve`), `src/wm/gridworld.py` (`revisit_walks`)
- Data: generated on the fly; train maps 0-1999, held-out maps 2000-2199

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
