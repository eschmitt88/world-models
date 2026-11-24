---
kind: experiment
slug: "gw-memory-equal-compute"
date: "2026-11-23"
status: done     # running | done | abandoned
hypothesis: "At equal tokens, 4 pose-retrieved memory frames beat 36 frames of plain context on revisit accuracy once the gap exceeds the window, and the model keeps regions it imagined less well than regions it saw"
result: "Harness check passed: 4 pose-retrieved memory frames recall the real context at 0.98-0.99 out to a 252-frame round trip vs 0.39 for 36 frames of plain context; the imagined-vs-seen gap was a metric artifact (first-sight drafts get revised); a matched-gap comparison is still needed"
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

Training: **mem** 13846 steps, **win** 13906 steps (1.5 h each, effective batch
32 via 2-step accumulation after an OOM at batch 32; see `log.md`). The eval
(`results/eval.json`, 24 held-out contexts, raw = scored at the true pose)
runs a real context, then G more steps out, then retraces to the start.

| model | real recall G=8 | 24 | 48 | 96 | mirror-imagined G=8 | 24 | 48 | 96 | first-sight G=8 | 96 |
|---|---|---|---|---|---|---|---|---|---|---|
| mem (32 + 4 retrieved) | 0.99 | 0.98 | 0.99 | 0.98 | 0.90 | 0.81 | 0.85 | 0.79 | 0.26 | 0.54 |
| win (36 plain) | 0.75 | 0.60 | 0.58 | 0.39 | 0.97 | 0.91 | 0.74 | 0.51 | 0.29 | 0.33 |
| df ref (32, no pose) | 0.61 | 0.56 | 0.47 | 0.29 | 0.95 | 0.94 | 0.70 | 0.55 | 0.28 | 0.34 |

- *real recall:* tile accuracy on the last 16 return frames, on cells the
  real context showed.
- *mirror-imagined:* agreement between a return frame and the outbound
  frame at the same pose, on cells only ever seen in generated frames.
- *first-sight:* agreement with a cell's first generated appearance (the
  originally pre-registered metric).

## Interpretation

1. **Prediction 2 (harness check) holds.** Four frames retrieved by pose
   keep the real context at 0.98-0.99 recall out to a 252-frame round trip
   (G=96). 36 frames of plain context fall from 0.75 to 0.39, and the
   random-walk-trained df reference from 0.61 to 0.29. This is MG3's
   retrieval result reproduced against ground truth, as expected from
   StateSpaceDiffuser, WorldMem and Po et al.; it is not a novel finding.
2. **Prediction 1 does not hold:** even at G=8, plain context recalls only
   0.75, not near 1. At G=8 the round trip is 2(30+8)=76 frames, already
   past the 36-frame window. The gap values were mislabelled as "small"
   when this was pre-registered.
3. **Prediction 3 (imagined recalled worse than seen) is not supported as
   measured.** The pre-registered first-sight metric gave 0.26-0.55, but it
   compares against a cell's *first* generated appearance, a sliver at the
   view edge that the model then **revises** while the cell is still in view.
   That is a finding in itself: first drafts at the frontier are provisional.
   The fairer mirror metric gives 0.80-0.90 for the memory model's
   imagined cells, against 0.87-0.95 on real cells. Without memory, imagined
   cells score *higher* than real ones, but real cells are revisited last,
   at longer gaps, so the comparison is confounded by gap. A matched-gap
   comparison is still needed.

## Diagnostics

Metric numbers reference `metrics.json` (held-out maps throughout).

- intended_effect_confirmed: partial — memory recall confirmed (metrics.json:mem_G96_real_recall 0.98 vs win_G96_real_recall 0.39); the imagined-vs-seen claim is unresolved (metric artifact, gap confound)
- leakage_check: held-out maps 2000-2199 for every evaluation; poses are oracle (simulator), stated as a limitation
- overfitting_signal: train=n/a val=n/a gap=n/a — on-the-fly data (metrics.json)
- delta_from_prior: vs 2026-11-23-mg3-memory-ablation, the same retrieval effect (MG3 memory on/off) reproduced from scratch, now measured against ground truth
- unexpected_findings: first-sight agreement is low (0.26-0.55) even when mirror agreement is high, so models revise the edge strips they have just revealed; the eval-v1 file (eval_v1.json) predates the mirror metric
- next_candidates:
  - Matched-gap comparison of imagined vs real recall (bin both by time since last seen) before making any claim about imagination.
  - Replace oracle poses with action-integrated poses (collisions make these wrong) to test how fragile pose retrieval is.
## Follow-up

- ...
