---
kind: experiment
slug: "mg2-context-noise-drift"
date: "2026-11-23"
status: done     # running | done | abandoned
hypothesis: "Re-noising cached context (context_noise > 0) trades per-frame sharpness for slower long-horizon drift in MG2"
result: "Partly confirmed: context noise 100-200 cuts darkness collapse from 53% to 8-22% of frames, but trades it for a saturated, detail-poor abstract world (colourfulness x1.8-2.9, sharpness x0.36-0.80); no setting keeps the start scene past ~15 s; 400 is worse again (U-shape)"
related_concepts: ["Exposure Bias", "Diffusion Forcing"]
related_literature: ["chen2024diffusion", "huang2025self", "zhang2025matrix"]
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

20 rollouts of 2,397 frames (96 s at 25 fps; upstream demos are 597):
2 images (stylized village 0013, real road 0005) × 2 seeds × context noise
∈ {0, 50, 100, 200, 400}, all with one fixed walking action track (W held,
sinusoidal yaw, alternating strafes; `results/actions.npz`). Each run took
283-287 s on the RTX 5080 (8.4 fps). Traces are in `results/traces.json`;
ratios are late (≥ 80 s) means over the first-2-s mean of the same run.

| context noise | dark frames | dark frames after 48 s | sharpness, late/start | colourfulness, late/start | 1-s LPIPS change, late |
|---|---|---|---|---|---|
| 0 | 53% | 67% | 0.34 | 0.16 | 0.15 |
| 50 | 43% | 65% | 0.11 | 0.05 | 0.06 |
| 100 | 22% | 25% | 0.80 | 1.82 | 0.24 |
| 200 | 8% | 5% | 0.36 | 2.92 | 0.26 |
| 400 | 39% | 37% | 0.26 | 1.52 | 0.23 |

"Dark" means mean grey below 20/255. n = 4 runs per setting, with large
between-run spread (`summary.json:per_run_dark_frac`), so treat this as
exploratory.

## Interpretation

The prediction was that context noise would slow drift at a cost in detail.
It holds only partly. What actually happens is that **every setting loses
the start scene within about 15 s**, and context noise chooses *which
attractor* the rollout falls into:

- **0 (upstream default):** a *darkness attractor*. The walk drifts into an
  interior or under an eave, the frame goes black, and black context makes
  more black the easiest prediction. More than half of all frames are black
  (village: 76-88%). This is [[Exposure Bias]] in its purest form: a fixed
  point of the model's own output distribution.
- **100-200:** the model gets out of the dark (dark frames 8-22%) but slides
  into a saturated, flat-shaded "abstract game" world. Colourfulness rises
  1.8-2.9× and sharpness drops to 0.36-0.80× of the start. Re-noising the
  context helps against black-frame lock-in, as [[Diffusion Forcing]] would
  suggest, and at the same time strips the detail that would keep the scene
  anchored.
- **400:** murky low-contrast frames come back (dark 39%). Too much noise and
  the cached context stops carrying information.

So the noise knob is real and U-shaped, but it can't replace training-time
robustness. MG2 was distilled with Self Forcing on clean self-generated
context ([[huang2025self]]), and dialling noise in at inference trades one
failure mode for another. That is consistent with Self Forcing's argument that
noisy-context conditioning trades away consistency.

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

- intended_effect_confirmed: partial — dark-frame share 0.53 (cn 0) → 0.08 (cn 200) → 0.39 (cn 400), but colourfulness late ratio 2.9× at cn 200 (metrics.json:cn*_dark_frac, cn200_colour_late_ratio)
- leakage_check: n/a — frozen public checkpoint, no training or holdout
- overfitting_signal: train=n/a val=n/a gap=n/a — no training (metrics.json)
- delta_from_prior: vs 2026-11-23-mg2-memory-horizon — the 6/120 dark frames seen there are the onset of the darkness attractor that dominates 96-s rollouts at the default setting (results/summary.json:0.per_run_dark_frac)
- unexpected_findings: the U-shape (400 worse than 200) and the switch from a dark to a saturated attractor; the real-road start (0005) seed 0 never went dark at cn 0 but did at cn 50 (72%), so per-run variance is large
- next_candidates:
  - Apply context noise only to the oldest cached block (keeping the newest clean) to test whether the saturation cost comes from noising recent context.
  - Measure the same long-rollout attractors in MG3, whose memory retrieval re-injects early clean frames and may pull rollouts back toward the start scene.
## Follow-up

- 2x2 time-lapse clip in the project report; full videos in `results/videos/` (gitignored, about 50 MB each).
