---
kind: experiment
slug: "mg3-memory-ablation"
date: "2026-11-23"
status: done     # running | done | abandoned
hypothesis: "MG3's camera-FOV retrieval memory keeps revisited views consistent long after the attention window; with retrieval disabled the same model forgets like MG2"
result: "Confirmed: with retrieval memory MG3 recognisably returns to its start view after a 180-degree turn (median 92 SIFT inliers, 7/8 runs) while memory-off MG3 and MG2 return to an unrelated place (5-7 inliers, 0/8); LPIPS understates the effect"
related_concepts: ["Novel View Prediction", "Exposure Bias", "Bidirectional vs Causal Video Diffusion"]
related_literature: ["zhang2025matrix", "song2025history", "zhou2025stable"]
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

64 rollouts: 4 images × 2 seeds × T ∈ {24, 48, 96, 120} × memory {on, off},
at 704×1280 on one RTX 5080. MG2 numbers come from
`2026-11-23-mg2-memory-horizon`, re-scored with the same metrics.

**Place recognition** (`geomatch_return.py`): SIFT keypoints, ratio test,
RANSAC-homography inliers between the start frame and the return frame.
Unrelated scenes score median 6, max 17.

| | T=24 | T=48 | T=96 | T=120 (180° and back) |
|---|---|---|---|---|
| MG3 memory **on**: median inliers (runs ≥30) | 290 (8/8) | 112 (6/8) | 128 (6/8) | **92 (7/8)** |
| MG3 memory **off** | 290 (8/8) | 20 (3/8) | 7 (0/8) | 5 (0/8) |
| MG2 (T=48 / 96 / 192) | 60 at T=24 | 8.5 (0/12) | 5 (0/12) | 6 at T=192 (0/12) |

T=24 fits in MG3's first 57-frame window, so on and off are the same run.
The ≥30 threshold was chosen after seeing the data. It sits above every
unrelated pair (max 17), and the medians tell the same story without it.

**Pixel metrics understate it.** Return-vs-start LPIPS is 0.45 / 0.49 / 0.49
(on) vs 0.49 / 0.54 / 0.56 (off) at T=48 / 96 / 120. Memory is better in all
24 paired comparisons, but only by 0.05 on average. The coarse-layout colour
distance gives 26.8 vs 32.9 ΔE at T=120 (unrelated 51.5; `layout.json`).
Seeds converge with memory on: LPIPS between two seeds' return frames at
T=120 is 0.22 with memory on and 0.47 with it off.

**Why LPIPS misses it.** Contact sheets for all four images show memory-on
returning to the same objects (the GAS sign and orange wall, the gravestone
tree, the fence with its log pile), but a few degrees off and re-textured.
MG3 integrates +yaw at about 10.4 px/frame and -yaw at about 9.7, so the
return heading lands 80-ish px short, and each render re-synthesises the
texture. A shift-registered LPIPS (`aligned_return.py`, `src/wm/aligned.py`)
did not fix it: phase correlation locks onto spurious shifts and made the
T=24 score worse. That metric is recorded as a failed attempt.

**Engineering.** MG3 does not fit 16 GB as released. What made it fit
(`src/wm/mg3.py`): T5 on the CPU; int8 for every large DiT linear *except*
the action module; token-chunked FFN; the VAE encoder parked on the CPU; and
the last 10 blocks streamed from pinned CPU memory. It runs at about 1.8 fps
at 720p with an 11-12 GB peak. One bug cost a launch: int8 with per-token
activation quantization on the action MLP rounded the ±0.1 mouse signal to
zero, so the camera never turned (LPIPS 0.003, `results/raw_invalid_int8_action.jsonl`).

## Interpretation

The hypothesis holds, and the effect is much bigger than the pixel metrics
suggest. Pose-indexed retrieval of a few past latents (5 per window,
chosen by camera field-of-view overlap) is enough for MG3 to recognisably
return to a place after a 180° turn and back. The same model with retrieval
off forgets as completely as MG2. Its memory is *place*, not pixels: objects
and layout come back, texture and exact framing don't.

This is the explicit-vs-implicit-3D question from [[welch2026fei-fei-li-atlas]]
in miniature. Neither model has a 3D state. MG3 gets consistency by indexing
its own past frames by camera pose, which is a lightweight explicit spatial
index. MG2 has nothing past ~1 s.

## Diagnostics

Metric numbers reference `metrics.json` (no held-out split exists for these probes).

- intended_effect_confirmed: yes — place-recognition inliers at T=120: memory on median 92.5 (7/8 ≥30) vs off 5 (0/8), unrelated median 6 (metrics.json:geomatch_median_MG3_mem*_T120, geomatch_unrelated_*)
- leakage_check: n/a — frozen public checkpoint probed with scripted actions; no training or holdout
- overfitting_signal: train=n/a val=n/a gap=n/a — no training (metrics.json)
- delta_from_prior: vs 2026-11-23-mg2-memory-horizon, long-pan return inliers 5-8.5 (MG2) → 92.5 (MG3 memory on), attributed to retrieval memory, because MG3 memory-off matches MG2 (metrics.json:geomatch_median_MG2_on_T192 = 6)
- unexpected_findings: LPIPS shows only a 0.05 memory benefit while geometric matching shows a ~15× inlier gain; the ≥30 recognition threshold was set post hoc (above the unrelated max of 17)
- next_candidates:
  - Loop-closure stress test: walk forward and back along a path (translation, not just rotation) to see if MG3's memory holds under parallax, where FOV retrieval is harder.
  - Fix the ±yaw asymmetry by integrating poses from the rendered frames rather than the action table, then re-score with LPIPS to separate heading error from content error.
## Follow-up

- Full videos in `results/videos/` (640x352, gitignored); contact sheets and clips are in the project report.
