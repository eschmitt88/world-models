---
kind: paper
title: "WorldProbe: When Video World Models Fall Apart under Long-Horizon Spatial Exploration"
authors: ["Minqi Wang", "Jie Peng", "Hejun Dong", "Ruchang Yao", "Puyi Wang", "Siyuan Huang", "Jihai Zhang", "Yafu Li", "Qi Dou", "Xiaoye Qu", "Yu Cheng"]
institutions: ["The Chinese University of Hong Kong", "Peking University", "University of Science and Technology of China", "Shanghai Artificial Intelligence Laboratory", "Nanyang Technological University"]
year: 2026
venue: "arXiv v1 (preprint, 29 Oct 2026)"
peer_reviewed: false
url: https://arxiv.org/abs/2610.39721
code_url: https://github.com/Untitled-W/WorldProbe
citations:
source: "raw/papers/wang2026worldprobe.pdf"
added: "2026-11-24"
relevance: 4
credibility: 3
status: read
related_experiments: ["2026-11-23-gw-weather-climate-eval", "2026-11-23-gw-memory-equal-compute", "2026-11-23-mg2-memory-horizon", "2026-11-23-mg3-memory-ablation"]
related_concepts: ["Long-Rollout Evaluation", "Spatial Memory", "Exposure Bias"]
tags: [benchmark, world-model, long-horizon, memory, agentic-evaluation, matrix-game, degradation-slope, vlm-judge]
---

# WorldProbe: When Video World Models Fall Apart under Long-Horizon Spatial Exploration

## TL;DR

A benchmark that drives 10 public camera- or action-conditioned video world
models for up to 180 s with a goal-conditioned VLM agent that picks actions
online. Per-frame visual and VLM content scores are fitted with a per-rollout
slope against time. Memory and "evolution fidelity" are scored by a VLM on
revisited viewpoints. Finding: local quality degrades more slowly in newer
models, but "long-range memory and evolution remain weak", most models
scoring around or below the midpoint of a 0 to 10 scale.

## Claims

- Existing benchmarks use short rollouts, camera paths fixed before
  generation, and local pairwise metrics, so they miss long-range failures.
- "Visual and content quality significantly degrade as the rollout proceeds.
  Newer models often degrade more slowly." Earlier models "frequently exhibit
  abrupt collapse"; recent ones keep "a non-trivial quality floor".
- "All models also exhibit substantial long-range inconsistencies." Leave and
  return trajectories keep coarse geometry and lose texture; revisiting from a
  new direction "often produces an entirely different scene"; models "behave
  more like local re-renderers than systems with persistent global spatial
  memory".
- Objects split, vanish or duplicate, and static objects move with the camera.

## Methods

- 164 test cases in four families (geometry-constrained approach, open-ended
  traversal, cross-view inspection, successive region exploration), built from
  initial frames of public datasets plus hand-written task prompts. Rollout
  duration ranges are 10 to 60, 20 to 120, 20 to 90 and 30 to 180 s per family.
- Agent: Qwen3.6-27B picks one discrete action per chunk from recent frames;
  SAM3.1 tracks the target, Depth-Anything-3 gives local geometry. Rollouts end
  on task completion, tracking failure or oscillation, and are filtered by a
  minimum duration and a coarse VLM screen. Short rollouts are retried up to
  three times and the longest attempt kept.
- Models (10): Matrix-Game 2.0, WorldPlay (HY), WorldPlay (Wan), LingBot-World,
  Matrix-Game 3.0, SANA-WM, DreamX-World, AlayaWorld, LingBot-World v2,
  ABot-World. Every public pipeline was refactored to accept an action per
  chunk; the authors report black-box checks that behaviour is preserved.
- Metrics. Visual: mean of normalised MUSIQ, MANIQA and Q-Align on sampled
  frames. Content: VLM 0 to 10 ratings of geometry, object validity and scene
  richness, per frame, with no earlier frame as context. Memory: ViPE poses
  estimated from the generated video, up to five spatially separated anchors,
  up to eight frames each (within 30 degrees and 3 m, FoV overlap above 0.30,
  at least 8 s apart), scored by a VLM for scene, layout, entity and texture
  consistency. Evolution: VLM over sampled frames plus the action sequence
  (entity, scene, physics, action-outcome causality). Slope is the per-rollout
  least-squares slope against elapsed seconds, averaged with equal weights over
  rollouts.
- Compute: about four days on 24 A100 GPUs. More than 1,600 rollouts, about 800
  min.

## Results

- Table 4 (local): visual slopes run from +0.76e-3/s (Matrix-Game 3.0, the only
  positive one) to -5.48e-3/s (ABot-World); content slopes from -0.015
  points/s (LingBot-World v2) to -0.093 (AlayaWorld). Matrix-Game 2.0 has the
  second-flattest visual slope (-0.77e-3/s, after MG3). ABot-World and
  DreamX-World have the lowest visual means (0.439, 0.458).
- Table 5 (memory overall, 0 to 10): AlayaWorld 6.53, Matrix-Game 3.0 6.31,
  LingBot-World v2 5.92, DreamX-World 5.87, SANA-WM 5.62, WorldPlay (HY) 5.37,
  ABot-World 5.20, WorldPlay (Wan) 4.16, Matrix-Game 2.0 4.01, LingBot-World
  3.78. Layout, entity and texture sub-scores sit lower than scene
  (Matrix-Game 3.0: scene 7.14, layout 5.75, entity 5.81, texture 6.54).
- Evolution overall ranges from 2.51 (WorldPlay Wan) to 5.17 (DreamX-World).
- Local slope and memory do not line up: Matrix-Game 2.0 has a near-flat visual
  slope and a memory score of 4.01, while AlayaWorld has a steep content slope
  (-0.093) and the best memory (6.53).
- The probing agent ran a DreamX rollout to 367.9 s (491 actions), beyond the
  180 s cap.

## Critique / open questions

- **No ground truth.** Everything is scored by IQA models, a VLM or poses
  estimated from the generated video. There is no known generator, no
  state-level scoring, no independent real samples and so no noise floor.
  The VLM scores come from one judge; no human agreement or judge variance is
  reported in the main text.
- **No weather versus climate split.** The slope is a trend of per-frame
  quality over elapsed time, which is a drift measure. It is not initial-value
  skill (no reference trajectory to match), and not a long-run distribution
  statistic: content and visual scores are per-frame and independent of the
  model's earlier frames. Memory is a revisit consistency score, a different
  thing from distributional fidelity. No ensembles, no calibration, no
  spectra, no contraction measure.
- **Survivorship and selection.** Rollouts that fail early are retried (up
  to three times, longest kept), invalid ones filtered, and the agent can stop
  a rollout. Table 2 shows a pass rate of 48.8% to 89.6% and 8.5% to 26.8% of
  rollouts above 60 s. Slopes are then over the survivors, with rollouts of
  different lengths pooled, so a slope is not step-binned and a collapse is
  cut off. Table 6 uses 327 rollouts with complete scores. Our own
  Matrix-Game 2.0 trials collapsed within about 15 s at default settings
  (`_meta/index.md`), which sits oddly with a near-flat visual slope here; the
  likely difference is the agent-driven action path, the retry and filter
  rules, and the settings.
- Rollouts differ per model (different actions come from the agent), so
  between-model differences mix model quality with action distribution. The
  maximum horizon is 180 s for one family only, and mean lengths are 40 to 60 s.
- No uncertainty intervals on any table cell.
- It is the empirical long-horizon evaluation that ADR 0002 most needs to be
  distinguished from, but it is a benchmark of video models in open scenes,
  with a VLM in the loop.

## Trust signals

- **Credibility:** 3, a multi-institution group (CUHK, Shanghai AI Lab, NTU) with code and benchmark released and a large evaluation (24 A100 GPUs, 1,600+ rollouts), but an unreviewed preprint with VLM-only scoring, no intervals and no citations yet.

## Follow-up

**Relevance:** 4, the nearest long-horizon evaluation of world models and material new evidence for [[Spatial Memory]] and [[Long-Rollout Evaluation]], but it does not touch ADR 0002's state-level statistics.

- **Pre-empts, partly:** "judge world models over long horizons and report
  the slope of quality against time, not only a short-horizon score" is now
  published at 180 s scale for video world models. ADR 0002 should not claim
  long-horizon degradation curves for world models as new, and should cite
  WorldProbe for the observation that local quality and memory dissociate.
- **Supports:** ADR 0002's premise that short-horizon or local quality does
  not determine long-horizon behaviour. Memory and local slope do not rank
  models alike (see Results). It is evidence from video models for what
  [[pedersen2025thermalizer]] and [[wattmeyer2024ace2]] show in weather
  emulators, though by a loose metric.
- **Does not pre-empt:** the weather/climate split (no reference trajectory,
  no state distribution), ensemble calibration, one-step contraction,
  spectral prediction, or a generator with ground truth for imagined cells.
  The ground-truth gridworld remains the edge: WorldProbe's metrics are all
  model-judged, with no noise floor.
- **Memory and our experiments:** in WorldProbe, Matrix-Game 3.0 (retrieval
  memory) scores 6.31 on memory against 4.01 for Matrix-Game 2.0, the same
  direction as our trials (MG3 retrieval 7/8 against 0/8 off, MG2 forgetting
  past its roughly 1 s window; `2026-11-23-mg3-memory-ablation`,
  `2026-11-23-mg2-memory-horizon`). The memory scores are VLM-based and the
  best (AlayaWorld, 6.53) is not a retrieval design we tested. The paper's
  finding that coarse layout survives while texture and entity details drift,
  and that revisits from a new direction fail, is the parallax case still open
  for pose-indexed retrieval (see the open question in `_meta/index.md`).
- **Experiment hook:** its memory protocol (poses estimated from the
  generated video, not commanded poses, to keep control error out of the
  memory score) is worth borrowing for `gw-memory-equal-compute`, where the
  true pose is known anyway.
