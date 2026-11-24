---
kind: paper
title: "Fréchet Forcing: Training Long-Horizon Streaming World Models from Short Clips"
authors: ["Yian Zhao", "Yu Yan", "Hongcan Guo", "Ruochong Zheng", "Jian Zhang", "Jie Chen"]
institutions: []    # the PDF author block has none; same authors as zhao2026miniworld (Peking University)
year: 2026
venue: "arXiv v1 (preprint, 19 Oct 2026)"
peer_reviewed: false
url: https://arxiv.org/abs/2610.25666
code_url: https://github.com/zhao-yian/Frechet_Forcing
citations:
source: "raw/papers/zhao2026frechet.pdf"
added: "2026-11-24"
relevance: 4
credibility: 3
status: read
related_experiments: ["2026-11-23-gw-weather-climate-eval"]
related_concepts: ["Distributional Rollout Training", "Exposure Bias", "Diffusion Forcing", "Long-Rollout Evaluation"]
tags: [world-model, action-conditioned, long-rollout, frechet-distance, distribution-matching, drift, evaluation]
---

# Fréchet Forcing: Training Long-Horizon Streaming World Models from Short Clips

## TL;DR

A fine-tuning recipe that trains a block-causal video world model (MiniWorld-1B,
pretrained on 253-frame clips) on its own free rollouts by minimising
Fréchet distances to real-data feature statistics, plus a flow-matching loss on a
self-generated history. After at most 36 GPU-hours on 381-frame clips, rollouts
stay stable beyond 4,000 frames on DROID (robot manipulation) and Sekai
(camera-controlled exploration). The Fréchet distance is a *training loss and
evaluation metric on feature statistics of video frames*; there is no
ground-truth state generator, no calibration and no long-run-statistics
suite beyond image and video feature distances.

## Claims

- Long rollouts need not match a teacher or a paired recording. They must stay a
  realistic continuation of their own scene and follow the actions. Realism and
  scene fidelity "can be measured directly against real-data statistics, which
  remain valid at any horizon"; action following is kept by the conditional
  loss.
- DMD teachers are a bottleneck for action-conditioned models: an external teacher
  is action-agnostic, the model's own pretrained network drifts like the
  student. "None of the DMD configurations we evaluated improved over the base
  model", so DMD baselines are omitted.
- Three failure modes of naive distribution matching on free rollouts: context
  decoupling (a marginal FD "pulls every rollout towards a dataset-average
  appearance"), gradient-statistic misalignment, and feature-space exploitation
  (a low-order feature space admits degenerate frames).
- Training on self-generated history that ends in a real chunk avoids a
  chunk-boundary "redraw" flicker.

## Methods

- Loss L = L_anc + L_ctx + L_real, unit weights. L_anc: flow matching on 4 real
  chunks after a history re-noised to t = 0.7 and denoised by the deployed
  sampler, final history chunk clean. L_ctx: Fréchet distance on latent
  channel mean and standard deviation of the last K=8 chunks of a free
  317-frame rollout, centred on the clip's own ground-truth mean. L_real: Fréchet
  distance of frozen DINOv2 ViT-B/14 CLS embeddings of decoded frames against
  VAE-reconstructed real frames. Running EMA moments (beta 0.99) let FD be
  differentiable at batch size 1 per GPU; loss is FD / stopgrad(FD).
- Data: DROID (8,958 training episodes, 2,774 held out, 15 fps, end-effector
  pose and gripper actions) and Sekai ("game-navigation videos paired with
  camera poses", 30 fps, 1,462 training and 156 held-out clips of 1,800 frames).
  MiniWorld-1B checkpoints are released; 225 steps on 8 GPUs, 34 to 36 GPU-hours.
- Evaluation: 32 held-out clips per domain rolled out from one real latent for
  4,001 frames, in six horizon bins. Metrics: CLIP-FD to real frames (reported
  beside the CLIP-FD of ground-truth frames at the same sample count as a
  floor), I3D FVD on 16-frame windows, brightness ratio, chunk-boundary flicker,
  PSNR against a forward-backward episode replay (DROID), VGGT-estimated yaw
  correlation and forked left/right turn tests (Sekai). Plus manual inspection of 96
  DROID rollouts. Baselines: base MiniWorld-1B and an FM-only ablation.

## Results

- DROID, final bin (frames 2,997 to 4,001): CLIP-FD 44 (base 431), FVD 81 (base
  2,058); PSNR 21.5 dB (base 8.9). Across bins CLIP-FD stays 30 to 44.
- Sekai: CLIP-FD 226 against 609 and FVD 441 against 1,193 in the final bin;
  brightness 0.92 (base 0.40). FVD "still exhibits a gradual upward trend from
  186 to 441", which the authors call a residual slow drift.
- Action following survives: Sekai turn branches forked at frame 3,581 turn in
  the commanded direction in 97% of cases, but reach only 49% of the commanded
  angle; yaw correlation 0.71 to 0.75 in the last bins.
- Flow matching alone (FM-only) barely changes long-horizon behaviour: DROID
  final-bin CLIP-FD 401 against 431.
- Manual inspection of 96 DROID rollouts: 98% artifact-free at frame 1,500, 79% at
  frame 4,000.

## Critique / open questions

- The ADR 0002 distinction is confirmed, with two corrections. The paper does match
  *feature* distributions (DINOv2 and latent-moment Fréchet distances), as a
  training loss and as the evaluation metric (CLIP-FD, FVD). It never scores
  decoded state against a ground-truth generator. Corrections: (a) it is not
  games-only: one domain is real-robot video (DROID) and the other (Sekai) is
  game-navigation video with camera-pose conditioning, no discrete or tile
  world; (b) it does centre statistics on each clip's own ground truth in
  training (context fidelity), so it is not purely a marginal match, though the
  eval FD is marginal and pooled over rollouts.
- No calibration, ensembles, spread, CRPS, rank histograms or contraction
  measure appear in the paper (searched for each). Each rollout is run once
  per clip. Its "climate" proxy is population FD plus brightness and flicker.
- The noise floor is there in embryo: CLIP-FD of ground-truth frames at the same
  sample count is reported for each bin. That is a precedent for ADR 0002's
  two-real-samples floor, in feature space.
- The evaluation metrics can be fooled where the training features cannot reach:
  the authors admit the perceptual space is coarse, "allowing some bright
  rollouts to develop fine periodic textures that none of our metrics
  registers", that population metrics "can mask rare localised failures"
  (20% of DROID rollouts show local artifacts by frame 4,000), and that a scene
  revisited much later need not look the same. These are the cases where
  decoded-state statistics against a known generator would add information.
- Context decoupling, where a marginal FD is satisfied by dataset-average frames,
  is the video analogue of "legal frames in the wrong place" in ADR 0002.
- Ground truth beyond the recorded length is synthetic: a forward-backward
  replay on DROID (approximate; objects may not return to the same pose) and
  concatenated held-out camera paths on Sekai. DROID held-out episodes were also
  part of pretraining, and only one 1B model is tested. No independent
  reproduction yet.
- "Stable" means flat CLIP-FD and FVD; Sekai FVD still rises over the horizon.
  The headline 4,000 frames is over 10x the fine-tune length, but it is one
  rollout per clip.

## Trust signals

- **Credibility:** 3, an unreviewed arXiv v1 from the MiniWorld group with code, models
  and a demo page released and a bootstrap over rollouts plus ablations, but
  no affiliations in the PDF, no citations yet, one model size and synthetic
  long-horizon references.

## Follow-up

**Relevance:** 4, the nearest published neighbour to ADR 0002's climate-statistics
idea, adding material evidence on long-horizon drift for [[Exposure Bias]] and
[[Long-Rollout Evaluation]], and seeding [[Distributional Rollout Training]].
- **Confirms ADR 0002's novelty distinction (climate statistics):** the
  distributions matched are image and video *feature* statistics against real data,
  used mainly as a training objective and scored with CLIP-FD and FVD, not
  structural statistics of decoded state (occupancy, wall/floor patterns,
  adjacency) against a known generator. ADR 0002's text should drop any
  implication that FF is games-only or purely marginal (see Critique).
- **Pre-empts, partly:** "long-rollout quality must be judged against real-data
  statistics at every horizon, with a ground-truth sample-size floor" is no
  longer new in world models, and FF reports metrics binned by rollout step. The
  paper's claim to novelty is the training method, so the evaluation framing
  ("weather versus climate") remains open, but cite FF for the binned feature
  statistics and the floor.
- **Does not pre-empt:** calibration of ensembles, one-step contraction
  probe, spectral predictor, a generator with ground truth for imagined cells,
  and the split between short-horizon accuracy and long-horizon statistics.
- **Supports (design caution):** a marginal match collapses to dataset-average
  appearance; clip-conditional centring is what prevents it. Relevant to ADR
  0002's "legal frames in the wrong place" and to the choice of conditioning
  when comparing DF with teacher forcing.
- **Experiment hook:** FF is built on the MiniWorld recipe ([[zhao2026miniworld]]) at
  video scale (1B, 8 GPUs), so it is a reference point, not a drop-in arm for
  the gridworld drift experiment.
