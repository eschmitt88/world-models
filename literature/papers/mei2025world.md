---
kind: paper
title: "World Models That Know When They Don't Know: Controllable Video Generation with Calibrated Uncertainty"
authors: ["Zhiting Mei", "Tenny Yin", "Micah Baker", "Ola Shorinwa", "Anirudha Majumdar"]
institutions: ["Princeton University"]
year: 2025
venue: "arXiv v2 (preprint, v1 5 Dec 2025, v2 10 Mar 2026)"
peer_reviewed: unknown
url: https://arxiv.org/abs/2512.05927
code_url: https://github.com/irom-princeton/c-cubed
citations:
source: "raw/papers/mei2025world.pdf"
added: "2026-11-24"
relevance: 3
credibility: 3
status: read
related_experiments: ["2026-11-23-gw-weather-climate-eval"]
related_concepts: ["Ensemble Calibration", "Long-Rollout Evaluation", "Diffusion World Model", "Diffusion Forcing"]
tags: [calibration, uncertainty, proper-scoring-rule, action-conditioned, robot, video-diffusion, ECE, latent-space]
---

# World Models That Know When They Don't Know: Controllable Video Generation with Calibrated Uncertainty

The arXiv abs page titles it with a dash ("... Know - Controllable Video Generation ..."); the PDF uses a colon. The method is called C3 (the paper writes C^3).

## TL;DR

C3 adds a transformer "UQ probe" to an action-conditioned latent video diffusion
model. The probe predicts, for each latent subpatch, the probability that the
generated value is within an L1 error threshold of the ground truth. It is
trained with strictly proper scoring rules (Brier, cross-entropy, BCE). The
confidence map is decoded to an RGB heatmap. Calibration is measured with ECE
and MCE on Bridge and DROID, plus a correlation between confidence and error and
an OOD test on a real WidowX arm. There is no ensemble-based calibration
diagnostic and no analysis against rollout horizon.

## Claims

- "C3, the first method for training video models that know when they don't
  know" (Fig. 1 caption); the only prior video-model UQ they cite
  (2510.02571) gives one composite, task-level confidence per video.
- Training the probe with strictly proper scoring rules gives calibrated
  confidence: ECE and MCE are low for all three variants (Fig. 3), and "neither
  underconfident nor overconfident" in reliability diagrams.
- Estimating uncertainty in latent space avoids the cost and instability of
  pixel-space UQ. Latent L1 error correlates with SSIM (-0.54), PSNR (0.83) and
  LPIPS (0.73) (Table 2; the paper's own sign comments are inconsistent: it
  says SSIM and PSNR should correlate negatively with error and lists PSNR at
  +0.83).
- Heatmaps localise hallucinations (invented objects, deformed toys, occlusion).
- Calibration holds with "only a very small drop" in OOD real-robot settings
  (ECE 9.98e-2, MCE 1.71e-1 there).

## Methods

- **What is calibrated.** Not the sampled frames' distribution. The target is a
  per-subpatch binary event in latent space, acc(x_hat, x*, eps) = [|x_hat - x*|
  <= eps] (Eq. 4), and the probe outputs q_hat = P(acc). Three variants: FSC
  (one fixed eps, sigmoid, Brier loss), MCC (error bins, softmax, cross-entropy)
  and CS-BC (eps is an input, so confidence is a function of the accuracy
  threshold; Brier or BCE).
- Backbone: DiT over SVD-VAE latents (49 layers, 4 heads, 512 dim, 256x256, no
  temporal compression), action embedding summed with the timestep embedding,
  trained with diffusion forcing for per-frame noise levels. Probe reads
  penultimate-layer features with stop-gradient. Trained 50k iterations on 8
  L40 GPUs.
- **Evaluation protocol.** Videos are generated "conditioned on the initial
  video frame and the entire action trajectory"; one pass, no feedback of
  generated frames into a later context. 110 Bridge and 83 DROID test
  trajectories. ECE and MCE with 20 bins, averaged over 10 thresholds (CS-BC).
- Diagnostics: ECE, MCE, reliability diagrams, and the Shepherd's Pi correlation
  between confidence and latent error.
- Baselines (App. D, Table 3): an "approximate deep ensemble" (variance across
  several generations using the model's own stochasticity) and a heuristic
  (raw diffusion noise), compared by correlation with error only, because ECE
  and MCE "do not apply".

## Results

- Confidence-error correlation (Bridge): FSC -0.373, CS-BC -0.172 (both 99%
  significant); MCC comes out positive until the maximum bin edge is capped at
  0.2 (then -0.130). Table 3: C3 -0.36, ensemble variance +0.42 (the right sign
  for an uncertainty; the sign flips because C3 outputs confidence), heuristic
  -0.01 (not significant). The paper calls the ensemble "comparable" and
  dismisses it on cost.
- Video quality is unchanged by the probe (SSIM 0.75 vs 0.76, PSNR 18.4 vs 18.6,
  LPIPS 0.28 both; Table 1).
- Ablations: BCE versus Brier differ by about 3e-4 in ECE. Adding diffusion
  forcing for the confidence maps raises ECE to about 0.33 and MCE to about 0.55
  ("increases underconfidence").
- Calibration is worse at tight thresholds (eps <= 0.3: underconfident).

## Critique / open questions

- **No horizon dependence.** No calibration, ECE or correlation is reported
  against frame index, lead time or rollout length; the numbers are averages
  over every frame of a trajectory. Figures show frames up to about t=21; the
  rollout length is not stated. The authors list it as future work:
  "long-duration temporal consistency of the confidence estimates ... is limited
  by the history length of the conditioning inputs" and "Long-duration video
  generation remains an open research problem". One qualitative remark (Fig. 8)
  notes uncertainty rising over time under unseen lighting, as an observation.
- **Open-loop, teacher-forced-style evaluation.** Actions are given for the
  whole trajectory and the model generates once; there is no autoregressive
  feedback of the model's own outputs beyond the model's internal frame
  conditioning, so exposure bias and drift are not what is measured.
- **Scoring rules are training losses, not forecast scores.** Brier and BCE
  train the probe on a thresholded error event. CRPS, spread-skill ratio and
  rank histograms are never used. A learned confidence head is calibrated to its
  own error indicator, and says nothing about whether samples from the model
  span the truth.
- **Ensembles as a foil.** The ensemble baseline is a variance-versus-error
  correlation on a handful of samples (count not stated), judged impractical
  for "billions of parameters". It has a larger correlation magnitude (0.42)
  than C3 (-0.36) on the same latent error, which weakens the claim.
- Small test sets (110 and 83 trajectories), one backbone, ECE binned at 20
  bins over a heavily imbalanced error distribution (most patches have small
  error), MCC needed a bin-edge cap to get the right sign. The paper's Proposition
  1 gives calibration "provided that phi converges to an optimal solution",
  which is in-distribution only; the authors concede that.
- The generator has no ground-truth state: error is latent L1 to a recorded
  video, so there is no noise floor or independent real samples.

## Trust signals

- **Credibility:** 3, a single university group (Princeton, Majumdar lab) with code released and real-robot tests, but an unreviewed preprint (no venue named), small test sets, an inconsistent sign remark in Table 2, and no citation count established.

## Follow-up

**Relevance:** 3, the nearest published calibration work on an action-conditioned world model and cite-worthy prior art for [[Ensemble Calibration]], but it uses ECE on a learned confidence head and shifts no concept.

- **Confirms ADR 0002's framing, with one correction.** "Scoring rules, no long
  rollouts" is accurate, but the scoring rules are Brier, cross-entropy and
  BCE used as training losses for a per-subpatch accuracy classifier, not the
  CRPS, spread-skill ratio and rank-histogram family that [[price2023gencast]]
  defines for ensembles. C3 evaluates with ECE, MCE, reliability diagrams and a
  confidence-error correlation. It also does not use ensembles (it treats them
  as too expensive), and has no lead-time axis.
- **Action-conditioned: yes** (action sequence embedded and fed to the DiT).
  What is calibrated: per-subpatch latent accuracy at a threshold, from
  hidden features of the model; not the sampled frame distribution and not
  decoded state.
- **Pre-empts, partly:** the idea of scoring calibration of an
  action-conditioned world model with proper scoring rules, and the test that
  uncertainty correlates with error (Shepherd's Pi). ADR 0002 should cite C3
  for both. Its "spread must correlate with error" kill criterion has a close
  relative here: the ensemble-variance correlation of +0.42 on latent error is
  weak evidence that sample spread tracks error in a video world model, on
  short rollouts and with an unreported ensemble size.
- **Does not pre-empt:** ensemble spread-skill ratio against lead time, rank
  histograms, CRPS, any horizon-resolved calibration, long rollouts with the
  model fed its own outputs, or calibration scored against a known generator
  with a noise floor. The novelty claim in part 3 of ADR 0002 stands if it is
  worded as "spread-skill and rank-histogram calibration as a function of
  rollout step in an action-conditioned model", not "calibration of world
  models".
- **Worth borrowing:** the confidence-versus-accuracy-threshold view (CS-BC,
  calibration as a function of eps) as a second axis next to rollout step, and
  the observation that diffusion forcing worsened calibration (ECE 0.33),
  relevant to the DF versus TF comparison in
  `2026-11-23-gw-drift-df-vs-tf`, though it was not analysed.
