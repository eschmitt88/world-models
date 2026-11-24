---
kind: concept
name: "Ensemble Calibration"
status: seedling
added: "2026-11-24"
sources: [price2023gencast, lippe2023pderefiner, mei2025world]
related_concepts: ["Long-Rollout Evaluation", "Diffusion World Model", "Conservation by Construction"]
related_experiments: ["2026-11-23-gw-weather-climate-eval"]
tags: [weather-climate, ensemble, CRPS, spread-skill, rank-histogram, evaluation]
---

# Ensemble Calibration

## Definition

Whether the spread of a set of sampled rollouts (same context, different noise) is
as large as the model's actual error, and whether the truth looks like one more
member. Scored with CRPS (accuracy of the whole predictive distribution), the
spread-skill ratio (average spread against ensemble-mean RMSE) and rank histograms
(where the truth falls among the members).

## Why it matters here

ADR 0002 part 3 runs 8-member ensembles and asks whether per-cell disagreement
tracks error. [[price2023gencast]] is the reference for the estimators:

- CRPS: mean |member - truth| minus half the mean pairwise |member - member|.
  Lower is better; the traditional estimator is for a finite M, a "fair" variant
  estimates the infinite-ensemble value.
- Spread-skill ratio = sqrt((M+1)/M) * spread / skill, with spread the root of mean
  ensemble variance and skill the ensemble-mean RMSE. About 1 is calibrated, below 1
  over-confident, above 1 under-confident (assuming small bias).
- Rank histogram: flat is calibrated, U-shaped under-dispersed, hump-shaped
  over-dispersed. Needs M+1 bins, so M = 8 is coarse.

GenCast (50 members) is near 1 and flat after 2 to 3 days, over-dispersed earlier
because the target is a single deterministic analysis. All three diagnostics are
averages: none checks that spread tracks error case by case, which is what the ADR's
kill criterion needs, so that needs a separate correlation test.

A trajectory-level precedent for that correlation test: [[lippe2023pderefiner]] samples
32 rollouts per test trajectory and finds the time when samples decorrelate predicts
the time the rollout stays accurate (R^2 = 0.86, KS equation, single model). It is
whole-rollout decorrelation time, not per-cell spread against error.

The nearest world-model neighbour is [[mei2025world]] (C3): an action-conditioned
latent video model with a probe trained with Brier or cross-entropy loss to predict
whether each latent subpatch is within an L1 threshold of the truth. It is scored
with ECE, MCE, reliability diagrams and a confidence-error correlation (about -0.17
to -0.37), averaged over all frames of single-pass rollouts of unstated length (no
lead-time axis). It uses no CRPS, spread-skill ratio or rank histogram, and treats
ensembles as too costly; its own ensemble-variance baseline correlates +0.42 with
latent error (ensemble size not stated), a weak data point for the ADR's "spread
tracks error" kill criterion. It also reports that diffusion forcing raised ECE to
about 0.33.

## Connections

- [[Long-Rollout Evaluation]]: calibration is the third of ADR 0002's four parts, and
  GenCast stops at 15 days.
- [[Diffusion World Model]]: noise draws give the ensemble for free.
