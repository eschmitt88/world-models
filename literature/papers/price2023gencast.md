---
kind: paper
title: "GenCast: Diffusion-based ensemble forecasting for medium-range weather"
authors: ["Ilan Price", "Alvaro Sanchez-Gonzalez", "Ferran Alet", "Tom R. Andersson", "Andrew El-Kadi", "Dominic Masters", "Timo Ewalds", "Jacklynn Stott", "Shakir Mohamed", "Peter Battaglia", "Remi Lam", "Matthew Willson"]
institutions: ["Google DeepMind"]
year: 2023
venue: "arXiv 2312.15796 (Dec 2023; v2 May 2024); Nature 2025 as 'Probabilistic weather forecasting with machine learning', doi 10.1038/s41586-024-08252-9"
peer_reviewed: true
url: https://arxiv.org/abs/2312.15796
code_url: https://github.com/google-deepmind/graphcast
citations: 485
source: "raw/papers/price2023gencast.pdf"
added: "2026-11-24"
relevance: 4
credibility: 5
status: read
related_experiments: ["2026-11-23-gw-weather-climate-eval"]
related_concepts: ["Ensemble Calibration", "Long-Rollout Evaluation", "Diffusion World Model"]
tags: [weather-climate, ensemble, CRPS, spread-skill, rank-histogram, diffusion, calibration]
---

# GenCast: Diffusion-based ensemble forecasting for medium-range weather

## TL;DR

A conditional diffusion model of the next weather state given the previous two,
sampled autoregressively to 15 days. Different noise draws give an ensemble; 50
members beat the ECMWF ensemble (ENS) on CRPS in 97.4% of 1320 variable, lead and
level targets, and the calibration diagnostics (spread-skill, rank histograms) are
near ideal after the first 2 to 3 days. Its Appendix E gives the exact estimators,
which is why it is the evaluation template for ADR 0002's calibration part.

## Claims

- First ML ensemble more skilful than ENS at 0.25 degrees: CRPS better on 97.4% of
  1320 targets (99.8% beyond 36 h), ensemble-mean RMSE as good or better on 96%
  and significantly better on 82%.
- Calibrated: spread-skill ratio near 1 and flat rank histograms, "as good or
  better" than ENS and far better than a perturbed deterministic ML baseline.
- Samples are sharp and have spherical power spectra matching ERA5; the ensemble
  mean is blurry, like MSE-trained models. A model trained only on 12 h steps gives
  stable, skilful 15-day rollouts, which the authors read as the samples being
  realistic trajectories.
- Better extremes, tropical-cyclone tracks (relative economic value) and wind
  power (CRPS about 20% better than ENS up to 2 days).

## Methods

- Model: p(X_{t+1} | X_t, X_{t-1}), a diffusion denoiser with an
  encoder-processor-decoder: grid to a 6-refined icosahedral mesh, a graph
  transformer, back to the grid. State is 6 surface and 6 atmospheric variables on
  13 levels (0.25 degrees, 12 h steps, T = 30 steps to 15 days). Trained on ERA5
  1979 to 2018 with the Karras denoising objective, single-step only; 2018
  validation, 2019 test.
- Sampling: probability-flow ODE with second-order DPMSolver++2S plus stochastic
  churn, N = 20 solver steps (about 40 denoiser calls) per forecast step. Every step
  starts from fresh noise, so repeating the process gives an ensemble. About 8
  minutes per 15-day member on a TPUv5.
- Initial conditions: ERA5 plus perturbations from the 9-member ERA5 EDA. Needs
  an NWP data-assimilation system for inputs.
- **Ensemble size: 50**, matched to ENS's 50 perturbed members. Baselines: ENS and
  GraphCast-Perturbed (deterministic GraphCast with EDA plus ad hoc Gaussian-process
  perturbations).
- Each model is scored against its own analysis (ENS against HRES-fc0, ML models
  against ERA5); ML models start from 06 and 18 UTC to equalise look-ahead.
- **CRPS (App. E.2.1):** the traditional finite-ensemble estimator, area-weighted
  and averaged over initialisations: mean over members of |x_m - y| minus
  1/(2 M^2) times the sum over member pairs of |x_m - x_m'|. They deliberately
  reject the "fair" CRPS estimator because ENS analyses are a fixed finite set,
  and larger GenCast ensembles would reuse them.
- **Spread-skill ratio (E.2.3):** Skill is the ensemble-mean RMSE. Spread is the
  root of the mean ensemble variance (with 1/(M-1)), taking the mean over grid and
  initialisations before the root. Ratio = sqrt((M+1)/M) * Spread / Skill, which
  is about 1 for an ensemble exchangeable with the truth. Above 1 is
  over-dispersed, below 1 under-dispersed, assuming small bias.
- **Rank histogram (E.2.4):** the rank of the truth among M members (1 to M+1) per
  cell and time, pooled and normalised. Flat is ideal, U-shaped under-dispersed,
  hump-shaped over-dispersed.
- Also: Brier score and relative economic value for events, and CRPS on spatially
  pooled (average and max) fields, to test spatial structure that per-cell CRPS
  misses.

## Results

- CRPS: GenCast beats ENS on 97.4% of 1320 targets; beats GraphCast-Perturbed on
  99.6%; GraphCast-Perturbed beats ENS on only 27%. Biggest gains of 10 to 30% on
  surface, temperature and humidity at 3 to 5 days.
- Pooled CRPS: GenCast beats ENS on 98.1% (average-pooled) and 97.6% (max-pooled)
  targets, with the gap growing at larger regions.
- Calibration (Sec. 6, App. F.2): spread-skill close to 1 from 2 to 3 days;
  over-dispersed at short leads. The paper explains this as a property of scoring
  against a deterministic analysis. Rank histograms flat from 1 to 3 days, flatter
  than ENS in most cases. GraphCast-Perturbed is consistently overconfident
  (ratio below 1, U-shaped).
- ENS is mostly near 1 but under-dispersed on a few variables (t300, q700, 2t).

## Critique / open questions

- Horizon is 15 days: this is the "weather" half. It says nothing about
  long-run statistics, drift or climate, and the stability claim rests on 30
  steps. It is not a long-rollout paper.
- One test year (2019), authors' own baselines, and the GraphCast-Perturbed
  baseline needed ad hoc tuning. CRPS is marginal and per cell, so it cannot see
  errors in cross-cell structure; the pooled scores are the partial remedy.
- Calibration is judged on aggregates by variable and lead time, not conditional
  on state. A model can have ratio 1 on average and still have spread
  uncorrelated with error case by case. ADR 0002's "spread correlates with
  error" kill criterion is stronger than anything checked here.
- Verification against a single deterministic analysis favours under-dispersion at
  short leads, the paper says so. The same concern applies to a gridworld scored
  against the one true next frame.
- Inputs come from NWP data assimilation, so initial-condition uncertainty is
  imported, not learned. Compute is about 40 denoiser calls per step.
- The raw PDF is 40.9 MB and was left uncommitted per the size rule, so
  `source:` points to a local file that git does not hold. Re-fetch from
  arxiv.org/pdf/2312.15796 to reproduce.

## Trust signals

- **Credibility:** 5, peer-reviewed (Nature) from Google DeepMind with code
  released, about 485 citations (Crossref, for the Nature version), and a
  full appendix of estimators; the arXiv v2 read here is the preprint of that work.

## Follow-up

**Relevance:** 4, the canonical source for the definitions that ADR 0002 part 3
uses, and it seeds [[Ensemble Calibration]]; it is weather-only, with no rollout
beyond 15 days and no actions.
- **Supports ADR 0002 (part 3, calibration):** gives the working estimators
  (CRPS, spread-skill with the sqrt((M+1)/M) factor, rank histogram) and the
  reading rules. For the planned 8-member ensembles, note (my inference, not in the
  paper): the correction factor is 1.06 at M = 8, a rank histogram has only 9 bins,
  and the traditional CRPS estimator is biased at small M. The gridworld has
  unlimited independent samples of the true generator, so the "fair" CRPS estimator
  GenCast had to reject becomes usable.
- **Informs ADR 0002:** the over-dispersion at short leads, from scoring against a
  deterministic target, should be expected for "seen" cells and not read as
  miscalibration. Cell classes (seen, unseen) need Brier or ranked-probability
  scores, as tile types are categorical; GenCast's Brier score (E.2.5) is the
  starting point.
- **Does not test:** long-run climate, contraction, or conservation; nothing
  here bears on kill criteria 1 to 3. It bears only on the fourth (spread against
  error), and only indirectly.
- Related: [[Diffusion World Model]] is the same family (a diffusion step model
  sampled autoregressively, ensemble by noise draw); see [[Long-Rollout Evaluation]]
  for what this paper does not cover.
