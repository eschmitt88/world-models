---
kind: paper
title: "Can AI Weather Models Predict Beyond Two Weeks? A Quantitative Benchmark and Analysis of Long Rollouts"
authors: ["Fanny Lehmann", "Firat Ozdemir", "Yun Cheng", "Torsten Hoefler", "Sebastian Schemm", "Benedikt Soja", "Siddhartha Mishra"]
institutions: ["ETH AI Center, ETH Zurich", "Swiss Data Science Center, ETH Zurich and EPFL", "Scalable Parallel Computing Lab, ETH Zurich", "Dept. of Applied Mathematics and Theoretical Physics, University of Cambridge", "Institute of Geodesy and Photogrammetry, ETH Zurich", "Seminar for Applied Mathematics, ETH Zurich"]
year: 2026
venue: "arXiv v2 (preprint, 5 Jun 2026)"
peer_reviewed: false
url: https://arxiv.org/abs/2605.30184
code_url: https://github.com/lehmannfa/ai-weather-stability
citations:
source: "raw/papers/lehmann2026can.pdf"
added: "2026-11-24"
relevance: 4
credibility: 4
status: read
related_experiments: ["2026-11-23-gw-weather-climate-eval"]
related_concepts: ["Long-Rollout Evaluation", "Exposure Bias"]
tags: [weather-climate, rollout-stability, benchmark, denoising, noise-robustness, spectra, seasonality]
---

# Can AI Weather Models Predict Beyond Two Weeks? A Quantitative Benchmark and Analysis of Long Rollouts

## TL;DR

A benchmark of nine AI weather models (3.5M to 4.7B parameters) rolled out for
two years (2920 six-hour steps) against ERA5, with three quantitative failure
modes (blow-up time, loss of seasonality, small-scale spectral energy) and a
noise-injection study. It is the source of the "stable models denoise noisy
input" observation behind ADR 0002's contraction probe, but what it measures
is the divergence of noisy and clean rollouts over days, not a one-step gain.

## Claims

- Long-rollout failure falls into three regimes: blow-up, drift, loss of
  seasonality.
- Stability hinges on the treatment of small scales: GraphCast and AIFS amplify
  high-frequency energy (spectral ratios above 1, up to 1e2 to 1e6 for
  GraphCast, FuXi and FourCastNet in places); the stable models are blurrier
  than ERA5 (ratios below 1), and blur is mostly intrinsic to one-step
  predictions rather than added by rollout.
- Stable models (Aurora, SFNO, and Pangu less so) "denoise" noisy initial
  states into realistic evolving fields; the stable models do not replay
  training data (nearest-neighbour ratio near 1; trajectory spread from
  white-noise starts comparable to training std at the surface).
- The time embedding drives seasonality: removing it leaves a stable rollout
  with no seasonal cycle.
- Short-term error has only a moderate correlation with long-term seasonal
  accuracy across training checkpoints (Fig. 5b).
- Stability is insensitive to most architecture changes tried (window size,
  patch size, norm, levels, static variables).

## Methods

- Nine models: Aurora, SFNO, Pangu, AIFS, GraphCast, FourCastNet, FuXi,
  DLESyM, GenCast. 2-year rollout from 2021-01-01 (outside training for all
  but AIFS); ERA5 as reference. Initial-date robustness is in Tables A.2 and
  A.3.
- Blow-up time: first 30-day window where the global min or max shows a high
  log-linear correlation (coefficient of determination above 0.9). Loss of
  seasonality: large-scale (wavelength above 5000 km) energy spectrum off ERA5
  by more than twice the climatological range for at least 45 days. Small-scale
  energy ratio: last-30-day spectrum (wavelength below 250 km) over ERA5's.
- Noise probe: add Gaussian noise (1 sigma and 10 sigma of each variable) to
  the initial state, then compare the noisy rollout to the clean rollout over
  time. Also pure-noise starts, a cat image, corrupted static fields, a
  seasonal-shift experiment (January state with June time embedding).
- Ablations on AuroraS (113M, trained on ERA5 1979 to 2019, 60k steps); 4-year
  rollouts. 10-year rollouts of Aurora, SFNO and DLESyM for extreme-event
  statistics (regional QQ plots).

## Results

- Blow-up (Table 1): Aurora, SFNO and Pangu never blow up in 730 days;
  FourCastNet blows up in about 8 days; GraphCast in 270 to 470 days; GenCast
  and AIFS vary a lot by variable (AIFS Z500 at 16 days, other variables
  stable).
- Seasonality (Table 2): Pangu loses it at 181 to 525 days (it has no time
  embedding), GenCast at 79 to 105 days, FourCastNet at about 60 days.
- Noise (Fig. 2, A.6, A.8): with 1 sigma noise Aurora, SFNO and Pangu settle to
  a constant error; GraphCast's error grows exponentially. The error is
  V-shaped over the first ten days (falls as the model denoises, then rises as
  trajectories decorrelate). At 10 sigma SFNO's residual diverges while Aurora
  and AIFS stay bounded. Aurora denoises structured noise and even a cat image.
- DLESyM is stable but amplifies initial noise rapidly; Pangu reduces noise
  but converges to unrealistic fields. So bounded error is not the same as
  denoising to a realistic state.
- Ablations: all four AuroraS resolution and lead-time variants give stable
  4-year rollouts; 6-hour models flag loss of seasonality at 381 days (0.25
  degrees) and 244 days (1.5 degrees); 21M parameters loses the seasonal
  cycle.
- Extremes (10-year): Aurora, SFNO and DLESyM produce plausible extremes but
  less extreme and less frequent than ERA5.

## Critique / open questions

- The noise probe is a rollout-level comparison of noisy against clean
  trajectories over days and one initial date (plus robustness of the
  stability metrics across dates); it is not a one-step gain or Jacobian norm.
  ADR 0002's one-step contraction measure must be validated as a predictor
  here, not assumed from it.
- It does not show contraction predicts stability. It shows co-occurrence over
  about five models: stable-and-denoising (Aurora, SFNO), stable but
  noise-amplifying (DLESyM), stable but unrealistic after noise (Pangu).
  DLESyM is a direct counterexample to "stable implies contractive".
- Nine heterogeneous models differ in data, resolution, training and variables;
  no matched-capacity comparison outside the Aurora ablations. Causal claims
  rest on those ablations only, and only for ViT architectures.
- The short-term vs seasonal-accuracy link (Fig. 5b) is a single model family
  across checkpoints; "moderate correlation" gives no R-squared in the text I
  read.
- Abstract says "year-long" rollouts; the main benchmark is two years, with
  ten-year runs for three models only.
- "Not stochastic parrots" rests on two weak memorization tests (a distance
  ratio near 1 and white-noise-start variance).
- No peer review yet; code is for the metrics, with models run through their
  public repositories and Earth2Studio.

## Trust signals

- **Credibility:** 4, an ETH Zurich/Cambridge group (Hoefler, Mishra) preprint
  with released metric code, public models and ERA5 and quantitative,
  date-robust metrics; not peer reviewed (v2 of an arXiv preprint) and no
  citation count I could establish.

## Follow-up

**Relevance:** 4, the cited source of the "stable models act as denoisers"
observation behind ADR 0002's contraction probe, and a direct instance of its
weather-versus-climate split; it adds new evidence to [[Long-Rollout Evaluation]].
- **Supports ADR 0002 (claim 1):** across training checkpoints, one-step error
  falls steadily while seasonal-cycle error is only moderately correlated with
  it (Fig. 5b). The 2-year benchmark is also a template for the climate part:
  blow-up time, seasonality loss and small-scale spectral ratio as metrics.
- **Supports ADR 0002 (contraction):** the noise study is the only evidence
  for a noise-response signature of stability, and the V-shaped error curve
  is the shape the probe should expect.
- **Undercuts it:** the probe is a multi-day rollout comparison, not a one-step
  measurement without rollouts. DLESyM (stable, noise-amplifying) and Pangu
  (stable, noise-reducing but unrealistic, then seasonality lost) mean
  contraction need not predict survival or climate. Expect to need
  on-manifold perturbations, which ADR 0002 already includes, and to treat a
  null as plausible.
- **Related papers named but not ingested:** Gallusser et al. on design choices
  for 10-year stability, FouRKS (Chattopadhyay et al.), cited in the paper, and ArchesWeather
  under climate runs (2605.29976), from the candidate file.
