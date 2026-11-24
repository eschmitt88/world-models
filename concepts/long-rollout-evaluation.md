---
kind: concept
name: "Long-Rollout Evaluation"
status: seedling
added: "2026-11-24"
sources: [li2025smallworlds, pedersen2025thermalizer, lehmann2026can, wattmeyer2024ace2, price2023gencast, lippe2023pderefiner, zhao2026frechet]
related_concepts: ["Exposure Bias", "Diffusion Forcing", "Spatial Memory"]
related_experiments: ["2026-11-23-gw-drift-df-vs-tf", "2026-11-23-gw-weather-climate-eval"]
tags: [evaluation, benchmark, rollout, horizon, state-space, weather-climate]
---

# Long-Rollout Evaluation

## Definition

Scoring a world model on how its predictions degrade as it runs on its own
outputs for many steps, rather than on one-step or short-horizon accuracy or
downstream reward.

## Why it matters here

The project's drift and weather/climate experiments (ADR 0001 and ADR 0002)
rest on this. The published reference point is [[li2025smallworlds]]: four
architectures in fully observable, isolated worlds, state-space MSE over 90
steps and error against horizon to 990. It finds that short-horizon rank does
not carry over (a Neural-ODE model is best at 90 steps and blows up
exponentially with horizon, while the others plateau). It only measures
mean error against the true trajectory; distributional statistics, ensemble
calibration and contraction, the parts ADR 0002 adds, are not covered there.

## Weather versus climate (from weather/climate ML)

Weather-ML splits rollout quality in two: *weather* (trajectory accuracy from
the initial state, which chaos eventually destroys) and *climate* (whether the
long-run state distribution, spectra and seasonal cycle stay right). ADR 0002
imports the split. Evidence so far:

- Short-horizon error does not order long-horizon stability. In
  [[pedersen2025thermalizer]] (App. B.3) a 48M Unet with much lower validation
  MSE than a 0.5M ResNet goes unstable sooner, and stability varies with weight
  seed. Past the decorrelation time, MSE plateaus near 1 even for a model that
  is fine on climate, so MSE cannot be the long-horizon score.
- Out-of-distribution drift is the failure to watch: the emulator's own
  stationary law differs from the training law (Thermalizer Eqs. 2 and 3).
- A prior over the stationary distribution, used as a denoiser every few
  steps, restores the attractor but not the trajectory (Thermalizer: stable to
  1e4 steps against divergence at about 2e3 to 5e3; to 1e5 in one figure). The
  headline "10x" (v1 abstract) and "100x" (v2 abstract) are lengths tested
  against the divergence step, with the thermalized runs stopped, not failed.
- Failure-mode metrics worth borrowing come from [[lehmann2026can]], a
  two-year benchmark of nine AI weather models: blow-up time (log-linear growth
  of the extremes), loss of seasonality (large-scale spectrum off climatology
  for 45 days), and small-scale spectral ratio against reference. Across
  training checkpoints, one-step error falls steadily while seasonal-cycle
  error is only moderately correlated with it.
- Noise response as a stability signature: stable models (Aurora, SFNO)
  pull noisy inputs back to realistic states, and the noisy-minus-clean error
  is V-shaped over days. This is measured by comparing full rollouts, not by a
  one-step gain, and the evidence is loose: DLESyM is stable but amplifies
  noise, and Pangu is stable but denoises to unrealistic fields
  ([[lehmann2026can]]). A one-step contraction probe (ADR 0002) is still
  untested as a predictor of survival. If it earns its own page, split it out
  as a concept then.
- Training loss versus climate, again: [[wattmeyer2024ace2]] (App. A.4) trains an
  emulator on two-step rollouts; all models trained are reported indefinitely stable
  (one run to 1000 years), yet 10-year climate error does not fall with more training while
  validation loss does, and one of four seeds is much worse. The authors select
  checkpoints on time-mean RMSE, and report error as a multiple of the reference
  model's own two-sample variability (1.1 to 1.5x), a noise-floor yardstick.
- Stability claims need a drift test, not only "did not blow up": the first ACE
  was "stable for 100 years" yet showed seasonal-amplitude swings and
  surface-pressure drift (as reported in ACE2).
- A spectral view of why short-horizon accuracy misleads: [[lippe2023pderefiner]]
  finds one-step MSE training fits only the dominant low-wavenumber band and
  neglects low-amplitude components that limit rollout time, and that pushforward
  and exact invariance correction did not fix it (PDEs, correlation-time
  metric). See [[Spectral Rollout Diagnostics]]; no per-step spectrum is shown.
- The nearest world-model neighbour: [[zhao2026frechet]] evaluates 4,000-frame
  action-conditioned rollouts in six horizon bins with CLIP-FD and FVD against real
  frames, plus brightness, flicker and action-following scores, and reports a
  ground-truth frames floor at the same sample count. Everything is feature or
  pixel level; no decoded-state statistics against a generator, no calibration
  or ensembles. Its own limitations (population metrics mask local failures, a
  coarse feature space misses fine periodic texture) mark where ADR 0002's
  state-level statistics add information. See [[Distributional Rollout Training]].
- Ensemble calibration, the third part of ADR 0002, is covered in
  [[Ensemble Calibration]]: [[price2023gencast]] defines CRPS, spread-skill and rank
  histograms for 50-member, 15-day forecasts. It does not reach long horizons.

## Connections

- [[Exposure Bias]]: the mechanism that makes long-rollout error compound.
- [[Diffusion Forcing]]: one of the compared backbones; the paper does not
  hold one-step accuracy fixed against teacher forcing.
- [[Spatial Memory]]: recall tests are the other long-horizon measure; they
  score revisits, not free-running drift.
