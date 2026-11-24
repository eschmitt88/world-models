---
kind: concept
name: "Long-Rollout Evaluation"
status: seedling
added: "2026-11-24"
sources: [li2025smallworlds, pedersen2025thermalizer]
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
- Not yet covered here: measuring contraction of the emulator itself. The
  weather benchmark that motivates it is not yet ingested.

## Connections

- [[Exposure Bias]]: the mechanism that makes long-rollout error compound.
- [[Diffusion Forcing]]: one of the compared backbones; the paper does not
  hold one-step accuracy fixed against teacher forcing.
- [[Spatial Memory]]: recall tests are the other long-horizon measure; they
  score revisits, not free-running drift.
