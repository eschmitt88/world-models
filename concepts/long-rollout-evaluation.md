---
kind: concept
name: "Long-Rollout Evaluation"
status: seedling
added: "2026-11-24"
sources: [li2025smallworlds]
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

## Connections

- [[Exposure Bias]]: the mechanism that makes long-rollout error compound.
- [[Diffusion Forcing]]: one of the compared backbones; the paper does not
  hold one-step accuracy fixed against teacher forcing.
- [[Spatial Memory]]: recall tests are the other long-horizon measure; they
  score revisits, not free-running drift.
