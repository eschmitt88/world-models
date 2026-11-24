---
kind: moc
name: "Long-Rollout Stability"
status: active
added: "2026-11-24"
concepts:
  - "[[concepts/long-rollout-evaluation]]"
  - "[[concepts/ensemble-calibration]]"
  - "[[concepts/spectral-rollout-diagnostics]]"
  - "[[concepts/conservation-by-construction]]"
  - "[[concepts/distributional-rollout-training]]"
tags: [moc, long-rollout-stability, weather-climate, evaluation, rollout-stability]
---

# Long-Rollout Stability

**How to tell whether a world model run for thousands of steps on its own
outputs is still right, and what keeps it there.** Short-horizon accuracy does
not answer this. Several sources show one-step error and long-run behaviour
coming apart ([[li2025smallworlds]], [[pedersen2025thermalizer]],
[[lehmann2026can]], [[wattmeyer2024ace2]]). Weather and climate ML has the most
mature toolkit for the problem; action-conditioned world models have borrowed
only parts of it. These five concepts are that toolkit, split into ways of
measuring a long rollout and ways of constraining one. Together they are the
literature behind ADR 0002 (`docs/decisions/0002-weather-climate-evaluation-of-world-models.md`)
and `experiments/2026-11-23-gw-weather-climate-eval/`.

The failure mechanism they all respond to is [[concepts/exposure-bias|Exposure Bias]],
and [[concepts/diffusion-forcing|Diffusion Forcing]] is one training-side answer.
Both are mapped in [[mocs/interactive-world-models]], so they are linked here but
not duplicated.

## Measure it

- [[concepts/long-rollout-evaluation|Long-Rollout Evaluation]]: the frame. It
  separates *weather* (trajectory skill from the initial state) from *climate*
  (whether long-run statistics stay right). It also collects the borrowed
  failure-mode metrics: blow-up time, loss of seasonality, and a two-sample
  noise floor. World-model neighbours: [[zhao2026frechet]] (feature statistics
  binned by rollout step) and [[wang2026worldprobe]] (degradation slopes for 10
  models to 180 s).
- [[concepts/ensemble-calibration|Ensemble Calibration]]: whether the spread of
  sampled rollouts matches their error. It is scored with CRPS, the spread-skill
  ratio and rank histograms ([[price2023gencast]]). The only world-model
  calibration work found, [[mei2025world]], has no lead-time axis.
- [[concepts/spectral-rollout-diagnostics|Spectral Rollout Diagnostics]]: the
  band energy of each predicted frame against a reference, used as a diagnostic
  or a leading indicator of failure. The sources disagree on which band fails:
  [[lippe2023pderefiner]] finds a missing high tail, [[lehmann2026can]] an
  excess one, and [[li2026freqforcing]] a lost low band.

## Constrain it

- [[concepts/conservation-by-construction|Conservation by Construction]]: enforce
  a known invariant exactly at every step. ACE2's corrector is the reference
  ([[wattmeyer2024ace2]]); in the gridworld, snap-to-tile is the analogue. The
  caution: it removes budget drift but not a wrong climate, and invariance
  correction did not lengthen rollouts in [[lippe2023pderefiner]].
- [[concepts/distributional-rollout-training|Distributional Rollout Training]]:
  train on free rollouts against distribution statistics, since long rollouts
  have no paired ground truth ([[zhao2026frechet]]). The learned counterpart at
  inference is a stationary-distribution prior used as a denoiser
  ([[pedersen2025thermalizer]]).

## Open thread

ADR 0002's bet is that a property measurable in one step, without a rollout,
predicts long-run climate: contraction under context perturbation, or a
spectral ratio that moves before failure. Neither has been tested as a
predictor in any source here. The nearest evidence is loose: stable weather
models are not uniformly denoisers ([[lehmann2026can]]), and the sources
disagree on which band fails. A gridworld with a known generator can settle
both, because its climate has a ground truth and a measurable noise floor.
