---
kind: concept
name: "Spectral Rollout Diagnostics"
status: seedling
added: "2026-11-24"
sources: [lippe2023pderefiner, lehmann2026can, li2026freqforcing]
related_concepts: ["Long-Rollout Evaluation", "Exposure Bias", "Diffusion World Model"]
related_experiments: ["2026-11-23-gw-weather-climate-eval"]
tags: [weather-climate, spectrum, high-frequency, rollout-stability, evaluation]
---

# Spectral Rollout Diagnostics

## Definition

Using the radial power spectrum of a predicted state, as a ratio against the
reference spectrum, to diagnose or predict rollout failure: energy missing from
or excess in a band of spatial frequencies, tracked along rollout steps.

## Why it matters here

ADR 0002's "spectral failure predictor" crossover would compute a radial
spectrum per rollout step and ask whether a rising high-frequency energy ratio
precedes the first illegal state. What the sources show so far:

- [[lippe2023pderefiner]] (PDEs, NeurIPS 2023): one-step MSE models are accurate
  only in the dominant low-wavenumber band and neglect low-amplitude (mostly
  high-frequency) components, which limit rollout time through nonlinear
  interaction. A model that fits the whole spectrum (denoising refinement) rolls
  out about a third longer on KS. The evidence is the spectrum of one-step
  prediction and error and a resolution ablation; no spectrum per rollout step
  is shown to rise ahead of failure.
- [[lehmann2026can]] (weather, two-year rollouts): the small-scale energy ratio
  against ERA5 separates regimes: unstable models amplify small scales (ratios
  above 1, up to 1e2 to 1e6) and stable models are blurry (below 1).

- [[li2026freqforcing]] (text-to-video, Self-Forcing 1.3B, 60 s runs): the
  drifting band is the lowest one. A short-time Fourier transform of attention
  outputs and decoded frames over generation time shows DC and low-band
  energy falling away (colour fades, layout loosens) while the high band
  "becomes unstable" (jitter). A larger attention sink slows the drift. This
  is a time-resolved band picture shown as curves only, with no per-step
  ratio against a reference, no noise floor and no lead-time test; the
  spectrum is used to design a training-free correction (anchoring the low
  band to early frames), not to predict failure.

The first two disagree in sign (missing tail against excess tail), and
FreqForcing adds a third case (lost low band), so a usable
predictor is the ratio's distance from 1 in log terms, in either direction,
and has to be calibrated against the noise floor of two real samples. Neither
source tests lead time: whether the spectrum moves before the first failure.

## Connections

- [[Long-Rollout Evaluation]]: spectra are one of the borrowed failure-mode
  metrics.
- [[Exposure Bias]]: spectral drift is one signature of compounding error.
  [[li2026freqforcing]] is the first video-rollout source here that frames
  it that way.
- A counterpoint from video: the Fréchet Forcing authors note their coarse
  perceptual feature space lets "fine periodic textures" develop that none of
  their metrics registers ([[zhao2026frechet]]), which is what a spectrum would catch.
