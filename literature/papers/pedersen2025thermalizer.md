---
kind: paper
title: "Thermalizer: Stable autoregressive neural emulation of spatiotemporal chaos"
authors: ["Christian Pedersen", "Laure Zanna", "Joan Bruna"]
institutions: ["Courant Institute of Mathematical Sciences, New York University", "Center for Data Science, New York University"]
year: 2025
venue: "ICML 2025 (PMLR 42; arXiv v2, 8 Jul 2025)"
peer_reviewed: true
url: https://arxiv.org/abs/2503.18731
code_url: https://github.com/Chris-Pedersen/thermalizer
citations:
source: "raw/papers/pedersen2025thermalizer.pdf"
added: "2026-11-24"
relevance: 4
credibility: 4
status: read
related_experiments: ["2026-11-23-gw-drift-df-vs-tf", "2026-11-23-gw-weather-climate-eval"]
related_concepts: ["Long-Rollout Evaluation", "Exposure Bias", "Diffusion Forcing"]
tags: [weather-climate, invariant-measure, denoiser, rollout-stability, chaotic-systems, diffusion]
---

# Thermalizer: Stable autoregressive neural emulation of spatiotemporal chaos

## TL;DR

A pretrained black-box autoregressive emulator is stabilised at inference time
by a separately trained diffusion model of the system's stationary
distribution. When a classifier head says the current state looks noisy
(off the attractor), a few reverse-diffusion steps pull it back ("thermalizing").
Tested on two 2D turbulent PDE systems; emulators that diverge at about 2k to
5k steps stay stable to 10k steps, and to 1e5 steps in one spectrum figure.
It is the nearest prior art for an invariant-measure prior used as a denoiser
during rollout (ADR 0002's method arm).

## Claims

- Unstable emulator rollouts are out-of-distribution drift: the emulator's own
  stationary law differs from the training law, so the test-error distribution
  is shifted (Eqs. 2 and 3).
- Reverse diffusion started at the right low noise level approximates a
  transport back to the stationary measure and, if the noise level is
  estimated correctly, contracts KL towards equilibrium (Eq. 7, data-processing
  argument; it assumes Gaussian-like errors).
- Thermalizing extends stable rollouts by "two orders of magnitude" (v2
  abstract). Version 1's abstract says "an order of magnitude". The quoted
  10x and 100x are the same claim at two arXiv versions, not a lay-versus-
  abstract split.
- The correction is cheap and does not disturb dynamics: autocorrelation is
  unchanged (Fig. 5).
- Validation MSE does not predict long-term stability (App. B.3, Fig. 7): the
  48M Unet has far lower short-horizon MSE than the 0.5M dilated ResNet yet
  goes unstable sooner, and stability varies with weight initialisation.

## Methods

- Systems: 2D Kolmogorov flow and a 2-layer quasi-geostrophic flow. 500k
  numerical trajectories, 450k for training.
- Emulators: a 48M Unet and a 0.5M dilated ResNet, residual one-step
  prediction trained on 2 to 4 unrolled steps plus tiny noise (tau = 1e-5).
- Thermalizer: a 53M Unet diffusion model of single snapshots with an added
  classifier head that predicts the noise level of the input. At each step the
  head runs (60% of an emulator step's cost); if predicted level s_pred exceeds
  s_init, run s_pred - s_stop DDPM denoising steps. Training spans all noise
  levels even though inference uses s of about 20 or less.
- Metrics: MSE to the numerical model, radially averaged kinetic-energy
  spectra over 20 to 40 trajectories, autocorrelation, count of thermalization
  steps.

## Results

- Emulators diverge at about 2000 to 5000 steps (Kolmogorov Unet visibly at
  about 3500 to 5000; QG at about 4000), except the Kolmogorov DRN, which
  often lasts longer. All thermalized trajectories stay stable for the full 10k
  steps (Fig. 4).
- Thermalized MSE sits near a constant of about 1: stable but decorrelated from
  the numerical trajectory, as expected for chaos. The method restores the
  attractor, not the trajectory.
- QG Unet needs only about 140 thermalization steps per 10k-step trajectory.
- At 1e5 steps (Fig. 6) the thermalized spectra match the numerical model, with
  a persistent low-energy offset for QG. No un-thermalized baseline is shown at
  that length, since it diverged long before.

## Critique / open questions

- What "extends by 10x or 100x" measures: the ratio of the tested thermalized
  length (1e4 main, 1e5 once) to the emulators' divergence step (about 2e3 to
  5e3). The thermalized runs were stopped, not observed to fail, so it is a
  lower bound, and the ratio is closer to 2 to 5x at 1e4 and 20 to 50x at 1e5
  than a clean 100x.
- Hyperparameters s_init and s_stop are tuned per emulator and thermalizer
  pair by brute-force search (their stated limitation).
- Narrow test bed: 20 to 40 trajectories, two systems, both time-stationary,
  fully observed 2D fields with periodic boundaries.
- Theory assumes isotropic Gaussian errors; real emulator errors are not.
  Their remedy is future work (flow matching).
- It restores the marginal distribution, and says so; it does not claim to
  fix errors that are in-distribution but wrong for the conditioning context.
  An action-conditioned world model's legal-but-wrong frame is exactly that,
  and a context-free frame prior cannot see it.
- The prior is a model of single snapshots; there is no conditioning on
  forcing or actions (they list non-autonomous forcing as the next step).
- No comparison with simpler stabilisers (training noise, longer unrolls) at
  matched compute, apart from the baseline emulators themselves.

## Trust signals

- **Credibility:** 4, a peer-reviewed ICML 2025 paper from NYU Courant and
  CDS with code released; narrow experiments (two systems, no seeds
  reported) and no citation count I could establish.

## Follow-up

**Relevance:** 4, the nearest published form of the "invariant-measure prior
as denoiser during rollout" crossover in ADR 0002, and independent evidence
for its first claim; it shifts [[Long-Rollout Evaluation]] but no video
version exists.
- **Supports ADR 0002 (claim 1):** App. B.3 shows lower validation MSE with
  earlier long-run instability, a clean physics instance of "short-horizon
  accuracy does not determine long-horizon behaviour". It also shows that
  stability depends on seed.
- **Supports ADR 0002 (method arm):** the idea transfers directly. Snap-to-tile
  is an exact projection onto the frame manifold, the thing Thermalizer
  approximates with a learned prior; Thermalizer's own metric (spectra,
  MSE plateau) is the template for distance to the climate.
- **Undercuts or limits it:** the prior only fixes off-attractor states. The
  ADR's caution (legal frames in the wrong place) is consistent with this;
  a learned prior earns its night only if snap-to-tile helps first. Its
  noise-level classifier is also a usable drift detector, a cheaper alternative
  to the contraction probe for deciding when to correct.
- **Does not test** contraction of the emulator itself (the contraction in
  Eq. 7 is of the thermalizer step), so it does not bear on the contraction-probe
  claim. Cite as [[Exposure Bias]] prior art from the physics side.
- Correction to the candidate entry: quote "two orders of magnitude" (v2) and
  note v1 said "an order of magnitude"; say what the length ratio actually is.
