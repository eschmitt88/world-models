---
kind: paper
title: "PDE-Refiner: Achieving Accurate Long Rollouts with Neural PDE Solvers"
authors: ["Phillip Lippe", "Bastiaan S. Veeling", "Paris Perdikaris", "Richard E. Turner", "Johannes Brandstetter"]
institutions: ["Microsoft Research AI4Science"]
year: 2023
venue: "NeurIPS 2023 (arXiv v2, 21 Oct 2023)"
peer_reviewed: true
url: https://arxiv.org/abs/2308.05732
code_url: https://github.com/microsoft/pdearena
citations:
source: "raw/papers/lippe2023pderefiner.pdf"
added: "2026-11-24"
relevance: 4
credibility: 4
status: read
related_experiments: ["2026-11-23-gw-weather-climate-eval"]
related_concepts: ["Spectral Rollout Diagnostics", "Long-Rollout Evaluation", "Exposure Bias", "Ensemble Calibration", "Conservation by Construction"]
tags: [pde, rollout-stability, spectrum, high-frequency, diffusion, refinement, uncertainty, weather-climate]
---

# PDE-Refiner: Achieving Accurate Long Rollouts with Neural PDE Solvers

## TL;DR

On the 1D Kuramoto-Sivashinsky (KS) equation and 2D Kolmogorov flow, the paper
argues that one-step MSE training fits only the high-amplitude, low-wavenumber
part of the spectrum and neglects low-amplitude (mostly high-frequency)
components that still drive long-run error. PDE-Refiner replaces the one-step
prediction with a few denoising refinement steps at a decreasing noise ladder
and stays accurate longer (about 98 s against 75 s on KS). It is the evidence
behind ADR 0002's spectral predictor idea, but it shows the spectrum of
*one-step predictions and errors*, not a spectrum tracked along a rollout.

## Claims

- All tested unrolling schemes (history input, pushforward, Sobolev loss, MNO,
  invariance correction) share one pitfall: they neglect low-amplitude spatial
  frequencies, which "have minimal immediate impact" but "still impact long-term
  dynamics".
- Mechanism offered: in KS the nonlinear term u*u_x makes all frequencies
  interact, so "high-frequency errors" propagate into lower frequencies and
  matter more as the rollout lengthens. Stated as a deduction from Fig. 1,
  tested by ablation (below), not by a direct per-step measurement.
- A denoising objective with a very low minimum noise (sigma_min^2 = 2e-7 on
  KS) weights all frequencies equally, because independent Gaussian noise is
  flat across frequency. The added noise also acts as spectral data
  augmentation.
- The refinement ensemble's own sample divergence time predicts how long a
  rollout stays accurate (R^2 = 0.86).

## Methods

- Backbone: a modern U-Net (hidden 64, 3 downsamplings). FNO and dilated
  ResNet in the appendix; FNO gains are smaller because its layers cut high
  frequencies.
- One-step prediction at step k=0 (as in MSE training), then K=3 refinement
  steps in which the previous prediction is noised with sigma_k (exponentially
  decreasing) and the model predicts the added noise. Training uses ground
  truth at every refinement level.
- KS: 2048 training and 128 test trajectories, 256 grid points, every fourth
  time step. 2D: Kolmogorov flow at Reynolds number 1000, 64x64, 128 training
  and 16 test trajectories. Metric: "high-correlation time", when mean Pearson
  correlation with the ground truth drops below 0.9 and 0.8; five seeds on KS.
- Baselines: MSE, two-step history, 4x parameters, 5-model ensemble,
  pushforward, Sobolev loss, MNO, invariance correction (zero mean and cut
  wavenumbers above 60), error-prediction model, plain DDPM variants.

## Results

- KS: MSE U-Net 75 s; PDE-Refiner nearly 100 s (Fig. 3). One refinement step is
  already 20% above the best baseline; gains flatten at 3 to 4 steps. 4x
  parameters gives about 5%; a 5-model MSE ensemble stays under 80 s.
- Spectrum evidence (Fig. 1 and 4): the MSE model is accurate only in a small,
  high-amplitude band (wavenumbers 1 to about 25 for the dominant dynamics);
  PDE-Refiner's refinement steps 2 and 3 cut error at low-amplitude wavenumbers
  without improving the dominant wavenumber 7 beyond step 1. The improvement
  includes the lowest wavenumbers (1 to 6), not only high ones.
- Resolution ablation (Fig. 5): MSE rollout time is flat from 32 to 256 grid
  points, PDE-Refiner gains as resolution (and so high-frequency content)
  rises; at 32 points the two match. This is the cleanest causal support.
- Pushforward does not help here (cannot learn the low-amplitude content as no
  gradient flows through the predicted input). Zeroing the mean and truncating
  above wavenumber 60 on the MSE model "does not improve", so the problem is not
  an overestimate of high frequencies but accurate modelling of a broader spectrum.
- 2D Kolmogorov: correlation above 0.8 for 10.66 s against 9.66 s for the MSE
  U-Net, beating DNS and hybrid solvers. Smaller gain: a shorter, flatter
  spectrum.
- Uncertainty (Fig. 6): over 32 sampled rollouts per test trajectory, the time
  when samples decorrelate (cross-correlation below 0.8) tracks the
  ground-truth accurate time with R^2 = 0.86, better than input modulation, with
  a single trained model.
- Averaging 16 samples per rollout step ("3 steps - Mean") barely helps, so the
  model mostly predicts single modes.

## Critique / open questions

- The spectra shown are of the one-step prediction, its error and the added
  noise. Fig. 4's right panel plots the spectrum of the prediction "at 100s"
  for the MSE model, PDE-Refiner and ground truth, but the text does not
  describe it, and nowhere is a spectrum tracked per rollout step or shown to
  rise before the first failure. So the paper supports "neglected
  low-amplitude components limit rollout length", not a rising high-frequency
  energy ratio as a per-step failure predictor.
- "Failure" is correlation dropping below 0.8 or 0.9 on chaotic trajectories,
  an initial-value (weather) criterion. It says nothing about long-run
  statistics (climate) or states that stay plausible but wrong.
- Direction is mixed. Here the failure is a *missing* low-amplitude signal
  (an under-fit tail). [[lehmann2026can]] finds weather models blowing up with
  *excess* small-scale energy (ratios above 1) and stable ones blurry
  (below 1). A spectral predictor must be two-sided: ratio away from 1 in
  either direction.
- Deterministic, low-dimensional PDEs on fixed periodic grids with test data
  "from a similar domain as the training"; extrapolation and image-like data
  are untested. The 2D set is tiny (16 test trajectories). Transferring to
  discrete tile worlds, where high frequency means edges and tile boundaries,
  is an assumption.
- Gain over a strong MSE baseline in 2D is 10%, and PDE-Refiner is 4x slower per
  step (four model calls).
- Citation count not established.

## Trust signals

- **Credibility:** 4, a Microsoft Research AI4Science paper at NeurIPS 2023
  (peer reviewed), code released in PDEArena, strong tuned baselines and
  five-seed error bars on KS; held at 4 because I could not establish a citation
  count and the 2D evidence is thin.

## Follow-up

**Relevance:** 4, material new evidence for a spectral failure diagnostic
(new concept [[Spectral Rollout Diagnostics]]) behind an ADR 0002 crossover, plus a
usable uncertainty-tracks-error result; does not itself test a per-step predictor.
- **Supports ADR 0002 (spectral failure predictor), partly:** the explanation that
  rollout error comes from neglected low-amplitude frequencies is stated and
  ablated (resolution, loss comparisons, invariance correction). The
  quantity to compute per rollout step is not shown here, so the predictor
  itself is still untested. Treat the radial spectrum ratio as a hypothesis
  that needs a null (no ratio rise before the first illegal state) in the kill
  criteria.
- **Supports ADR 0002 (kill criterion 4, calibration):** the R^2 = 0.86 between sample
  decorrelation time and true accurate time is a precedent for "spread tracks
  error", at trajectory level on PDEs. See [[Ensemble Calibration]].
- **Undercuts the method arm only mildly:** exact invariance correction did not extend
  the rollout (see [[Conservation by Construction]]); the same may hold for
  snap-to-tile.
- **Does not pre-empt:** no action conditioning, no climate statistics, no
  contraction probe.
- Pushforward did nothing here, which is a data point for [[Exposure Bias]] in
  PDEs, against the usual claim that training on own outputs is the fix.
