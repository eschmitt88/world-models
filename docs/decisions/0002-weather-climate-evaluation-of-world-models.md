# 0002 — "Weather and climate" evaluation of world models (weather-ML crossover)

- **Date:** 2026-11-23
- **Status:** accepted (agency max). The paper direction is conditional on the first evaluation's kill criteria.

## Context

The user asked whether the ML weather/climate literature, which has worked
hard on long autoregressive rollouts, offers a publishable crossover for world
models. Inputs: a two-round Fable consultation and a verified web survey
(`raw/_candidates/2026-11-23-weather-rollout-crossover.md`).

What the survey found in weather/climate ML that action-conditioned world
models lack:
- a **weather vs climate** split: initial-value skill vs long-run statistics
  (ACE/ACE2, LUCIE, Spherical DYffusion, ArchesWeather multi-decadal);
- **calibration of ensembles**: spread-skill ratio vs lead time, rank
  histograms, CRPS (GenCast, AIFS-CRPS, FGN, FourCastNet 3);
- **contraction** as a predictor of stability: stable emulators denoise
  noisy inputs (year-long benchmark 2605.30184);
- **invariant-measure priors** used as a denoiser during rollout
  (Thermalizer 2503.18731).

None of these was found applied to action-conditioned world models.
Nearest neighbours: Fréchet Forcing 2610.25666 (matches *feature*
distributions), FreqForcing 2607.27110 (low-frequency drift), CaliBench
2608.16829 (single-event outcome calibration), and C3 2512.05927 (scoring
rules, no long rollouts).

## Decision

Pursue **one paper**. Claim: *in action-conditioned world models,
short-horizon accuracy does not determine long-horizon climate, and a one-step
contraction property measured without rollouts predicts which models
survive.* The gridworld's edge is ground truth. Decoded state is scored
against a known generator, with unlimited independent samples, a measurable
estimator noise floor, and a controllable climate (generator knobs).

The evaluation has four parts:
1. **Weather:** determined-cell accuracy after re-registration, as *skill*
   against a chance/map-prior baseline, and the lead at which skill hits zero.
2. **Climate:** structural statistics of *imagined* cells (never shown in the
   real context), binned by rollout step: tile occupancy (sanity tier),
   2×2 wall/floor/landmark patterns, same-colour floor adjacency (rooms are
   one colour), and landmark density. Distance is JS divergence against real
   frames through the same crop and decoder, with a noise floor from two
   independent real samples.
3. **Calibration:** ensembles of 8 members (same context and actions),
   per-cell disagreement by class (seen / unseen), and its correlation with
   error.
4. **Contraction probe:** a one-step gain under context perturbation.
   Gaussian ε ∈ {0.05, 0.1, 0.2}, with DF told the true noise level and told
   "clean"; plus on-manifold perturbations (a one-tile shift, a one-tile
   flip). The gain is correlated with survival across checkpoint snapshots.

**Method arm** (inside the paper only if it helps): *snap-to-tile* every 8
steps, i.e. decode and re-render, the free exact projection onto the frame
manifold. A learned Thermalizer prior earns a night only if snapping
improves climate or survival. Fable's caution: the dominant failure is legal
frames in the wrong place, which a context-free prior cannot fix.

## Kill criteria (any one drops the framing)

- Climate adds nothing: DF and TF order the same on every long-horizon
  measure, *and* climate distance stays at the noise floor until accuracy has
  already collapsed.
- No climate drift at all to 384 frames in either arm, so the only failure is
  odometry.
- Contraction gain does not discriminate, or orders survival the wrong way.
- Spread on determined cells does not correlate with error.

Confound to check first: DF vs TF is valid only at matched one-step accuracy.

If both of the first two kills land, drop the crossover and keep memory and
imagination (`2026-11-23-gw-memory-equal-compute`) as the main line.
