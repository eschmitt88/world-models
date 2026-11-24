---
kind: experiment
slug: "gw-weather-climate-eval"
date: "2026-11-24"
status: done     # running | done | abandoned
hypothesis: "Short-horizon accuracy does not determine long-horizon climate (DF vs TF or snap vs no-snap dissociate on weather vs climate), and one-step contraction gain predicts rollout survival across checkpoints"
result: "Framing survives with a changed contraction claim: climate drifts and warns early, spread predicts error (rho 0.50-0.87), and the two training regimes fail differently: tf off-manifold (snap-to-tile rescues it) vs df-at-noise-0 on-manifold into a wrong climate (snap useless, context noise fixes it); Gaussian contraction gain orders survival the wrong way, on-manifold shift recovery orders it right"
related_concepts: ["Exposure Bias", "Diffusion Forcing"]
related_literature: ["chen2024diffusion", "huang2025self"]
tags: [gridworld, weather-crossover, calibration, climate, contraction]
# members: only set when kind: ensemble — list parent experiment slugs.
# parent:  only set when this experiment was produced via /propose --expand.
---

# gw-weather-climate-eval

## Hypothesis

Pre-registered from ADR 0002 (weather-ML crossover). It evaluates tonight's
DF and TF checkpoints (`2026-11-23-gw-drift-df-vs-tf`) with no new training.

Predictions:
1. **Dissociation:** at least one pair of arms (DF vs TF, context noise 0 vs
   the DF optimum, snap vs no snap) orders differently on weather (skill
   horizon) than on climate (JS distance of imagined-cell statistics from real).
2. **Climate drifts before collapse:** climate distance rises above the
   real-sample noise floor several frames *before* determined-cell skill
   reaches zero, so it is an early warning.
3. **Calibration:** per-cell ensemble disagreement on determined cells
   correlates positively with error.
4. **Contraction predicts survival:** across TF snapshots, a lower one-step
   gain goes with longer survival (later off-manifold or uniform-frame entry).
5. **Snap** improves survival and the off-manifold rate but does not fix
   accuracy, because the dominant failure is legal frames in the wrong place.

The kill criteria are in ADR 0002. Any one drops the framing.

## Setup

- Config: `config.yaml`
- Code: `src/wm/climate_eval.py`
- Data: held-out maps 2000-2199, generated on the fly

## Result

**Ensemble evaluation** (`results/ensemble.json`): 16 held-out contexts × 8
members, 256 generated frames on the walk script, 32-frame bins (every second
bin shown). Skill = (accuracy − chance) / (1 − chance) on context-determined
cells after re-registration. Climate = JS divergence of 2×2
wall/floor/landmark patterns on *imagined* cells vs real frames along the same
trajectories; the real-vs-real noise floor averages 0.010.

| arm | skill (bins 1,3,5,7) | climate JSD (bins 1,3,5,7) | off-manifold | spread/error ρ | uniform-frame survival (median frames) |
|---|---|---|---|---|---|
| df_cn0 | 0.84 / 0.31 / -0.05 / -0.16 | 0.034 / 0.106 / 0.152 / 0.126 | 0.06 | 0.52 | 82 |
| df_cnbest | 0.94 / 0.63 / 0.48 / 0.30 | 0.021 / 0.025 / 0.022 / 0.017 | 0.05 | 0.87 | 256 |
| tf_cn0 | 0.94 / 0.61 / 0.28 / -0.08 | 0.018 / 0.053 / 0.052 / 0.093 | 0.21 | 0.76 | 213 |
| df_cn0_snap8 | 0.84 / 0.32 / 0.00 / -0.14 | 0.032 / 0.102 / 0.140 / 0.101 | 0.06 | 0.50 | 86 |
| tf_cn0_snap8 | 0.94 / 0.63 / 0.45 / 0.36 | 0.018 / 0.044 / 0.047 / 0.056 | 0.11 | 0.83 | 179 |

`df_cnbest` = df at context noise 0.2, picked automatically from the long eval.

**Contraction probe** (`results/contraction.json`): one generated step from
256 real clips. Gain = output change / input change for Gaussian ε = 0.1 on
the last context frame, with the model told the frame is clean vs told its
true noise level. Plus an on-manifold probe: shift the last context frame by
one tile, then measure tile accuracy of the next frame. Survival comes from
single-member 256-frame walks.

| checkpoint | gain, told clean | gain, told true ε | after a one-tile shift | survival, off-manifold | survival, uniform |
|---|---|---|---|---|---|
| df 46515 | 1.57 | 0.45 | 0.956 (base 0.986) | 256 | 74 |
| tf 8000 | 0.37 | 1.74 | 0.857 (base 0.998) | 92 | 174 |
| tf 16000 | 0.31 | 1.76 | 0.847 (base 1.000) | 113 | 134 |
| tf 24000 | 0.28 | 1.74 | 0.848 (base 1.000) | 74 | 132 |
| tf 24807 | 0.27 | 1.73 | 0.851 (base 1.000) | 76 | 140 |

## Interpretation

Against the ADR 0002 kill criteria:
- **Climate adds information: not killed.** Climate distance rises far
  above the noise floor (up to 25× at the end), and it rises *before* skill
  reaches zero (tf: climate 5× floor at frames 64-96 while skill is still
  0.6). Prediction 2 (early warning) holds.
- **Climate drifts: not killed.**
- **Spread is informative: not killed.** Ensemble disagreement on
  determined cells rank-correlates with error, ρ 0.50-0.87. It is best
  calibrated for df at its noise optimum (0.87).
- **Gaussian contraction: killed in its original form.** Told "clean",
  df *amplifies* a Gaussian perturbation (gain 1.57) and tf *damps* it
  (0.27-0.37), yet df survives off-manifold far longer. Within tf's snapshots
  the gain falls 0.37 → 0.27 with no survival trend (4 points). The
  **on-manifold** probe orders things correctly: after a one-tile shift, df
  recovers (0.956 vs 0.986 base), while tf copies the error (0.85 vs 1.00 base).
  The paper should claim *on-manifold perturbation recovery*, not Gaussian
  contraction. That needs more than two arms to establish.

**The new main finding: two failure modes, and they need different cures.**
- **tf fails off the frame manifold** (off-manifold up to 35% late).
  Truth-free snap-to-tile every 8 frames, the invariant-measure
  projection, rescues it: final skill 0.00 → 0.27 and late climate JSD
  0.18 → 0.10. Prediction 5 ("snap doesn't fix accuracy") is **refuted for
  tf**. This is exactly the Thermalizer's failure class.
- **df at context noise 0 fails *on* the manifold.** Its frames are valid
  (off-manifold 6-8%) but its climate is the worst of all arms (late JSD 0.17).
  It collapses into near-uniform frames (median uniform-frame survival
  ~74 frames). Snapping does nothing for it, and inference context noise
  fixes it (skill 0.29 at the end, climate within ~2-3× the floor).
- So **frame-validity metrics** (off-manifold rate, and by extension
  FID/FVD-style realism) would rank df@0 above tf, while **climate** ranks
  them the other way and matches long-horizon skill. That is the
  weather-crossover paper's claim, supported on 5 arms × 1 seed.

Caveats: one training seed per arm; 16 contexts × 8 members; tf saw about
half as many samples as df.

## Diagnostics

Metric numbers reference `metrics.json` (held-out maps throughout).

- intended_effect_confirmed: partial — climate early warning and calibration hold (metrics.json:*_climate_pat_*, *_spread_err_rho); Gaussian contraction is refuted (gain_df_46515_eps0.1_clean 1.57 vs gain_tf_24807_eps0.1_clean 0.27, opposite to survival)
- leakage_check: all evaluation on held-out maps 2000-2199; climate reference from real frames on the same trajectories plus independent halves for the noise floor (`src/wm/climate_eval.py`)
- overfitting_signal: train=n/a val=n/a gap=n/a — evaluation only (metrics.json)
- delta_from_prior: vs 2026-11-23-gw-drift-df-vs-tf, same checkpoints; adds the failure-mode split (off-manifold for tf, on-manifold climate collapse for df@0) and shows snap helps only tf
- unexpected_findings: tf is contractive to Gaussian context noise yet collapses first; df@0 has the lowest off-manifold rate and the worst climate
- next_candidates:
  - More arms for the on-manifold recovery claim: df snapshots, 2-3 seeds per regime, and mixed-noise training, to see whether on-manifold shift recovery predicts climate survival across ≥6 models.
  - Replace snap with a learned frame prior (a Thermalizer) and test it on tf (off-manifold failure) and df@0 (on-manifold failure); the prediction is that it helps only tf.
## Follow-up

- Kill-criteria verdict and the revised claim feed ADR 0002 (status note added).
