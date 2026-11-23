---
kind: candidates
topic: "long autoregressive rollouts: what ML weather and climate models learned, and which ideas are untested in action-conditioned world models"
discovered: 2026-11-23
source: discover
n_requested: 20
n_returned: 20
---

All arXiv IDs were fetched and the title checked on 2026-11-23. "verified: abstract" means only the abstract page was read; numbers not on the abstract page are marked as such.

## 1. GraphCast: Learning skillful medium-range global weather forecasting

- url: https://arxiv.org/abs/2212.12794
- type: paper
- summary: GNN, 10-day 0.25 degree forecasts. Trained on autoregressive rollouts of increasing length (curriculum up to 12 steps, about 3 days; from the paper body, not re-fetched) so the loss sees its own outputs.
- reason: The canonical multi-step (unrolled) fine-tuning recipe; the weather analogue of Self Forcing.
- verified: abstract (title and 10-day horizon only)

## 2. Pangu-Weather: A 3D High-Resolution Model for Fast and Accurate Global Weather Forecast

- url: https://arxiv.org/abs/2211.02556
- type: paper
- summary: Separate models for 1 h, 3 h, 6 h and 24 h lead times, composed greedily ("hierarchical temporal aggregation") so that a 7-day forecast takes few autoregressive steps.
- reason: Fewer AR steps instead of better steps; no action-conditioned world model composes multi-stride models.
- verified: abstract (the aggregation claim; stride list from the paper body)

## 3. FuXi: A cascade machine learning forecasting system for 15-day global weather forecast

- url: https://arxiv.org/abs/2306.12873
- type: paper
- summary: Cascade of models fine-tuned per lead-time window, because one model could not be good at both short and long lead times. 15 days, 6 h steps; matches ECMWF ensemble mean at 15 days.
- reason: Evidence that a long horizon wants different specialists, not one model.
- verified: abstract

## 4. Stormer: Scaling transformer neural networks for skillful and reliable medium-range weather forecasting

- url: https://arxiv.org/abs/2312.03876
- type: paper
- summary: Randomized-horizon training objective; at inference several lead-time paths to the same target time are combined. Beats prior methods beyond 7 days on WeatherBench 2.
- reason: A cheap, single-model version of Pangu aggregation.
- verified: abstract

## 5. Message Passing Neural PDE Solvers (pushforward trick)

- url: https://arxiv.org/abs/2202.03376
- type: paper
- summary: Frames error accumulation as a zero-stability / domain-shift problem; trains on the model's own detached prediction (the "pushforward trick"). The abstract page does not name the trick; it is in the body.
- reason: Origin of stop-gradient self-rollout training. Video analogues: Self Forcing, below.
- verified: abstract (trick named only in paper body)

## 6. PDE-Refiner: Achieving Accurate Long Rollouts with Neural PDE Solvers

- url: https://arxiv.org/abs/2308.05732
- type: paper
- summary: Attributes long-rollout failure to neglected low-amplitude, mostly high-frequency components; multistep diffusion-style refinement of each step fixes it.
- reason: Mechanistic claim linking spectrum to rollout failure; testable on pixel gridworlds.
- verified: abstract

## 7. Spherical Fourier Neural Operators: Learning Stable Dynamics on the Sphere

- url: https://arxiv.org/abs/2306.03838
- type: paper
- summary: Spectral-domain operator with correct sphere geometry; stable autoregressive rollouts for 1,460 steps (one simulated year).
- reason: Architecture-level stability (no aliasing/blow-up). Gridworld analogue: translation-equivariant spectral token mixing.
- verified: abstract

## 8. DYffusion: A Dynamics-informed Diffusion Model for Spatiotemporal Forecasting

- url: https://arxiv.org/abs/2306.01984
- type: paper
- summary: Diffusion "time" is physical time: an interpolator and a forecaster mimic forward and reverse diffusion, so one denoising pass is a step in time. Stable multi-step probabilistic forecasts on SST, Navier-Stokes.
- reason: Differs from diffusion forcing (noise level per frame, independent of time). No action-conditioned world-model version found.
- verified: abstract

## 9. Probabilistic Emulation of a Global Climate Model with Spherical DYffusion

- url: https://arxiv.org/abs/2406.14798
- type: paper
- summary: DYffusion plus SFNO; stable 100-year, 6-hourly simulations; climate biases averaged over 34 fields within half of the reference model's. Code: github.com/Rose-STL-Lab/spherical-dyffusion (named by search result, not fetched).
- reason: Evidence that a diffusion rollout can stay stable for decades. Note: an earlier guess of ID 2506.13264 was wrong (a quantum-annealing paper).
- verified: abstract (via search result and fetch of arxiv.org/abs/2406.14798)

## 10. Rolling Diffusion Models

- url: https://arxiv.org/abs/2402.09470
- type: paper
- summary: Sliding window in which later frames carry more noise (monotone noise ramp across the window); beats standard diffusion when dynamics are complex.
- reason: Close cousin of diffusion forcing; see relation note below.
- verified: abstract

## 11. GenCast: Diffusion-based ensemble forecasting for medium-range weather

- url: https://arxiv.org/abs/2312.15796
- type: paper
- summary: Diffusion ensemble, 15-day, 12 h steps; better than ECMWF ENS on 97.4% of 1320 targets.
- reason: The evaluation template: CRPS, spread-skill, rank histograms over a whole ensemble. Ensemble size, CRPS and spread-skill figures are in the paper body, not re-fetched.
- verified: abstract

## 12. ArchesWeather & ArchesWeatherGen

- url: https://arxiv.org/abs/2412.12971
- type: paper
- summary: Flow-matching generative model maps a deterministic model's prediction to ERA5 distribution; beats IFS ENS and NeuralGCM on headline variables at 1.5 degrees on academic compute.
- reason: Small-compute template (relevant to a one-GPU lab): deterministic backbone plus generative residual.
- verified: abstract

## 13. AIFS-CRPS: Ensemble forecasting using a model trained with a loss function based on the CRPS

- url: https://arxiv.org/abs/2412.15832
- type: paper
- summary: Ensemble forecasts from a model trained directly on a CRPS loss; beats ECMWF's physics ensemble for most variables and lead times, including subseasonal.
- reason: Proper-scoring-rule training in place of diffusion; no world-model counterpart for rollouts found.
- verified: abstract

## 14. Skillful joint probabilistic weather forecasting from marginals (FGN)

- url: https://arxiv.org/abs/2506.10772
- type: paper
- summary: Ensembles via learned model perturbations, trained only on per-location CRPS; captures joint spatial structure despite marginal-only training.
- reason: Cheap one-step stochastic generator; fair-CRPS recipe.
- verified: abstract

## 15. FourCastNet 3: A geometric approach to probabilistic ML weather forecasting at scale

- url: https://arxiv.org/abs/2507.12144
- type: paper
- summary: CRPS in physical and spectral space; stable spectra and good calibration to 60 days; 60-day forecast in under 4 minutes on one GPU.
- reason: Spectral loss plus spectral diagnostics at long lead; the most direct "spectrum as stability signal" example.
- verified: abstract

## 16. ACE: A fast, skillful learned global atmospheric model for climate prediction / ACE2

- url: https://arxiv.org/abs/2310.02074
- type: paper
- summary: ACE stable for 100 years, near-conserves column moisture without explicit constraint. ACE2 (https://arxiv.org/abs/2411.11268, fetched) enforces exact dry-air mass and moisture conservation and steps "stably for arbitrarily many steps"; its CO2 and SST sensitivities are "not entirely realistic".
- reason: Hard conservation constraints plus climate (not weather) evaluation. Repo: github.com/ai2cm/ace (named by search result, not fetched).
- verified: abstract

## 17. Neural General Circulation Models for Weather and Climate

- url: https://arxiv.org/abs/2311.07222
- type: paper
- summary: Differentiable dynamical core plus learned physics; matches ML and physics models for deterministic weather, ensemble weather and climate.
- reason: Hybrid: hard-code what you know (the 2D grid physics), learn the rest. Our gridworld has exactly known dynamics, so this is a natural comparison arm.
- verified: abstract

## 18. LUCIE: Lightweight Uncoupled ClImate Emulator

- url: https://arxiv.org/abs/2405.16297
- type: paper
- summary: Small emulator trained on two years of 6-hourly ERA5; stable for 100 years with 100 members; climatology matches ERA5 including variability.
- reason: Small data and one-GPU scale with a climate-statistics evaluation.
- verified: abstract

## 19. Thermalizer: Stable autoregressive neural emulation of spatiotemporal chaos

- url: https://arxiv.org/abs/2503.18731
- type: paper
- summary: Separate diffusion model of the invariant measure applied as an on-the-fly denoiser during rollout; extends stable horizon by about 10x (abstract) to 100x (lay summary). Hyperparameters tuned per pair.
- reason: Treats the emulator as a black box and corrects drift toward the attractor set; no video or game version found. The most direct fix for "context noise swaps one attractor for another".
- verified: abstract

## 20. Can AI Weather Models Predict Beyond Two Weeks? A Quantitative Benchmark and Analysis of Long Rollouts

- url: https://arxiv.org/abs/2605.30184
- type: paper
- summary: Nine AI weather models, year-long rollouts; failure regimes blow-up, drift, loss of seasonality. Unstable models amplify high-frequency energy; stable models act as denoisers on noisy input.
- reason: A diagnostic taxonomy and a "noise contraction" probe to borrow. Related: https://arxiv.org/abs/2505.02506 (design choices for 10-year stability; stability depends on seed and variable set) and https://arxiv.org/abs/2605.29976 (ArchesWeather under 45-year climate runs).
- verified: abstract

---

# Video / world-model side (already crossed over, or near)

- Diffusion Forcing, https://arxiv.org/abs/2407.01392 (independent per-token noise levels; verified abstract).
- Self Forcing, https://arxiv.org/abs/2506.08009 (train on own outputs; the video pushforward trick; verified).
- Rolling Forcing, https://arxiv.org/abs/2509.25161 (rolling-window joint denoising, attention sink; verified) and FIFO-Diffusion, https://arxiv.org/abs/2405.11473 (diagonal denoising queue, training-free; verified).
- FreqForcing, https://arxiv.org/abs/2607.27110 (frequency analysis of drift: low-frequency energy drift; training-free anchoring; verified).
- Frechet Forcing, https://arxiv.org/abs/2610.25666 (action-conditioned; matches Frechet distance of latent and perceptual feature statistics; stable beyond 4,000 frames; verified).
- Multiplayer world model with RAE, https://arxiv.org/abs/2607.05352 ("distributional quality" steady to 5 minutes; metrics not read; verified abstract).
- WorldProbe, https://arxiv.org/abs/2610.39721 (10 models, 180 s horizons, memory weak; verified).
- Calibration: C3 https://arxiv.org/abs/2512.05927 (strictly proper scoring rules, action-conditioned, robot data, no long rollout); S-QUBED https://arxiv.org/abs/2510.02571; CaliBench https://arxiv.org/abs/2608.16829 (outcome distributions on single events: dice, roulette; no weather metrics); EO-WM https://arxiv.org/abs/2606.27277 (video DiT for Earth observation; reports no CRPS in abstract).
- Matrix-Game 2.0, https://arxiv.org/abs/2508.13009 (verified).

## Relation: Diffusion Forcing, Rolling Diffusion, DYffusion

Diffusion Forcing samples an independent noise level per frame in training, so any noise pattern is in-distribution; Rolling Diffusion fixes one pattern (linear ramp over a sliding window, noisier for later frames) in both training and sampling, which gives constant-cost infinite rollout and an uncertainty-grows-with-lead-time prior. So Rolling Diffusion is Diffusion Forcing with a fixed schedule; Rolling Forcing and FIFO are the video descendants. DYffusion ties noise level to physical time through an interpolator and is a different family. No source found applies these to action-conditioned game worlds with a state-decodable evaluation.

## Findings on evaluation (what has and has not crossed)

- Used in video/world-model work: CRPS-style proper scoring (C3), calibration and rank correlation (S-QUBED), outcome distribution TV (CaliBench), Frechet distance on feature statistics (Frechet Forcing), frequency analysis of drift (FreqForcing), slope of quality over time (WorldProbe).
- Not found in any world-model paper: rank histograms (only seen in radar nowcasting), spread-skill ratio over rollout lead time for an action-conditioned model, radial energy spectra as a long-rollout diagnostic or failure predictor, state-occupancy "climate" statistics versus ground truth, and "weather vs climate" separation.

## Candidate crossovers (action-conditioned, 2D gridworld, one GPU; most novel first)

1. Thermalizer-style denoiser: train a frame-prior diffusion model on true-state frames, apply it every k steps during rollout. Nearest: Thermalizer (fluids only). Test: first-illegal-state step and state-occupancy distance beat the context-noise knob at matched compute.
2. Climate statistics: decoded-state occupancy histogram over 10x-training-length rollouts against the true policy-induced distribution (TV or KL), plus drift of occupancy toward or away from the training marginal. Nearest: CaliBench, Frechet Forcing (feature statistics, not decoded state). Test: DF versus TF models can have equal short-horizon accuracy and different occupancy distance.
3. Spectral failure predictor: radial power spectrum per rollout step; high-frequency energy ratio crossing a threshold predicts first illegal state a fixed number of steps ahead (AUC above 0.8). Nearest: FreqForcing (low-frequency drift), the weather benchmark 2605.30184 (high-frequency amplification).
4. Contraction probe: add noise eps to the context, measure one-step output error ratio. Test: contraction below 1 predicts survival time across DF and TF checkpoints (rank correlation). Nearest: 2605.30184 (denoiser observation); no world-model use found.
5. Conservation by construction: invariant (one agent, fixed wall and item counts) enforced by projection onto valid layouts or an invariant-weighted loss. Nearest: ACE2. Test: invariant-violation rate over horizon, and whether it removes the darkness-like collapse analogue.
6. Fair-CRPS one-step generator (FGN or AIFS-CRPS style) in place of diffusion on a stochastic gridworld: spread-skill ratio and rank histograms versus lead time for DF, TF and CRPS models. Nearest: C3 (calibration but not rollouts), GenCast.
7. Multi-stride aggregation (Pangu or Stormer): models or one randomized-stride model at strides 1, 4, 16 frames composed greedily with action chunks. Test: horizon at fixed illegal-state rate versus a single-stride model with equal parameters.
8. Ensemble-mean feedback (ArchesWeatherGen long runs): average N sampled next frames before feeding back. Test: whether averaging causes blur-collapse (spectral decay) while reducing drift; a spread-skill trade-off.

Most novel: 1, 2, 4 (no near neighbour in video). 3 and 8 have partial neighbours. Not novel by itself: unrolled or self-rollout training (Self Forcing, MP-PDE) and rolling windows (Rolling Forcing).
