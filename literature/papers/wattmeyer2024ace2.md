---
kind: paper
title: "ACE2: Accurately learning subseasonal to decadal atmospheric variability and forced responses"
authors: ["Oliver Watt-Meyer", "Brian Henn", "Jeremy McGibbon", "Spencer K. Clark", "Anna Kwa", "W. Andre Perkins", "Elynn Wu", "Lucas Harris", "Christopher S. Bretherton"]
institutions: ["Allen Institute for Artificial Intelligence (Ai2), Seattle", "Geophysical Fluid Dynamics Laboratory, NOAA, Princeton"]
year: 2024
venue: "arXiv 2411.11268 (Nov 2024); npj Climate and Atmospheric Science 2025, doi 10.1038/s41612-025-01090-0"
peer_reviewed: true
url: https://arxiv.org/abs/2411.11268
code_url: https://github.com/ai2cm/ace
citations: 42
source: "raw/papers/wattmeyer2024ace2.pdf"
added: "2026-11-24"
relevance: 4
credibility: 4
status: read
related_experiments: ["2026-11-23-gw-weather-climate-eval", "2026-11-23-gw-drift-df-vs-tf"]
related_concepts: ["Long-Rollout Evaluation", "Conservation by Construction", "Exposure Bias"]
tags: [weather-climate, climate-emulator, conservation, hard-constraints, rollout-stability, SFNO]
---

# ACE2: Accurately learning subseasonal to decadal atmospheric variability and forced responses

## TL;DR

A 450M-parameter spherical Fourier neural operator (SFNO) that steps a 1-degree,
8-layer atmosphere forward 6 hours at a time, trained on only two-step (12 h)
rollouts. A "corrector" module inside the model forces exact conservation of
global dry-air mass and column moisture. All eight trained models (four seeds
each on ERA5 and on SHiELD output) were stable indefinitely, one ran 1000 years, but
climate accuracy varied a lot between seeds and did not track validation loss.
It is the clearest worked example of "climate-style" evaluation of an emulator
and of conservation enforced by correction, both used by ADR 0002.

## Claims

- An autoregressive emulator trained on 81 years of ERA5 or of SHiELD (GFDL
  AMIP-style) output reproduces global-mean temperature and water trends
  (R2 of annual global-mean 2 m temperature 0.93 for ACE2-ERA5, against 0.97
  between two SHiELD ensemble members), the ENSO precipitation response, tropical
  cyclone climatology, the MJO and stratospheric polar-vortex variability.
- Dry-air mass and moisture are conserved exactly, by a hard corrector rather
  than a loss penalty (see Methods).
- Stable for 1000 years under repeating climatological forcing: no growing error,
  no drift in total water path, the 100-year time-mean climate is nearly
  identical across periods (Sec. 2.2.7, Fig. 10).
- Not claimed, and stated as limits: the response to SST changes alone or to CO2
  alone is not realistic (Sec. 2.4); energy is not conserved; no response to
  strong forcing (e.g. doubled CO2) is expected since training covers about 80
  years.

## Methods

- Architecture: SFNO, embedding dimension 384 (256 in ACE). 6-hourly steps,
  1 degree, 8 terrain-following layers. Inputs include prescribed SST, sea-ice
  and surface fractions, and CO2 (new in ACE2). About 1500 simulated years per
  wall-clock day.
- Loss: MSE summed over two autoregressive 6 h steps, with per-variable weights.
- **Conservation mechanism (Sec. 4.3, Eqs. 1 to 7).** A corrector sits inside the
  model and runs before the loss, so the network is trained through it and "no
  model bias" is introduced. In order: (1) clip negative moisture, precipitation
  and radiative fluxes to zero; (2) subtract a globally constant surface-pressure
  offset so the global mean of dry-air surface pressure (ps minus g times total
  water path) equals its previous value (Eq. 5); (3) multiply precipitation by one
  global scalar so global-mean moisture closes against evaporation (Eq. 6);
  (4) set the predicted advective water tendency to the residual of the column
  budget, so each column closes exactly (Eq. 7). Order is chosen so later steps do
  not undo earlier ones. This is a sequence of scalar and residual adjustments in
  physical units, not an orthogonal projection, and it is exact only for two
  budgets. Global energy is excluded (its budget is more complex and SHiELD itself
  does not close it).
- Checkpoint selection on climate skill: the channel-mean global RMSE of time
  means over 1940 to 2000 (Eq. 9), not validation loss.
- Evaluation: 81-year runs initialised in 1940; a held-out 10-year test period
  (2001 to 2010) scored by time-mean RMSE against ERA5 or SHiELD; "SHiELD
  reference variability" (the difference between two SHiELD ensemble members) as a
  noise floor; 10-day weather skill on 2020 against GraphCast and IFS; 1000-year
  run. Ensembles are three initial-condition members started a day apart.

## Results

- Time-mean errors over the test period are 1.1 to 1.5 times the SHiELD
  reference variability, "nearly indistinguishable" from the reference climate by
  that yardstick; ACE2-ERA5 total-water-path error over 2020 is 1.05 mm against
  1.09 mm for NeuralGCM. ENSO precipitation-regression RMSE 0.46 and 0.48
  mm/day/K against a 0.54 internal-variability floor, and 0.64 for ACE-climSST.
- Tropical cyclone frequency is overestimated by about 28% against ERA5.
- Weather skill is secondary: about half a day behind IFS at 5 days (850 hPa
  temperature) and another day behind GraphCast.
- Ablation (App. A.5, SHiELD): with no constraints, global dry-air surface pressure
  deviates up to 20 Pa and single-step moisture-budget violations exceed 0.2
  mm/day; with the corrector both close. Time-mean surface-pressure bias drops
  robustly; other climate metrics do not change robustly. A residual drift under
  2 Pa per 10 years remains, which the authors attribute to rounding.
- Seeds and checkpoints (App. A.4): all trained models were indefinitely stable
  but climate accuracy "can vary significantly". Validation loss falls steadily
  while climate error does not, and can deteriorate with more training. One of
  four SHiELD seeds (RS3) has much poorer skill.
- ERA5 itself does not close its budgets (dry-air pressure varies by up to 400 Pa
  early on); the corrector still makes the emulator conserve exactly.
- Coarser training: 4-degree emulators are slightly worse than 1-degree but far
  better than a 4-degree SHiELD run.

## Related prior work: ACE (v1)

ACE (arXiv 2310.02074, https://arxiv.org/abs/2310.02074), "ACE: A fast, skillful
learned global atmospheric model for climate prediction", Watt-Meyer et al. 2023,
a 200M-parameter emulator, called ACE-climSST in this paper. Its abstract claims
"stable for 100 years" and "nearly conserves column moisture without explicit
constraints". Not fetched or ingested separately. ACE2 qualifies the stability
claim: ACE-climSST's 100-year runs had unrealistic swings in the global-mean
seasonal-cycle amplitude, global-mean surface pressure drifted, and the global
moisture budget broke by up to 0.1 mm/day on single steps. So "stable" in ACE
means "did not blow up", which is weaker than "no drift". ACE2 adds the hard
corrector, historical SST variability and CO2 as input, and an 81-year evaluation.

## Critique / open questions

- The conservation is narrow. Two global or column budgets are closed; energy,
  angular momentum and regional budgets are not. A "conservation by construction"
  claim has to name which invariants it enforces.
- Exact conservation does not by itself give a good climate: biases in
  precipitation and evaporation were not robustly reduced, and the pattern error
  did not change. Constraints fix the budget, not the pattern.
- Selection by climate skill uses the evaluation period (1940 to 2000), so "best
  inference performance" is partly fitted to the metric; the held-out 10 years
  and 2020 are the clean tests. Checkpoint and seed variance means a single
  trained model is weak evidence.
- Ensembles are three initial-condition members, used for error bars. There is no
  probabilistic skill score, spread-skill ratio or rank histogram, so this paper
  speaks to climate, not to calibration.
- The model is deterministic and forced by prescribed SST, CO2 and surface fields:
  the boundary conditions play the role that actions play in a world model, which
  is the real analogy, but they are smooth, low-dimensional and known in advance.
- "Stable indefinitely" is established by running up to 1000 years under repeating
  forcing, not by a proof, and only on this SFNO recipe.

## Trust signals

- **Credibility:** 4, peer-reviewed (npj Climate and Atmospheric Science 2025) from
  Ai2 and NOAA GFDL with code and training data released and an ablation and seed
  ensemble reported, but self-evaluated, 42 citations (Crossref), and a narrow
  model family.

## Follow-up

**Relevance:** 4, material new evidence for [[Long-Rollout Evaluation]] (climate
accuracy decoupled from training loss, noise-floor scoring) and the source for
[[Conservation by Construction]], which this paper seeds.
- **Supports ADR 0002 (claim 1 and part 2, climate):** Appendix A.4 gives a second
  physics instance of "short-horizon accuracy does not determine long-horizon
  statistics" (two-step loss falls while 10-year bias does not). Its yardstick, an
  error expressed as a multiple of the reference model's own two-sample variability,
  is the template for the JS-divergence noise floor in the ADR.
- **Informs ADR 0002 (method arm):** the corrector is the nearest precedent for
  snap-to-tile, as it is applied inside the model and before the loss so the
  network trains through it. That is a design choice to copy or test: snap at
  train time, or only at inference. It also shows an exact fix leaves pattern error
  alone, which is the ADR's own caution about legal frames in the wrong place.
- **Does not cover:** ensemble calibration (part 3) or contraction (part 4).
- Correction to the entry: ACE's "stable for 100 years" is true only in the sense
  of no blow-up; ACE2 reports seasonal-cycle amplitude swings and pressure drift
  in the same runs.
