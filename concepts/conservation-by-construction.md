---
kind: concept
name: "Conservation by Construction"
status: seedling
added: "2026-11-24"
sources: [wattmeyer2024ace2, lippe2023pderefiner]
related_concepts: ["Long-Rollout Evaluation", "Exposure Bias"]
related_experiments: ["2026-11-23-gw-weather-climate-eval"]
tags: [weather-climate, conservation, hard-constraints, projection, rollout-stability]
---

# Conservation by Construction

## Definition

Enforcing a known invariant of the true system (a conserved total, a state that
must lie on a manifold) with a fixed correction inside or after the model step, so
the invariant holds exactly at every rollout step instead of being learned or
penalised.

## Why it matters here

ADR 0002's method arm, snap-to-tile (decode and re-render every 8 steps), is the
gridworld's exact projection onto the frame manifold. The reference instance is
the corrector in [[wattmeyer2024ace2]]: clip negatives, shift global dry-air
pressure, rescale precipitation, and take advective moisture as the budget
residual. It runs before the loss, so training sees it; it conserves dry-air mass
and moisture to numerical precision. Two cautions from the same paper: it covers
only the invariants it names (not energy), and exact budgets did not robustly
improve pattern error. Constraints remove one failure mode (budget drift); they do
not make the climate right. Compare the learned alternative, a prior over the
stationary distribution used as a denoiser ([[pedersen2025thermalizer]]).

One negative result: on the KS equation, correcting MSE-trained rollouts for the
known invariants (zero mean, truncate wavenumbers above 60) did not lengthen the
accurate rollout, because the limit was missing low-amplitude content, not an
overestimate ([[lippe2023pderefiner]]). A projection is no cure for a spectrum
that is wrong in the first place.

## Connections

- [[Long-Rollout Evaluation]]: the constraint is judged by whether the long-run
  statistics improve, not only the budget.
- [[Exposure Bias]]: an exact projection stops one class of compounding error but
  cannot fix errors that are legal yet wrong for the context.
