---
kind: paper
title: "SmallWorlds: Assessing Dynamics Understanding of World Models in Isolated Environments"
authors: ["Xinyi Li", "Zaishuo Xia", "Weyl Lu", "Chenjie Hao", "Yubei Chen"]
institutions: ["University of California, Davis", "Open Path AI Foundation"]
year: 2025
venue: "arXiv v1 (preprint)"
peer_reviewed: false
url: https://arxiv.org/abs/2511.23465
code_url:
citations:
source: "raw/papers/li2025smallworlds.pdf"
added: "2026-11-24"
relevance: 4
credibility: 2
status: skimmed
related_experiments: ["2026-11-23-gw-drift-df-vs-tf", "2026-11-23-gw-weather-climate-eval"]
related_concepts: ["Long-Rollout Evaluation", "Exposure Bias", "Diffusion Forcing"]
tags: [benchmark, long-horizon, state-space, rssm, neural-ode, diffusion-forcing, isolated-dynamics]
---

# SmallWorlds: Assessing Dynamics Understanding of World Models in Isolated Environments

## TL;DR

A benchmark of small, fully observable, deterministic worlds (ten MuJoCo
physics tasks plus a keypoint-reprojection task) for scoring world models in
state space with no reward and no policy-collected data. It compares four
backbones (DreamerV3 RSSM, an IRIS-style Transformer, Diffusion Forcing, and a
Neural-ODE motion simulator) on 90-step imagination and on error growth out to
990 steps. It is the nearest published prior art for long-rollout degradation
across architectures in small worlds, and the basis of [[Long-Rollout Evaluation]].

## Claims

- Evaluating a world model inside an RL loop hides its dynamics errors;
  isolating it on clean, parameterised dynamics makes failures diagnosable.
- Diffusion Forcing and MoSim get the lowest error across domains; the
  Transformer beats RSSM.
- MoSim is most precise on physical tasks but its error grows exponentially
  with horizon (one-step-from-previous-state); the other three plateau.
- Diffusion trades precision for robustness: best on Atari, worse than MoSim
  on tasks needing fine accuracy.

## Methods

- Tasks: ten physics tasks (free fall, circular, projectile, inclined plane,
  pendulum, rolling, rotation, spin, elastic collision, bouncing ball), a
  keypoint-reprojection task, plus Atari, Go, Panda, DMControl and
  Memory-Maze converted to state space.
- Models trained on the same randomly collected episodes, no rewards.
- Metric: state-space MSE over 90 imagined steps after 10 ground-truth warm-up
  steps; error vs horizon (1 to 990) on three tasks.
- Diffusion arm is Diffusion Forcing with a 2D transformer denoiser, adapted
  to state vectors.

## Results

- Table 1 (MSE, 90 steps), e.g. Atari Boxing: RSSM 993.6, Transformer 820.1,
  MoSim 1448.9, DF 670.0; Pong: 1094.9 / 545.7 / 2683.9 / 332.8; Spin:
  341.6 / 64.1 / 0.632 / 2.32. MoSim wins on most physics tasks, DF on Atari.
- Fig. 4: MoSim error rises exponentially with horizon; the other three show
  a smaller change from horizon 100 to 990. The authors suggest they "rely
  only on their fundamental understanding of the dynamics" once context is
  lost.

## Critique / open questions

- Single set of numbers with no seeds or error bars; Table 1 reports "best
  imagination performance", so the selection rule is unclear.
- Fully observable, deterministic and state-space: no partial observability,
  no pixels for the main results, so the failure modes of a pixel-space
  generative model (blur, saturation, legal-but-wrong frames) are absent.
- The architectures differ in everything (data tokenisation, size, objective),
  so the DF result is not a matched DF vs teacher-forcing comparison and does
  not isolate [[Exposure Bias]].
- The plateau in Fig. 4 is read as loss of information, but the paper does
  not compare against a climatology or mean-prediction baseline, so it cannot
  say whether a plateau is skill or the model's invariant distribution.
- Text says "six domains" and points to Section 2 for them; they are defined
  in Section 3. Only four of the ten physics tasks appear in Table 1.
- No code or artifacts are named.

## Trust signals

- **Credibility:** 2, a UC Davis arXiv preprint (the group also wrote the
  CVPR 2025 MoSim paper it evaluates, so it is not an independent test of
  that model), with no named code, no peer review, no seeds and no citation
  count I could establish.

## Follow-up

**Relevance:** 4, the closest published long-rollout degradation comparison
across architectures in small worlds, including Diffusion Forcing; it seeds
[[Long-Rollout Evaluation]] and is the prior art ADR 0002 must cite.
- **Overlaps ADR 0002**: error-versus-horizon out to hundreds of steps; DF as
  one arm; ground truth available because the world is small and known;
  state-space scoring rather than reward; the observation that models settle
  into a regime where horizon no longer matters.
- **Differs from ADR 0002**: it scores only mean MSE against the true
  trajectory (the "weather" half), with no distributional "climate"
  statistics against a noise floor, no ensemble calibration, no contraction
  probe and no snap-to-tile arm. It is fully observed and deterministic, where
  ours is a partially observed pixel-space gridworld with imagined (never
  seen) cells. It compares unmatched architectures, not DF against teacher
  forcing at matched one-step accuracy. It does not support a claim that
  weather-ML evaluation is untried for world models, but it does show the
  weather-only half is already published, so ADR 0002's novelty rests on
  the climate, calibration and contraction parts.
- Its MoSim result (exponential growth from one-step-from-previous-state) is
  a ready positive case for the contraction probe: a model with no
  self-correction.
- Cite it in the drift write-up, and use its error-vs-horizon plot shape as
  the baseline figure.
