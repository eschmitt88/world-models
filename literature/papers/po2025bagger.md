---
kind: paper
title: "BAgger: Backwards Aggregation for Mitigating Drift in Autoregressive Video Diffusion Models"
authors: ["Ryan Po", "Eric Ryan Chan", "Changan Chen", "Gordon Wetzstein"]
institutions: ["Stanford University"]
year: 2025
venue: "arXiv v1 (preprint)"
peer_reviewed: false
url: https://arxiv.org/abs/2512.12080
code_url:
citations:
source: "raw/papers/po2025bagger.pdf"
added: "2026-11-24"
relevance: 4
credibility: 3
status: skimmed
related_experiments: ["2026-11-23-gw-drift-df-vs-tf"]
related_concepts: ["Exposure Bias", "Diffusion Forcing", "Few-Step Distillation for Real-Time Rollout"]
tags: [exposure-bias, dagger, reversed-rollouts, self-correction, no-distillation, autoregressive-video]
---

# BAgger: Backwards Aggregation for Mitigating Drift in Autoregressive Video Diffusion Models

## TL;DR

A DAgger-style fix for [[Exposure Bias]] that needs no teacher and no
distribution-matching loss. Roll out the current causal model from a real
first chunk, time-reverse the drifted rollout so it runs from degraded to
clean, add it to the training set with a reversed prompt, and fine-tune with
the ordinary diffusion-forcing loss. Three rounds beat DF, DF with context
noise and History Guidance on 50 s text-to-video, at 1.3B scale.

## Claims

- Reversed self-rollouts are free "corrective trajectories": they show how to
  move from a drifted state back toward clean data, with no expert.
- Training only needs the standard flow-matching loss, so it avoids a
  bidirectional teacher, backprop through time and mode-seeking losses that
  can hurt diversity.
- It beats DF and History Guidance at equal data and iterations, and has
  higher motion scores than Self Forcing without a 14B teacher.

## Methods

- Base: Wan2.1 1.3B made block-causal (3 latent frames per chunk, 7 chunks),
  trained with DF on 55K Pexels clips (16K iterations, batch 96).
- Each round: generate 27K rollouts (half the seed size) from one real chunk
  with the current model, decode, reverse in pixel space, re-encode, aggregate
  with the seed data, and fine-tune. In corrective clips the first 4 chunks are
  clean prefix with no loss and the loss covers the last 3.
- Prompt changed to a "reversed video of ..." form. Each round restarts from
  the base model for 16K steps rather than continuing.
- Compute: 16 H100 80GB per round. Inference uses 20 steps per chunk and
  recomputes the KV cache per window.

## Results

- VBench on 50 s videos (Table 1): drift metrics (first 20% minus mean, lower
  is better) for aesthetic / imaging: DF sigma=0 5.12 / 7.34; DF sigma=0.2
  7.83 / 12.27; History Guidance 8.83 / 14.56; BAgger R1 4.84 / 11.83; R2
  3.79 / 5.92; R3 3.29 / 3.57.
- Round 1 alone can be worse than the seed model (dynamic degree falls to
  61.7 and outputs under-saturate); gains need several rounds.
- Table 2: motion quality 87.59 vs Self Forcing 73.93, frame quality 59.38 vs
  64.81 (Self Forcing is higher).

## Critique / open questions

- One seed, no error bars; VBench aesthetic and imaging scores are weak
  proxies for state-level correctness.
- The recipe rests on time-reversal invariance of the data. That fits natural
  video with a reworded prompt but not an action-conditioned world with
  irreversible dynamics, where a reversed clip needs reversed actions that may
  not exist (see Follow-up).
- Not compared against Self Forcing under matched compute, and each round
  retrains from the base, so cost scales with rounds (three rounds of 16 H100
  runs plus 27K generations each).
- The authors say drift still appears over very long horizons and the
  seed/corrective ratio is untuned.

## Trust signals

- **Credibility:** 3, a Stanford (Wetzstein group) arXiv preprint with clear
  ablations over rounds and matched-budget baselines, but no peer review, no
  released code (the project page says "Code (Coming Soon)") and no citation
  count I could establish.

## Follow-up

**Relevance:** 4, a new exposure-bias remedy without distillation, in the
family we care about (block-causal DF), and a candidate extra arm against DF
and TF; it has not yet shifted any concept.
- **Testing it as an arm on our gridworld is moderately cheap, with one
  catch.** Rollouts, aggregation and fine-tuning with the existing DF loss
  fit the 64x64 model in `src/wm/df.py`, and "clean prefix, loss on the rest"
  is a mask change. Cost: about three rounds of rollout generation plus
  fine-tuning on a small model, which is hours rather than days on one GPU.
- **The catch is time reversal.** Reversing a gridworld rollout is only valid
  if the reversed action sequence is a legal inverse. Moves into walls,
  collisions, partial observability and imagined unseen cells are not
  invertible, so reversed clips may be out-of-distribution. Options: restrict
  to invertible moves, define an explicit inverse-action map, or reverse only
  the frames and train with a "reverse" action token. Check that the corrective
  clips are valid before scoring any arm.
- Compare as an arm against DF and TF at matched compute, scored by true-state
  error versus horizon ([[Long-Rollout Evaluation]]); expect the
  round-count effect (R1 worse than seed) to appear.
- Wait for code before porting; the algorithm is simple enough to implement
  from Algorithm 1 if not.
