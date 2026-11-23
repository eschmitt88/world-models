---
kind: paper
title: "Matrix-Game 2.0: An Open-Source, Real-Time, and Streaming Interactive World Model"
authors: ["Xianglong He", "Chunli Peng", "Zexiang Liu", "Boyang Wang", "Yifan Zhang", "Yahui Zhou"]
institutions: ["Skywork AI"]
year: 2025
venue: "arXiv (v5)"
peer_reviewed: false
url: https://arxiv.org/abs/2508.13009
code_url: https://github.com/SkyworkAI/Matrix-Game
citations:
source: "raw/papers/zhang2025matrix.pdf"
added: "2026-11-23"
relevance: 5
credibility: 3
status: skimmed
related_experiments: []
related_concepts: ["Bidirectional vs Causal Video Diffusion", "Exposure Bias", "Few-Step Distillation for Real-Time Rollout", "Latent Action Model"]
tags: [matrix-game, interactive-world-model, self-forcing, action-injection, open-weights]
---

# Matrix-Game 2.0: An Open-Source, Real-Time, and Streaming Interactive World Model

## TL;DR

A 1.3B action-conditioned world model that generates minute-long video
causally at 25 FPS on one H100. It takes a bidirectional I2V backbone, adds
frame-level mouse and keyboard injection, and distils it into a few-step
causal student trained with Self-Forcing-style DMD. MIT code, MIT weights.

## Claims

- Prior interactive world models depend on bidirectional attention and many steps,
  which blocks real time.
- Three components: a scalable Unreal-Engine and game-capture data pipeline (about 1200 hours
  with action labels), an action-injection module, and few-step causal distillation.
- Self-generated history during the DMD phase "mitigates" the error accumulation of
  Teacher Forcing and Diffusion Forcing (an explicit adoption of the
  [[huang2025self]] argument).

## Methods

- Backbone: SkyReels-V2-I2V-1.3B (Wan 2.1 architecture), text branch removed.
- Mouse actions concatenated to the input latents; keyboard actions via
  cross-attention, with RoPE on the keyboard stream for long sequences.
- Student initialised from ODE trajectories, block-wise causal mask with KV cache
  and a rolling window (evicts the oldest tokens), then DMD with self-rollouts.

## Results

- 25 FPS on a single H100, minute-level videos; compares favourably with Oasis in Minecraft
  and with Matrix-Game 1 in wild scenes (author-run comparisons).

## Critique / open questions

- Self-reported numbers, no independent reproduction seen here. Qualitative drift
  past a minute is not quantified.
- Repo README: "Nvidia GPU with at least 24 GB memory (A100, and H100 are tested)", so 16 GB is
  unsupported as shipped.
- The repo has since moved on: Matrix-Game 3.0 (a 5B model with long-horizon memory,
  Apache-2.0) and 4.0 exist, so check which version an experiment should use.

## Trust signals

- **Credibility:** 3, a company preprint with open code and weights but no peer
  review and author-only benchmarks.

## Follow-up

**Relevance:** 5, the nearest thing to the project's target: open, causal, action-conditioned,
and post-trained from a bidirectional model.
- First candidate to try on the 16 GB card (1.3B, MIT).
