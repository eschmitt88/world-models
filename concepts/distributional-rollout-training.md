---
kind: concept
name: "Distributional Rollout Training"
status: seedling
added: "2026-11-24"
sources: [zhao2026frechet]
related_concepts: ["Exposure Bias", "Diffusion Forcing", "Long-Rollout Evaluation"]
related_experiments: ["2026-11-23-gw-weather-climate-eval"]
tags: [training-objective, distribution-matching, frechet-distance, long-rollout, world-model]
---

# Distributional Rollout Training

## Definition

Training an autoregressive model on its own free rollouts with a loss on the
*distribution* of the rollout (feature means and covariances, a score from a
teacher) instead of a per-sample regression target, because a rollout thousands
of steps long has no paired ground truth.

## Why it matters here

[[zhao2026frechet]] is the instance: Fréchet distances to real-data feature
statistics (frozen DINOv2 embeddings, clip-centred latent moments) applied to
free rollouts, with flow matching on a self-generated history to keep action
conditioning. It takes a 1B video world model from drift after 253 frames to
stable CLIP-FD beyond 4,000 frames on DROID and Sekai, where distribution-matching
distillation gave no gain.

Two cautions for this project. A marginal match is satisfiable by dataset-average
frames (the paper's "context decoupling"), so the statistic must be conditioned on
the clip; and the same feature distances used as the training loss are also the
headline evaluation, so they say little about state-level structure, which
ADR 0002's decoded-state statistics against a known generator would check.

## Connections

- [[Exposure Bias]]: the train/test mismatch it addresses, as an alternative to
  self-resampled history and time-reversal fixes.
- [[Long-Rollout Evaluation]]: it reports metrics binned by rollout step, with a
  ground-truth same-sample-size floor.
- [[Diffusion Forcing]]: the base model is a DF-pretrained block-causal DiT.
