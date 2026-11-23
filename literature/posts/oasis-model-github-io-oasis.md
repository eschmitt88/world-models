---
kind: post
title: "Oasis: A Universe in a Transformer (Decart and Etched)"
author: "Decart and Etched"
url: https://oasis-model.github.io
source: "raw/web/oasis-model-github-io-oasis.md"
added: "2026-11-23"
institutions: ["Decart", "Etched"]
peer_reviewed: false
code_url: "https://huggingface.co/Etched/oasis-500m"
relevance: 4
credibility: 2
related_experiments: []
related_concepts: ["Diffusion Forcing", "Exposure Bias", "Bidirectional vs Causal Video Diffusion"]
tags: [oasis, minecraft, diffusion-forcing, open-weights, action-conditioned]
---

# Oasis: A Universe in a Transformer (Decart and Etched)

## TL;DR

October 2024 launch post for Oasis, a real-time action-conditioned Minecraft
world model: a ViT spatial autoencoder plus a DiT latent-diffusion backbone,
autoregressive over frames, **trained with Diffusion Forcing**, with inference-time
"dynamic noising". Only a 500M-parameter model was released.

## Key points

- Architecture is Transformer-only (ViT autoencoder, DiT backbone); chosen over
  GameNGen and DIAMOND-style models for predictable scaling and speed on Etched's ASIC.
- Autoregressive per-frame generation conditioned on keyboard input, in contrast to
  bidirectional models such as Sora ([[Bidirectional vs Causal Video Diffusion]]).
- Training uses [[Diffusion Forcing]] (independent per-token noise); at inference,
  "dynamic noising" injects noise in the first diffusion passes and removes it in later ones
  to reduce error accumulation ([[Exposure Bias]]). Noise seen in training makes
  the model robust to noisy context.
- Claims 20 FPS with Decart's inference stack on the large demo model; the released 500M is the
  runnable one. Self-listed weaknesses: fuzzy distant detail, limited long-horizon memory,
  poor domain generalisation and imprecise inventory and object control.
- Released: weights for the 500M model (Hugging Face, MIT, gated with automatic approval:
  two files `oasis500m.pt/.safetensors` and a ViT VAE `vit-l-20`); the GitHub inference repo that the
  post links to returns 404 on 2026-11-23, so code must come from forks or a
  mirror. Third-party numbers: 2.58 FPS in [[guo2025mineworld]]'s Table 1 (author-run).

## Follow-up

**Relevance:** 4, the first widely used open model trained with diffusion forcing, and the
baseline Matrix-Game 2.0 and MineWorld both compare to.
**Credibility:** 2, company blog with self-reported results and no paper; useful because
the weights are checkable.
- Verify VRAM by running the 500M model (not stated in the post).
