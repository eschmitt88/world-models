---
kind: paper
title: "MineWorld: a Real-Time and Open-Source Interactive World Model on Minecraft"
authors: ["Junliang Guo", "Yang Ye", "Tianyu He", "Haoyu Wu", "Yushu Jiang", "Tim Pearce", "Jiang Bian"]
institutions: ["Microsoft Research"]
year: 2025
venue: "arXiv"
peer_reviewed: false
url: https://arxiv.org/abs/2504.08388
code_url: https://github.com/microsoft/mineworld
citations:
source: "raw/papers/guo2025mineworld.pdf"
added: "2026-11-23"
relevance: 4
credibility: 3
status: skimmed
related_experiments: []
related_concepts: ["Exposure Bias", "Bidirectional vs Causal Video Diffusion"]
tags: [mineworld, autoregressive-transformer, discrete-tokens, diagonal-decoding, minecraft]
---

# MineWorld: a Real-Time and Open-Source Interactive World Model on Minecraft

## TL;DR

A non-diffusion foil: a LLaMA-style autoregressive transformer over interleaved
discrete image tokens and action tokens, trained with plain next-token
prediction on Minecraft, with training-free "diagonal decoding" for speed.
It reports better action-following and FVD than Oasis at 300M to 1.2B
parameters. **The released checkpoints have been taken down** (see Follow-up).

## Claims

- A visual-action autoregressive transformer (VQ tokenizer, 16x spatial compression, 8k codebook,
  14 x 24 tokens per frame) learns both game state and action conditioning.
- Diagonal decoding exploits spatial dependencies between adjacent image tokens to
  decode in parallel, giving real-time interaction (4 to 7 FPS per the README).
- Authors propose action-following benchmarks (precision, recall, F1 on discrete
  actions, L1 on camera control).

## Methods

- VQ-VAE initialised from a public checkpoint and fine-tuned; per-frame tokenization,
  no temporal compression. Action tokenizer for keys and camera rotation.
- 300M, 700M and 1.2B models, trained on 32 A100-40G GPUs for 200k steps.

## Results

- Table 1 (author-run): Oasis 500M 2.58 FPS, F1 0.41, FVD 377; MineWorld 300M 5.91 FPS,
  F1 0.70, FVD 246; 1.2B 3.01 FPS, F1 0.73, FVD 227.
- Authors list as a limitation that it trains only on Minecraft at a fixed downsampled resolution.

## Critique / open questions

- The comparison with Oasis is self-run and the FPS numbers are on A100/H100.
- Discrete tokens avoid continuous-latent drift in a different way from diffusion; whether
  this removes [[Exposure Bias]] is untested here.

## Trust signals

- **Credibility:** 3, Microsoft Research, MIT licence, but a preprint and
  the weights are currently unavailable, which blocks independent checking.

## Follow-up

**Relevance:** 4, the only discrete-token autoregressive alternative in the set.
- Code is MIT-licensed on GitHub. The README says (News, May 2025) "The model checkpoints in
  the Huggingface repo have been temporally taken down", and the Hugging Face model API
  returned an authentication error on 2026-11-23, so assume no weights.
- README: "We recommend using high-end GPU for inference. We have done all testing and
  development using A100 and H100 GPU."
