---
kind: paper
title: "Genie: Generative Interactive Environments"
authors: ["Jake Bruce", "Michael Dennis", "Ashley Edwards", "Jack Parker-Holder", "Yuge (Jimmy) Shi", "Edward Hughes", "Tim Rocktaschel"]
institutions: ["Google DeepMind", "University of British Columbia"]
year: 2024
venue: "arXiv (venue not stated in the PDF)"
peer_reviewed: unknown
url: https://arxiv.org/abs/2402.15391
code_url: null
citations:
source: "raw/papers/bruce2024genie.pdf"
added: "2026-11-23"
relevance: 5
credibility: 5
status: skimmed
related_experiments: []
related_concepts: ["Latent Action Model", "Exposure Bias", "Real-to-Sim-to-Real"]
tags: [genie, world-model, latent-actions, maskgit, platformer]
---

# Genie: Generative Interactive Environments

## TL;DR

An 11B-parameter "foundation world model" that turns a text, image or sketch
prompt into a frame-by-frame playable environment. It is trained from
unlabelled Internet video: a **latent action model** infers discrete
actions between frames with no action labels, a video tokenizer discretises
frames, and an autoregressive MaskGIT dynamics model predicts the next frame.

## Claims

- First generative interactive environment trained without ground-truth
  action labels; the latent action space (8 codes) is consistent enough
  to train agents to imitate behaviour from unseen videos.
- Scales smoothly from 40M to 2.7B in ablations, then to a final 11B model.
- Data: a 30,000-hour curated filtered subset (hundreds of 2D platformer games)
  of a larger 200,000-hour set; also RT-1 robot videos.

## Methods

- Video tokenizer: VQ-VAE with an ST-transformer (spatial attention per frame, causal
  temporal attention, so cost is linear in frames).
- Latent action model: VQ-VAE style, learns to encode the change between
  frames; a small vocabulary forces controllability.
- Dynamics model: MaskGIT over discrete tokens conditioned on past tokens and latent actions.

## Results

- Controllable generation from OOD prompts including sketches and photos.
- Robotics-domain experiment shows consistent latent actions across embodiments.

## Critique / open questions

- Discrete tokens, not continuous diffusion, so the exposure-bias story
  in [[Exposure Bias]] applies differently; the paper keeps only 16 frames
  of memory and notes it can hallucinate unrealistic futures.
- Not released (no weights or code), so it cannot be run locally; the line of
  open reproductions (Oasis, MineWorld, Matrix-Game) follows its recipe.

## Trust signals

- **Credibility:** 5, Google DeepMind, widely built upon; peer-review status
  not established from the PDF. No code released.

## Follow-up

**Relevance:** 5, seeds [[Latent Action Model]] and is the reference point
for action-conditioned world models.
