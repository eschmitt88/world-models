---
kind: paper
title: "Diffusion for World Modeling: Visual Details Matter in Atari"
authors: ["Eloi Alonso", "Adam Jelley", "Vincent Micheli", "Anssi Kanervisto", "Amos Storkey", "Tim Pearce", "Francois Fleuret"]
institutions: ["University of Geneva", "University of Edinburgh", "Microsoft Research"]
year: 2024
venue: "NeurIPS 2024"
peer_reviewed: true
url: https://arxiv.org/abs/2405.12399
code_url: https://github.com/eloialonso/diamond
citations:
source: "raw/papers/alonso2024diffusion.pdf"
added: "2026-11-24"
relevance: 4
credibility: 5
status: skimmed
related_experiments: []
related_concepts: ["Diffusion World Model", "Exposure Bias"]
tags: [diamond, edm, atari, csgo, small-scale, frame-stacking, few-step-sampling]
---

# Diffusion for World Modeling: Visual Details Matter in Atari

## TL;DR

DIAMOND is an RL agent trained inside a pixel-space diffusion world model.
It conditions a small U-Net on a stack of past frames and actions, and uses
the EDM formulation so that 3 denoising steps per frame stay stable over
long autoregressive rollouts. It is the template for a
[[Diffusion World Model]] trained from scratch on one consumer GPU.

## Claims

- Discrete-latent world models lose visual detail that can matter for the
  policy; modelling pixels with diffusion keeps it.
- The diffusion formulation is a design choice that decides rollout
  stability: EDM stays stable at few denoising steps where DDPM compounds
  error and drifts out of distribution.
- Mean human-normalised score 1.46 on Atari 100k, a best for agents trained
  entirely inside a world model.
- The same model, scaled, serves as an interactive game engine (CS:GO, Dust II).

## Methods

- Next-frame diffusion: denoise x_t given the clean last L frames and actions
  (frame stacking as the only memory), autoregressive in environment time.
  Euler sampling, 3 NFE per frame for Atari (IRIS needs 16).
- EDM (Karras et al. 2022) preconditioning instead of DDPM; a 4M-parameter
  U-Net for Atari, 381M for CS:GO (including a 51M upsampler, 10 denoising
  steps for the upsampler).
- Agent: actor-critic with reward/termination heads, trained purely in
  imagination.

## Results

- Atari 100k (26 games, 5 seeds): mean HNS 1.46. Each run used about 12 GB
  VRAM and about 2.9 days on one RTX 4090.
- Stability study (Fig. 3, Breakout data): imagined trajectories to t = 1000;
  DDPM at low step counts drifts out of distribution, EDM does not.
- CS:GO: 87 h of 16 Hz human gameplay, trained 12 days on an RTX 4090, runs at
  10 Hz on an RTX 3090. Stable over hundreds of steps but drifts
  out of distribution more often in some regions.

## Critique / open questions

- Atari and CS:GO are visually simple or short-memory; the stability claim is
  shown by qualitative trajectory figures, not a drift metric against
  ground truth.
- Frame stacking is a minimal memory. The authors list this as a limitation
  and suggest a transformer over environment time. Nothing tests revisits.
- The model is trained with clean ground-truth history (teacher forcing). The
  paper's stability comes from the sampler and parametrisation, not from
  a fix to [[Exposure Bias]]; no comparison with
  [[Diffusion Forcing]] exists here.
- The 2.9 days on a 4090 per game is the budget to compare against our
  16 GB RTX 5080 runs.

## Trust signals

- **Credibility:** 5, NeurIPS 2024, University of Geneva / Edinburgh / Microsoft
  Research, MIT-licensed code, agents and playable models released (repo about
  2.1k stars), widely used as a baseline (StateSpaceDiffuser builds on it).

## Follow-up

**Relevance:** 4, the reference recipe for training a small diffusion world
model on one consumer GPU and material evidence on rollout stability, but it
is a frame-stacked U-Net without diffusion forcing or a memory test.
- Compare its EDM preconditioning and step count with the causal DiT in
  `src/wm/df.py`.
- Use the "stable to t = 1000" protocol as a baseline for the weather/climate
  long-rollout evaluation (ADR 0002), with a ground-truth drift metric.
