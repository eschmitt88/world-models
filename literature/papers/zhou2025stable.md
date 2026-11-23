---
kind: paper
title: "Stable Virtual Camera: Generative View Synthesis with Diffusion Models"
authors: ["Jensen (Jinghao) Zhou", "Hang Gao", "Vikram Voleti", "Aaryaman Vasishta", "Chun-Han Yao", "Mark Boss", "Philip Torr", "Christian Rupprecht", "Varun Jampani"]
institutions: ["Stability AI", "University of Oxford", "University of California, Berkeley"]
year: 2025
venue: "arXiv (venue not stated in the PDF)"
peer_reviewed: unknown
url: https://arxiv.org/abs/2503.14489
code_url: https://github.com/Stability-AI/stable-virtual-camera
citations:
source: "raw/papers/zhou2025stable.pdf"
added: "2026-11-23"
relevance: 5
credibility: 4
status: skimmed
related_experiments: []
related_concepts: ["Novel View Prediction", "Real-to-Sim-to-Real"]
tags: [seva, novel-view-synthesis, camera-conditioned, multi-view-diffusion, open-weights]
---

# Stable Virtual Camera: Generative View Synthesis with Diffusion Models

## TL;DR

SEVA is a single 1.3B-parameter diffusion model that renders novel views from
**any number of input views and target camera poses**, with no explicit 3D
representation inside the network and no per-scene NeRF distillation. It
covers both "set" NVS (unordered images) and "trajectory" NVS (smooth video),
and produces videos of up to half a minute with loop closure.

## Claims

- A "P-in Q-out" generative renderer: variable-length sequences of posed input
  views and target cameras, using SD 2.1 as the base and growing the parameter
  count from 870M to 1.3B.
- Existing diffusion NVS models (Zero123, ReconFusion, CAT3D, SV3D, ViewCrafter)
  each lack one of generation capacity, interpolation smoothness or input
  flexibility; SEVA targets all three.
- State of the art across a unified benchmark of 10 datasets, e.g. +1.5 dB PSNR over
  prior methods on average in the set-NVS tables, +6.0 dB on LLFF with 3 views.
- Consistency comes without 3D-representation distillation.

## Methods

- View-selection strategy during training covers both small and large viewpoint changes.
- Two-pass procedural sampling: anchor views first, then trajectory
  interpolation, giving smooth video of arbitrary camera paths.
- Optional temporal pathway of 3D convolutions turns the model into a video model.
- Trained with a two-stage curriculum, batch size 512 for 600K iterations (first stage).

## Results

- Outperforms CAT3D by about +0.6 dB PSNR on set NVS and wins most splits across
  OO3D, GSO, RE10K, LLFF, DTU, CO3D, WRGBD, Mip360, DL3DV, T&T.
- Long trajectories up to 30 s with seamless loop closure.

## Critique / open questions

- It is a **bidirectional, batch generator**, not an autoregressive rollout:
  the camera path is supplied upfront, which the project's own framing says rules
  out interactivity ([[Bidirectional vs Causal Video Diffusion]]).
- The repo's licence is a non-commercial one (output follows the same
  licence per its FAQ) and the weights are gated on Hugging Face.
- Implicit 3D consistency only; geometry is not an output (contrast Atlas, which
  per its blog also emits point clouds and splats).

## Trust signals

- **Credibility:** 4, Stability AI with Oxford and Berkeley authors, weights and
  code released, a 10-dataset benchmark; peer-review status not established.

## Follow-up

**Relevance:** 5, an open, runnable reference point for
[[Novel View Prediction]] that a 16 GB card can plausibly load at 1.3B
(not measured; the README has no VRAM figure).
