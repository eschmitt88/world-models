---
kind: post
title: "Can Fei-Fei Li Reinvent AI (Again)?"
author: "Stephen Welch (Welch Labs), interviewing Fei-Fei Li, Justin Johnson, Ben Mildenhall, Christoph Lassner (World Labs)"
url: https://www.youtube.com/watch?v=hORDMRY_bZ0
source: "raw/web/welch2026fei-fei-li-atlas.md"
added: "2026-11-22"
institutions: ["Welch Labs", "World Labs"]
peer_reviewed: false
code_url:
relevance: 4
credibility: 2
related_experiments: []
related_concepts: ["Novel View Prediction", "Diffusion Forcing", "Exposure Bias", "Bidirectional vs Causal Video Diffusion", "Real-to-Sim-to-Real"]
tags: [world-labs, atlas, marble, novel-view-synthesis, diffusion-forcing, autoregressive-diffusion, video-model, robotics]
---

# Can Fei-Fei Li Reinvent AI (Again)?

## TL;DR

An explainer video on World Labs' **Atlas**, built around three design
decisions. (1) *Prediction target*: Marble emitted explicit Gaussian splats.
Atlas instead predicts RGB (and optionally depth) at a requested camera pose,
so the 3D structure becomes an implicit internal state. (2) *Architecture*: an
autoregressive diffusion transformer trained with
[[Diffusion Forcing]]-style independent noise levels, extended from frames to
modalities ("latent forcing" / "modality forcing"). (3) *Scale*: trained from
scratch rather than post-training a bidirectional video model. The video's
best technical segment explains why bidirectional video DiTs can't be
interactive and why causal teacher forcing drifts ([[Exposure Bias]]).

## Key points

- **Explicit 3D caps quality.** Marble's splat bottleneck limited visual
  fidelity and produced artifacts. [[Novel View Prediction]] treats "the view
  from pose P" as the next-token analogue and leaves geometry implicit.
- **Bidirectional attention rules out interactivity.** In Wan or Sora,
  frame *t* attends to frame *t+k*, so the whole camera or action trajectory
  has to be fixed upfront. The video shows Wan attention heads tracking static
  background across all frames and others looking ahead to a ball's future
  position. See [[Bidirectional vs Causal Video Diffusion]].
- **Causal masking alone drifts.** Teacher forcing conditions on clean ground
  truth. At rollout the model sees its own continuous-valued outputs and the
  errors compound. Discrete LLM tokens hide this; RGB and latents don't.
- **Diffusion forcing fixes the train/test gap.** Each frame gets its own
  noise level *k_i*, fed to the model so it knows how much to trust each
  context frame (Chen et al., MIT 2024; Minecraft rollouts far past the
  training horizon). World Labs' claimed extension is a per-*modality* noise
  level (text, RGB, depth, pose). One model then covers text-to-image, NVS,
  360° panoramas (six orthogonal poses) and interactive rollouts, chosen by
  which inputs are clean and which are noised.
- **From scratch, compute-bound.** Johnson says Atlas shows clean scaling
  laws and is limited by compute, not data. The video presents this as the
  motive for the reported $8B AMD deal (Sept 2026).
- **[[Real-to-Sim-to-Real]] (July 2026 demo).** Three iPhone videos of a
  manipulation scene → Atlas reconstruction → fast physics sim → millions of
  counterfactual RL iterations → policy deployed back on the real robot.
- **Landscape, Feb 2024.** Sora (text-to-video framed as a world simulator),
  V-JEPA (predicts in latent space, not pixels), Genie (playable worlds from
  200k h of gameplay). NeRF serves as the per-scene reconstruction contrast.

## Trust signals

**Credibility:** 2. A secondary explainer (and this copy is a user-supplied
chaptered summary of it, not the verbatim transcript), with primary-source
interviews but no paper, no code and no peer review. Architecture claims about
Atlas ("modality forcing", from-scratch training, scaling laws) are the
company's own account and unverified. The deal and demo dates are past the
assistant's knowledge cutoff and were not checked. The explanations of
diffusion forcing, exposure bias and bidirectional attention match the
public literature and are the trustworthy part.

## Follow-up

**Relevance:** 4. Seeds the project's core architectural thread
(bidirectional → causal → diffusion-forced world models) and the
explicit-vs-implicit-3D question. It sits below 5 only because the canonical
evidence for these concepts is the primary papers, not this video.

- Fetch primary sources: Diffusion Forcing (Chen et al. 2024), its successor
  the Diffusion Forcing Transformer / history guidance, Self Forcing and
  CausVid (bidirectional→causal distillation, the "post-train" route Atlas
  rejected), Genie 1–3, V-JEPA / V-JEPA 2, NeRF, 3D Gaussian Splatting,
  CAT3D / camera-controlled video NVS.
- Look for any World Labs technical report on Atlas, "latent forcing" or
  "modality forcing".
- Open-model trial candidates for 16 GB: the Diffusion Forcing reference code
  (small Minecraft/DMLab models), Self Forcing on Wan 2.1 1.3B (real-time
  causal rollout), Oasis 500M, V-JEPA 2. Probe drift as a function of
  rollout length and of the context-noise level.
- Open question: does implicit 3D (NVS) beat explicit splats once you need
  geometry for physics, as in Real-to-Sim-to-Real? The demo still has to
  lift *explicit* geometry into the simulator.
