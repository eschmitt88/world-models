---
kind: paper
title: "From Slow Bidirectional to Fast Autoregressive Video Diffusion Models (CausVid)"
authors: ["Tianwei Yin", "Qiang Zhang", "Richard Zhang", "William T. Freeman", "Fredo Durand", "Eli Shechtman", "Xun Huang"]
institutions: ["MIT", "Adobe"]
year: 2024
venue: "arXiv (v4; venue not stated in the PDF)"
peer_reviewed: unknown
url: https://arxiv.org/abs/2412.07772
code_url: https://github.com/tianweiy/CausVid
citations:
source: "raw/papers/yin2024from.pdf"
added: "2026-11-23"
relevance: 5
credibility: 4
status: skimmed
related_experiments: []
related_concepts: ["Bidirectional vs Causal Video Diffusion", "Exposure Bias", "Diffusion Forcing", "Few-Step Distillation for Real-Time Rollout"]
tags: [causvid, distillation, dmd, causal-video-diffusion, streaming]
---

# From Slow Bidirectional to Fast Autoregressive Video Diffusion Models (CausVid)

## TL;DR

Distil a slow bidirectional video diffusion teacher into a **causal,
4-step student** with KV caching, using an asymmetric Distribution Matching
Distillation (DMD): bidirectional teacher, causal student. It streams
video after 1.3 s initial latency at about 9.4 FPS on one GPU, VBench-Long 84.27.

## Claims

- Bidirectional attention forces the whole sequence to be processed for any
  one frame, so it blocks streaming and interactivity (the 128-frame
  full-sequence latency is 219 s for the teacher).
- Asymmetric distillation, from a bidirectional teacher into a causal
  student, mitigates autoregressive error accumulation and is "surprisingly
  helpful".
- Student initialised from the teacher's ODE trajectories for stability.
- Trained only on text-to-video, it runs zero-shot image-to-video and
  supports video-to-video and a game-engine-to-video demo.

## Methods

- Block-wise causal attention over 3D-VAE chunks of 5 latent frames, KV cache at inference.
- Roughly 400K single-clip training samples; 10-second clips, longer rollouts by sliding window.
- DMD extended to video: a 50-step teacher into a 4-step generator.

## Results

- VBench-Long total 84.27 reported at the time as best among compared models.
- 9.4 FPS streaming, 1.3 s first-frame latency.
- Quality still degrades beyond about 30 seconds (authors' own note).

## Critique / open questions

- [[huang2025self]] argues the DMD loss here matches the wrong distribution
  because training rollouts come from a Diffusion Forcing scheme, not the
  model's own inference distribution, and shows CausVid accumulating error
  in long rollouts at the same 1.3B size.
- No action or camera conditioning: it is a video generator, not yet a world
  model. The repo licence field is unspecified.

## Trust signals

- **Credibility:** 4, MIT and Adobe, code public; peer-review status not
  established from the PDF; licence unspecified. Cited and extended by [[huang2025self]].

## Follow-up

**Relevance:** 5, canonical evidence for the post-train route in
[[Bidirectional vs Causal Video Diffusion]] and for
[[Few-Step Distillation for Real-Time Rollout]].
