---
kind: paper
title: "History-Guided Video Diffusion"
authors: ["Kiwhan Song", "Boyuan Chen", "Max Simchowitz", "Yilun Du", "Russ Tedrake", "Vincent Sitzmann"]
institutions: ["MIT", "Carnegie Mellon University", "Harvard University"]
year: 2025
venue: "ICML 2025"
peer_reviewed: true
url: https://arxiv.org/abs/2502.06764
code_url: https://github.com/kwsong0113/diffusion-forcing-transformer
citations:
source: "raw/papers/song2025history.pdf"
added: "2026-11-23"
relevance: 5
credibility: 4
status: skimmed
related_experiments: []
related_concepts: ["Diffusion Forcing", "Bidirectional vs Causal Video Diffusion", "Exposure Bias", "Novel View Prediction"]
tags: [diffusion-forcing, dfot, history-guidance, long-video, realestate10k]
---

# History-Guided Video Diffusion

## TL;DR

The Diffusion Forcing Transformer (DFoT) applies per-frame independent
noise to a **non-causal** video transformer, so any subset of frames can be
conditioned on at sampling time. This enables "history guidance", a CFG
variant that treats the history as the conditioning variable, and stable
long rollouts (862 frames from one image on RealEstate10K).

## Claims

- Dropping history CFG-style performs badly, and fixed-size conditioning
  architectures cannot take variable-length history. DFoT solves both.
- Vanilla history guidance (HG-v) improves quality and consistency.
  Temporal (HG-t) and fractional (HG-f, history corrupted at partial noise
  like a low-pass filter) guidance improve dynamics and out-of-distribution
  history. Compositions: HG-tf.
- Works as fine-tuning of pretrained video diffusion models; compatible
  with DiT and U-ViT.

## Methods

- "Noising as masking": independent noise per frame in training; at
  sampling a history window is masked by noising it. The unconditional CFG
  score is the model with all history masked.
- Architecture is bidirectional over the context window, which is a chunk-level
  design with no KV-cached causal rollout.

## Results

- Kinetics-600 frame prediction: DFoT with HG reaches FVD that matches or beats
  the compared video-diffusion baselines (see Fig. 5 and the Kinetics-600 table).
- RealEstate10K navigation: 862-frame rollout from a single frame; robustness
  to out-of-distribution history with HG-t.

## Critique / open questions

- Shows that diffusion forcing does **not** require causal attention, which
  sharpens the video's pairing of "diffusion forcing = causal":
  see [[Bidirectional vs Causal Video Diffusion]].
- Rollouts are windowed re-conditioning, not strictly streaming; latency is
  not the focus (compare [[huang2025self]]).
- Camera-pose-conditioned RealEstate10K is the nearest to
  [[Novel View Prediction]], which makes it a useful probe of long-range consistency.

## Trust signals

- **Credibility:** 4, ICML 2025 from MIT with released code; I did not check
  citation counts.

## Follow-up

**Relevance:** 5, the video-scale demonstration of [[Diffusion Forcing]] that
the Atlas explainer leans on.
- Check whether DFoT checkpoints (RE10K, Kinetics) fit a 16 GB card.
