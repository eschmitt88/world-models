---
kind: post
title: "Atlas: A World Model for Spatial Intelligence (World Labs blog)"
author: "World Labs"
url: https://worldlabs.ai/blog/atlas
source: "raw/web/worldlabs-ai-atlas.md"
added: "2026-11-23"
institutions: ["World Labs"]
peer_reviewed: false
code_url:
relevance: 5
credibility: 2
related_experiments: []
related_concepts: ["Novel View Prediction", "Diffusion Forcing", "Bidirectional vs Causal Video Diffusion", "Real-to-Sim-to-Real"]
tags: [world-labs, atlas, novel-view-synthesis, primary-source, company-claims]
---

# Atlas: A World Model for Spatial Intelligence (World Labs blog)

## TL;DR

The company's own launch post for Atlas (2026-09-01): a "multimodal autoregressive
diffusion transformer" pretrained from scratch on text, images, video and 3D, with
camera poses and depth as inputs and a shared spatial context. There is **no technical
report or arXiv paper**; the post offers only a BibTeX entry for itself. The raw file is
a fetch-tool summary, not verbatim page text.

## Key points

- Rectified-flow latent diffusion; borrows KV caching and serving tricks from LLMs and
  distillation and CFG from diffusion models.
- Outputs are novel-view images **and explicit 3D** (point clouds or Gaussian splats), from two to over a hundred
  inputs. This is sharper than the explainer video's "implicit 3D, no splats" framing
  ([[Novel View Prediction]]).
- Company-reported benchmarks: camera-controlled generation preferred 75% to 94% of the time
  by third-party raters; lowest mean pointmap error (about 8.6 against about 9.7 for VGGT).
- No parameter counts, dataset sizes or compute figures. "Improves with increased compute"
  is asserted without numbers.
- The summary of the post does not mention diffusion forcing, per-modality noise or "latent
  forcing"; those claims remain attributed only to the video
  ([[welch2026fei-fei-li-atlas]]). Not evidence against them, since I only have a summary.
- Real-to-sim: RGB and depth sensor data for robot navigation and manipulation.
- Availability: web app and API only; no open weights.

## Follow-up

**Relevance:** 5, the only primary statement about the project's namesake system.
**Credibility:** 2, company marketing with self-reported benchmarks and no paper or code.
- Re-fetch the page text verbatim to check for the diffusion-forcing wording.
