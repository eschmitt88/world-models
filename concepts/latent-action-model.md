---
kind: concept
name: "Latent Action Model"
status: seedling
added: "2026-11-23"
sources: [bruce2024genie, assran2025vjepa]
related_concepts: ["Latent-Predictive World Models", "Real-to-Sim-to-Real"]
related_experiments: []
tags: [action-conditioning, unsupervised, genie, world-model]
---

# Latent Action Model

## Definition

A model that infers a small discrete action code between two consecutive
frames, with no action labels, so that a dynamics model can be made
controllable from raw video. Genie uses a VQ-VAE-style encoder with only 8
codes, so the code must carry just what the history cannot predict
([[bruce2024genie]]).

## Why it matters here

Real action labels are scarce, but video is not. A latent action space
turns unlabelled video into action-conditioned training data, and the paper
reports the learned actions are consistent across a robotics dataset and
usable for imitating unseen videos. The same data-efficiency logic appears in
[[assran2025vjepa]], which attaches real robot actions to a video-pretrained
model using under 62 hours of data. Open reproductions (Oasis, MineWorld, Matrix-Game)
mostly use real keyboard or mouse actions instead, so check what a given
checkpoint actually conditions on before calling it "action-controllable".

## Connections

- [[Latent-Predictive World Models]]: the other low-label route to action conditioning
- [[Real-to-Sim-to-Real]]: robot use of action-conditioned predictors
