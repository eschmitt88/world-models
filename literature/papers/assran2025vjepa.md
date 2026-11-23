---
kind: paper
title: "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning"
authors: ["Mido Assran", "Adrien Bardes", "David Fan", "Quentin Garrido", "Nicolas Ballas", "Michael Rabbat", "Yann LeCun"]
institutions: ["FAIR at Meta", "Mila - Quebec AI Institute", "Polytechnique Montreal"]
year: 2025
venue: "arXiv preprint"
peer_reviewed: false
url: https://arxiv.org/abs/2506.09985
code_url: https://github.com/facebookresearch/vjepa2
citations:
source: "raw/papers/assran2025vjepa.pdf"
added: "2026-11-23"
relevance: 4
credibility: 4
status: skimmed
related_experiments: []
related_concepts: ["Latent-Predictive World Models", "Real-to-Sim-to-Real", "Latent Action Model"]
tags: [jepa, v-jepa-2, latent-prediction, robot-planning, droid]
---

# V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning

## TL;DR

An action-free encoder-predictor (JEPA) pre-trained on over 1 million hours
of Internet video predicts in **representation space**, not pixels. A
second stage, V-JEPA 2-AC, adds action conditioning using under 62 hours of
unlabelled Droid robot video and is used for zero-shot planning of pick-and-place
on Franka arms in two labs.

## Claims

- SSv2 top-1 77.3 and Epic-Kitchens-100 action-anticipation recall@5 39.7
  (state of the art at the time); aligned with an LLM, SOTA on video QA at 8B.
- Scaling data (2M to 22M videos), model (ViT-L 300M to ViT-g 1B) and
  training length each add measurable points.
- The AC world model plans with image goals via the cross-entropy method.

## Methods

- Mask-denoising objective in latent space; ViT encoders up to 1B parameters.
- V-JEPA 2-AC: action-conditioned predictor post-trained on frozen encoder features;
  planning samples actions and scores them by distance to the goal embedding.

## Results

- Planning compared with an action-conditioned Cosmos baseline (latent-diffusion 7B):
  Cosmos needs about 4 minutes per action and does worse on object interaction;
  V-JEPA 2-AC needs 16 seconds per action.

## Critique / open questions

- Author-stated limitation: sensitivity to camera positioning, so camera
  placement needed manual tuning per lab.
- It predicts embeddings, so there is no pixel rollout to look at; it is
  a planner's world model, not a playable simulator. This paper only
  compares against one diffusion baseline (Cosmos).
- Preprint from Meta FAIR, not peer reviewed per the PDF.

## Trust signals

- **Credibility:** 4, FAIR with MIT-licensed code and Hugging Face
  checkpoints, large evaluations, but an arXiv preprint.

## Follow-up

**Relevance:** 4, seeds [[Latent-Predictive World Models]] and gives a
robotics evidence point for [[Real-to-Sim-to-Real]].
- Can the ViT-g encoder run on 16 GB for embedding-space rollouts?
