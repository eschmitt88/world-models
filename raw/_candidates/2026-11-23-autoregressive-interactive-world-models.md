---
kind: candidates
topic: "autoregressive / interactive / action-conditioned world models and causal video diffusion (2024-2026)"
discovered: 2026-11-23
source: discover
n_requested: 20
n_returned: 20
---

## 1. Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion

- url: https://arxiv.org/abs/2407.01392
- type: paper
- summary: Per-token independent noise levels unify next-token prediction and full-sequence diffusion (Chen et al., MIT, 2024).
- reason: Primary source for the diffusion-forcing and exposure-bias concepts, which currently cite only a video.
- verified: page

## 2. History-Guided Video Diffusion (Diffusion Forcing Transformer)

- url: https://arxiv.org/abs/2502.06764
- type: paper
- summary: Diffusion Forcing Transformer plus history guidance for variable-length conditioning and long, consistent video rollouts.
- reason: Scales diffusion forcing to video; the "DFoT" lineage Atlas is said to extend.
- verified: page

## 3. Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion

- url: https://arxiv.org/abs/2506.08009
- type: paper
- summary: Trains autoregressive video diffusion on its own rollouts with KV caching, closing the exposure-bias gap in real time.
- reason: The main alternative to diffusion forcing for exposure bias; open Apache-2.0 code.
- verified: page

## 4. From Slow Bidirectional to Fast Autoregressive Video Diffusion Models (CausVid)

- url: https://arxiv.org/abs/2412.07772
- type: paper
- summary: Distils a bidirectional video DiT into a few-step causal student for streaming generation.
- reason: Canonical evidence for the bidirectional-vs-causal trade-off.
- verified: page

## 5. Genie: Generative Interactive Environments

- url: https://arxiv.org/abs/2402.15391
- type: paper
- summary: 11B latent-action world model that makes playable worlds from unlabelled video.
- reason: Foundational action-conditioned world model; latent-action idea.
- verified: page

## 6. V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning

- url: https://arxiv.org/abs/2506.09985
- type: paper
- summary: Latent-predictive video model with an action-conditioned variant (V-JEPA 2-AC) used for robot planning.
- reason: The latent-predictive branch named in the project scope; real-to-sim-to-real link.
- verified: page

## 7. Stable Virtual Camera: Generative View Synthesis with Diffusion Models

- url: https://arxiv.org/abs/2503.14489
- type: paper
- summary: Stability AI multi-view diffusion model for camera-conditioned novel view synthesis from any number of views.
- reason: Open-weights camera-conditioned NVS; evidence for novel-view-prediction.
- verified: page

## 8. CAT3D: Create Anything in 3D with Multi-View Diffusion Models

- url: https://arxiv.org/abs/2405.10314
- type: paper
- summary: Multi-view diffusion generates consistent views that feed a 3D reconstruction.
- reason: The explicit-3D-via-diffusion baseline Atlas moves away from; no weights expected.
- verified: snippet

## 9. Atlas: A World Model for Spatial Intelligence (World Labs blog)

- url: https://worldlabs.ai/blog/atlas
- type: post
- summary: World Labs' launch post for Atlas, a multimodal autoregressive diffusion transformer; no technical report exists.
- reason: The primary company statement behind the video; checks the video's claims.
- verified: page

## 10. Diffusion Models Are Real-Time Game Engines (GameNGen)

- url: https://arxiv.org/abs/2408.14837
- type: paper
- summary: Fine-tuned Stable Diffusion plays DOOM at 20 fps, conditioned on past frames and actions.
- reason: Early action-conditioned diffusion game engine; shows rollout drift mitigation by noise augmentation. Weights not public.
- verified: page

## 11. Diffusion for World Modeling: Visual Details Matter in Atari (DIAMOND)

- url: https://arxiv.org/abs/2405.12399
- type: paper
- summary: Diffusion world model for RL agents; small, runnable.
- reason: Smallest runnable open diffusion world model (MIT code); borderline with ai-robotics.
- verified: page

## 12. MineWorld: a Real-Time and Open-Source Interactive World Model on Minecraft

- url: https://arxiv.org/abs/2504.08388
- type: paper
- summary: Autoregressive transformer over visual and action tokens, Microsoft, open weights.
- reason: A non-diffusion autoregressive foil; runnable.
- verified: page

## 13. Matrix-Game: Interactive World Foundation Model

- url: https://arxiv.org/abs/2506.18701
- type: paper
- summary: Skywork's 17B action-conditioned Minecraft world foundation model.
- reason: Open weights, but likely too large for 16 GB without quantization.
- verified: page

## 14. Matrix-Game 2.0: An Open-Source Real-Time and Streaming Interactive World Model

- url: https://arxiv.org/abs/2508.13009
- type: paper
- summary: Few-step causal distilled action-conditioned model at 25 fps with open weights.
- reason: Closest match to "causal, interactive, open"; the follow-up to #13.
- verified: snippet

## 15. Cosmos World Foundation Model Platform for Physical AI

- url: https://arxiv.org/abs/2501.03575
- type: paper
- summary: NVIDIA platform of diffusion and autoregressive world foundation models.
- reason: Physical-AI world models with open weights (Cosmos-Predict2 is Apache-2.0).
- verified: page

## 16. Oasis (open-oasis repo)

- url: https://github.com/etched-ai/open-oasis
- type: repo
- summary: 500M DiT Minecraft world model by Decart and Etched; the weights are on a gated Hugging Face repo.
- reason: Smallest action-conditioned DiT with a diffusion-forcing-style schedule; listed in the project scope. No paper exists.
- verified: snippet

## 17. Hunyuan-GameCraft: High-dynamic Interactive Game Video Generation with Hybrid History Condition

- url: https://arxiv.org/abs/2506.17201
- type: paper
- summary: Tencent action-conditioned game video model with hybrid history conditioning.
- reason: History-conditioning comparison; weights status unchecked.
- verified: snippet

## 18. Advancing Open-source World Models (LingBot-World)

- url: https://arxiv.org/abs/2601.20540
- type: paper
- summary: Robbyant open-sourced world simulator with a causal-attention adaptation and few-step distillation.
- reason: 2026 open world model; weights claimed.
- verified: snippet

## 19. minWM: A Full-Stack Open-Source Framework for Real-Time Interactive Video World Models

- url: https://arxiv.org/abs/2605.30263
- type: paper
- summary: Framework that converts bidirectional video models into few-step camera-controllable AR world models.
- reason: Recipe for exactly the bidirectional-to-causal conversion on small hardware.
- verified: snippet

## 20. Marble (World Labs)

- url: https://www.worldlabs.ai/
- type: post
- summary: Commercial 3D world generation; a search found no technical report or paper.
- reason: Recorded as checked; no primary paper to ingest.
- verified: snippet
