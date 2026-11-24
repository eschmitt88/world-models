---
kind: candidates
topic: "small-world world models: training from scratch on 2D, grid and low-resolution environments on one consumer GPU"
discovered: 2026-11-23
source: discover
n_requested: 20
n_returned: 20
curated: 2026-11-24
---

## 1. DIAMOND: Diffusion for World Modeling, Visual Details Matter in Atari

- url: https://arxiv.org/abs/2405.12399
- type: paper
- summary: Pixel diffusion world model, 64x64 Atari (4M-param UNet, ~2.9 days on one RTX 4090 per run) and CS:GO (381M, 56x30 + upsampler, 12 days on one 4090).
- reason: The reference point for training a diffusion world model from scratch on one consumer GPU; code MIT at github.com/eloialonso/diamond.
- verified: page

## 2. Delta-IRIS: Efficient World Models with Context-Aware Tokenization

- url: https://arxiv.org/abs/2406.19320
- type: paper
- summary: Discrete-token transformer world model on Crafter, 64x64, one A100 40GB, about 5x faster than IRIS.
- reason: Cheap token-based baseline and compute accounting for a Crafter-class 2D world; code github.com/vmicheli/delta-iris.
- verified: page

## 3. Genie: Generative Interactive Environments

- url: https://arxiv.org/abs/2402.15391
- type: paper
- summary: 11B latent-action world model; the CoinRun case study (appendix) shows the design working on a small 2D platformer.
- reason: Origin of the latent-action recipe; the CoinRun case is the small-scale part of interest. No official code.
- verified: page

## 4. Jafar: an open-source Genie reimplementation in JAX

- url: https://github.com/FLAIROx/jafar
- type: repo
- summary: Apache-2.0 JAX Genie; README says the CoinRun reproduction runs on one L40S in under a week (about 100 GB dataset).
- reason: Closest to a single-GPU Genie-on-CoinRun reproduction.
- verified: page

## 5. Learning Generative Interactive Environments By Trained Agent Exploration (GenieRedux)

- url: https://arxiv.org/abs/2409.06445
- type: paper
- summary: PyTorch Genie reimplementation on CoinRun; RL-agent data collection beats random exploration.
- reason: Shows data-collection policy matters for world-model quality; code github.com/insait-institute/GenieRedux.
- verified: page

## 6. TinyWorlds: a minimal Genie world model

- url: https://github.com/AlmondGod/tinyworlds
- type: repo
- summary: MIT, PyTorch, action-less video on PicoDoom, Pong, Zelda, Pole Position, Sonic; weights on Hugging Face.
- reason: Educational small Genie clone; parameter count and compute are not in the README (secondary sources say about 3M).
- verified: page

## 7. Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion

- url: https://arxiv.org/abs/2407.01392
- type: paper
- summary: Per-token noise levels; DMLab model 24M params at 64 px and Minecraft model 36M, 8 A100s, about 12 h to converge.
- reason: Primary source for diffusion forcing, with small-scale numbers showing it is cheap to reproduce.
- verified: page

## 8. How Far is Video Generation from World Model: A Physical Law Perspective

- url: https://arxiv.org/abs/2411.02385
- type: paper
- summary: 2D ball-physics videos at 128x128, DiT-S 22.5M to DiT-XL 456M; scaling fixes in-distribution error but not OOD, case-based generalisation.
- reason: The controlled 2D-physics study of what video models learn; compute is large (32-64 A100s), so a smaller replication is open. Project page phyworld.github.io.
- verified: page

## 9. Long-Context State-Space Video World Models

- url: https://arxiv.org/abs/2505.20171
- type: paper
- summary: Block-wise SSM plus local attention; Memory Maze and TECO Minecraft retrieval tasks at low resolution.
- reason: Memory-on-revisit with a small-world benchmark and a stated cost curve.
- verified: page

## 10. StateSpaceDiffuser: Bringing Long Context to Diffusion World Models

- url: https://arxiv.org/abs/2505.22246
- type: paper
- summary: Adds Mamba state to a diffusion world model; MiniGrid 7x7 partial-view maze (85x85 maze) and CS:GO; 51.9% PSNR gain on the maze; 8 A100s (4 for the MiniGrid SSM).
- reason: The nearest existing 2D-maze revisit experiment with ground truth.
- verified: page

## 11. WorldMem: Long-term Consistent World Simulation with Memory

- url: https://arxiv.org/abs/2504.12369
- type: paper
- summary: Pose and timestamp-indexed memory bank on Minecraft (Oasis backbone, 4 GPUs, ~500K steps) and RealEstate10K.
- reason: Explicit pose-indexed memory, the mechanism MG3 probes resemble.
- verified: page

## 12. Evaluating Long-Term Memory in 3D Mazes (Memory Maze)

- url: https://arxiv.org/abs/2210.13383
- type: paper
- summary: Randomised 3D mazes with object-location and localisation probes; offline dataset and probing evaluation.
- reason: Standard memory testbed for world models; code github.com/jurgisp/memory-maze.
- verified: page

## 13. BAgger: Backwards Aggregation for Mitigating Drift in Autoregressive Video Diffusion Models

- url: https://arxiv.org/abs/2512.12080
- type: paper
- summary: Self-supervised corrective trajectories from the model's own rollouts, plain flow-matching loss.
- reason: Exposure-bias fix that needs no distillation; a candidate to test on a toy world.
- verified: page

## 14. LeWorldModel: Stable End-to-End JEPA from Pixels

- url: https://arxiv.org/abs/2603.19312
- type: paper
- summary: 15M-param JEPA, two loss terms, trains on one GPU in a few hours, 2D and 3D control tasks; code github.com/lucas-maes/le-wm.
- reason: Explicit small-compute latent baseline for the implicit-versus-explicit state question.
- verified: page

## 15. SmallWorlds: Assessing Dynamics Understanding of World Models in Isolated Environments

- url: https://arxiv.org/abs/2511.23465
- type: paper
- summary: Six isolated fully observed domains; RSSM, transformer, diffusion and neural-ODE compared on long-rollout degradation.
- reason: Closest existing benchmark of rollout drift across architectures in small worlds.
- verified: page

## 16. MiniWorld: Democratizing the Training of Video World Models from Scratch

- url: https://arxiv.org/abs/2608.01127
- type: paper
- summary: Block-causal video DiT with diffusion forcing and rolling KV cache; several days on one 8-GPU server; code and checkpoints promised.
- reason: Open from-scratch recipe, but 8-GPU scale, so only partly single-GPU.
- verified: page

## 17. Scaling Laws for Pre-training Agents and World Models

- url: https://arxiv.org/abs/2411.04434
- type: paper
- summary: Power laws in world modelling and imitation; coefficients depend on tokenizer, task and architecture.
- reason: Main evidence on whether small-scale scaling results carry over.
- verified: page

## 18. Craftax: A Lightning-Fast Benchmark for Open-Ended RL

- url: https://arxiv.org/abs/2402.16801
- type: paper
- summary: JAX Crafter rewrite up to 250x faster; MIT; persistent, partially observed 2D world.
- reason: Candidate data generator with revisitable state; code github.com/MichaelTMatthews/Craftax.
- verified: page

## 19. Kinetix: Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks

- url: https://arxiv.org/abs/2410.23208
- type: paper
- summary: Jax2D engine: 9.05M engine-only steps/s and 824K/s inside an RL loop (best case); pixel renderer; MIT.
- reason: Controllable 2D rigid-body physics data generator; code github.com/FLAIROx/Kinetix.
- verified: page

## 20. XLand-MiniGrid: Scalable Meta-Reinforcement Learning Environments in JAX

- url: https://arxiv.org/abs/2312.12044
- type: paper
- summary: JAX grid worlds with partial observability; paper reports tens of millions of steps/s over many devices, about 1M steps/s training throughput on one device; Apache-2.0.
- reason: Partially observed, rule-configurable grid world for memory tests; code github.com/corl-team/xland-minigrid.
- verified: page

## Curation

Curated 2026-11-24 under `agency: max`, verdict slow/low (GPU busy with the overnight gridworld run), so ingests ran two at a time. The selection follows the build path in ADR 0001/0002: what the gridworld DiT, the DF-vs-TF drift test, the memory arms and the climate evaluation lean on.

1. DIAMOND: ingested → alonso2024diffusion (declined 2026-11-23 as RL-adjacent; it is now the single-GPU from-scratch reference for our own training)
2. Delta-IRIS: declined — discrete-token Crafter world model aimed at RL sample efficiency (`../ai-robotics` territory); DIAMOND already covers the single-GPU reference role
3. Genie: already in graph (bruce2024genie)
4. Jafar: declined — reimplementation of an in-graph source; ADR 0001 builds a causal DiT, not a latent-action model; revisit if a latent-action arm is added
5. GenieRedux: declined — same reason as Jafar; its one transferable finding (an RL exploration policy beats random data collection) is recorded here
6. TinyWorlds: declined — educational repo; parameter count and compute are not documented in the README
7. Diffusion Forcing: already in graph (chen2024diffusion)
8. How Far is Video Generation from World Model (physical laws): declined — the OOD-generalisation question is not on the active drift/memory/climate track, and at 32–64 A100s it is no compute reference; revisit if generalisation over generator knobs becomes a study
9. Long-Context State-Space Video World Models: declined — StateSpaceDiffuser covers SSM-state memory with a ground-truth maze revisit, which is closer to our setup; revisit if an SSM memory arm is added
10. StateSpaceDiffuser: ingested → savov2025statespacediffuser (seeded `spatial-memory`; code not yet released, NeurIPS'25 per README)
11. WorldMem: ingested → xiao2025worldmem
12. Memory Maze: declined — an RL-agent memory benchmark; our memory probes use decoded state in our own gridworld
13. BAgger: ingested → po2025bagger (no code yet; the reversal trick needs an action-inverse map in our gridworld)
14. LeWorldModel: declined — no latent-predictive arm in ADR 0001/0002; the V-JEPA line is covered by assran2025vjepa; revisit for an implicit-vs-explicit-state comparison
15. SmallWorlds: ingested → li2025smallworlds (seeded `long-rollout-evaluation`; benchmark named "SmallWorld" in the text)
16. MiniWorld: ingested → zhao2026miniworld (correction: code and 0.5B/1B checkpoints are released, not "promised")
17. Scaling Laws for Pre-training Agents and World Models: declined — no active question needs scaling evidence yet; revisit when gridworld models are scaled
18. Craftax: declined — not used; the gridworld in `src/wm/gridworld.py` is custom
19. Kinetix: declined — rigid-body physics generator, not on the build path
20. XLand-MiniGrid: declined — ADR 0001 proposed layering on it, but the project built its own gridworld instead

