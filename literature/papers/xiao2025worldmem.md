---
kind: paper
title: "WORLDMEM: Long-term Consistent World Simulation with Memory"
authors: ["Zeqi Xiao", "Yushi Lan", "Yifan Zhou", "Wenqi Ouyang", "Shuai Yang", "Yanhong Zeng", "Xingang Pan"]
institutions: ["S-Lab, Nanyang Technological University", "Wangxuan Institute of Computer Technology, Peking University", "Shanghai AI Laboratory"]
year: 2025
venue: "NeurIPS 2025"
peer_reviewed: true
url: https://arxiv.org/abs/2504.12369
code_url: https://github.com/xizaoqu/WorldMem
citations:
source: "raw/papers/xiao2025worldmem.pdf"
added: "2026-11-24"
relevance: 4
credibility: 4
status: skimmed
related_experiments: ["2026-11-23-gw-memory-equal-compute", "2026-11-23-mg3-memory-ablation"]
related_concepts: ["Spatial Memory", "Diffusion Forcing", "Diffusion World Model"]
tags: [memory-bank, pose-retrieval, minecraft, oasis, cdit, diffusion-forcing, revisit]
---

# WORLDMEM: Long-term Consistent World Simulation with Memory

## TL;DR

An autoregressive diffusion-forcing world model (Oasis-style CDiT) gets an
external memory bank of past latent frames, each stored with its pose and
timestamp. Per step it retrieves a few units by field-of-view overlap and
recency, and reads them through cross-attention blocks whose queries and keys
carry pose and time embeddings. On Minecraft this lifts PSNR on frames 600+
steps later from 17.3 (DF) to 24.0. It is the explicit pose and
timestamp-indexed memory that [[Spatial Memory]] says Matrix-Game 3's
retrieval resembles.

## Claims

- A token-level memory bank with state cues (pose, timestamp) restores
  long-term 3D consistency without explicit geometry.
- State-aware attention (dense Plucker pose embedding, relative encoding)
  beats sparse or absolute pose cues.
- Adding timestamps lets the model track changing scenes (placed objects,
  growing wheat), not only static ones.

## Methods

- Base: conditional DiT with spatial and temporal blocks, causal temporal
  attention, 25-d action input via AdaLN, trained with
  [[Diffusion Forcing]] noise levels. Oasis initialisation for Minecraft;
  DFoT for RealEstate10K.
- Memory unit: (latent tokens, 5-d pose x y z pitch yaw, timestamp).
- Retrieval (greedy): confidence = FOV overlap (Monte Carlo) x 1 minus
  0.2/tc x time gap; pick the top unit, drop units with similarity above 0.9
  to it, repeat for 8 units.
- Conditioning: memory frames are fed as clean (lowest noise) frames; an
  attention mask lets memory affect only the memory blocks. Query pose is
  reset to identity and key poses made relative.
- Training: 12K Minecraft videos of 1500 frames (MineDojo), 8-frame context
  plus 8-frame memory, about 500K steps on 4 GPUs. Progressive sampling
  (memory within 2 m, then 8 m) is used.

## Results

- Beyond the context window (600-frame bank, 100 future frames, ground-truth
  poses, 300 test videos): DF 17.32 PSNR / 0.4376 LPIPS / 51.28 rFID; WorldMem
  23.98 / 0.1429 / 15.37. Within the window: 24.11 vs 25.98 PSNR.
- Ablations: random retrieval 18.32 PSNR, confidence filter 23.12, plus
  similarity filter 23.98. Dense relative pose 23.98 vs sparse absolute 20.67.
  Progressive sampling 23.98 vs 19.23 small-range or 21.11 large-range.
- RealEstate10K loop-closure trajectories (37 to 60 frames): PSNR 23.34 vs
  ViewCrafter 21.72 and DFoT 16.42.

## Critique / open questions

- Poses come from the simulator. Predicting the next pose from action alone
  is only explored in the supplement, and in a first-person world with
  accumulating odometry error that is the hard part.
- Evaluation is 100 frames after a 600-frame bank, teacher-provided poses,
  PSNR against ground truth: recall, not a long free-running rollout. Nothing
  here measures drift over thousands of imagined steps.
- Revisit tests reuse memory frames that were themselves ground truth in
  this setup; whether memory of the model's own imagined frames stays
  consistent is not separated from memory of real ones.
- Retrieval is hand-built geometry (FOV overlap needs a pose and a camera
  model), so it does not transfer unchanged to a 2D top-down or egocentric
  grid; the analogue would be tile-position overlap.
- Full-sequence baseline is dropped from the long test, so the comparison is
  one baseline (DF).

## Trust signals

- **Credibility:** 4, NeurIPS 2025 paper from NTU S-Lab with released code
  (xizaoqu/WorldMem, 404 stars) and a project page; results are one
  customised Minecraft set plus RealEstate10K, and I did not establish
  citations.

## Follow-up

**Relevance:** 4, the canonical explicit pose and timestamp-indexed memory
bank for diffusion-forcing world models, and the mechanism that
[[Spatial Memory]] attributes to Matrix-Game 3's retrieval; it also uses the
same base recipe as our causal DiT with DF.
- In `gw-memory-equal-compute`, copy the retrieval rule in a 2D form (key =
  tile position and step, score = view-window overlap and recency) and
  compare against plain longer context at equal tokens.
- Test whether retrieval helps under free rollout, where memory entries are
  the model's own imagined frames (see [[Exposure Bias]]).
- Timestamps as a retrieval key are relevant to ADR 0002's climate arm: a
  changing world needs a "when" as well as a "where".
