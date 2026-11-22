---
kind: concept
name: "Novel View Prediction"
status: seedling
added: "2026-11-22"
sources: [welch2026fei-fei-li-atlas]
related_concepts: ["Diffusion Forcing", "Bidirectional vs Causal Video Diffusion", "Real-to-Sim-to-Real"]
related_experiments: []
tags: [nvs, spatial-intelligence, implicit-3d, objective]
---

# Novel View Prediction

## Definition

A world-model training objective: given context (text, reference views and
their poses) and a target camera pose, predict the image (optionally depth)
seen from that pose. It plays the role next-token prediction plays for LLMs
and next-frame prediction plays for video models. 3D structure becomes an
implicit internal state rather than an explicit output such as splats, meshes
or a NeRF.

## Why it matters here

World Labs moved from Marble (explicit Gaussian splats) to Atlas (NVS)
because the explicit representation capped visual fidelity
([[welch2026fei-fei-li-atlas]]). This is the project's central design axis:
**explicit vs implicit 3D**. The open tension is that downstream uses needing
physics (see [[Real-to-Sim-to-Real]]) still have to recover explicit geometry.

## Connections

- [[Diffusion Forcing]]: the training scheme that lets one NVS model run
  autoregressively, one pose at a time
- [[Bidirectional vs Causal Video Diffusion]]: camera-conditioned
  bidirectional video models need every pose upfront; causal NVS doesn't
- NeRF / 3DGS: per-scene optimization, the explicit-representation baseline
