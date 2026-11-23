---
kind: concept
name: "Few-Step Distillation for Real-Time Rollout"
status: seedling
added: "2026-11-23"
sources: [yin2024from, huang2025self]
related_concepts: ["Bidirectional vs Causal Video Diffusion", "Exposure Bias", "Diffusion Forcing"]
related_experiments: []
tags: [distillation, dmd, kv-cache, real-time, streaming]
---

# Few-Step Distillation for Real-Time Rollout

## Definition

Compressing a many-step video diffusion model into a 4-step causal generator
(Distribution Matching Distillation, DMD, or similar) and caching keys and values
across frames, so each new frame costs a few forward passes.

## Why it matters here

It is what makes a diffusion world model interactive on one GPU. CausVid
distils a 50-step bidirectional teacher into a 4-step causal student
(1.3 s first-frame latency, about 9.4 FPS) ([[yin2024from]]). Self Forcing keeps the
4-step student but trains it on its own KV-cached rollouts, reaching 17 FPS
and 0.69 s latency on an H100 at 1.3B parameters ([[huang2025self]]).
Both papers report H100-class numbers, and the Self Forcing README asks for
24 GB, so 16 GB speeds are a project measurement still to be made.

## Connections

- [[Bidirectional vs Causal Video Diffusion]]: the teacher/student pair
- [[Exposure Bias]]: the training/inference mismatch the loss must respect
