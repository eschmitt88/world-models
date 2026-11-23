---
kind: moc
name: "Interactive World Models"
status: active
added: "2026-11-23"
concepts:
  - "[[concepts/bidirectional-vs-causal-video-diffusion]]"
  - "[[concepts/diffusion-forcing]]"
  - "[[concepts/exposure-bias]]"
  - "[[concepts/few-step-distillation-for-real-time-rollout]]"
  - "[[concepts/latent-action-model]]"
tags: [moc, interactive-world-models, autoregressive-diffusion, rollout]
---

# Interactive World Models

**What it takes to turn a video generator into something you can play.** A
world model you can steer has to produce frames one step at a time, condition
each on an action, stay coherent when it eats its own outputs, and do all of
this fast enough to respond. Each concept below is one of those requirements,
or the main way it fails. The open checkpoints this project runs (Matrix-Game
2.0 and 3.0, Self Forcing, Oasis) are each a particular set of choices across
the five.

## Make it steppable

- [[concepts/bidirectional-vs-causal-video-diffusion|Bidirectional vs Causal Video Diffusion]]:
  why Sora- and Wan-style models need the whole trajectory upfront, and the two
  routes out: post-train to causal ([[yin2024from]], [[huang2025self]]) or
  train causal from scratch (Atlas, [[welch2026fei-fei-li-atlas]]).

## Keep it stable over long rollouts

- [[concepts/exposure-bias|Exposure Bias]]: the failure. Continuous outputs
  compound their own errors.
- [[concepts/diffusion-forcing|Diffusion Forcing]]: per-frame noise levels as
  the training fix, with a context-noise knob at inference
  ([[chen2024diffusion]], [[song2025history]]). [[huang2025self]] says it is a
  partial cure and trains on the model's own rollouts instead.

## Make it fast and controllable

- [[concepts/few-step-distillation-for-real-time-rollout|Few-Step Distillation for Real-Time Rollout]]:
  DMD to 3-4 steps plus a KV cache. This is what makes 17-25 FPS on an H100
  possible ([[yin2024from]], [[huang2025self]], [[zhang2025matrix]]).
- [[concepts/latent-action-model|Latent Action Model]]: where the action labels
  come from when the training video has none ([[bruce2024genie]]).

## Open thread: memory is the missing requirement

None of the five concepts gives a model **memory beyond its attention
window**. A KV-cached causal model sees only the last ~1 s (Matrix-Game 2.0:
6 latent frames). In this project's first trial
(`experiments/2026-11-23-mg2-memory-horizon`), MG2 kept no memory of a view
after a ~4 s look-away-and-back: return-view LPIPS matched the
unrelated-scene level. Matrix-Game 3.0 adds pose-indexed retrieval of past
latents (`experiments/2026-11-23-mg3-memory-ablation`). Once that is
measured, "spatial memory" is likely the sixth concept here.
