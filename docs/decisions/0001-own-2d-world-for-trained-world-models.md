# 0001 — Train our own world models on a partially observed 2D gridworld

- **Date:** 2026-11-23
- **Status:** proposed (awaiting go-ahead to build)

## Context

The first trials (`experiments/2026-11-23-*`) could only *probe* large
pretrained models. Their findings (MG2 forgets after ~1 s; MG3's pose
retrieval recognises revisited places; long rollouts fall into darkness or
saturation attractors) had to be measured with proxies (LPIPS, SIFT), because
a pretrained model's imagined world has no ground truth. The question was
whether a lower-dimensional world would let us *train* world models from
scratch on one RTX 5080 and measure these phenomena against true state.

Inputs: a Fable design consultation (two rounds), a web literature sweep
(`raw/_candidates/2026-11-23-small-world-world-models.md`), and a measured
training-throughput probe (`scripts/bench_dit.py`).

## Decision (proposed)

1. **Keep the state 2D, keep the view egocentric and partial.** A fully
   visible top-down map removes look-away-and-back, which is the phenomenon
   under study. The observation is an agent-centred pixel crop of a larger
   persistent map. Kept from 3D video: partial observability, revisits,
   pose-indexed retrieval, exposure bias, action conditioning. Lost:
   perspective, occlusion, parallax, texture re-synthesis. A low-res
   first-person raycast of the same grid is the later second renderer.
2. **Layer on an existing JAX gridworld rather than writing a world.** Use
   NAVIX or XLand-MiniGrid (MiniGrid also gives direct comparability with
   StateSpaceDiffuser, arXiv 2505.22246). Add our own pieces on top: a fast
   tile-atlas renderer, map-generator knobs (landmark density, uniform-area
   fraction, a continuous nuisance such as shading), and scripted
   trajectories with a controlled revisit-gap distribution.
3. **Observation contract (the decision that matters most):** every
   generated frame decodes back to symbolic state by nearest-tile match. This
   gives tile accuracy, an off-manifold fraction and odometry error in place
   of LPIPS/SIFT, plus agent-placed objects as memory probes that the map
   prior cannot guess.
4. **Model: a pixel-space causal DiT with diffusion forcing, no autoencoder
   at first.** Continuous outputs, not discrete tokens, because the target
   phenomena (exposure bias, the context-noise knob) are continuous-output
   effects. 64×64 frames, patch 8 → 64 tokens/frame, 32-frame window, plus
   K retrieved frames for the memory arm.
5. **Budget:** one 15-h night per trained model. Ablation arms at 10-20M
   parameters, three seeds each. Kill rule: if the window-only baseline is
   not near ceiling on in-context tile accuracy after one night, shrink the
   world rather than extend training.

## Measured throughput (RTX 5080, bf16, AdamW, `scripts/bench_dit.py`)

| model | tokens | batch | throughput | peak VRAM |
|---|---|---|---|---|
| 47.5M, patch 8, 64 tok × 32 frames, FlexAttention block-causal | 2,048 | 16 | 78.7 seq/s, 161k tok/s, **2,517 frames/s** | 11.8 GB |
| same, SDPA with boolean mask | 2,048 | 16 | 57.7 seq/s, 1,847 frames/s | 11.9 GB |
| 47.3M, patch 4, 256 tok × 16 frames, FlexAttention | 4,096 | 8 | 34.4 seq/s, 551 frames/s | 11.8 GB |

A 15-h night is about 135M frames (4.2M 32-frame clips) at patch 8. For
comparison, Diffusion Forcing's 24M DMLab model took about 96 A100-hours,
but at higher image entropy.

## Experiment order

1. **Drift and attractors with known state:** teacher forcing vs diffusion
   forcing, an inference context-noise sweep, and an attractor rate as a
   function of the generator's uniform-area fraction. This fills two of the
   reported gaps. Go/no-go: drift must exist in the baseline. Report state
   error and the off-manifold fraction.
2. **Equal-compute memory comparison:** plain long context vs pose-indexed
   retrieval vs SSM/recurrent state, on ground-truth revisit accuracy and
   marker recall. This is the stated benchmark gap. A one-night window-only
   model doubles as the harness check, and is not reported as a finding,
   since StateSpaceDiffuser, WorldMem and Po et al. already show
   window-vs-memory.
3. **Imagination consistency:** does the model recall regions it *invented*
   as well as regions it *saw*? No prior work found.

## Consequences and risks

- Tile worlds may be too easy to drift in, because crisp tiles let the model
  snap back. Hence the nuisance knob, and the go/no-go on drift.
- Memory can be faked by the map prior: always report against a
  no-context baseline and use agent-placed markers.
- Imagined regions must be scored against the model's own earlier
  commitment, not the simulator map.
- Transfer to 3D video is mechanism-level only. Each claim should come back
  as one prediction tested on MG2/MG3.
- MiniGrid moves a whole tile per step and turns in 90° snaps. Smooth
  scrolling is a later variable.
- `hexphys-rl` (design only) is not the substrate now. Keep the world behind
  a `state -> render(pose)` interface so it could plug in later, for "the
  world evolves while you look away" studies.
