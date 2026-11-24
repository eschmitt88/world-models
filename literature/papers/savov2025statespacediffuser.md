---
kind: paper
title: "StateSpaceDiffuser: Bringing Long Context to Diffusion World Models"
authors: ["Nedko Savov", "Naser Kazemi", "Deheng Zhang", "Danda Pani Paudel", "Xi Wang", "Luc Van Gool"]
institutions: ["INSAIT, Sofia University", "ETH Zurich", "TU Munich"]
year: 2025
venue: "arXiv v3 (repo README says NeurIPS 2025; the PDF says under review)"
peer_reviewed: unknown
url: https://arxiv.org/abs/2505.22246
code_url:
citations:
source: "raw/papers/savov2025statespacediffuser.pdf"
added: "2026-11-24"
relevance: 4
credibility: 3
status: skimmed
related_experiments: []
related_concepts: ["Spatial Memory", "Diffusion World Model", "Exposure Bias"]
tags: [state-space-model, mamba, minigrid, csgo, long-context, revisit, diamond]
---

# StateSpaceDiffuser: Bringing Long Context to Diffusion World Models

## TL;DR

A short-window diffusion world model (DIAMOND) forgets what it saw a few
steps ago. StateSpaceDiffuser adds a Mamba state-space model that runs over
the whole history and feeds its features to the diffusion branch. On a
partially observed MiniGrid maze with a go-and-return protocol, this lifts
recall PSNR from about 26 to about 40 dB. It seeds [[Spatial Memory]]:
the nearest existing revisit test with ground truth.

## Claims

- Diffusion world models lose long-term context because they see only a
  short window of recent frames (K = 4 for DIAMOND), so content drifts from
  what was observed.
- A persistent compact state from an SSM restores long-term recall while
  keeping diffusion image quality; the SSM adds under 2% of inference compute.
- A forward-then-reversed-actions protocol measures recall of seen content.

## Methods

- Long-Context Branch: each frame becomes one vector (Cosmos continuous
  tokenizer, scale 16), concatenated with an action embedding and fed to
  Mamba, trained to predict the next frame's features with MSE.
- Generative Branch: the DIAMOND U-Net (EDM, 3 denoising steps) over 4 frames,
  conditioned on the last 4 SSM outputs through a two-layer MLP fusion.
- Two-stage training: SSM first, then freeze it and train the diffusion
  branch. End-to-end training was unstable and diffusion learned to ignore
  the SSM features.
- Data: MiniGrid mazes (85x85 grid, 7x7 egocentric view, colour markers),
  100-step episodes, 50 steps out and 50 back along the same path; CS:GO with
  mirrored action sequences.

## Results

- MiniGrid, context 16 (PSNR avg / final): DIAMOND 27.13 / 25.44;
  SSM-only 33.40 / 33.17; StateSpaceDiffuser 41.01 / 40.55. Context 50:
  26.13 / 25.15 vs 39.68 / 39.32. Zeroing the state features drops it below
  the baseline (23.68 / 20.95).
- Trained at context 50, it keeps the gain at 100 and 150 (37.99 and 30.75
  average PSNR vs 26.39 and 24.35 for DIAMOND).
- CS:GO: 12-participant user study, mean preference +0.20 (frame 15) and
  +0.24 (frame 17) on a [-1, 1] scale; no fidelity metric.

## Critique / open questions

- MiniGrid predicts one frame at a time from the ground-truth history, so most
  of each frame is copied from the previous one and only the newly revealed
  cells need recall. This tests memory, not free-running rollout, so it says
  little about [[Exposure Bias]] drift over imagined rollouts.
- The return path retraces the outbound path exactly. Recall of a *different*
  route to a seen place is not tested.
- The CS:GO evidence is a small user study with a modest preference score.
- Reported gains are PSNR on a simple grid render, where the dataset
  statistics make the metric easy to move.
- No released code: the repo README says "Expect the code here soon".

## Trust signals

- **Credibility:** 3, INSAIT/ETH group (Van Gool) and a repo (MIT, no code
  yet) that claims NeurIPS 2025, but the PDF says "under review", the code
  is not released and I could not establish citations.

## Follow-up

**Relevance:** 4, the closest published 2D-maze revisit experiment with ground
truth, and it seeds [[Spatial Memory]]; it uses DIAMOND as the generative
branch ([[alonso2024diffusion]], [[Diffusion World Model]]).
- Reuse the protocol shape in `gw-memory-equal-compute`: out-and-back and
  also revisit via a different route, scored against the true grid state.
- Ask whether our causal DiT's full-context attention already gives the
  recall that the SSM supplies here.
