---
kind: experiment
slug: "mg2-memory-horizon"
date: "2026-11-23"
status: done     # running | done | abandoned
hypothesis: "MG2's spatial memory is bounded by its KV attention window: after an out-and-back camera pan, revisited views match the original only when the time gap is within the window"
result: "Confirmed: MG2 keeps a view only within its ~1 s attention window; after a >=5 s look-away-and-back the return view differs from the original as much as two independent samples do (LPIPS 0.64 vs seed spread 0.55)"
related_concepts: ["Exposure Bias", "Bidirectional vs Causal Video Diffusion", "Novel View Prediction"]
related_literature: ["zhang2025matrix", "huang2025self", "chen2024diffusion"]
tags: [matrix-game-2, open-model-trial, rollout]
# members: only set when kind: ensemble — list parent experiment slugs.
# parent:  only set when this experiment was produced via /propose --expand.
---

# mg2-memory-horizon

## Hypothesis

Matrix-Game 2.0 (Skywork, 1.8B causal Wan-based, distilled to 3 steps) attends
over a sliding window of `local_attn_size = 6` latent frames (~21-24 video
frames, about 1 s at 25 fps; `configs/distilled_model/universal/config.json`).
Its only longer-range memory is the CLIP embedding of the start image,
injected by cross-attention (`inference.py`, `visual_context`).

**Prediction.** In an out-and-back pan (camera right for T frames, then left for
T frames), the frames at `t_peak - k` and `t_peak + k` should show the same view.
Their perceptual distance (LPIPS) should stay low while the gap `2k` is under
about 24 frames, then rise steeply toward the *unrelated-scene* baseline (LPIPS
between different seeds at the same timestep). The return frame (k = T) should
match frame 0 only for small T. A CLIP-level "same place, different
details" plateau above the noise floor would mean the image embedding carries
coarse memory past the window.

## Setup

- Config: `config.yaml`
- Code: `src/wm/mg2.py` (loads MG2 once, runs scripted action episodes), `run.py` in this folder
- Model: `Skywork/Matrix-Game-2.0` base_distilled_model, upstream commit 51913b2 (vendor/, gitignored)
- Data: start images from MG2 `demo_images/universal/` (no holdout; not an HCE project)

## Result

120 rollouts: 4 images × 3 seeds × T ∈ {12, 24, 48, 96, 192} × {out-and-back, still}.
Full table in `results/summary.json`; headline numbers in `metrics.json`.
LPIPS (AlexNet, 176×320) between frames that share a camera heading:

| round trip (frames) | 24 | 48 | 96 | 192 | 384 |
|---|---|---|---|---|---|
| return vs start, out-and-back | 0.10 | 0.25 | 0.54 | 0.63 | 0.64 |
| same interval, camera still | 0.03 | 0.08 | 0.21 | 0.41 | 0.58 |

Reference levels: independent seeds' return frames at T=192 differ by
**0.55** [0.47, 0.63]. Frame 0 of different start images differ by **0.72**.

Mirror-pair curve (all T pooled), by gap: ≤16 frames 0.05–0.07, then 24 → 0.11,
32 → 0.18, 64 → 0.37, 128 → 0.56, 256 → 0.61. The still-camera control rises
more slowly over the same gaps: 0.04, 0.07, 0.18, 0.36, 0.53.

Setup notes: the image condition is a 21-latent VAE-encoded prefix extended
with its converged black-frame latent (mean abs difference 0.0085 vs latent
std 1.26 against a longer encode; `src/wm/mg2.py`). Yaw 0.1/frame moves the
view about 8.75 px/frame, so the start view is fully off-screen once T ≥ 73.
At 8.5 fps and a 13.7 GB peak on the RTX 5080, MG2 runs at about a third of
its H100 real-time speed.

## Interpretation

The hypothesis holds, and the break sits where the architecture says it
should. Revisited views stay consistent while the round trip fits inside the
6-latent KV window (~21-24 frames). They degrade steeply past it and reach
the independent-sample level by a ~5 s round trip. At that point MG2 is
*re-imagining* the place it left rather than remembering it. The CLIP
image-embedding cross-attention keeps the style (village stays village) but
not the layout; there is no plateau below the seed-spread line.

Two things I didn't expect:

1. **The world drifts even when you stand still.** With no input, the frame
   after 15 s is about as far from the start (0.58) as an independent sample.
   So "memory" fails in time as well as in space. The out-and-back excess
   over that baseline is largest at mid gaps (32-96 frames), where moving
   the camera adds its own forgetting on top of the generic drift.
2. **The Minecraft start (0016) degrades least** (0.43 at T=192 vs
   0.64-0.77 for the others). Repetitive block textures make re-imagined and
   remembered views look alike to LPIPS. So LPIPS understates forgetting on
   self-similar scenes, and per-image reporting matters.

The Monet start image (smoke test) also shows the time drift directly: within
~4 s MG2 turns an impressionist haystack into a game-engine snowy village.
The start image anchors only the first second.

## Diagnostics

Fill in after the run. One line per field; leave `n/a` rather than
blank. `next_candidates` must list ≥2 concrete one-sentence proposals.
Every concrete claim below needs a **citation anchor** — a code
reference like `train.py:42-58`, a metrics file path like
`metrics.json:val_acc`, or a wikilink into `literature/`. Unanchored
assertions are flagged by `/lint` (Kosmos, arXiv 2511.02824).

Unless otherwise noted, metric numbers here reference `metrics.json`
(validation split). Cite `final_metrics.json` only if this experiment
is itself the final-scoring pass.

- intended_effect_confirmed: yes — return LPIPS 0.10 (24-frame round trip) → 0.54 (96) → 0.64 (384), reaching the seed-spread level 0.55 (metrics.json:return_lpips_outback_T*, ref_seed_spread_lpips)
- leakage_check: n/a — no holdout; this is a probe of a frozen public checkpoint, not a trained model
- overfitting_signal: train=n/a val=n/a gap=n/a — no training (metrics.json)
- delta_from_prior: vs none (first experiment)
- unexpected_findings: the still-camera control drifts to near the independent-sample level by 384 frames (metrics.json:return_lpips_still_T192 = 0.58); Minecraft start image underreports forgetting (results/summary.json:per_image.0016)
- next_candidates:
  - Run the same protocol on Matrix-Game 3.0 with retrieval memory on vs off to isolate what pose-indexed memory buys (experiments/2026-11-23-mg3-memory-ablation).
  - Replace LPIPS with a layout-sensitive metric (e.g. DINOv2 patch-feature correspondence or depth-map agreement) so self-similar scenes stop masking forgetting.
## Follow-up

- Figures and videos: see the project report artifact; full videos in `results/videos/` (gitignored, regenerable with `run.py`).
