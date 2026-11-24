---
kind: paper
title: "FreqForcing: Autoregressive Long Video Generation via Spectral Self-Anchoring"
authors: ["Jiatong Li", "Leo Liang", "Linghe Kong", "Yulun Zhang"]
institutions: ["Shanghai Jiao Tong University", "Tencent HY Team"]
year: 2026
venue: "arXiv v2 (preprint, 3 Aug 2026); AAAI-style format, venue not stated"
peer_reviewed: unknown
url: https://arxiv.org/abs/2607.27110
code_url: https://github.com/jiatongli2024/FreqForcing
citations:
source: "raw/papers/li2026freqforcing.pdf"
added: "2026-11-24"
relevance: 3
credibility: 3
status: read
related_experiments: ["2026-11-23-gw-weather-climate-eval", "2026-11-23-gw-drift-df-vs-tf"]
related_concepts: ["Spectral Rollout Diagnostics", "Exposure Bias", "Long-Rollout Evaluation", "Diffusion Forcing"]
tags: [video-diffusion, autoregressive, drift, frequency-domain, attention-sink, training-free, self-forcing, long-rollout]
---

# FreqForcing: Autoregressive Long Video Generation via Spectral Self-Anchoring

## TL;DR

A frequency-domain account of drift in Self-Forcing rollouts (Wan2.1-1.3B,
text-to-video, 5 s training clips): error accumulation "appears as a
pronounced energy drift in the low-frequency bands", with the DC and low bands
collapsing and the high band jittering. A training-free fix, Spectral
Self-Anchoring (SSA), pulls the low-frequency part of the attention output
back toward that of a few early, high-quality "anchor" frames. It reaches 120 s
(24x the training length). The spectrum is a diagnostic picture and a
correction target. It is never used as a predictor of failure.

## Claims

- Error accumulation in streaming video generation shows up as energy drift
  in the DC and low-frequency bands, which the authors map to "color tone,
  layout, identity". High-frequency components ("local motion, fine detail")
  "become unstable" rather than drifting monotonically.
- Attention sink (keeping a few initial frames in the KV cache) slows the
  spectral drift, and a larger sink (S = 12 against 3) slows it more, but
  "cannot fully resolve it".
- Deep Forcing (a KV-compression baseline) "suffers from frequency jitter"
  in attention outputs, which shows up as pixel-space flicker.
- A training-free spectral fusion beats other training-free methods on
  VBench-Long and is "competitive with" training-based ones.

## Methods

- Analysis (Sec. "Error Accumulation in the Frequency Domain"): 60 s videos
  from Self-Forcing on MovieGen prompts. They record the self-attention output
  of every DiT layer and the decoded pixel frames, then apply a short-time
  Fourier transform "to these sequences to track how the frequency band
  energy changes over time". Plotted: relative spectral energy of DC, low and
  high bands over generation time, in latent (attention output) and pixel
  space (Fig. 3, Fig. 6). The main text does not define the band edges or say
  whether the bands are temporal or spatial frequency; no table of numbers
  accompanies the curves.
- SSA: a local branch (sliding window with attention sink) and an anchor
  branch (6 latent frames, one in every three of the first 21 generated, then
  frozen). Both attention outputs go through a 3D FFT over space and time; the
  fused output is A_loc + lambda * H_lp * (A_anc - A_loc) with a Gaussian
  low-pass H_lp (sigma_xy = sigma_t = 0.125, lambda = 0.6), applied only in the
  first two of four denoising steps. About 16.5% extra latency on A6000s.
  Temporal RoPE indices are shifted (Rolling-Forcing style).
- Evaluation: 128 MovieGen prompts, seven VBench-Long metrics at 60 s and 120 s,
  against Self-Forcing, LongLive, Rolling Forcing (training-based), Infinity-RoPE
  and Deep Forcing (training-free), all on Wan2.1-T2V-1.3B.

## Results

- 60 s: Dynamic Degree 59.58 against 33.29 for Self-Forcing, Overall
  Consistency 20.94 against 18.81; 120 s: 58.97 and 20.98 against 27.42 and
  16.17. Subject and background consistency are on par with the baselines
  (97.27 and 96.47 at 60 s; Rolling Forcing and LongLive are higher on some).
  Best on Dynamic Degree and Overall Consistency only; Motion Smoothness is
  lower than Self-Forcing's.
- Ablation (60 s): without SSA, Dynamic Degree 56.85 and Overall Consistency
  20.26; without the attention sink, 44.18 and 20.00.
- Hyperparameters trade consistency against diversity: raising sigma or
  lambda lifts consistency and the "repetitive rate" (0.675 to 0.877 across
  the grid; 0.749 at the chosen setting, against 0.812 for Deep Forcing).
- The spectral claims rest on curves in Fig. 3 and 6, not on any reported
  statistic. No spectral number is given in a table.

## Critique / open questions

- **Which band, and in which direction.** The drifting band is the lowest one
  (DC plus low), and its energy falls ("collapses rapidly" without a sink) as
  colour fades. The high band is described as unstable or jittery, with no
  consistent sign. This is a third pattern next to the two in
  [[Spectral Rollout Diagnostics]]: PDE-Refiner has a neglected high tail at
  one step, Lehmann 2026 an excess small-scale ratio in blow-ups, and
  FreqForcing a lost large-scale (DC) component in a fading video. Not a
  contradiction, since the domain and the failure mode differ (fading
  text-to-video against numerical blow-up), but it means no single signed
  band works across systems, and the distance-from-reference-in-log-terms
  design the concept note proposes is the one that fits all three.
- **Per rollout step?** Time-resolved but not step-resolved: STFT over the
  generated sequence, shown as curves over a 60 s run. Which prompts or how
  many rollouts are averaged is not stated in the main text. Nothing is
  binned by rollout step against a reference, and no noise floor from
  independent real samples is given (the reference is the model's own early
  frames).
- **Prediction.** None. The spectrum is observed after the fact and used as a
  correction target. There is no lead-time analysis and no test of whether a
  spectral change precedes visible failure. The claimed link between spectral
  energy and "visual quality" is shown qualitatively (Fig. 3, 5).
- **Not action-conditioned.** The setting is prompt-driven text-to-video,
  without actions, poses or a revisit test. The paper calls itself a step
  toward interactive world models, but "world model" is only the
  motivation.
- **The fix may trade away variation.** The authors' own repetitive-rate
  column rises with stronger anchoring: anchoring the low band toward early
  frames is a memory of the first seconds, which suppresses drift partly by
  suppressing change.
- **Metric limits.** VBench-Long metrics are model-based image and video
  scores; Dynamic Degree and Overall Consistency are the only ones where it
  wins. No ground truth, no state-level statistics, one 1.3B base model, one
  seed per prompt.

## Trust signals

- **Credibility:** 3, a university group with a Tencent co-author, code released and the numbers consistent between tables, but unreviewed as far as stated, no citations yet, one base model, and the central spectral claim is shown only as figures.

## Follow-up

**Relevance:** 3, useful prior art for ADR 0002's "spectral failure predictor" and for [[Exposure Bias]] in streaming video, but it is text-to-video, descriptive and shifts no concept.

- **Undercuts (partly) the novelty of "spectra for rollout drift in video":**
  a per-time band-energy analysis of autoregressive rollouts is published,
  and a correction target is built on it. ADR 0002 should cite it and not
  claim the first spectral view of video drift.
- **Does not pre-empt the predictor:** it reports no lead-time test, no
  threshold, no noise floor, no calibration against a reference spectrum and
  no relation to the first illegal state. The "predicts failure" claim stays
  open, and so does the reference-ratio design.
- **Adds a sign case to the conflict:** low band falls (fading) in a video
  model. Record it in [[Spectral Rollout Diagnostics]]; when the gridworld
  spectra are computed, the predictor should be two-sided and cover the
  lowest band (DC and mean tile statistics) as well as the high tail.
- **Supports, as a baseline idea:** snapping or re-anchoring the state toward
  an early or exact reference is the training-free analogue of ADR 0002's
  snap-to-tile arm. FreqForcing's anchoring pulls only the low band toward
  early frames; snap-to-tile projects onto the frame manifold.
- Related: [[huang2025self]] (base model), [[zhao2026frechet]] (the training
  alternative that matches feature distributions).
