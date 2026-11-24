# NOTES

Running log of work sessions. `/wrap` appends a new dated section at the
end of each session with **Did / Findings / Next** subsections. The
SessionEnd hook backstops this if you forget.

<!-- entries go below this line, newest at bottom -->

## 2026-11-23

### Did
- Scaffolded the project (public, experiment skills) on 2026-11-22 and ingested the Welch Labs / World Labs Atlas video as the first source. Switched to `agency: max` on request.
- Literature sweep (Sonnet subagent): ingested 11 sources (Diffusion Forcing, DFoT, Self Forcing, CausVid, Genie, V-JEPA 2, SEVA, Matrix-Game 2.0, MineWorld, Oasis blog, Atlas blog), gave all 5 seed concepts primary sources, and seeded 3 new concepts. Promoted `mocs/interactive-world-models.md` (5 concepts).
- Got Matrix-Game 2.0 (README: ≥24 GB) and Matrix-Game 3.0 (~6.5B, tested on A/H-series) running on the 16 GB RTX 5080. Wrappers are `src/wm/mg2.py` and `src/wm/mg3.py`; `src/wm/shims/flash_attn` is an SDPA stand-in.
- Ran three experiments, all `done`, 204 rollouts in total: `2026-11-23-mg2-memory-horizon` (120), `2026-11-23-mg3-memory-ablation` (64), `2026-11-23-mg2-context-noise-drift` (20 × 96 s).
- Published the report "Look Away, Look Back" (claude.ai artifact U7TB1szRCswYH9dAtdMyxR; source in `reports/2026-11-23-first-trials/`).

### Findings
- MG2's spatial memory is its ~1 s KV window. After a look-away-and-back of ≥ 4 s the return view is at the independent-sample level (LPIPS 0.64 vs seed spread 0.55; 0/12 runs recognised by SIFT+RANSAC). Even standing still, it drifts that far in about 15 s.
- MG3's pose-indexed retrieval memory works. After a 180° turn and back, the median is 92 matched keypoints and 7/8 runs are recognised. With retrieval off in the same model: 5 matches, 0/8. Unrelated scenes: median 6, max 17.
- LPIPS badly understates the effect (0.49 vs 0.56): the model remembers places, not pixels. Use geometric matching for memory tests.
- In 96-s MG2 rollouts, context noise picks the attractor. At 0 there is darkness lock-in (53% of frames black); at 100-200 it escapes into a saturated, flat-shaded world; at 400 it turns murky. No setting keeps the scene past ~15 s.
- Engineering lessons: int8 W8A8 on MG3's action MLP rounded the mouse signal to zero, so the camera didn't turn. Always verify that actions still act after quantizing. Upstream MG2 keeps all decoded frames on the GPU, which OOMs long rollouts.

### Next
- Loop closure with translation (walk out and back), to test MG3 retrieval under parallax.
- Long-rollout attractors in MG3: does retrieving early clean frames pull rollouts back?
- A layout-aware metric (depth agreement, DINOv2 patches) to replace LPIPS in memory tests.
- Try Self Forcing (Wan 1.3B) and Matrix-Game 4.0 (engine-grounded, explicit geometry; officially supports the 5080). Ingest the MG3 tech report and seed a "spatial memory" concept for the MoC.
- Gated checkpoints (Oasis, SEVA, Cosmos) need a Hugging Face token on this box.

## 2026-11-24

### Did
- Asked whether a lower-dimensional world would let us *train* world models. Consulted Fable twice and ran a web sweep (`raw/_candidates/2026-11-23-small-world-world-models.md`), then measured throughput (47M DiT ≈ 2,500 frames/s on the 5080). Wrote ADR 0001.
- Built the world and models: `src/wm/gridworld.py` (persistent tile maps, egocentric 64-px view, ~400k frames/s render, exact decode to tile ids, revisit walks, phase-correlation frame shift), `src/wm/df.py` (22M pixel block-causal DiT; diffusion or teacher forcing; pose-retrieved memory slots), `train_df.py`, `train_mem.py`, `climate_eval.py`.
- User asked for a weather-ML crossover. Consulted Fable twice and ran a web sweep (`raw/_candidates/2026-11-23-weather-rollout-crossover.md`). Wrote ADR 0002 (weather vs climate evaluation, kill criteria pre-registered).
- Overnight: 4 models trained, 3 experiments done: `2026-11-23-gw-drift-df-vs-tf`, `2026-11-24-gw-weather-climate-eval`, `2026-11-23-gw-memory-equal-compute`.
- Published the report "Small World, Long Rollouts" (claude.ai artifact TsgA6iexv5JRuSHHVJRUjJ; source in `reports/2026-11-24-small-world/`).

### Findings
- Teacher forcing is best short-horizon (32-frame accuracy 0.92 vs 0.73) and collapses long (0.15-0.20 at 384 frames). Diffusion forcing at context noise 0.1 stays exact for 384 frames when still or pacing (12× the window), but drifts at noise 0. The context-noise curve is U-shaped, the same as MG2's, now measured against ground truth.
- Weather/climate: three of four kill criteria passed (climate drifts and warns early; spread predicts error, ρ 0.50-0.87). Gaussian contraction failed (it orders survival the wrong way); on-manifold shift recovery orders it right.
- Two failure modes. tf goes **off-manifold**, and truth-free snap-to-tile rescues it (skill 0 → 0.27). df@0 collapses **on-manifold** into a wrong climate (uniform frames): snapping is useless and context noise fixes it. Validity metrics rank df@0 above tf; climate ranks them the other way.
- Memory: 4 pose-retrieved frames hold recall at 0.98 to a 252-frame round trip vs 0.39 for 36 frames of plain context (expected; a harness check). "Imagined regions forgotten" was a first-sight metric artifact: models revise their first drafts of newly revealed edges.
- Process lessons: concurrent GPU smoke tests OOM'd a production run (rule: none while a run is active). tf and memory arms need gradient accumulation at batch 32. `set -e` in a chain script skips everything after one failure. Background waits on `systemctl` misfire.

### Next
- Paper (ADR 0002): 2-3 seeds per regime plus mixed-noise arms (≥6 models) to test on-manifold recovery as the survival predictor; a learned Thermalizer prior vs snap (predicted to help only tf); matched-sample df vs tf.
- Memory: a matched-gap imagined-vs-real comparison; action-integrated poses instead of oracle ones.
- Take the climate split back to MG2's attractors.
