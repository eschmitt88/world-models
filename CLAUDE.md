# Project: world-models

Orientation only. `~/.claude/CLAUDE.md` holds the durable principles; this
file refines them for this project.

## What this project is about

Generative, interactive **world models**: models that learn to predict what a
scene or environment looks like next, conditioned on actions or camera poses,
and can be rolled out as a simulator. In scope: action-conditioned video
models (Genie, Cosmos, Oasis-style), novel-view / spatial models (World Labs
Atlas and Marble, NeRF and 3DGS lineage), causal and autoregressive diffusion
(diffusion forcing, self-forcing, exposure bias), latent-predictive models
(V-JEPA), and real-to-sim-to-real uses for robotics. Three jobs: log recent
research, run open models locally on one 16 GB GPU, and prototype new ideas.
Dreamer-style latent dynamics for RL control belong to `../ai-robotics`;
cross-link rather than duplicate.

## Layout

- `raw/` — immutable source material. Read only.
- `literature/` — processed notes on papers, repos, posts.
- `concepts/` — atomic ideas. Promote to `mocs/` when ≥5 cluster.
- `experiments/YYYY-MM-DD-<slug>/` — self-contained runs.
- `docs/decisions/` — lightweight ADRs.
- `journal/` — daily session files (hook-written).
- `_meta/` — index, log, templates.

## Rules

`.claude/rules/` holds scoped conventions. Each loads when you read or edit
a matching path (its `paths:` frontmatter): `experiments.md` for
`experiments/**`, `notebooks.md` for notebooks, `data.md` for `raw/**` and
data files.

Framework rules live at `~/claude-system/claude/rules/` and load only on
demand: read `evaluation.md` before designing or judging an evaluation,
`agency.md` before an autonomous run. Skills that list them under
`respects:` do this themselves.

## Budget & compute

`budget.yaml` sets the hard ceilings (wall time, tokens, disk), the
autonomy level and the model roles. Anything with real resource demands
has to fit under it. If it doesn't, say so and scope down, or flag the
ceiling.

@budget.yaml

## Project-specific facts

- Primary language: Python (PyTorch). 16 GB VRAM: prefer small/distilled
  checkpoints, quantization, short rollouts; ask `/jobs` for a window first.
- Environment: managed by `uv`; run `make env` to sync.
- Data: tracked by DVC. Large artifacts on SN850X via `~/projects/`.

## Housekeeping

- End sessions with `/wrap`. The SessionEnd hook backstops this.
- Experiments are scaffolded by `/new-experiment <slug>` (part of the
  experiment skill group; claude-system README, "Growing a lit repo").
- Run `/lint` weekly.
