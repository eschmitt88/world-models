# world-models

**Exploring generative, interactive world models: tracking the research, running open models on a single GPU, and prototyping ideas.**

📂 **[Browse this repo →](https://eschmitt88.github.io/world-models/)** —
interactive, always-live view of experiments, concepts, literature, and maps of
content. Served via GitHub Pages from `docs/index.html`; reads the live file
tree, no build step. _(Link is live once the repo is public and Pages is enabled
— `/new-project` does both by default.)_

## What this is

A world model predicts how a scene changes as you act in it or move through
it: the next frame given an action, or the view from a new camera pose. These
models are converging on causal, autoregressive diffusion transformers you can
roll out interactively, like a learned simulator. This repo keeps a literature
log of the field, runs open checkpoints locally to see what they actually do
(consistency, drift over long rollouts, controllability), and prototypes small
ideas that fit on one 16 GB GPU.

## How it's organized

Plain Markdown + flat YAML frontmatter, cross-linked with `[[wikilinks]]`:

- `concepts/` / `mocs/` — atomic ideas; promoted to a map of content when ≥5 cluster.
- `literature/` — processed notes on papers, repos, posts (0–5 relevance scored).
- `experiments/YYYY-MM-DD-<slug>/` — self-contained runs (hypothesis → result, config, metrics, log).
- `raw/` — immutable source captures · `docs/decisions/` — ADRs · `_meta/` — index, log, templates.

## Local use

```sh
make env    # uv sync
make lint   # knowledge-graph / experiment health check
```

Built on the [claude-system](https://github.com/eschmitt88/claude-system)
research framework (upstream attribution — this project is its own repo).
See `CLAUDE.md` for the agent-facing orientation and `~/.claude/CLAUDE.md`
for the framework's durable principles.
