# Information-flow visualization for a real Llama

This repo is replacing a synthetic React demo of per-token information flow
through a transformer with a **real-model** version. A Python backend runs
`meta-llama/Llama-3.2-3B` (via [`nnsight`](https://nnsight.net)), linearizes the
forward pass with **frozen-QK + frozen-gate + frozen-RMSNorm-scale** so the
residual stream at every (layer, position) can be decomposed exactly into
per-source-token contributions, and a React frontend renders the result
interactively (prompt in → bars out, with a selectable norm and click-to-trace).

## Layout

- `synthetic_demo/` — the original synthetic React demo, kept for reference. Run
  with `cd synthetic_demo && npm run dev`.
- `real/` — the real-Llama implementation. Currently only `real/docs/` exists;
  backend (`real/backend/`) and frontend (`real/frontend/`) land in later phases.
- `backup/` — legacy `CLAUDE.md`.
- `CLAUDE.md`, `DECISIONS.md`, `RECORD.md` — project memory (working agreement,
  locked design decisions, research log).

## Where the math lives

- `real/docs/decomposition_math.tex` — canonical formal spec of the
  linearization (notation, frozen-QK recurrence, frozen-RMSNorm, frozen-gate
  SwiGLU derivation, full algorithm, correctness criteria C1–C4). Rendered to
  `real/docs/decomposition_math.pdf`.
- `real/docs/building_blocks.md` — transformer primer.

## Status

Phase 0 (repo hygiene) is done. Phase 1 (nnsight probe — confirm the cache
points we need are exposed at the right granularity) is next.
