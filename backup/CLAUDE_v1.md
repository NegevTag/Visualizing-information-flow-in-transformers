# Project: Per-token information flow on real Llama (frozen-QK decomposition)

This file is the system prompt for any Claude session working in this repo. The global `~/.claude/CLAUDE.md` still applies; this file specializes it for *this* research project.

---

## Research goal

Take the synthetic React demo of token information flow (in `synthetic_demo/`) and reproduce it with **real numbers from a real Llama forward pass**. For an input sequence, decompose the residual stream at every (layer, position) into the contributions of each *source* token, with attention patterns (the output of the QK circuit) **frozen** so the decomposition is well-defined.

The output is a (layer × target-position × source-position) tensor of contribution magnitudes, plus a visualization that mirrors the demo's structure but with real model numbers.

## Core methodology

### Frozen QK → linear-in-source-embeddings

When the attention pattern `A = softmax(QK^T / √d)` is held fixed (computed once from the real forward pass and reused), each attention operation becomes a *linear* function of the residual stream:

```
attn_out[p] = Σ_s A[p,s] · (W_OV · residual[s])
```

If we also linearize RMSNorm (freeze the per-token scale factor from the actual forward pass — standard interpretability practice from the Anthropic Mathematical Framework), then *everything except MLPs* is linear in the source-token embeddings. The residual at every layer/position can be written as a sum of per-source contributions:

```
residual[L, p] = Σ_s contribution[L, p, s]              ∈ ℝ^{d_model}
```

The visualization shows the normalized magnitudes `||contribution[L, p, s]||` per source `s`, exactly analogous to the synthetic demo's bars.

### "Each token effect"

For now we treat it as the **vector contribution** of each source token to the residual stream at every (L, p), measured by some scalar norm (L2 or unembed-projected L1). This is *not* the same as causal ablation; see open questions for whether we should also run ablation as a sanity check.

## Working mode

**The user is the research supervisor.** I am the junior researcher. That means:

- **Ask questions liberally.** When a design choice is non-trivial and not already resolved in this file, ask before coding. Better to interrupt with a clarification than to silently pick the wrong path.
- **Challenge assumptions actively.** When I think something will work, also write down the cheapest argument or experiment that would *disprove* it, and run that first.
- **Verify before claiming.** Never assert that a tensor has a shape, an attention pattern looks a certain way, or a sanity check passes without having actually run the code. Precision over recall. If I'm guessing, say "I'm guessing."
- **Empirical loop.** Code small → run → inspect intermediate outputs (shapes, distributions, plots) → decide next step. Save artifacts so they can be revisited.
- **Lots of asserts.** Shape asserts (use `jaxtyping`), normalization invariants (rows of attention sum to 1, source contributions reconstruct the residual within tolerance), causality asserts (no future-token leakage).
- **Test the math, not just the plumbing.** The central correctness check is: `Σ_s contribution[L, p, s] ≈ actual_residual[L, p]` from the real forward pass, within numerical tolerance. If that fails the entire decomposition is wrong.
- **Keep `RECORD.md` as a research notebook.** Append-only log of what was tried, what worked, what didn't, and *why*. Snapshot key numbers and plots.
- **Don't yak-shave.** Maximum information per line of code. No premature abstractions. Three similar lines beat a half-baked helper.
- **Reuse aggressively.** If `transformer_lens` already exposes attention patterns, OV matrices, or hook points, use them. Don't reimplement.

## Open design questions (resolve with supervisor before committing to code)

These are the questions I'll bring to the supervisor at the start. Each one materially shapes what gets built.

1. **Model + hardware.** Llama-3.2-1B (fastest, easy on CPU/small GPU), 3.2-3B, or 3-8B? Is there a GPU in this environment, or are we CPU-bound? Smaller is better for iteration speed.
2. **Framework.** `transformer_lens` (clean hooks, attention pattern access, already supports Llama) vs raw HuggingFace `transformers` with custom forward hooks. TL is strongly preferred unless there's a reason against.
3. **MLP treatment.** Options ordered by fidelity:
   - (a) Pass-through: MLPs leave source-attribution unchanged at each position. Loses MLP effects but cleanest math.
   - (b) Proportional redistribution (what the synthetic demo does): MLP output at p is attributed to the existing source mix at p.
   - (c) Linearize at the actual activation: `MLP(x) ≈ J(x*)·x` with the Jacobian frozen. Most rigorous, more code, more compute.
   - (d) Skip MLPs entirely (attention rollout only). Simplest.
4. **RMSNorm.** Freeze the scale factor from actual activations (standard) — yes/no?
5. **Heads.** Aggregate attention pattern across heads per layer, or keep per-head and show per-head decomposition?
6. **Norm for "magnitude".** L2 of the contribution vector, L1 of the unembed projection (= "direct logit effect"-style), or something else?
7. **Input prompt.** Fixed sentence for parity with the synthetic demo, or user-configurable via CLI?
8. **Visualization pipeline.** Python → JSON → existing React frontend (keeps the polished UI), or Python-only matplotlib/plotly (faster iteration)?
9. **Scope of v1.** Just compute + sanity-check + one plot? Or full CLI + tests + multiple prompts?

## Planned repo layout

```
/
├── CLAUDE.md                       (this file)
├── RECORD.md                       (research notebook)
├── README.md                       (top-level overview)
├── synthetic_demo/                 (existing React demo, moved here as-is)
│   ├── info_flow_demo.jsx
│   ├── package.json, vite.config.js, index.html, src/, ...
└── real/                           (new Python package — the actual work)
    ├── pyproject.toml              (uv)
    ├── src/info_flow/
    │   ├── __init__.py
    │   ├── model.py                (load Llama, cache activations + attn patterns)
    │   ├── decompose.py            (the frozen-QK source decomposition)
    │   ├── visualize.py            (matplotlib or JSON-for-React export)
    │   └── cli.py                  (click CLI)
    └── tests/
        ├── test_decompose.py       (reconstruction sanity, normalization, causality)
        └── scratchpad/             (exploratory scripts, throwaway plots)
```

## Tooling

- `uv` for Python deps. Always `uv run <script>.py`, never plain `python`.
- `torch` + `transformer_lens` (preferred) or `transformers` + hooks.
- `jaxtyping` + `beartype` for tensor type/shape annotations.
- `pytest` (+ `hypothesis` where it fits) for tests.
- `click` for CLI, `rich` for output.
- `seaborn`/`matplotlib` for plots; Tufte-style — high data-ink ratio, no chartjunk.
- `polars` if dataframes appear.

## Definition of "done" for v1

A CLI that takes a prompt, runs Llama, computes the frozen-QK per-source decomposition for every (layer, position), passes the residual-reconstruction sanity test within tolerance, and produces a plot/JSON that drops into the visualization. Plus a short `RECORD.md` entry with one example prompt's results and any surprises.

## Definition of "done" for the research direction

Beyond v1, the question we're really after is: *can this decomposition tell us something non-trivial about how information moves through Llama?* That's an open-ended research question. v1 is the apparatus; the interesting work is what we look at with it.
