# `test_warmup_sanity.py` — what each check is asserting

A local-only sanity harness for `calc_contribution_per_layer_per_residual`. Runs
on the tiny [`ToyLlama`](toy_llama.py) (same architectural shape as real Llama:
GQA, RMSNorm, eager attention so attention weights are materialised) without
touching NDIF, so it iterates in seconds.

## The claim under test

`calc_contribution_per_layer_per_residual` is supposed to decompose, for each
layer $\ell$, the self-attention output at every query position $q$ into a sum
over source positions $k$:

$$\text{self\_attn.output}_\ell[q] \;=\; \sum_{k} \text{contrib}_\ell[q, k]$$

If that identity holds, the rest of the project (information-flow visualization)
is built on solid ground. If it doesn't, the decomposition has a math or
indexing bug and everything downstream is suspect.

## How the script is organised

- `capture_reference(model, prompt)` — runs one trace and grabs the real
  `self_attn.output[0]` (the attention output) and `self_attn.output[1]` (the
  softmax pattern) for every layer. These are the **ground truth** the
  implementation must match.
- `calc_contribution_per_layer_per_residual(model, prompt, remote=False)` —
  produces the per-source decomposition `contrib[l][q, k, :]` we want to check.
- A handful of `check_*` functions, each returning a list of `Result(name, ok, detail)`
  so failures don't short-circuit the rest.
- Two `report_*` functions that just dump diagnostic tables — no pass/fail.

Each check is intentionally narrow: a failure in any one check points at a
different class of bug, so isolating which check fails is half the diagnosis.

## The checks

### 1. `shapes`
Verifies `len(per_layer) == LAYERS_NUM` and each `per_layer[l]` has shape
$(Q, Q, d_\text{model})$. This catches "the trace didn't actually populate the
list" (a real NDIF-vs-local hazard with `list().save()`) and any rank/axis
mistakes before deeper checks run on garbage shapes.

### 2. `attn pattern is causal softmax`
Sanity on the **reference capture** itself, not the implementation under test.
Two sub-checks:

- $\sum_k a_{h, q, k} \approx 1$ for every $(h, q)$ — softmax-normalised.
- $a_{h, q, k} = 0$ for $k > q$ — causal mask is honored.

If this fails, we're not looking at the model we think we're looking at, and
every other check is meaningless. It exists purely so a confusing downstream
failure doesn't send us chasing a phantom bug.

### 3. `reconstruction (main claim)`
The headline test. For each layer, computes
$\text{recon}[q] = \sum_k \text{contrib}[q, k]$ and compares to the captured
`self_attn.output[q]` with `torch.allclose(atol=1e-2, rtol=1e-2)`. Also reports
the global relative error
$\lVert \text{recon} - \text{ref} \rVert / \lVert \text{ref} \rVert$.

This is the only check whose passing actually validates the decomposition.

### 4. `contrib causal structure`
Checks that $\text{contrib}[q, k] = 0$ for all $k > q$ (strictly upper
triangle in the $(q, k)$ plane). The implementation builds `contrib` with a
`torch.zeros(...)` then writes into a lower-triangular subset, so this should
trivially pass — but if it ever doesn't, an indexing bug is overwriting the
wrong cells and the main claim will be wrong for a different reason than you'd
expect.

## Diagnostic tables (not pass/fail)

### `per-query reconstruction error`
For each $(\ell, q)$ prints $\lVert\text{recon}[q]\rVert$,
$\lVert\text{ref}[q]\rVert$, and their relative error. When the main claim
fails, this table tells you **which queries are most wrong** — a flat error
profile and a strongly $q$-dependent one point at different bugs.

### `diagonal contribution norms`
For each $(\ell, q)$ prints $\lVert \text{contrib}[q, q] \rVert$. The diagonal
is the "self-token" channel of the decomposition; whether it's populated or
empty is the most direct signal about how the loop bounds are written.

## How to run

```bash
cd real/backend
uv run python tests/scratchpad/test_warmup_sanity.py
```

No HF token, no NDIF key, no network for the model itself (ToyLlama is built
from a `LlamaConfig` with random weights). The tokenizer is pulled from HF on
first run and cached.

## Adding more checks

Each check is a function taking `(per_layer, ref, d_model)` and returning
`list[Result]`. Add a new entry to the `CHECKS` list at the bottom of the
script. If a check is more of a "look at numbers" affair than a pass/fail, make
it a `report_*` function called explicitly from `main`.
