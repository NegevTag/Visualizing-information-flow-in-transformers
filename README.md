# Information Flow Visualization

Visualize, per token, how information flows through a real Llama model. For a
given prompt the backend decomposes the residual stream at every
layer/position into **per-source-token contributions**, and the frontend
renders those contributions as stacked bars so you can read which earlier
tokens fed into each position.

The core trick: by **freezing** the attention pattern (QK circuit) and the MLP
gate values from a normal forward pass, the residual stream becomes a *linear*
function of the per-source contributions. That linearity is what lets us
attribute every position's residual back to the source tokens that produced it.

---

## Repository layout

```
real/
  backend/                 FastAPI service that computes the decomposition
    src/api_checks/         the API + decomposition logic (see below)
    src/info_flow/config.py runtime config (Config, from .env.local)
    pyproject.toml          deps (uv); also declares info_flow package
  frontend/                React + Vite UI that calls the backend and draws bars

project_scratchpad/        archived research: experiments, tests, notes,
                           docs, the synthetic demo, and earlier prototypes.
                           Not needed to run the app.
```

### `api_checks` (the backend that matters)

| File | Role |
|------|------|
| `api.py` | FastAPI app. `GET /?prompt=...` returns attention/MLP contribution **norms** `(layer, position, source)`, the tokenized prompt, and top next-token predictions. |
| `model.py` | `ModelInformationCalculatorF32` + `calc_contribution_per_layer_per_residual`. Runs the model via **nnsight** (remote on NDIF by default), captures the frozen attention pattern and MLP gate, and reconstructs each layer's residual as a sum over source tokens. RMSNorm is folded in analytically. All math is done in `float32`. |
| `api_cache.py` | `APICache` — memoizes full runs to `.pt` files keyed by `(model_name, prompt)`, so repeated prompts skip recomputation. |
| `full_run_result.py` | Pydantic containers: `FullRunResults`, `Contributions` (post-attention / post-MLP, shape `(layer, position, source, d_model)`), `ResidualStream` (the true residuals, for precision checks), `ResultsDimentions`. |
| `utils.py` | `get_model` (loads an nnsight `LanguageModel`) and cache-file timestamp helper. |

### `frontend`

`src/InfoFlow.jsx` fetches the backend JSON and draws, for each layer/position,
a stacked bar where each segment is a source token's normalized contribution
norm. Attention and MLP bands are colored distinctly (Tufte-ish muted palette).
`ZoomPanVanilla.jsx` adds zoom/pan over the grid.

---

## Running

### Backend

Requires a `real/backend/.env.local` providing the `Config` fields:
`hf_token`, `ndif_api_key`, `info_flow_model`, `result_cache_path`,
`default_atol`, `default_rtol`.

```bash
cd real/backend
uv sync
uv run uvicorn api_checks.api:app --reload   # serves on http://127.0.0.1:8000
# or: uv run python -m api_checks.api
```

`GET http://127.0.0.1:8000/?prompt=The capital of France is` returns:

```json
{
  "attention_norms": [[[...]]],   // (layer, position, source)
  "mlp_norms":       [[[...]]],   // (layer, position, source)
  "tokens":          ["The", " capital", ...],
  "top_perdictions": {" Paris": 0.81, ...}
}
```

### Frontend

```bash
cd real/frontend
npm install
npm run dev          # Vite dev server; hits the backend directly (CORS is open)
```

---

## How the decomposition works (sketch)

For a prompt of length $P$ we track a contribution tensor of shape
$(\text{layer}, \text{position}, \text{source}, d_{\text{model}})$, initialized so
position $p$'s only contribution is its own embedding. Then per layer:

1. **Attention** — apply RMSNorm (`input_layernorm`) to the contributions, push
   them through $W_V$, weight each source by the **frozen** attention pattern
   $\text{softmax}(QK^\top)$, and project with $W_O$. Because the pattern is held
   fixed, this is linear in the source contributions.
2. **MLP** — apply RMSNorm (`post_attention_layernorm`), up-project, multiply by
   the **frozen** gate activations $g$, then down-project. Again linear given $g$.
3. Accumulate into the next layer's contribution tensor.

A final RMSNorm + LM head gives logits. The true residual stream is also captured
(`ResidualStream`) so the reconstruction can be checked against the real forward
pass. The API exposes the per-source **L2 norms** of these contributions, which is
what the UI visualizes.
