# Plan — Real Llama frozen-QK / frozen-gate decomposition v1

## Context

The repo holds a synthetic React demo of token information flow (each bar shows the mixture of source-token information carried at that residual-stream position after each layer). The synthetic data was hand-crafted to *look* like attention rollout. We are now replacing the synthetic numbers with **real values from a Llama-3.2-3B forward pass**, computed via a frozen-QK + frozen-gate + frozen-RMSNorm-scale linear decomposition of the residual stream. The product is an **interactive web tool**: the user types a prompt into the React UI, a Python backend runs Llama and returns per-source-contribution magnitudes per (layer, target-position, source-position), and the existing bar visualization renders them with a user-selectable norm.

Why this matters: the synthetic demo is illustrative but tells us nothing about a real model. The real decomposition is a genuine interpretability instrument — it tells us, at every layer and position, how much each source token contributes to the residual stream, with the model linearized at the actual activation point so the decomposition is exact at that point.

---

## Locked decisions (will be saved to `DECISIONS.md`)

| # | Question | Decision | Rationale |
|---|----------|----------|-----------|
| 1 | Model | `meta-llama/Llama-3.2-3B`, configurable via `model_name` string. No hardcoded dims. | Supervisor preference. Cost of full configurability within Llama family is essentially zero with nnsight. |
| 2 | Framework | `nnsight`. | Supervisor pick. |
| 3 | MLP handling | **Frozen-gate (A′).** `s* = SiLU(W_gate · x*)` per position from real forward pass; MLP becomes `W_down · (s* ⊙ W_up · x)`, exactly linear, exact at x*. | Most simple. Structurally symmetric with frozen-QK. No autograd. Not Jacobian, not pass-through. |
| 4 | RMSNorm | Freeze scale `1/√mean(x²)` per token from real forward pass. | Standard practice. Required for the linearization to compose with frozen-QK and frozen-gate. |
| 5 | Hardware | Device-aware (`cuda` / `mps` / `cpu`), default `cpu`. Auto-detect with override. | Supervisor's hardware not yet pinned down. Code stays portable. |
| 6 | Heads | Aggregate across heads in viz; compute and cache per-head on backend. | Drill-in available later without recomputing. |
| 7 | Norms | Four selectable in UI: **L0, L1, L2, L∞** over the d_model axis. No functional norms (cosine, unembed) in v1. | Supervisor list. Functional norms deferred to v2. |
| 8 | Prompt input | **Interactive** — text box in the React UI, sent to backend on submit. | Supervisor: this is an interactive tool. |
| 9 | Viz | **Reuse the existing React UI**, replace synthetic data source with backend fetch. No matplotlib. | Supervisor pick. |
| 10 | Perturbation | **Deferred to v2.** Will design together. | Supervisor preference. |

---

## v1 refactor-friendliness for known v2 features

We will design v1's API and frontend so the following v2 features are additive, not rewrites:

| v2 feature | Pure viz? | v1 design accommodation |
|---|---|---|
| **Perturbation UI** (causal + linearized, click-to-ablate) | No (causal rerun is a backend roundtrip) | Backend endpoint pattern: `/decompose?prompt=…` is one call. v2 adds `/decompose?prompt=…&ablate=2,4` and `/decompose?prompt=…&causal_ablate=2,4`. Cache the per-source vector contributions on the backend so linearized ablation is in-memory math, not a re-decomposition. |
| **Functional norms** (cosine, unembed-projected) | Yes | Norm selector is already a dropdown; add new entries. Backend `norms.py` is a registry — add functions, register names. |
| **Color-grouping (visual only)** | Yes | Frontend renders one bar segment per source; coloring is just a per-source → color mapping. v2: replace with per-source → group → color. No backend change. |
| **Color-grouping (mathematically merged)** | **No** | Need to sum vector contributions *before* taking the norm. API accommodation: optional `groups` parameter on `/decompose`; v1 leaves it unset (per-source norms). v2: pass groups → backend sums vectors per group then norms. |
| **Prompt → response, hide response sources** | Largely yes | Backend returns tokens with positions, no `prompt`/`response` label baked in. Frontend tracks the split locally and filters which source columns to render. Optional `/generate` endpoint for auto-response (math unchanged; just a generation step before the decomposition). Hide-and-renormalize is a frontend toggle. |
| **Per-head drill-in** | No | Per-head contributions are already computed and cached server-side in v1. v2: add `/decompose_per_head?layer=L&prompt=…` reading from the cache. |
| **Cross-architecture (Qwen-2.5)** | No (different model) | `model_name` is a config knob; all dims read from `model.cfg`. v2 is "swap the string + rerun the nnsight probe." |

The single most important refactor-readiness move: **the v1 backend, internally, builds and caches the full per-source vector contributions `c[L, p, s, :]`** (shape `[L, N, N, d_model]`, ~130 MB at fp32 for 28 × 20 × 20 × 3072). It only reduces to scalars when serializing the response. This makes grouped norms, per-head drill-in, linearized ablation, and functional norms cheap additions because the raw material is in memory.

---

## Repo layout after v1

```
/
├── CLAUDE.md
├── DECISIONS.md                       (new — decisions table above)
├── RECORD.md                          (new — research notebook, first entry is the SwiGLU lesson)
├── README.md                          (new — top-level overview)
├── backup/                            (unchanged)
├── synthetic_demo/                    (existing React files moved here, untouched)
│   ├── info_flow_demo.jsx
│   ├── index.html
│   ├── package.json
│   ├── package-lock.json
│   ├── vite.config.js
│   └── src/main.jsx
└── real/
    ├── docs/
    │   └── mlp_nonlinearity.md        (the formal math note)
    ├── backend/
    │   ├── pyproject.toml             (uv)
    │   ├── src/info_flow/
    │   │   ├── __init__.py
    │   │   ├── model.py               (load Llama via nnsight; device-aware)
    │   │   ├── decompose.py           (frozen-gate decomposition math)
    │   │   ├── norms.py               (L0 / L1 / L2 / L∞ reductions; registry-style for v2 additions)
    │   │   ├── server.py              (FastAPI: /decompose endpoint)
    │   │   └── cli.py                 (click CLI for one-off runs)
    │   └── tests/
    │       ├── test_decompose.py      (reconstruction, causality, norm sanity)
    │       └── scratchpad/            (probes, throwaway scripts)
    └── frontend/
        ├── package.json
        ├── vite.config.js
        ├── index.html
        └── src/
            ├── main.jsx
            └── InfoFlow.jsx           (adapted from synthetic info_flow_demo.jsx)
```

---

## Implementation phases for v1

> **Branch convention:** each phase lives on its own branch, merged to `main` via PR when the phase is complete and tests pass. Branch names are listed per phase below. Start each phase with `git checkout main && git pull && git checkout -b <branch>`. **When a phase finishes, always push the branch to `origin` (`git push -u origin <branch>` on first push of the branch, plain `git push` after that).** Pushing is not the same as merging — the branch stays unmerged until the supervisor reviews and merges it, but pushing makes the work visible to the supervisor and backs it up off the local machine. Do this immediately when the phase commit lands; do not wait to be asked.

### Phase 0 — Repo hygiene `branch: phase/0-repo-hygiene`
1. Move React files into `synthetic_demo/`. Reinstall its `node_modules` there.
2. Write `DECISIONS.md`, `RECORD.md` (with the SwiGLU lesson as first entry), top-level `README.md`.
3. Create `real/docs/mlp_nonlinearity.md` (formal math note).

### Phase 1 — nnsight probe (highest-risk step; do first) `branch: phase/1-nnsight-probe`
Throwaway script `real/backend/tests/scratchpad/probe_nnsight.py`. For a short prompt, print shapes of:

- per-layer residual stream (pre and post each sublayer)
- per-(layer, head, source, target) attention pattern
- per-(layer, position) SwiGLU gate activation (post-SiLU, pre-Hadamard)
- per-(layer, position) RMSNorm scale factor (the `1/√mean(z²)` term, both norm1 and norm2 in each block)

**Decision gate:** if any of these isn't available at the granularity we need, stop and replan before writing decomposition code.

### Phase 2 — Decomposition math (`decompose.py`) `branch: phase/2-decomposition-math`

> **Before writing any code in this phase, read [`real/docs/decomposition_math.tex`](../real/docs/decomposition_math.tex) (rendered: `real/docs/decomposition_math.pdf`) in full.**
> That file is the authoritative formal specification of the linearization: notation, frozen-QK attention recurrence, frozen-RMSNorm construction, frozen-gate SwiGLU derivation, the full algorithm, and all four correctness criteria (C1–C4). The pseudocode below is a summary only — treat `decomposition_math.tex` as ground truth for the math.

Pipeline for a prompt of length `N`, model with `L` layers, `H` heads, `d_model = d`:

1. Run real forward pass. Cache: residual `x[L, p, d]`, attention pattern `A[L, H, p, s]`, gate activation `s*[L, p, d_ff]`, RMSNorm scales `r1[L, p]` and `r2[L, p]`.
2. Initialize `c[0, p, s, :] = δ(p == s) · embed(token_p)`.
3. For each layer L:
   - Attention sub-block: `c_attn[p, s] = c[L, p, s] + Σ_h Σ_s' A[L, h, p, s'] · W_O_h · W_V_h · (γ1 ⊙ c[L, s', s] · r1[L, s'])`
   - MLP sub-block: `c[L+1, p, s] = c_attn[p, s] + W_down · (s*[L, p] ⊙ (W_up · (γ2 ⊙ c_attn[p, s] · r2[L, p])))`
4. Output full tensor `c[L, p, s, :]` kept in memory; reduced per-norm versions serialized to frontend.

### Phase 3 — Sanity tests (`test_decompose.py`) `branch: phase/3-sanity-tests`
- **Reconstruction**: `‖Σ_s c[L, p, s] − x_L[p]‖∞ < 1e-3` at every (L, p).
- **Causality**: `c[L, p, s] == 0` for `s > p`.
- **Norm sanity**: non-negative; standard inequalities hold.
- **Identity at layer 0**: `c[0, p, s]` is `embed(token_p)` when `p==s`, zero otherwise.
<!--  -->
### Phase 4 — Norm reductions (`norms.py`) `branch: phase/4-norm-reductions`
Registry-style: `register("l2", fn)`. v1 ships L0/L1/L2/L∞. L0 needs a threshold (`|component| > 1e-6 · max|c|`).

### Phase 5 — FastAPI server (`server.py`) `branch: phase/5-fastapi-server`
- Load model once at startup; keep warm.
- Endpoint `GET /decompose?prompt=…&norm=l2`:
  ```json
  { "tokens": [...], "n_layers": 28, "norm": "l2",
    "contributions": [ /* [n_layers, n_target, n_source] */ ] }
  ```
- **Internally cache the full vector contributions** keyed by prompt hash, so v2 grouped/perturbation/per-head features are cheap additions.

### Phase 6 — React frontend (`InfoFlow.jsx`) `branch: phase/6-react-frontend`
Adapt synthetic `info_flow_demo.jsx`:
- Prompt input box + "Run" button (with loading spinner; CPU 5–15s).
- Norm selector: dropdown (L0 / L1 / L2 / L∞).
- Replace `BASE_LAYERS` with `fetch('/decompose?prompt=…&norm=…')`.
- Tokens come from backend (real tokenizer output).
- Keep click-to-trace.
- Vite proxy `/decompose` → `http://localhost:8000`.
- Remove the synthetic-only MLP-toggle (Llama always has MLPs).

### Phase 7 — Docs and final polish `branch: phase/7-docs-polish`
- Top-level `README.md`: how to run, where the math lives.
- `RECORD.md`: empirical first-prompt results, reconstruction tolerance, surprises.

---

## Files to create / modify

**Create:**
- `DECISIONS.md`, `RECORD.md`, `README.md`
- `real/docs/mlp_nonlinearity.md`
- `real/backend/pyproject.toml`
- `real/backend/src/info_flow/{__init__,model,decompose,norms,server,cli}.py`
- `real/backend/tests/test_decompose.py`
- `real/backend/tests/scratchpad/probe_nnsight.py`
- `real/frontend/{package.json, vite.config.js, index.html}`
- `real/frontend/src/{main.jsx, InfoFlow.jsx}`

**Move (no content edit):**
- `info_flow_demo.jsx`, `index.html`, `package.json`, `package-lock.json`, `vite.config.js`, `src/main.jsx` → `synthetic_demo/`
- `node_modules/` either moved or reinstalled fresh under `synthetic_demo/`.

**Untouched:** `CLAUDE.md`, `backup/`.

---

## Existing code / utilities to reuse

- **The synthetic React UI structure** (`info_flow_demo.jsx`): bar layout, color coding, click-to-trace, toggle controls all transfer. Changes: data source (fetch vs hardcoded), tokens (real vs fixed), addition of prompt input + norm selector, removal of MLP toggle.
- **nnsight's `LanguageModel` + `.trace()`** is the entire backend integration surface — no custom hooks.
- **Llama's weights** (`W_V`, `W_O`, `W_up`, `W_down`, `γ1`, `γ2`) are read directly off the model; no reimplementation.

---

## Verification (end-to-end)

1. `cd real/backend && uv run pytest tests/` — sanity tests pass.
2. `uv run uvicorn info_flow.server:app --reload --port 8000` — server starts, model loads.
3. `curl 'http://localhost:8000/decompose?prompt=the%20cat%20sat&norm=l2'` returns valid JSON, non-degenerate.
4. `cd real/frontend && npm install && npm run dev` — UI opens, prompt → bars render, click-to-trace works.
5. Compare visually to synthetic demo: deeper layers should show more mixing; layer 0 pure-self.
6. Append result + surprises to `RECORD.md`.

---

## Open caveats and things I have not yet verified

- **nnsight probe (Phase 1)** is genuinely the riskiest step.
- **Llama-3.2-3B latency on CPU** estimated 5–15s/prompt, not measured. Mitigations if much worse: bf16, downsize to 1B for v1, or require GPU.
- **GQA** (24 query heads, 8 KV heads in 3B): doesn't change the linearization, but I'll confirm head/kv-head shapes in Phase 1 before writing `decompose.py`.

---

## v2 backlog (do NOT build in v1; design together when v1 ships)

- **Perturbation UI**: causal + linearized, click-to-ablate (math note in `mlp_nonlinearity.md`).
- **Functional norms**: cosine-to-final-residual, unembed-projected L1 (direct logit effect).
- **Color-grouping (mathematical merge)**: needs API `groups` parameter; norms computed after summing vectors. (Pure-color-only grouping is frontend-only.)
- **Prompt → response with hidden-response viz**: largely frontend; optional `/generate` endpoint for auto-response.
- **Per-head drill-in**: secondary endpoint using already-cached per-head contributions.
- **Multi-prompt comparison**, **cross-architecture (Qwen-2.5)**.

---

## Artifacts to commit on plan exit

### `RECORD.md` — first entry

> **Lesson (Q3 design discussion, MLP handling).**
> When asked how to handle the MLP nonlinearity, I jumped to a generic textbook tool — full Jacobian linearization at the actual activation — and offered it as the recommended option. I did this **without first writing out SwiGLU's explicit form on the page**. When the supervisor asked for a formal explanation, I had to write `MLP(x) = W_down · (SiLU(W_gate · x) ⊙ (W_up · x))` properly. At that point the structure-specific option — *freeze the gate*, i.e. treat `s* = SiLU(W_gate · x*)` as a constant so the MLP becomes the linear map `W_down · diag(s*) · W_up`, exactly linear, exact at x* — became immediate. This option (frozen-gate, A′) is strictly simpler than the Jacobian, is structurally symmetric with the frozen-QK approach we were already using elsewhere, and did **not** appear in my original menu.
>
> **Rule for future me: before reaching for a generic mathematical tool, read the actual math of the operation under analysis. Write out the architecture's explicit form first. Look for structure-specific simplifications first; generic tools second.** The "selection × write" factorization in SwiGLU (gate selects features, W_up/W_down write them) mirrors the QK/OV split in attention — that pattern is everywhere in modern decoder transformers and is the first thing to look for.

### `real/docs/mlp_nonlinearity.md` — formal note

Contents (to be written on plan exit):
- SwiGLU definition with shapes for Llama-3.2-3B.
- The two sources of nonlinearity (SiLU on the gate; Hadamard product is bilinear).
- The frozen-gate construction with the linearity proof.
- Why not the full Jacobian (gate-sensitivity term it would capture; we chose simplicity).
- The two perturbation modes (causal rerun vs linearized counterfactual) and what their difference measures.
