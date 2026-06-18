# Decisions

Locked design decisions for the real-Llama information-flow decomposition project.
Append new decisions at the bottom. Do not silently overwrite — if a decision is
revised, add a new entry with the date and a pointer back to the superseded one.

---

## v1 locked decisions (from plan `ok-lets-make-it-vast-goose.md`)

| # | Question | Decision | Rationale |
|---|----------|----------|-----------|
| 1 | Model | `meta-llama/Llama-3.2-3B`, configurable via `model_name` string. No hardcoded dims. | Supervisor preference. Cost of full configurability within Llama family is essentially zero with nnsight. |
| 2 | Framework | `nnsight`. | Supervisor pick. |
| 3 | MLP handling | **Frozen-gate (A′).** $s^* = \mathrm{SiLU}(W_{\mathrm{gate}} \cdot x^*)$ per position from real forward pass; MLP becomes $W_{\mathrm{down}} \cdot (s^* \odot W_{\mathrm{up}} \cdot x)$, exactly linear, exact at $x^*$. | Most simple. Structurally symmetric with frozen-QK. No autograd. Not Jacobian, not pass-through. |
| 4 | RMSNorm | Freeze scale $1/\sqrt{\mathrm{mean}(x^2)}$ per token from real forward pass. | Standard practice. Required for the linearization to compose with frozen-QK and frozen-gate. |
| 5 | Hardware | Device-aware (`cuda` / `mps` / `cpu`), default `cpu`. Auto-detect with override. | Supervisor's hardware not yet pinned down. Code stays portable. |
| 6 | Heads | Aggregate across heads in viz; compute and cache per-head on backend. | Drill-in available later without recomputing. |
| 7 | Norms | Four selectable in UI: **L0, L1, L2, L∞** over the $d_{\mathrm{model}}$ axis. No functional norms (cosine, unembed) in v1. | Supervisor list. Functional norms deferred to v2. |
| 8 | Prompt input | **Interactive** — text box in the React UI, sent to backend on submit. | Supervisor: this is an interactive tool. |
| 9 | Viz | **Reuse the existing React UI**, replace synthetic data source with backend fetch. No matplotlib. | Supervisor pick. |
| 10 | Perturbation | **Deferred to v2.** Will design together. | Supervisor preference. |

---

## Phase 0 decisions (2026-05-13)

### 11. Math note: single source of truth is the LaTeX

`real/docs/decomposition_math.tex` (rendered to `decomposition_math.pdf`) is the
canonical math document. `real/docs/decomposition_math.md` has been removed.
No separate `real/docs/mlp_nonlinearity.md` is created — the SwiGLU / frozen-gate
derivation lives in §3.4 of the LaTeX.

**Why:** the `.md` and `.tex` carried duplicate content. Two files drift apart;
one file does not. The PDF render is what gets actually read by humans, the
`.tex` is what gets edited, and the `.md` was redundant overhead.

### 12. `node_modules` location

`node_modules/` was moved (filesystem `mv`) into `synthetic_demo/` along with
the rest of the React files, not reinstalled fresh.

**Why:** the existing install is known-good; `mv` on the same volume is near-
instant and avoids re-fetching ~21 packages. `node_modules/` is gitignored, so
the move is invisible to git.

### 13. Always push each phase branch when the phase is finished

When a phase's final commit lands, immediately push the branch to `origin`
(`git push -u origin <branch>` on first push of the branch, plain `git push`
after that). Do **not** wait to be asked. Pushing is not merging: the branch
stays unmerged until the supervisor reviews it.

**Why:** the supervisor cannot review work that only exists on the agent's
local machine, and a single-machine commit is one disk-failure away from
being lost. Pushing is cheap, non-destructive, and makes the phase commit
visible. The previous default ("commit, don't push, don't open a PR") was too
conservative — it conflated "don't merge" (correct) with "don't push" (wrong).

---

## SAE feature lens decisions (2026-06-18)

SAE analog of the logit-lens cell (`sae_feature_lens.py` + `.ipynb`).

| # | Question | Decision | Rationale |
|---|----------|----------|-----------|
| 14 | Which SAE | Llama Scope **8x / 32,768-feature residual** SAEs (`release="llama_scope_lxr_8x"`, `sae_id="l{L}r_8x"`, HF `fnlp/Llama3_1-8B-Base-LXR-8x`). | These are exactly the SAEs Neuronpedia indexes as `llama3.1-8b/{L}-llamascope-res-32k`, so local feature indices line up 1:1 for description lookup. `R` (post-block residual) matches our `is_mlp=True` contributions; SAE layer `L` ↔ `LLMResidualPosition(layer=L, is_mlp=True)`. Verified by loading (`d_in=4096, d_sae=32768`). |
| 15 | Local SAE vs Neuronpedia inference | **Run the SAE encoder locally** (`sae_lens`); use Neuronpedia only to fetch feature **descriptions** via the public GET endpoint. | Neuronpedia's activation API takes *text*, not vectors. Our contributions are *decomposed* vectors, not real activations, so they can't go through the text endpoint. |
| 16 | Layer | **Parameter**, default last layer (31). | Matches the logit cell by default; sweepable. |
| 17 | BOS / source-0 removal | **Raw view only** (no projection-out of the start direction). | Supervisor preference. |
| 18 | Keep encoder bias? | **Yes — keep the full encoder** (`b_enc` + ReLU + TopK). | Verified Llama Scope *has* `b_enc` (paper Eqs. 1/5); it is the SAE's threshold/notion of "active". Dropping it would change the quantity to a linear alignment score and surface features the SAE calls inactive. (Supersedes the "TopK has no bias → scale-invariant" idea, which was based on a false premise.) |
| 19 | Magnitude normalization | **Rescale each contribution's direction to the real residual norm at that position** (`run.precise[pos].norm()`), with a `raw` toggle for comparison. | Because `b_enc` is a fixed threshold for full-strength activations, top-features is scale-dependent. Empirically (layer 31) the raw `hate` contribution (norm 11.2 vs 87.7) collapses to one feature + zeros; normalized it recovers a rich top-5. Direct analog of the RMSNorm the logit lens applies before unembedding. |
| 20 | Location | `project_scratchpad/` (helper module + notebook), `sae-lens` added to `real/backend/pyproject.toml`. | Supervisor preference; reuses the backend env/cache the existing notebook already uses. |
