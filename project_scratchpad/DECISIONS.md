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
