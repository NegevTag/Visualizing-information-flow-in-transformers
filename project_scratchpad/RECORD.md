# Research notebook

Append-only log of what I tried, what worked, what didn't, and *why* — including
failed approaches and the lesson learned. Newest entries at the bottom.

---

## 2026-05-13 — SwiGLU lesson (Q3 design discussion, MLP handling)

> When asked how to handle the MLP nonlinearity, I jumped to a generic textbook
> tool — full Jacobian linearization at the actual activation — and offered it
> as the recommended option. I did this **without first writing out SwiGLU's
> explicit form on the page**. When the supervisor asked for a formal
> explanation, I had to write $\mathrm{MLP}(x) = W_{\mathrm{down}} \cdot
> (\mathrm{SiLU}(W_{\mathrm{gate}} \cdot x) \odot (W_{\mathrm{up}} \cdot x))$
> properly. At that point the structure-specific option — *freeze the gate*,
> i.e. treat $s^* = \mathrm{SiLU}(W_{\mathrm{gate}} \cdot x^*)$ as a constant so
> the MLP becomes the linear map $W_{\mathrm{down}} \cdot \mathrm{diag}(s^*)
> \cdot W_{\mathrm{up}}$, exactly linear, exact at $x^*$ — became immediate.
> This option (frozen-gate, A′) is strictly simpler than the Jacobian, is
> structurally symmetric with the frozen-QK approach we were already using
> elsewhere, and did **not** appear in my original menu.
>
> **Rule for future me: before reaching for a generic mathematical tool, read
> the actual math of the operation under analysis. Write out the architecture's
> explicit form first. Look for structure-specific simplifications first;
> generic tools second.** The "selection × write" factorization in SwiGLU
> (gate selects features, $W_{\mathrm{up}}/W_{\mathrm{down}}$ write them)
> mirrors the QK/OV split in attention — that pattern is everywhere in modern
> decoder transformers and is the first thing to look for.

---

## 2026-05-13 — Phase 0 executed (repo hygiene)

Moved synthetic React demo (`info_flow_demo.jsx`, `index.html`, `package.json`,
`package-lock.json`, `vite.config.js`, `src/`, `node_modules/`) into
`synthetic_demo/`. Deleted the duplicated `real/docs/decomposition_math.md`
in favor of the LaTeX source. Created this file plus `DECISIONS.md` and
top-level `README.md`. Branch: `phase/0-repo-hygiene`. Next: Phase 1 nnsight
probe.

---

## 2026-05-13 — Phase 1 nnsight probe — DECISION GATE: PASS

### Setup

- `real/backend/` initialized with `uv`: nnsight 0.7.0, torch 2.12.0+cpu,
  transformers 5.8.1, accelerate 1.13.0, jaxtyping 0.3.9.
- Probe lives at `tests/scratchpad/probe/{00_smoke,01_load_check,02_capture,03_inspect}.py`,
  with shared `_common.py` (env-driven config, `save_tensor` helper that
  re-flushes a manifest after every write).
- Default model was `meta-llama/Llama-3.2-3B`; switched to **`meta-llama/Meta-Llama-3.1-8B`**
  on **NDIF remote execution** because NDIF doesn't host the 3.2 family.

### Architecture (Llama-3.1-8B)

L=32 layers, H_q=32 query heads, H_kv=8 KV heads (GQA 4:1), d=4096, d_ff=14336,
SwiGLU MLP confirmed.

### What worked

When each `.save()` runs in its own trace block, **all 17 candidate captures
succeed remotely on NDIF**, including every quantity the math doc names:

- residual stream pre- and post- each sublayer
  (`input_layernorm.input`, `post_attention_layernorm.input`, `layer.output`)
- SwiGLU post-SiLU gate (`mlp.act_fn.output`) — the frozen-gate `g^(ℓ)_p`
- Q, K, V, O projections (`self_attn.{q,k,v,o}_proj.output`)
- attention sublayer output (`self_attn.output` — same as `o_proj.output`)
- MLP sublayer output (`mlp.output`)
- embedding output and final-norm input

All shapes match expectations from the Llama-3.1-8B config.

### What didn't work — and the actual root cause

The first attempt (`02_capture.py` registering ~5 saves per layer × 32 layers
in one big trace) failed with `MissedProviderError`. The error message said
"this module was not called", which sent me down a wrong-track hypothesis
that transformers 5.x bypasses the `self_attn` Envoy.

**The real cause** is different: nnsight 0.7's remote backend doesn't handle
many-saves-in-one-trace cleanly. The first save that misses kills the
interleaving session, and *every subsequent save* in the same trace then
raises `ValueError: ... outside of interleaving`. The cascade made it look
like submodule captures were broken — they weren't. Isolated traces prove
every individual target works.

### Lesson

> **When a tool returns an error message about state ("module was not called",
> "outside of interleaving"), the error message often describes a *downstream
> symptom* of an earlier failure that the tool has lost track of. Reproduce the
> failure in the smallest possible isolation before believing the message at
> face value.** I spent an hour optimising for the wrong hypothesis (transformers
> internals refactor) when a 17-trace smoke probe would have settled it in two
> minutes.

### Open Phase-2-relevant items

1. ~~**Attention weights `A^(ℓ,h)_{p,s}`** are not directly captured anywhere.~~
   **CORRECTED below — see "2026-05-13 — Correction".** A *is* directly captured
   in `self_attn.output[1]`; the earlier claim was an artifact of my own
   post-processing, not an NDIF or transformers behavior.

2. **Batched trace strategy.** Per-target-isolated traces (~17 round trips
   here, more if per-layer × per-target) is correct but slow. Phase 2 needs
   to pick the batching grain — most likely **one trace per layer** bundling
   that layer's ~6 captures, totalling 32 traces × ~0.5s each = ~15s wall
   clock per decomposition.

### Other small finds

- NDIF still needs the **HuggingFace gate** accepted for gated models (it
  fetches `config.json` from HF locally to build the Envoy hierarchy, even
  though weights/compute live on NDIF servers). HF auth is required
  separately, via `HF_TOKEN` (or `HUGGING_FACE_TOKEN`, which we alias).
- Windows console blew up with `UnicodeEncodeError` on nnsight's status
  spinner (`◉`); fix is `PYTHONIOENCODING=utf-8 PYTHONUTF8=1` in the env.
- `transformers 5.8` deprecates `torch_dtype=` in favor of `dtype=`. Not
  blocking; clean up when we write `real/backend/src/info_flow/model.py`
  in Phase 2.

### Tiny credential-leak postmortem

I `xxd`'d `real/backend/.env.local` to verify encoding and **inadvertently
echoed the NDIF API key into the conversation transcript**. The supervisor
chose not to rotate. Going forward, credential files are inspected only
through commands that don't read their *contents* (`ls -la`, `wc -c`,
`stat`). Lesson logged here so it's not lost.

---

## 2026-05-13 — Correction: A *is* directly captured (my own bug)

When the supervisor asked me how certain I was that the attention-weights gap
was an NDIF behavior, I went to verify and found I had been wrong on three
levels at once. Two follow-up probes (`00b_output_attentions.py` and
`00c_attn_sanity.py`) settle the picture:

- `self_attn.output` on NDIF is **always a 2-tuple** of shape
  `( Tensor(B, S, d_model), Tensor(B, H_q, S, S) )`. The second element is
  available regardless of whether `output_attentions=True` is passed to
  `model.trace(...)`.
- Sanity checks confirm element `[1]` is the post-softmax causal-masked
  attention probability matrix `A^(ℓ,h)_{p,s}`: values in `[0, 1]`, row sums
  ≈ 1 within bf16 noise (±2e-3), strict zero above the diagonal, nonzero
  counts per row exactly `1, 2, 3, 4` for the 4-token prompt.

So `A^(ℓ,h)` requires **no reconstruction at all** — it's just
`self_attn.output[1]`. The whole earlier discussion of options (a) client-side
RoPE+QK^T, (b) deeper hook, (c) inside-the-trace computation is moot.

### Why I had it wrong — three errors stacked

1. **Claim I never tested.** I wrote "NDIF silently drops `output_attentions=True`"
   in the Phase-1 writeup, but I had not actually run any trace with that flag
   set on NDIF (the only run that had the flag had failed earlier with the
   cascade `MissedProviderError`, so I had no observation of what came back).
2. **Display masked the data I had.** In the original smoke probe
   (`00_smoke.py`), I post-processed every capture with `unwrap_first(...)`,
   a helper I'd written for `.input` proxies (which legitimately come back as
   tuples of positional args). I had lazily applied it to `.output` proxies
   too. For `self_attn.output`, the captured value was already
   `(attn_out, attn_weights)` — `unwrap_first` silently discarded
   `[1]`, the attention weights tensor. The data was there all along.
3. **Built the next layer of reasoning on the wrong floor.** From the
   "Tensor(1,4,4096)" the display gave me, I confidently wrote a whole Phase-1
   caveat with two options and even a recommended (c) option to compute A
   inside the trace. Every word of that was unnecessary.

### Lesson (corollary to the earlier one)

> **A display function is part of the experiment, not a separate thing.** If
> the way I report a tensor mutates the tensor (here, `unwrap_first` discards
> a tuple slot), I will draw conclusions about what's "in" my data that aren't
> actually about my data — they're about what survived my display. When in
> doubt, print `type(val)` and `repr(val)` raw, not a "nice" string.

Earlier lesson (Phase-1 writeup) was *don't believe error messages without
isolating*; this lesson is its sibling — *don't believe success summaries
without seeing the raw data either*. Both are forms of trusting a layer of
processing I introduced myself.

### Updated picture for Phase 2

Everything the math doc needs is captured by **two `.save()` calls per layer**
plus two one-offs:

| Source | What it gives |
|---|---|
| `embed_tokens.output`                            | source embeddings `e_p` |
| `norm.input`                                     | final residual (sanity) |
| `layers[ℓ].input_layernorm.input`                | `x^(ℓ)_p` |
| `layers[ℓ].post_attention_layernorm.input`       | post-attn residual |
| `layers[ℓ].self_attn.output`                     | `[0]` = attn sublayer out, `[1]` = `A^(ℓ,h)` |
| `layers[ℓ].mlp.act_fn.output`                    | post-SiLU gate `g^(ℓ)_p` |

RMSNorm scales `r^(ℓ,k)_p` are computed from the residual stream at trace
time (`1 / sqrt(mean(x^2) + eps)`), no separate capture needed.

The only Phase-2 design choice left from the original two open items is the
**batching strategy** (how many `.save()` calls per `model.trace()`). The
attention-weights question is closed.

---

## 2026-06-18 — SAE feature lens for per-source contributions

Built the SAE analog of the logit-lens cell: decode each per-source contribution
into its top-5 **SAE features** instead of top-5 logits. New files
`project_scratchpad/sae_feature_lens.py` (helper) + `sae_feature_lens.ipynb`;
added `sae-lens` to `real/backend/pyproject.toml`; checks in
`real/backend/src/tests/scratchpad/check_sae_lens.py`. Reuses the cached
`FullRunResults` — no remote run.

**Verified (ran `check_sae_lens.py`), not guessed:**
- SAE = Llama Scope 8x/32K residual: `release="llama_scope_lxr_8x"`,
  `sae_id=f"l{L}r_8x"`, HF `fnlp/Llama3_1-8B-Base-LXR-8x`. Loaded layer 31:
  `d_in=4096, d_sae=32768`. TopK k=50. These are exactly the SAEs Neuronpedia
  indexes as `llama3.1-8b/{L}-llamascope-res-32k`, so indices align for lookup.
- Neuronpedia public GET `…/api/feature/{model}/{L}-llamascope-res-32k/{idx}`
  returns an `explanations` list; `explanations[0]["description"]` is the text
  (e.g. feature 15278 → *"quantitative data and numerical references in
  financial contexts"*). Field path confirmed against a live response.

**Key finding — why magnitude normalization is mandatory.** The Llama Scope
encoder has a bias threshold ($f(x)=\mathrm{ReLU}(W^{enc}x+b^{enc})$, paper
Eqs. 1/5). A per-source contribution is only a fraction of a real residual
(`hate` contribution norm **11.2** vs full-residual norm **87.7** at layer 31).
Fed raw, it collapses: only **one** feature fires (act 2.27) and the rest hit the
bias floor `0.0`. Rescaling its *direction* to the real residual norm
(`normalize_to=resid_norm`) recovers a full, content-bearing top-5
(acts 17.3, 13.4, 11.6, 10.6, 9.0). This is the SAE analog of the RMSNorm the
logit lens applies before unembedding.

**Lesson / pitfall avoided:** the supervisor's idea of dropping the encoder bias
to get scale-invariance would have changed the quantity from "active SAE
features" to a linear alignment score (and Llama Scope *does* have `b_enc`, so the
"TopK has no bias" premise was false). Kept the real encoder + normalized the
magnitude instead.

**Gotcha:** `sae.encode` output keeps `requires_grad`; wrap in `torch.no_grad()`
(or `.detach()`) before `float()` to avoid the scalar-conversion warning.
