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

1. **Attention weights `A^(ℓ,h)_{p,s}`** are not directly captured anywhere.
   `self_attn.output` returns a single tensor on NDIF (i.e., `output_attentions=True`
   is silently dropped). Two paths for Phase 2:
   - reconstruct `A = softmax(QK^T/√d_h + causal_mask)` client-side from
     captured `q_proj.output` + `k_proj.output`, **after applying RoPE
     ourselves** (q_proj is pre-rotary in HF Llama).
   - or probe a deeper hook (e.g., `self_attn.attention_dropout.input` if
     dropout is a module on NDIF's build) that exposes `A` directly.

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
