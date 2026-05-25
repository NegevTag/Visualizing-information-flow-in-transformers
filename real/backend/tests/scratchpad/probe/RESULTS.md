# Phase 1 probe — full results

**Decision gate: PASS.** Every quantity the math doc names is reachable on NDIF
via nnsight, with a small set of `.save()` calls per layer.

This file consolidates what the four probe scripts established. Long-form
narrative (with mistakes I made along the way and the lessons) lives in the
top-level `RECORD.md` under the 2026-05-13 entries.

---

## Environment

| Component | Value |
|---|---|
| Model | `meta-llama/Meta-Llama-3.1-8B` |
| Execution | **remote on NDIF** (`?INFO_FLOW_REMOTE=1`, `model.trace(..., remote=True)`) |
| Prompt | `"The cat sat"` (4 tokens after Llama-3 tokenization) |
| nnsight | 0.7.0 |
| transformers | 5.8.1 |
| torch | 2.12.0+cpu |
| Local device | CPU (no GPU); irrelevant for remote runs |

## Architecture (read from `model.config` at load time)

| Constant | Value |
|---|---|
| L (num_hidden_layers) | 32 |
| H_q (num_attention_heads) | 32 |
| H_kv (num_key_value_heads) | 8  *(GQA 4:1)* |
| d_model | 4096 |
| d_ff (intermediate_size) | 14336 |
| MLP form | SwiGLU (`gate_proj`, `up_proj`, `down_proj`, `act_fn = SiLU`) |

## Per-script summary

| Script | Purpose | Result |
|---|---|---|
| `01_load_check.py` | Load model only, dump config. | ✅ Load OK in ~3–37s, architecture matches expectations. |
| `00_smoke.py` | Try 17 candidate captures, one save per isolated trace. | ✅ **17/17 captures succeed.** |
| `00b_output_attentions.py` | Compare `self_attn.output` with/without `output_attentions=True`. | ✅ `self_attn.output` is *always* a 2-tuple `(attn_out, attn_weights)` regardless of the flag. |
| `00c_attn_sanity.py` | Verify `self_attn.output[1]` is the post-softmax causal `A^(ℓ,h)`. | ✅ All four sanity checks pass within bf16 noise. |
| `02_capture.py` *(superseded)* | Initial multi-save trace per layer; failed with cascading `MissedProviderError` / "outside of interleaving". | ❌ Don't bundle many `.save()` calls in a single trace until Phase 2's batching strategy is settled. |

## Captures confirmed available (one-save-per-trace, NDIF)

| Capture | Shape (B=1, S=4) | Role in math doc |
|---|---|---|
| `embed_tokens.output` | `(1, 4, 4096)` | source embeddings `e_p` |
| `model.norm.input` | `(1, 4, 4096)` | final residual (last-layer output, sanity) |
| `layers[ℓ].input_layernorm.input` | `(1, 4, 4096)` | residual entering layer ℓ:  `x^(ℓ)_p` |
| `layers[ℓ].post_attention_layernorm.input` | `(1, 4, 4096)` | post-attn / pre-MLP residual |
| `layers[ℓ].output` | `(1, 4, 4096)` | residual after the full layer ℓ |
| `layers[ℓ].self_attn.output[0]` | `(1, 4, 4096)` | attention sublayer output |
| `layers[ℓ].self_attn.output[1]` | `(1, 32, 4, 4)` | **`A^(ℓ,h)_{p,s}` (post-softmax causal-masked)** |
| `layers[ℓ].self_attn.{q,k,v,o}_proj.output` | Q/O `(1, 4, 4096)`, K/V `(1, 4, 1024)` (GQA) | individual projections, useful for sanity/per-head |
| `layers[ℓ].mlp.act_fn.output` | `(1, 4, 14336)` | **frozen gate `g^(ℓ)_p`** = `SiLU(W_gate · x)` |
| `layers[ℓ].mlp.{gate,up,down}_proj.output` | 14336- or 4096-dim | individual MLP projections (sanity / debug) |
| `layers[ℓ].mlp.output` | `(1, 4, 4096)` | MLP sublayer output |

**Derived (no separate capture needed):**

| Quantity | How |
|---|---|
| `r^(ℓ,1)_p` (pre-attn RMSNorm scale) | `1 / sqrt(mean(input_layernorm.input² , dim=-1) + ε)` |
| `r^(ℓ,2)_p` (pre-MLP RMSNorm scale) | same with `post_attention_layernorm.input` |

## Attention-weights sanity (from `00c_attn_sanity.py`, layer 0)

Tensor: `self_attn.output[1]` for `prompt="The cat sat"` → `S = 4`.

| Check | Pass | Notes |
|---|---|---|
| Shape = `(B, H_q, S, S) = (1, 32, 4, 4)` | ✅ | matches `A^(ℓ,h)_{p,s}` from the math doc |
| All entries in `[0, 1]` | ✅ | min `0.0`, max `1.0` |
| Strict upper-triangle = 0 (causal mask) | ✅ | `max |A[p, s>p]| = 0.000e+00` |
| Row sums ≈ 1 | ✅ | `[0.998, 1.002]` — bf16 noise; cleanly within ±2.2e-3 |
| Nonzero count per row | ✅ | exactly `1, 2, 3, 4` for rows 0–3 |

## What this means for Phase 2

The minimum set of `.save()` calls per layer for the decomposition is **four**:

```python
with model.trace(prompt, remote=True):
    embed = model.model.embed_tokens.output.save()                     # once
    for ℓ in range(L):
        x          = layers[ℓ].input_layernorm.input.save()            # x^(ℓ)
        y          = layers[ℓ].post_attention_layernorm.input.save()   # post-attn
        attn_pair  = layers[ℓ].self_attn.output.save()                 # (out, A)
        g_star     = layers[ℓ].mlp.act_fn.output.save()                # post-SiLU gate
```

with RMSNorm scales computed *afterwards* from the residual tensors. `weights`
(W_V, W_O, γ1, γ2, W_up, W_down) are read from `model.named_parameters()` (no
trace needed).

The exact **batching** of these calls across one or more `model.trace(...)`
blocks is Phase 2's only remaining open question — the all-at-once strategy
in the original `02_capture.py` triggered a cascade, but one-per-trace works
unconditionally. Reasonable choices to evaluate in Phase 2:

- **One trace per layer** (32 traces; each holds 4 saves). Reasonable middle ground.
- **One trace with all L×4 saves** (the original approach; needs to be made to work, or proven impossible).
- **One save per trace** (~ L×4 = 128 traces). Always works, slowest.

## What's *not* in this probe

- Multi-batch prompts (we only ran B=1).
- Long contexts (S=4; we have not tested RoPE behavior at e.g. S=2k).
- Different prompts to check whether NDIF caches anything between calls.

These can be revisited if/when Phase 2 finds a surprise. They aren't blockers.
