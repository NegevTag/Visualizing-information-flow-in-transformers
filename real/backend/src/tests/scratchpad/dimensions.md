# Shapes & Dimensions Reference

## Parameters

| Symbol | Meaning | HF config key |
|---|---|---|
| $L$ | number of layers | `num_hidden_layers` |
| $H_q$ | number of query heads ($H_q \geq H_{kv}$) | `num_attention_heads` |
| $H_{kv}$ | number of key/value heads | `num_key_value_heads` |
| $G$ | GQA group size $= H / H_{kv}$ | — |
| $d$ | residual stream dimension | `hidden_size` |
| $d_v$ | hidden dimension after $W_V$ | `head_dim` |
| $d_{kv}$ | total kv dimension $= H_{kv} \cdot d_v$ | — |
| $S$ | sequence length (prompt\_len) | — |

---

## Weight matrices
?
| Tensor | Shape | Meaning |
|---|---|---|
| `q_proj.weight` | $(d,\ d)$ | all query heads stacked |
| `k_proj.weight` | $(d_{kv},\ d)$ | all key heads stacked |
| `v_proj.weight` | $(d_{kv},\ d)$ | all value heads stacked |
| `o_proj.weight` | $(d,\ d)$ | projects concat of all heads back to residual stream |
| `input_layernorm.weight` $\gamma_1$ | $(d,)$ | pre-attention RMSNorm scale |
| `post_attention_layernorm.weight` $\gamma_2$ | $(d,)$ | pre-MLP RMSNorm scale |

### Per-head slices (head $h$, kv-head $k = \lfloor h/G \rfloor$)

| Tensor | Shape | How to slice |
|---|---|---|
| $W_Q^h$ | $(d_v,\ d)$ | `W_Q[h*d_v:(h+1)*d_v, :]` |
| $W_K^k$ | $(d_v,\ d)$ | `W_K[k*d_v:(k+1)*d_v, :]` |
| $W_V^k$ | $(d_v,\ d)$ | `W_V[k*d_v:(k+1)*d_v, :]` |
| $W_O^h$ | $(d,\ d_v)$ | `W_O[:, h*d_v:(h+1)*d_v]` — **columns**, because input to o_proj is concat of heads |
| $OV^h = W_O^h W_V^k$ | $(d,\ d)$ | rank $\leq d_v$ |

---

## Captured activations

| nnsight capture | Shape | Dim meaning |
|---|---|---|
| `embed_tokens.output` | $(B,\ S,\ d)$ | batch, position, residual |
| `input_layernorm.input` | $(B,\ S,\ d)$ | residual entering layer $\ell$ |
| `post_attention_layernorm.input` | $(B,\ S,\ d)$ | residual after attention, before MLP |
| `self_attn.output[0]` | $(B,\ S,\ d)$ | attention sublayer output (after $W_O$); $B$ = batch, $S$ = position, $d$ = residual dim; added to residual |
| `self_attn.output[1]` | $(B,\ H,\ S,\ S)$ | attention weights $A^{\ell,h}_{p,s}$ — dim $-2$ is **query** $p$, dim $-1$ is **key** $s$ |
| `mlp.act_fn.output` | $(B,\ S,\ d_{ff})$ | post-SiLU gate $g^{(\ell)}$ |
| `layer.output` | $(B,\ S,\ d)$ | residual after full layer $\ell$ |
