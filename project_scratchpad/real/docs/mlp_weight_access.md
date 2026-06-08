# MLP weight & tensor access (Llama via nnsight)

Reference extracted from [ex2_calc_mlp_as_well.py](../backend/src/info_flow/ex2_calc_mlp_as_well.py).
All accesses assume `layer = model.model.layers[l]` inside a `with model.trace(prompt, remote=...):` block.

## Config / shape constants

| Name | Description | Access |
|---|---|---|
| `LAYERS_NUM` | number of transformer layers | `len(model.model.layers)` |
| `D_MODEL` | residual stream width ($d_{model}$) | `model.model.config.hidden_size` |
| `rms_eps` | epsilon for RMSNorm | `model.config.rms_norm_eps` |

## RMSNorm (pre-MLP, a.k.a. `post_attention_layernorm`)

Applied to the residual stream *before* the MLP. Llama RMSNorm:
$$y = w \odot \frac{x}{\sqrt{\operatorname{mean}(x^2) + \epsilon}}$$

| Name | Description | Access |
|---|---|---|
| `rms_weight` | learned scale $w$, shape `(d_model,)` | `layer.post_attention_layernorm.weight` |
| pre-RMS input | residual entering the norm, `(prompt_len, d_model)` | `layer.post_attention_layernorm.input[0]` |
| post-RMS output | normalized residual fed to MLP | `layer.post_attention_layernorm.output[0]` |

## MLP (SwiGLU)

Llama MLP: $\text{down}\big(\operatorname{SiLU}(\text{gate}(x)) \odot \text{up}(x)\big)$.
Let $d_{mlp}$ = MLP intermediate dim.

| Name | Description | Shape | Access |
|---|---|---|---|
| `W_gate` | gate projection (input side of SwiGLU gate) | `(d_mlp, d_model)` | `layer.mlp.gate_proj.weight` |
| `W_up` | up projection (input side multiplied with gate) | `(d_mlp, d_model)` | `layer.mlp.up_proj.weight` |
| `W_down` | down projection (back to residual) | `(d_model, d_mlp)` | `layer.mlp.down_proj.weight` |
| SiLU | nonlinearity on the gate branch | — | `torch.nn.functional.silu(...)` |

### Usage pattern (from the file)

```python
# x: (prompt_len, d_model) — post-RMSNorm residual
gate = torch.nn.functional.silu(layer.mlp.gate_proj.weight @ x.T)   # (d_mlp, prompt_len)
up   = x @ layer.mlp.up_proj.weight.T                                # (prompt_len, d_mlp)
mlp_out = (up * gate.T) @ layer.mlp.down_proj.weight.T               # (prompt_len, d_model)
```

For linearity-preserving decomposition, `up_proj` and `down_proj` are applied per-(query, key) contribution, while the SiLU(gate) factor is computed from the summed residual and treated as a frozen per-position scalar field — see [ex2_calc_mlp_as_well.py:104-113](../backend/src/info_flow/ex2_calc_mlp_as_well.py#L104-L113).
