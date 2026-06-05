# RMSNorm

RMSNorm uses the **RMS (root mean square)**, not the L2 norm.

## L2 norm vs RMS

**L2 norm** — a single scalar for the whole vector:
$$\|x\|_2 = \sqrt{\sum_i x_i^2}$$

**RMS** — same but divided by dimension $d$:
$$\text{RMS}(x) = \sqrt{\frac{1}{d}\sum_i x_i^2}$$

## RMSNorm formula

Each token vector $x \in \mathbb{R}^{d_{\text{model}}}$ is normalized by its own RMS, then scaled by a learned gain $w$:

$$\hat{x} = \frac{x}{\text{RMS}(x)} \cdot w$$

where $w \in \mathbb{R}^{d_{\text{model}}}$ is the learned weight (`layer.input_layernorm.weight`).

## vs LayerNorm

| | LayerNorm | RMSNorm |
|---|---|---|
| Mean subtraction | yes | **no** |
| Bias term | yes | **no** |
| Normalization | std | RMS |

RMSNorm is cheaper and works just as well in practice — that's why Llama uses it.
