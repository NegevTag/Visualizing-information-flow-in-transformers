<!-- # The shape of `cos` and `sin` in Llama's RoPE -->

All references are to the installed source:
`real/backend/.venv/Lib/site-packages/transformers/models/llama/modeling_llama.py`.
Read off the file (line numbers noted), not from memory.

## Notation

- $B$ — batch size
- $S$ — current sequence length (number of token positions in *this* forward pass)
- $d_{\text{model}}$ — `hidden_size` (the residual-stream width)
- $n_{\text{heads}}$ — `num_attention_heads`
- $d_{\text{head}}$ — `head_dim` $= d_{\text{model}} / n_{\text{heads}}$

## The answer

$$
\boxed{\;\cos,\ \sin \;:\; \big[\,B,\ S,\ d_{\text{head}}\,\big]\;}
$$

If you ignore batch (one sequence), it's $(S,\ d_{\text{head}})$.

Two easy-to-get-wrong points:

1. **The last dim is $d_{\text{head}}$, *not* $d_{\text{model}}$.**
2. **$S$ is the current sequence length, *not* the max context window** (`max_position_embeddings`).

## Why the last dim is `head_dim`, not `hidden_size`

RoPE is applied **per attention head**. Inside attention, the query/key tensors have shape

$$
q,\ k \;:\; \big[\,B,\ n_{\text{heads}},\ S,\ d_{\text{head}}\,\big].
$$

In `apply_rotary_pos_emb` (lines 164–165) `cos` and `sin` are unsqueezed to

$$
\big[\,B,\ 1,\ S,\ d_{\text{head}}\,\big]
$$

and then **broadcast across the head dimension** (the `1`). The exact same rotation angles are reused for every head, so `cos`/`sin` only ever need to carry $d_{\text{head}}$ entries per position — never the full $d_{\text{model}}$.

Concretely, the rotation acts on a single head's vector ($d_{\text{head}}$ values), and there are $n_{\text{heads}}$ such heads sharing one `(cos, sin)` table. So:

$$
d_{\text{head}} \cdot n_{\text{heads}} = d_{\text{model}},
$$

but the RoPE table is sized to **one head** ($d_{\text{head}}$), not the concatenation of all heads.

For the toy config: $d_{\text{model}} = 32$, $n_{\text{heads}} = 4 \Rightarrow d_{\text{head}} = 8$. So `cos`/`sin` are $[B, S, 8]$, not $[B, S, 32]$.

## Why $S$ is the current length, not the max context

`cos`/`sin` are computed fresh **each forward pass** from the actual `position_ids` passed in
(`LlamaRotaryEmbedding.forward`, lines 124–135). The angle table is

$$
\text{freqs}[m, j] = m\,\theta_j, \qquad m \in \text{position\_ids},
$$

so its position axis is exactly as long as the input ($S$). `max_position_embeddings` only bounds how large $m$ may legitimately get; it does **not** set the size of `cos`/`sin`. A 5-token input produces $S = 5$ rows regardless of the max context.

## Summary table

| tensor | shape |
|---|---|
| `position_ids` | $[B,\ S]$ |
| `cos`, `sin` (returned by `rotary_emb`) | $[B,\ S,\ d_{\text{head}}]$ |
| `cos`, `sin` (after unsqueeze in `apply_rotary_pos_emb`) | $[B,\ 1,\ S,\ d_{\text{head}}]$ |
| `q`, `k` (rotated) | $[B,\ n_{\text{heads}},\ S,\ d_{\text{head}}]$ |
