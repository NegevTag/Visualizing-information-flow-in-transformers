# What `cos` and `sin` are in Llama's RoPE, exactly

All references are to the installed source:
`real/backend/.venv/Lib/site-packages/transformers/models/llama/modeling_llama.py`.
Everything below is read off that file (lines noted), not from memory.

## Notation

- $B$ — batch size
- $S$ — sequence length
- $d$ — `head_dim` ($=$ `hidden_size / num_attention_heads`)
- $m, n$ — absolute token positions (integers, $0,1,2,\dots$)
- $\text{base}$ — `rope_theta` (default $10000$)

## 1. The frequencies $\theta_j$ (the `inv_freq` buffer)

The default RoPE init (line 117) builds `inv_freq`, a vector of $d/2$ frequencies:

$$
\theta_j \;=\; \text{base}^{-\,2j/d}, \qquad j = 0, 1, \dots, \tfrac{d}{2}-1 .
$$

So $\theta_0 = 1$ (fastest rotation) down to $\theta_{d/2-1} = \text{base}^{-(d-2)/d}$ (slowest). These depend only on the dimension index $j$, never on position.

## 2. How `cos` and `sin` are built (`LlamaRotaryEmbedding.forward`, lines 124–135)

The forward multiplies each frequency by each position to get an **angle per (position, frequency)**:

$$
\text{freqs}[m, j] \;=\; m \cdot \theta_j , \qquad \text{shape } [B, S, d/2].
$$

(In code: `inv_freq_expanded @ position_ids_expanded` then transpose — an outer product of $\theta$ and the positions.)

Then it **duplicates** the $d/2$ angles into a length-$d$ vector (line 131, `emb = cat(freqs, freqs)`):

$$
\text{emb}[m] \;=\; \big(\, m\theta_0,\; \dots,\; m\theta_{d/2-1},\;\; m\theta_0,\; \dots,\; m\theta_{d/2-1} \,\big).
$$

Finally (lines 132–133):

$$
\cos[m] = \cos(\text{emb}[m]) \cdot s, \qquad
\sin[m] = \sin(\text{emb}[m]) \cdot s,
$$

where $s =$ `attention_scaling` (a scalar; $s = 1$ for default RoPE, $\neq 1$ only for scaled variants like YaRN).

**So `cos` and `sin` are each of shape $[B, S, d]$**, and entry $k$ at position $m$ is:

$$
\cos[m]_k = s\cos\!\big(m\,\theta_{(k \bmod d/2)}\big), \qquad
\sin[m]_k = s\sin\!\big(m\,\theta_{(k \bmod d/2)}\big).
$$

The "$k \bmod d/2$" is exactly the `cat(freqs, freqs)` duplication: index $k$ and index $k + d/2$ share the same angle.

## 3. How they are applied (`apply_rotary_pos_emb` + `rotate_half`, lines 138–168)

For a query vector $q \in \mathbb{R}^d$ at position $m$:

$$
q' \;=\; q \odot \cos[m] \;+\; \text{rotate\_half}(q) \odot \sin[m],
$$

where $\odot$ is elementwise product and (lines 138–142):

$$
\text{rotate\_half}(q) \;=\; \big(\,-q_{d/2},\, \dots,\, -q_{d-1},\;\; q_0,\, \dots,\, q_{d/2-1}\,\big).
$$

(`cos`/`sin` are first unsqueezed to $[B,1,S,d]$ at lines 164–165 to broadcast over the head dimension; the math per element is unchanged.)

### What this does, component by component

Write the angle $\phi_j = m\theta_j$ and take $s=1$. Pair up dimension $j$ with dimension $j+d/2$. Working through the elementwise formula:

- For $k = j$ (first half): $\;q'_j = q_j\cos\phi_j - q_{j+d/2}\sin\phi_j$
- For $k = j + d/2$ (second half): $\;q'_{j+d/2} = q_{j+d/2}\cos\phi_j + q_j\sin\phi_j$

In matrix form, each pair $(q_j,\, q_{j+d/2})$ is **rotated by angle $\phi_j = m\theta_j$**:

$$
\begin{pmatrix} q'_j \\ q'_{j+d/2} \end{pmatrix}
=
\begin{pmatrix} \cos\phi_j & -\sin\phi_j \\ \sin\phi_j & \cos\phi_j \end{pmatrix}
\begin{pmatrix} q_j \\ q_{j+d/2} \end{pmatrix}.
$$

So RoPE splits the head into $d/2$ independent 2-D planes and rotates plane $j$ by an angle **proportional to the absolute position $m$**, with plane-specific speed $\theta_j$. Keys are rotated identically with their own position $n$.

> Note: HF pairs $j$ with $j+d/2$ (the two halves), **not** adjacent dims $(2j, 2j+1)$ as in the original RoPE paper. Same idea, different ordering — a consequence of `rotate_half` + `cat(freqs, freqs)`.

## 4. Why rotate, not add: the relative-position property

The point of rotating both $q$ (at $m$) and $k$ (at $n$) is what happens in the attention dot product. For one plane $j$, let $R(\phi)$ be the $2\times2$ rotation. The contribution to $q_m \cdot k_n$ from that plane is

$$
\big(R(m\theta_j)\,q_j\big)^{\!\top} \big(R(n\theta_j)\,k_j\big)
= q_j^{\top} R(m\theta_j)^{\top} R(n\theta_j)\, k_j
= q_j^{\top} R\big((n-m)\theta_j\big)\, k_j ,
$$

using $R(a)^\top R(b) = R(b-a)$. The absolute positions cancel; only the **relative offset $n-m$** survives. That is the whole reason RoPE encodes position multiplicatively inside the QK dot product rather than as an additive term on the residual stream.

## 5. The identity (disabled-RoPE) case

Setting every angle to $0$ — i.e. $\theta_j = 0$ (zeroing `inv_freq`) or directly forcing the outputs — gives

$$
\cos[m] = s\cdot\mathbf{1}, \qquad \sin[m] = \mathbf{0},
$$

so $q' = q \odot (s\mathbf 1) + \text{rotate\_half}(q)\odot \mathbf 0 = s\,q$. With $s=1$ (default) this is the **identity**: `apply_rotary_pos_emb` returns $q, k$ unchanged, and attention scores no longer depend on position (ordering is then enforced only by the causal mask).
