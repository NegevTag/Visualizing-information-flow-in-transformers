# Why ⊙ and not simple multiplication?

There are two uses of $\odot$ in `decomposition_math.tex`. Both cases share the same reason:
one operand is a **vector of per-feature scales**, the other is a **vector of activations**,
and you need to scale each feature independently — not collapse dimensions.

---

## Line 52 — RMSNorm normalization

$$\hat{v}^{(\ell,k)}_p := \gamma^{(\ell,k)} \odot \bigl(r^{(\ell,k)}_p \cdot v\bigr)$$

- $r^{(\ell,k)}_p \in \mathbb{R}_{>0}$ is a **scalar** (the inverse-RMS scale for that position/layer). So $r \cdot v$ is fine as plain scalar multiplication.
- $\gamma^{(\ell,k)} \in \mathbb{R}^d$ is the **learned gain vector** — one scale *per hidden dimension*. Dimension $i$ of $v$ gets multiplied by $\gamma_i$, independently of all other dimensions.

If you wrote $\gamma \cdot v$ with juxtaposition, it would look like a dot product (scalar) or matrix-vector product — neither of which is the right operation. $\odot$ makes it explicit: this is a **coordinate-wise** product, no summation.

---

## Line 71 — MLP SiLU gating

$$c^{(\ell+1)}_{p,s} = c^{\mathrm{mid},(\ell)}_{p,s} + W^{(\ell)}_{\mathrm{down}}\,\Bigl(g^{(\ell)}_p \odot W^{(\ell)}_{\mathrm{up}}\,\hat{c}^{\mathrm{mid},(\ell,2)}_{p,s}\Bigr)$$

- $W^{(\ell)}_{\mathrm{up}}\,\hat{c} \in \mathbb{R}^{d_{\mathrm{ff}}}$ — the up-projected contribution vector in the intermediate dimension.
- $g^{(\ell)}_p \in \mathbb{R}^{d_{\mathrm{ff}}}$ — the frozen SiLU gate, also a vector in the intermediate dimension.

This is the **SwiGLU/SiLU gating mechanism**: each intermediate feature is scaled by its own gate value before being projected back down by $W_{\mathrm{down}}$. Again, both operands are vectors of the same shape — $\odot$ signals element-wise, no summation.

---

## Summary

| Situation | Operator | Why |
|---|---|---|
| scalar × vector | $\cdot$ or juxtaposition | broadcasts naturally, no ambiguity |
| vector × vector, per-feature | $\odot$ | element-wise; plain juxtaposition would imply dot/matrix product |

In PyTorch this is just `*` on same-shape tensors, or `torch.mul`.
