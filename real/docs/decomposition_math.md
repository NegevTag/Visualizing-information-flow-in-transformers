# Method: Linearized Per-Source Decomposition of the Residual Stream

## 3.1 Setup and Notation

Let $\mathcal{M}$ be a decoder-only transformer with $L$ layers, $H$ attention heads, model dimension $d$, feed-forward dimension $d_{\text{ff}}$, and head dimension $d_h = d / H$. Given an input sequence of $N$ tokens, let $e_p \in \mathbb{R}^d$ denote the embedding of the token at position $p \in \{0, \dots, N-1\}$.

We write the residual stream at layer $\ell$ and position $p$ as $x^{(\ell)}_p \in \mathbb{R}^d$, with $x^{(0)}_p = e_p$.

Our goal is to decompose every $x^{(\ell)}_p$ into an **exact sum of per-source contributions**:

$$x^{(\ell)}_p = \sum_{s=0}^{N-1} c^{(\ell)}_{p,s}, \qquad c^{(\ell)}_{p,s} \in \mathbb{R}^d$$

where $c^{(\ell)}_{p,s}$ is interpreted as the contribution of source token $s$ to the residual stream at target position $p$ after layer $\ell$. The decomposition is made exact — not approximate — by **linearizing the model at the actual forward-pass activation point** $x^* = (x^{(\ell)}_p)_{\ell,p}$.

The scalar quantity displayed in the visualization is $\|c^{(\ell)}_{p,s}\|_q$ for a user-selected $q \in \{0, 1, 2, \infty\}$.

---

## 3.2 Frozen-QK Attention Linearization

Each transformer block applies multi-head attention followed by a position-wise MLP, with RMSNorm preceding each sub-block.

**Attention pattern.** In the real forward pass, each head $h$ computes the attention pattern

$$A^{(\ell,h)}_{p,s} = \text{softmax}\!\left(\frac{Q^{(\ell,h)}_p \cdot K^{(\ell,h)^{\top}}_s}{\sqrt{d_h}}\right) \in \mathbb{R}^{N \times N}$$

We freeze $A^{(\ell,h)}$ at the values produced by the real forward pass and treat them as fixed scalars henceforth.

**Attention output.** With $W^{(\ell,h)}_V \in \mathbb{R}^{d_h \times d}$ and $W^{(\ell,h)}_O \in \mathbb{R}^{d \times d_h}$, the attention output at position $p$ for head $h$ is

$$\text{Attn}^{(\ell,h)}_p = W^{(\ell,h)}_O \sum_{s=0}^{p} A^{(\ell,h)}_{p,s} \cdot W^{(\ell,h)}_V \hat{x}^{(\ell)}_s$$

where $\hat{x}^{(\ell)}_s$ is the RMSNorm-preprocessed residual (detailed below). Since $A^{(\ell,h)}_{p,s}$ is frozen, this expression is **linear** in $\hat{x}^{(\ell)}_s$.

**Source-contribution propagation through attention.** Defining $W^{(\ell,h)}_{\text{OV}} := W^{(\ell,h)}_O W^{(\ell,h)}_V$, the post-attention per-source contributions accumulate as

$$c^{\text{mid},(\ell)}_{p,s} = c^{(\ell)}_{p,s} + \sum_{h=1}^{H} \sum_{s'=0}^{p} A^{(\ell,h)}_{p,s'} \cdot W^{(\ell,h)}_{\text{OV}} \hat{c}^{(\ell)}_{s',s}$$

where the hat denotes passage through the frozen RMSNorm (Section 3.3). The causal mask enforces $A^{(\ell,h)}_{p,s} = 0$ for $s > p$, so $c^{(\ell)}_{p,s} = 0$ for all $s > p$ (Proposition 3.1).

---

## 3.3 Frozen RMSNorm

RMSNorm rescales the residual at each position before each sub-block:

$$\text{RMSNorm}(x_p) = \gamma \odot \frac{x_p}{\sqrt{\frac{1}{d}\|x_p\|_2^2 + \varepsilon}}$$

The denominator $r_p := \left(\sqrt{\frac{1}{d}\|x_p\|_2^2 + \varepsilon}\right)^{-1}$ is **position-specific and nonlinear** in $x_p$. We freeze $r_p$ at the value computed during the real forward pass and define the linearized RMSNorm action on each source contribution as

$$\hat{c}^{(\ell)}_{p,s} := \gamma \odot \left(r^{(\ell)}_p \cdot c^{(\ell)}_{p,s}\right)$$

This is exact at the actual activation point $x^* = x^{(\ell)}_p$, since $\sum_s c^{(\ell)}_{p,s} = x^{(\ell)}_p$ and linearity gives $\sum_s \hat{c}^{(\ell)}_{p,s} = \text{RMSNorm}(x^{(\ell)}_p)$.

---

## 3.4 Frozen-Gate MLP Linearization (SwiGLU)

Llama uses the SwiGLU feed-forward architecture. For each layer $\ell$ and position $p$:

$$\text{MLP}^{(\ell)}(x_p) = W_{\text{down}}^{(\ell)}\!\left(\text{SiLU}\!\left(W_{\text{gate}}^{(\ell)} x_p\right) \odot W_{\text{up}}^{(\ell)} x_p\right)$$

with $W_{\text{gate}}^{(\ell)}, W_{\text{up}}^{(\ell)} \in \mathbb{R}^{d_{\text{ff}} \times d}$ and $W_{\text{down}}^{(\ell)} \in \mathbb{R}^{d \times d_{\text{ff}}}$.

There are two sources of nonlinearity: (i) the SiLU activation on the gate branch, and (ii) the Hadamard product between the gate and the write branch, which is **bilinear** — linear in each argument separately, but not jointly.

**Frozen-gate construction.** We freeze the gate activations at the values from the real forward pass:

$$s^{*,(\ell)}_p := \text{SiLU}\!\left(W_{\text{gate}}^{(\ell)} x^{*,(\ell)}_p\right) \in \mathbb{R}^{d_{\text{ff}}}$$

Substituting $s^{*,(\ell)}_p$ as a fixed constant vector, the MLP becomes the **exactly linear map**:

$$\widetilde{\text{MLP}}^{(\ell)}(x_p) = W_{\text{down}}^{(\ell)}\!\left(s^{*,(\ell)}_p \odot W_{\text{up}}^{(\ell)} x_p\right) = W_{\text{down}}^{(\ell)} \operatorname{diag}(s^{*,(\ell)}_p)\, W_{\text{up}}^{(\ell)}\, x_p$$

The MLP contribution from source $s$ to target position $p$ (after the second frozen RMSNorm $r^{(\ell,2)}_p$) is therefore

$$\delta c^{\text{mlp},(\ell)}_{p,s} = W_{\text{down}}^{(\ell)} \operatorname{diag}(s^{*,(\ell)}_p)\, W_{\text{up}}^{(\ell)}\!\left(\gamma_2^{(\ell)} \odot r^{(\ell,2)}_p \cdot c^{\text{mid},(\ell)}_{p,s}\right)$$

and the full post-layer contribution is

$$c^{(\ell+1)}_{p,s} = c^{\text{mid},(\ell)}_{p,s} + \delta c^{\text{mlp},(\ell)}_{p,s}$$

Note that the frozen-gate construction is **structurally symmetric** to the frozen-QK construction: in both cases we freeze the "selection" weights (attention pattern / gate activations) and allow the "write" weights (OV circuit / up-down circuit) to act linearly on the source contributions.

---

## 3.5 Full Decomposition Algorithm

**Initialization** (layer $0$):

$$c^{(0)}_{p,s} = \begin{cases} e_p & \text{if } s = p \\ 0 & \text{otherwise} \end{cases}$$

**Recurrence** (for $\ell = 0, 1, \dots, L-1$):

$$c^{\text{mid},(\ell)}_{p,s} = c^{(\ell)}_{p,s} + \sum_{h=1}^{H} \sum_{s'=0}^{p} A^{(\ell,h)}_{p,s'} \cdot W^{(\ell,h)}_{\text{OV}} \!\left(\gamma_1^{(\ell)} \odot r^{(\ell,1)}_{s'} \cdot c^{(\ell)}_{s',s}\right)$$

$$c^{(\ell+1)}_{p,s} = c^{\text{mid},(\ell)}_{p,s} + W_{\text{down}}^{(\ell)} \operatorname{diag}(s^{*,(\ell)}_p)\, W_{\text{up}}^{(\ell)}\!\left(\gamma_2^{(\ell)} \odot r^{(\ell,2)}_p \cdot c^{\text{mid},(\ell)}_{p,s}\right)$$

The output is the **four-dimensional tensor** $c \in \mathbb{R}^{L \times N \times N \times d}$, kept in memory and reduced to scalar norms only at serialization time.

---

## 3.6 Correctness Criteria

The decomposition is deemed correct if and only if all of the following hold.

**(C1) Reconstruction.**

$$\left\|\sum_{s=0}^{N-1} c^{(\ell)}_{p,s} - x^{(\ell)}_p\right\|_\infty < \epsilon \quad \forall\, \ell \in \{0,\dots,L\},\; p \in \{0,\dots,N-1\}$$

with tolerance $\epsilon = 10^{-3}$ (floating-point accumulation over $L$ layers at fp32).

**(C2) Causality.**

$$c^{(\ell)}_{p,s} = 0 \quad \forall\, s > p$$

**(C3) Identity at layer $0$.**

$$c^{(0)}_{p,s} = e_p \cdot \mathbf{1}[s = p]$$

**(C4) Norm non-negativity.** $\|c^{(\ell)}_{p,s}\|_q \geq 0$ for all $q \in \{0,1,2,\infty\}$.

These four conditions are enforced as automated tests in `tests/test_decompose.py` and must all pass before any results are reported.
