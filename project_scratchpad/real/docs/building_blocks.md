# Transformer Building Blocks

## RMSNorm

$$\operatorname{RMSNorm}(x_p) = \gamma \odot \frac{x_p}{\sqrt{\frac{1}{d}\|x_p\|_2^2 + \varepsilon}}$$

Divide each vector by its RMS, then scale element-wise by learned $\gamma$.

---

## MLP (SwiGLU, as in Llama)

$$\operatorname{MLP}(x_p) = W_{\mathrm{down}}\Bigl(\operatorname{SiLU}(W_{\mathrm{gate}}\, x_p) \odot W_{\mathrm{up}}\, x_p\Bigr)$$

where $\operatorname{SiLU}(z) = z \cdot \sigma(z)$.

Three projections: $W_{\mathrm{gate}}$ and $W_{\mathrm{up}}$ both map $\mathbb{R}^d \to \mathbb{R}^{d_{\mathrm{ff}}}$, their outputs are multiplied element-wise (the gate controls how much of $W_{\mathrm{up}} x_p$ passes through), then $W_{\mathrm{down}}$ maps back $\mathbb{R}^{d_{\mathrm{ff}}} \to \mathbb{R}^d$.
$g^{(\ell)}p \in \mathbb{R}^{d{\mathrm{ff}}}$,/