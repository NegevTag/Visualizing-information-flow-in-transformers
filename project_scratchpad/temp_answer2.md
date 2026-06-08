# Why the "better precision" method is actually slightly worse

Looking at the two `_calculate_mlp_contribution` functions side-by-side, I think
I see what's happening — and it's a genuinely subtle and interesting effect.
**The old method has a hidden self-correcting property that the new method
gives up.**

## The setup

In both methods, we want $\sum_s c_s \approx \mathrm{RMSNorm}(r_{\text{real}})$
where $r_{\text{real}}$ is the true residual.

Let me write the reconstructed sum as
$r_{\text{rec}} = \sum_s c_s = r_{\text{real}} + \delta$
for some small error $\delta$.

- **Old** uses $\mathrm{rms\_factor}_{\text{old}} = 1/\sqrt{\|r_{\text{rec}}\|^2/d + \varepsilon}$
- **New** uses $\mathrm{rms\_factor}_{\text{new}} = 1/\sqrt{\|r_{\text{real}}\|^2/d + \varepsilon}$

Both then multiply by the **same** contributions (which sum to $r_{\text{rec}}$,
not $r_{\text{real}}$).

## What sums to what

- **Old:** $\sum_s = w \cdot \mathrm{rms\_factor}_{\text{old}} \cdot r_{\text{rec}} = \mathrm{RMSNorm}(r_{\text{rec}})$ — an *exact* RMSNorm applied to the wrong input.
- **New:** $\sum_s = w \cdot \mathrm{rms\_factor}_{\text{new}} \cdot r_{\text{rec}}$ — correct scale factor, but applied to the wrong input.
- **True:** $w \cdot \mathrm{rms\_factor}_{\text{new}} \cdot r_{\text{real}}$.

First-order expansion of $\mathrm{rms\_factor}_{\text{old}}$ around $r_{\text{real}}$:

$$\mathrm{rms\_factor}_{\text{old}} \approx \mathrm{rms\_factor}_{\text{new}} \cdot \left(1 - \frac{r_{\text{real}} \cdot \delta}{\|r_{\text{real}}\|^2 + d\varepsilon}\right)$$

Working through the errors:

$$\text{OLD} - \text{True} \approx w \cdot \mathrm{rms\_factor}_{\text{new}} \cdot \big(\delta - \mathrm{proj}_{r_{\text{real}}}(\delta)\big) = w \cdot \mathrm{rms\_factor}_{\text{new}} \cdot \delta_\perp$$

$$\text{NEW} - \text{True} = w \cdot \mathrm{rms\_factor}_{\text{new}} \cdot \delta$$

**The old method's error contains only $\delta_\perp$** — the component of
$\delta$ perpendicular to $r_{\text{real}}$.

## The intuition

RMSNorm is scale-invariant — feed it a vector with the wrong norm but the
correct direction and you get the *exact same* output. The old method
exploits this: by computing the rms factor from the *same* (wrong-scale)
reconstructed vector it later multiplies, scale errors cancel.

The new method breaks the cancellation: it uses the correct scale factor but
multiplies it by the wrong-scale vector, so the parallel component of $\delta$
now leaks through.

## Why it matches the data

- **Layers 0–7:** $\delta$ is tiny, both methods near-identical (diff $\approx 0$).
- **Layers 8–20:** error builds, new starts losing on `max_elm` (drift in the
  parallel direction).
- **Layers 21–30:** `max_elm` diff blows up to $+0.02$ to $+0.04$ —
  accumulated scale drift, exactly what you'd predict from losing the
  self-correction.
- `max_norm_rel` / `mean_norm_rel` move only slightly because the *whole
  vector* error doesn't change much (perpendicular vs full); but `max_elm`
  does, because the worst single element is dominated by the parallel-direction
  leak.

## Practical conclusion

Using the "real" RMS factor seems more principled, but it sacrifices the
algebraic identity

$$\sum_s c_s = \mathrm{RMSNorm}\!\left(\sum_s c_s\right)$$

that the old version satisfies exactly. The old version's wrongness in the
factor is **anti-correlated** with the wrongness in the vector, and they cancel.

If you actually want to use the real norm for some other reason (e.g., it's the
"true" model behavior), you'd need to *also* correct the contributions to sum
to $r_{\text{real}}$ rather than $r_{\text{rec}}$ — for example by adding a
small uniform correction
$\frac{r_{\text{real}} - r_{\text{rec}}}{n_{\text{sources}}}$
to each contribution, or attributing the residual error to some "noise" bucket.

## Cheap test to verify

At one layer, decompose $\delta = \delta_\parallel + \delta_\perp$, then check
that
$$\frac{|\text{OLD error}|}{|\text{NEW error}|} \approx \frac{|\delta_\perp|}{|\delta|}$$
If the ratio matches across layers, the analysis is right.
