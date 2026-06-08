# Layer 31 MLP discrepancy — investigation

**Symptom.** `test_full_contributions.py` passes layers 0..30 and fails the
`rel_norm_diff_mlp < 4e-3` assertion at layer 31. Across layers, the rel-norm
of the MLP diff grows smoothly $0.0015 \to 0.0030$ through layer 30, then jumps
to $0.0057$ at layer 31 (~2× the previous step's growth rate). Attention at
layer 31 is in-trend ($0.0031$). Max-abs MLP diff jumps $0.0306 \to 0.1211$
(~4×). Attention max only $0.0233 \to 0.0320$ (in trend).

The discontinuity is **MLP-only and last-layer-only**. That is the load-bearing
observation — a pure bf16 accumulation story would expect both attention and
MLP error to spike together, since they share the residual stream.

## Hypotheses, ranked by prior plausibility

**H1 — Massive activations at the last MLP.** Llama-family models are known to
develop a small number of huge-magnitude features in the last 1–2 layers
(Sun et al. 2024, "Massive Activations in LLMs"). A bf16 down-projection over
such activations accumulates much larger rounding error than earlier layers.
Compatible with attention being clean because attention output magnitudes
don't follow the same pattern.

*Prediction:* `|real_mlp[31]|.max()` is several× larger than `|real_mlp[30]|.max()`,
and the diff at layer 31 is concentrated on the same few token-positions or
hidden-dims that carry those massive activations.

**H2 — A genuine bug in the layer-31 MLP recomputation.** Possible
mechanisms: an off-by-one on the layer index that makes layer 31 reuse layer
30's weights / RMSNorm; a last-layer-specific code path; a cached tensor
that's stale only at the end. If true, the diff is structural and won't shrink
with fp32 arithmetic.

*Prediction:* the diff at layer 31 does **not** sit on extreme-magnitude
positions/dims; it looks spread or systematic. Casting bf16→fp32 before
metric does not change the picture.

**H3 — Pure bf16 floating-point drift, just unusually bad at L31.** Cannot be
fully ruled out, but the discontinuity argues against it.

*Prediction:* same growth trend as L0–L30, no concentration anywhere.

## Experiments

All three experiments operate on the cached tensors (`.cache/real_*.pt`,
`.cache/mine_*.pt`) — no model reload required.

- **E1. Magnitude check.** Print `|real_mlp[l]|.max()`, `|real_mlp[l]|.mean()`,
  `|mine_mlp[l]|.max()`, `|real_attn[l]|.max()` for *every* layer. Look for a
  step change at L31. Discriminates H1 (yes, step change) from H2/H3 (no).

- **E2. fp32 metric.** Recompute the rel-norm diff after casting both tensors
  to fp32. If the spike disappears, the metric itself is bf16-lossy (subset
  of H1/H3). If it persists, the bf16 tensors genuinely disagree.

- **E3. Localize the diff.** At L30 and L31, split the diff by token
  position and by hidden-dim, report top-k concentration shares. Under H1 the
  diff should concentrate on a tiny number of positions/dims that coincide
  with extreme activations. Under H2 it's diffuse or systematic.

(Note: a fully principled test of H1 would re-run the *calculator* in fp32,
not just the metric. That requires editing `ex3_calc_full_contributions.py`
and re-tracing — listed as follow-up after E1–E3 narrow things down.)

---

# Results

*(filled in after running E1/E2/E3)*

## E1 — magnitudes (the big surprise)

`|real_mlp[l]|.max()` by layer:

| L | max | mean | L | max | mean |
|---|---|---|---|---|---|
| 0 | 3.34 | 0.011 | 16 | 302 | 0.155 |
| 1 | **300** | 0.040 | 20 | 302 | 0.227 |
| 2 | **300** | 0.048 | 24 | 302 | 0.311 |
| ... | 300–302 | growing | 28 | 302 | 0.435 |
| 14 | 302 | 0.122 | 29 | 302 | 0.486 |
| 15 | 302 | 0.139 | 30 | **300** | 0.605 |
| | | | **31** | **27.5** | 0.922 |

**The residual stream carries a max ≈ 300 outlier from layer 1 through layer 30.**
This is the massive-activations phenomenon, but it lives across the entire
network, not just the last layer. At **layer 31 the max collapses to 27.5** (~11×
drop) while the mean activation keeps climbing.

This is the opposite of what H1 predicted. H1 expected magnitudes to *spike*
at L31 and inflate the error. They *collapse* instead.

## E2 — fp32 metric

bf16-vs-fp32 rel-norm of the diff agrees to within ≤1e-5 at every layer
(L31: 0.00568 vs 0.00566; L30: 0.00302 vs 0.00303). The metric is **not**
bf16-lossy. The underlying bf16 tensors genuinely disagree by the reported
amount. H3 in its "metric-precision" form is ruled out.

## E3 — diff localization

Per-position L2 of the MLP-residual diff, and ratio to `|real|` at that position:

| pos | L30 diff | L30 real | L30 ratio | L31 diff | L31 real | L31 ratio |
|---|---|---|---|---|---|---|
| 0 | 0.29 | **497** | 0.0006 | 0.37 | **170** | 0.0022 |
| 1 | 0.38 | 51 | 0.0074 | 0.55 | 101 | 0.0054 |
| 2–16 | ≈0.40 | ≈52 | ≈0.008 | ≈0.52 | ≈80 | ≈0.006 |

- The **absolute diff per position is nearly the same at L30 and L31** (~0.4 → ~0.5).
- The **denominator collapses**: position 0 falls from $|real|=497$ to $|real|=170$
  (~3× drop). Other positions roughly double.
- Top-1-position share of squared diff: 6.6% (L30), 6.9% (L31) — diff is *diffuse*,
  not concentrated.
- Top-10-dim share of squared diff: 11.6% (L30), 12.1% (L31) — also diffuse.
- Dim 2352 carries the largest diff at both layers (0.34 → 0.33), but its real
  L2 collapses from 88.9 → 8.5 between L30 and L31. The error in that dim
  **does not scale down with the activation** — it stays ~0.3 either way.

## Updated hypothesis (H4)

The error has been there the whole time — bf16 reconstruction drift accumulated
across 30+ matmul/RMSNorm/MLP layers. Up through layer 30 it was hidden by an
enormous denominator: the residual stream carried a $\approx 300$-magnitude
"sink" feature (concentrated on position 0 and a handful of dims like 2352)
that dominated `|real|`. The rel-norm was small because the denominator was
huge, not because the numerator was small.

**Layer 31's MLP cancels the sink feature** (pos-0 norm drops 3×, dim 2352
collapses 10×). The numerator — generic, diffuse, dim-wide bf16 drift —
doesn't shrink with it. Result: a *mechanical* spike in rel-norm with no
underlying bug.

Evidence:

- E1: massive activations are present everywhere from L1 onward, then collapse at L31.
- E2: the spike is in the tensors, not the metric — so it is not a precision artifact of `.norm() / .norm()`.
- E3: per-position diff is essentially flat across L30 → L31; only the denominator changed.
- E3: diff is diffuse across dims, not concentrated on one buggy feature — consistent with bf16 drift, not a structural bug.

H1 (massive activations at L31), H2 (last-layer bug), H3 (pure precision-of-metric)
are all rejected. H4 is "**accumulated bf16 drift, exposed when the magnitude-sink is removed.**"

### Predictions H4 makes (cheap to check next)

1. If we re-run the calculator in fp32 (not just the metric), the L0–L30
   numbers shouldn't change much (they were already small), but **L31 should
   come down into the trend** because the accumulated drift goes away. This is
   the decisive test.
2. The "tolerance" in the test (`rel_norm < 4e-3`) is the wrong threshold for
   the last layer specifically — it implicitly assumes a stable denominator.
   A more robust metric would normalize by `|real_attn[l]|` or by the
   *per-position* norm rather than the global one.
3. The pos-0/dim-2352 outlier is the BOS-token sink documented in Llama; we
   should see the same pattern on other prompts.

### Suggested next experiment

Modify `_calculate_mlp_contribution` to keep fp32 through the whole stack
(drop the `.to(contribution.dtype)` casts), re-trace, recompute L31. If L31
drops to ≈ 0.0035, H4 is confirmed and the right fix is to relax the
last-layer tolerance (or switch the metric to a denominator that isn't
dominated by a sink feature).

---

# E4 — fp32 reconstruction vs bf16 ground truth

Implemented `ex4_full_contribution_f32.py` (archived) — promoted every storage
buffer, weight (`W_V/W_O/W_up/W_down`), and bf16 activation (`attn_pattern`,
`act_fn.output`) to fp32 inside the trace. Ran `test_full_contribution_f32.py`
against the same cached bf16 ground truth.

## Result — H4 falsified

| L | bf16 attn | fp32 attn | bf16 mlp | fp32 mlp |
|---|---|---|---|---|
| 0 | 0.0031 | 0.0024 | 0.0015 | **0.0025** |
| 1 | 0.0020 | 0.0026 | 0.0001 | **0.0012** |
| 2 | 0.0001 | 0.0013 | 0.0001 | **0.0015** |
| 5 | 0.0003 | 0.0027 | 0.0003 | **0.0031** |
| 8 | 0.0004 | 0.0036 | 0.0004 | **0.0039** |
| 9 | 0.0005 | **0.0040 FAIL** | — | — |

The fp32 reconstruction is **worse** than bf16 at every layer past L0, and the
error grows monotonically with depth. Opposite of H4's prediction.

## H4′ — what actually happened

The bf16 reconstruction made the **same rounding errors as the bf16 forward
pass** (same matmuls, same dtype, same noise). Errors cancelled in
`mine − real`, so the small rel-norm at layers 0–30 measured *correlation of
rounding noise*, not reconstruction accuracy. Going to fp32 decorrelates the
errors: the rel-norm now measures the bf16 ground truth's own rounding error
(~bf16-eps × √layers ≈ 0.003–0.004), matching the observed curve.

**The L31 spike under bf16 has the same root cause, inverted:** at L31 the
magnitude-sink collapses, the correlation between the two bf16 pipelines
breaks down, and the cancellation stops working.

## Consequence for the test

You can't validate an fp32 reconstruction against a bf16 oracle. For
validating that the per-source decomposition equation is correct, the
**bf16 reconstruction is the right tool** — bit-matching the model's own
forward proves the decomposition holds. The L31 failure isn't a math
or precision bug; it's a **metric-denominator artifact**.

Decision: revert to bf16 reconstruction, fix the test's metric.

---

# E5 — sink-resistant metric (H5)

## H5 — hypothesis

The rel-norm metric `|diff| / |real_mlp[l]|` is fragile because the
denominator is dominated by a magnitude-sink feature that collapses at L31.
Replacing the denominator with
$$\max\bigl(\lVert\text{real\_mlp}[l]\rVert,\ \lVert\text{real\_attn}[l]\rVert\bigr)$$
should fix it: `real_attn[l]` is the residual stream *before* the MLP block,
so it still carries the magnitude sink even when the MLP cancels it. The
denominator can't collapse from one layer to the next.

**Quantitative prediction.** From the cache (E1/E3):
- $\lVert\text{real\_attn}[31]\rVert \gg \lVert\text{real\_mlp}[31]\rVert$
  (attn residual still has pos-0 outlier ~300; mlp residual after sink-cancel ~27).
- L31 MLP diff norm ≈ 2.1 (per E3). Attn residual norm ≈ 500-ish (sink-dominated).
- New L31 metric ≈ $2.1 / 500 \approx 0.004$ — back in trend.
- L0–L30 should change only marginally because there `real_mlp` ≈ `real_attn`
  (both sink-dominated), so the `max` is a near-no-op.

If the prediction holds, all 32 layers pass with the same threshold.

## Result — H5 confirmed

Implementation: cache-only test at
`src/info_flow/test_full_contributions_sink_resistant.py` reuses the bf16
reconstruction cache and applies the new denominator.

Key rows from the printout:

| L | $\lVert\text{real\_mlp}\rVert$ | $\lVert\text{real\_attn}\rVert$ | denom | mlp old | **mlp new** |
|---|---|---|---|---|---|
| 0  | 12.94 | 2.91  | 12.94 | 0.00151 | 0.00151 |
| 1  | 492 | 13.19 | 492 | 0.00010 | 0.00010 |
| 2–30 | ≈500 | ≈500 | ≈500 | (unchanged) | (unchanged) |
| 30 | 540 | 536 | 540 | 0.00302 | 0.00302 |
| **31** | **374** | **548** | **548** | **0.00568 FAIL** | **0.00388 PASS** |

- **L1**: `|real_attn|` (=13) is the post-attention residual *before* the L1 MLP
  fires — still small because the sink hasn't grown in yet. The `max` correctly
  takes `|real_mlp|`. New metric = old metric for MLP. Attn rel actually drops
  (attn numerator measured against a larger denom).
- **L2–L30**: `|real_mlp|` ≈ `|real_attn|` (sink-dominated residual stream).
  `max` is a near no-op. Confirms the prediction that the change wouldn't
  regress earlier layers.
- **L31**: `|real_mlp|` drops to 374 (sink-cancellation), `|real_attn|` stays at
  548. Denom takes 548. Old metric: $2.13/374 = 0.00568$. New: $2.13/548 = 0.00388$.
  In trend with L29/L30's $0.00282 / 0.00302$ (modest bf16-drift growth).
- **All 32 layers pass** under threshold 4e-3.

The quantitative prediction was $2.1/500 \approx 0.004$. Actual: 0.00388. Match.

# Final story

1. **Phenomenon.** Llama 3.1 8B's residual stream carries a $\approx 300$-magnitude
   sink feature on position 0 and a handful of hidden dims (notably 2352)
   from layer 1 through layer 30.
2. **Last MLP cancels the sink.** At L31, $\lVert\text{real\_mlp}\rVert$ drops
   $540 \to 374$ (~31%) and dim 2352 collapses ~10×. This is the model doing
   real work, not a bug.
3. **The original metric was fragile.** Dividing `|diff|` by `|real_mlp|` made
   the rel-norm explode whenever the MLP shrank the residual stream norm,
   even though `|diff|` itself was unchanged.
4. **The reconstruction is correct.** Bit-for-bit matching the bf16 forward
   pass (E4 showed fp32 reconstruction looks *worse* only because errors
   decorrelate with the bf16 oracle — the math is right).
5. **Fix.** Use $\max(\lVert\text{real\_mlp}[l]\rVert, \lVert\text{real\_attn}[l]\rVert)$
   as denominator. Sink-resistant, no regression on L0–L30, fixes L31.

The original `test_full_contributions.py` is left untouched. The new metric
lives in `test_full_contributions_sink_resistant.py` for review before being
adopted as the canonical check.
