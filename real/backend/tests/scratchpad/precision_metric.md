# Precision metric — definition and baseline

Purpose: a single, reusable score for *how close my per-source reconstruction is
to the model's actual forward pass at each layer*. The result is a `(L, 5)`
tensor so that future reconstructions can be compared against this baseline
column-by-column and layer-by-layer.

Implementation: [`src/info_flow/precision_metric.py`](../../src/info_flow/precision_metric.py).
Run via [`src/info_flow/test_full_contributions_residual_combined.py`](../../src/info_flow/test_full_contributions_residual_combined.py).

## What the metric measures

At every layer $l$ there are **two residual-stream snapshots**:

- post-attention residual: `real_attn[l]` (the model's value), `mine_attn[l]` (my reconstruction summed over sources).
- post-MLP residual: `real_mlp[l]`, `mine_mlp[l]`.

For each layer we produce two "views" of the error.

### Per-position view (vector-norm rel-error per token position)

For each token position $p$ and each stage:

$$\text{rel\_pos}[l,p,\text{attn}] = \frac{\lVert \text{mine\_attn}_p - \text{real\_attn}_p \rVert_2}{\max\bigl(\lVert\text{real\_attn}_p\rVert_2,\ \text{POS\_DENOM\_FLOOR}\bigr)}$$

$$\text{rel\_pos}[l,p,\text{mlp}] = \frac{\lVert \text{mine\_mlp}_p - \text{real\_mlp}_p \rVert_2}{\max\bigl(\max(\lVert\text{real\_mlp}_p\rVert_2,\lVert\text{real\_attn}_p\rVert_2),\ \text{POS\_DENOM\_FLOOR}\bigr)}$$

Two non-obvious denominator choices:
- For MLP we use $\max(\lVert\text{real\_mlp}\rVert, \lVert\text{real\_attn}\rVert)$ because
  at layer 31 the MLP cancels the magnitude-sink feature and $\lVert\text{real\_mlp}\rVert$
  collapses (~497 → ~170 at position 0). The attention residual still carries the
  sink, so this `max` keeps the denominator stable layer-to-layer. See
  `layer31_mlp_investigation.md` for the full story.
- `POS_DENOM_FLOOR = 1.0` protects against per-position norms ever being tiny.
  In practice positions are all $\geq 3$, so this is effectively a no-op for
  this prompt — kept for safety across other prompts.

Aggregations per layer:
- `max_rel`: $\max_{p,\text{stage}} \text{rel\_pos}[l,p,\text{stage}]$ — worst single position.
- `mean_rel`: $\text{mean}_{p,\text{stage}} \text{rel\_pos}[l,p,\text{stage}]$ — average over residual states.

### Per-element view (scalar rel-error for every coordinate)

For every individual element $i$ of every position of both stages:

$$\text{rel\_elem}_i = \frac{|\text{mine}_i - \text{real}_i|}{\max(|\text{real}_i|,\ \text{ELEM\_DENOM\_FLOOR})}$$

`ELEM_DENOM_FLOOR = 1.0` is doing real work here. Most elements have
$|\text{real}_i| \ll 1$, so a small absolute error would produce a meaningless
huge relative error. With the floor, tiny elements are scored *as if their
"reasonable" magnitude were 1.0*. Elements with $|\text{real}_i| \geq 1$ are
scored on their true rel-error.

Aggregations per layer (over all elements across both stages, all positions, all dims):
- `p98_elem`: 98th percentile — robust upper-tail signal.
- `p99_elem`: 99th percentile — more aggressive tail.
- `max_elem`: worst single element — most sensitive to outliers.

## Returned tensor shape

`compute_precision_table(...) -> Tensor` of shape `(L, 5)`. Column order
(`info_flow.precision_metric.COL_NAMES`):

```
[max_rel, mean_rel, p98_elem, p99_elem, max_elem]
```

The function accepts a `Callable[[], (post_mlp, post_attn)]` so any
reconstruction can be evaluated — pass a different callable to score a
different implementation, then compare the two tensors element-wise.

## Baseline result — bf16 reconstruction (current `ex3` calculator)

Prompt: `"The cat sat on the mat, and then afterword, he decided that "`
Model: `meta-llama/Meta-Llama-3.1-8B-Instruct`. bf16 throughout.

| L | max_rel | mean_rel | p98_elem | p99_elem | max_elem |
|---:|---:|---:|---:|---:|---:|
|  0 | 0.00557 | 0.00305 | 0.00012 | 0.00024 | 0.00195 |
|  1 | 0.00665 | 0.00523 | 0.00028 | 0.00049 | 0.00730 |
|  2 | 0.00720 | 0.00570 | 0.00049 | 0.00049 | 0.00781 |
|  3 | 0.00732 | 0.00608 | 0.00073 | 0.00098 | 0.00794 |
|  4 | 0.00782 | 0.00637 | 0.00098 | 0.00098 | 0.01172 |
|  5 | 0.00842 | 0.00667 | 0.00098 | 0.00146 | 0.01266 |
|  6 | 0.00847 | 0.00686 | 0.00146 | 0.00195 | 0.01562 |
|  7 | 0.00862 | 0.00718 | 0.00183 | 0.00195 | 0.01953 |
|  8 | 0.00873 | 0.00749 | 0.00195 | 0.00195 | 0.01953 |
|  9 | 0.00932 | 0.00776 | 0.00195 | 0.00195 | 0.02344 |
| 10 | 0.00950 | 0.00792 | 0.00195 | 0.00244 | 0.02308 |
| 11 | 0.00969 | 0.00804 | 0.00195 | 0.00293 | 0.02344 |
| 12 | 0.00959 | 0.00796 | 0.00220 | 0.00293 | 0.01953 |
| 13 | 0.00949 | 0.00778 | 0.00244 | 0.00293 | 0.01899 |
| 14 | 0.00969 | 0.00758 | 0.00269 | 0.00391 | 0.02051 |
| 15 | 0.00856 | 0.00704 | 0.00293 | 0.00391 | 0.02051 |
| 16 | 0.00862 | 0.00680 | 0.00293 | 0.00391 | 0.02732 |
| 17 | 0.00841 | 0.00661 | 0.00391 | 0.00391 | 0.02415 |
| 18 | 0.00810 | 0.00642 | 0.00391 | 0.00391 | 0.02273 |
| 19 | 0.00788 | 0.00644 | 0.00391 | 0.00439 | 0.02649 |
| 20 | 0.00806 | 0.00660 | 0.00391 | 0.00586 | 0.03125 |
| 21 | 0.00831 | 0.00676 | 0.00488 | 0.00684 | 0.02930 |
| 22 | 0.00843 | 0.00683 | 0.00586 | 0.00781 | 0.03788 |
| 23 | 0.00879 | 0.00692 | 0.00685 | 0.00781 | 0.03516 |
| 24 | 0.00884 | 0.00708 | 0.00781 | 0.00781 | 0.03711 |
| 25 | 0.00908 | 0.00721 | 0.00781 | 0.00781 | 0.03356 |
| 26 | 0.00893 | 0.00742 | 0.00781 | 0.01083 | 0.03906 |
| 27 | 0.00929 | 0.00743 | 0.00879 | 0.01172 | 0.03896 |
| 28 | 0.01110 | 0.00761 | 0.01172 | 0.01172 | 0.04348 |
| 29 | 0.00970 | 0.00761 | 0.01172 | 0.01367 | 0.05469 |
| 30 | 0.00845 | 0.00723 | 0.01212 | 0.01493 | 0.05469 |
| 31 | 0.00810 | 0.00648 | 0.01493 | 0.01562 | **0.15625** |
| **MEAN** | **0.00859** | **0.00688** | **0.00449** | **0.00531** | **0.02955** |
| MEDIAN | 0.00859 | 0.00698 | 0.00293 | 0.00391 | 0.02344 |
| MAX | 0.01110 | 0.00804 | 0.01493 | 0.01562 | 0.15625 |

## Reading the baseline

- **`max_rel` ≈ 0.86% on average, peaks at 1.11% (L28).** Per-position worst-case
  is steady-state from L7 onward — flat at ~0.9%. This is at the bf16
  reconstruction floor: see `layer31_mlp_investigation.md` E4/H4′ — the bf16
  forward and the bf16 reconstruction make different rounding paths through the
  same matmuls, and the discrepancy steady-states at $\sim 1\%$ after a few layers.
- **`mean_rel` ≈ 0.69%.** Plateaus around L10. Same story — bf16 floor.
- **`p98_elem` / `p99_elem` grow with depth**, from ~0 at L0 to ~1.5% by L31.
  Per-element drift compounds because the recurrence carries forward all
  earlier-layer errors. At L0 there is no recurrence yet, so per-element errors
  are smallest.
- **`max_elem` blows up at L31 (15.6%).** Almost certainly one element of the
  sink-feature (e.g. dim 2352) where the MLP cancels a value of magnitude ~50
  down to ~5 with bf16 precision — a small rounding error against an originally
  huge number becomes a sizeable rel-error against the post-MLP value. The
  per-element floor mitigates this but doesn't prevent it because the post-MLP
  element is still ≥ 1. A useful diagnostic, not a regression.

## How to compare a new reconstruction against this baseline

```python
from info_flow.precision_metric import compute_precision_table, print_table

new_table = compute_precision_table(
    my_new_get_contributions,           # callable returning (post_mlp, post_attn)
    real_mlp=real["mlp"], real_attn=real["attn"],
)
baseline = torch.load(".cache/precision_table_bf16.pt")

improvement = baseline - new_table                 # positive = better than baseline
print_table(improvement, label="improvement over bf16 baseline")
```

Goal of any precision-improvement work: get the **`MEAN`** row (especially
`max_rel`, `p98_elem`, `p99_elem`) below the baseline values. The bf16 floor
is around `max_rel ≈ 0.7%` for this prompt — getting meaningfully below that
requires either (a) running both sides in fp32 (not just one — see E4), or
(b) reorganising the reconstruction to match the model's reduction order
(see the three non-associativity sources at end of `layer31_mlp_investigation.md`).