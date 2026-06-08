"""Per-layer reconstruction-precision metric for the residual stream.

Two views of the same error at each layer l:
  - Per-position view: |diff_p| / max(|real_p|, POS_DENOM_FLOOR), aggregated over
    positions and over the two stages (post-attn, post-mlp). Floor prevents tiny
    per-position-norms from blowing up the ratio.
  - Per-element view: |diff_i| / max(|real_i|, ELEM_DENOM_FLOOR), aggregated as
    {p98, p99, max} over every element of every position of both stages.
    Floor protects against the many ~0 elements producing fake huge ratios.

For the MLP-stage per-position denominator we use
    max(|real_mlp|_p, |real_attn|_p)
because at the last layer the MLP cancels the magnitude sink and |real_mlp|
collapses; the pre-MLP attn residual still carries the sink and keeps the
denominator stable (see layer31_mlp_investigation.md).

Use `compute_precision_table(get_contributions, real_mlp, real_attn)` to produce
a (L, 5) tensor — columns: [max_rel, mean_rel, p98_elem, p99_elem, max_elem].
"""

from typing import Callable

import torch
from torch import Tensor

# Column indices for the returned table
COL_MAX_REL = 0
COL_MEAN_REL = 1
COL_P98_ELEM = 2
COL_P99_ELEM = 3
COL_MAX_ELEM = 4
COL_NAMES = ["max_rel", "mean_rel", "p98_elem", "p99_elem", "max_elem"]


def compute_precision_table(
    get_contributions: Callable[[], tuple[list[Tensor], list[Tensor]]],
    real_mlp: list[Tensor],
    real_attn: list[Tensor],
    *,
    pos_denom_floor: float = 1.0,
    elem_denom_floor: float = 1.0,
) -> Tensor:
    """Compute the precision table for a reconstruction.

    Args:
        get_contributions: callable -> (post_mlp_contribution, post_attention_contribution).
            Each is list[L] of tensors (1, pos, source, d) — i.e. NOT yet summed over sources.
        real_mlp:  list[L] of (1, pos, d) — model post-MLP residual per layer.
        real_attn: list[L] of (1, pos, d) — model post-attention residual per layer.
        pos_denom_floor: floor on per-position denom (default 1.0).
        elem_denom_floor: floor on per-element denom (default 1.0).

    Returns:
        Tensor of shape (L, 5), columns per COL_NAMES.
    """
    post_mlp_contrib, post_attn_contrib = get_contributions()
    mine_mlp = [t.sum(dim=1) for t in post_mlp_contrib]   # sum over sources -> (1, pos, d)
    mine_attn = [t.sum(dim=1) for t in post_attn_contrib]

    L = len(real_mlp)
    out = torch.zeros(L, len(COL_NAMES))

    for l in range(L):
        rm = real_mlp[l].float().squeeze(0)
        ra = real_attn[l].float().squeeze(0)
        mm = mine_mlp[l].float().squeeze(0)
        ma = mine_attn[l].float().squeeze(0)

        # --- per-position metric ---
        norm_attn = ra.norm(dim=-1)
        norm_mlp = rm.norm(dim=-1)
        denom_pos_attn = norm_attn.clamp_min(pos_denom_floor)
        denom_pos_mlp = torch.maximum(norm_mlp, norm_attn).clamp_min(pos_denom_floor)

        attn_rel = (ma - ra).norm(dim=-1) / denom_pos_attn   # (pos,)
        mlp_rel = (mm - rm).norm(dim=-1) / denom_pos_mlp     # (pos,)
        rel_stack = torch.stack([attn_rel, mlp_rel], dim=0)  # (2, pos)

        # --- per-element metric ---
        elem_attn = (ma - ra).abs() / ra.abs().clamp_min(elem_denom_floor)
        elem_mlp = (mm - rm).abs() / rm.abs().clamp_min(elem_denom_floor)
        all_elem = torch.cat([elem_attn.flatten(), elem_mlp.flatten()])

        out[l, COL_MAX_REL] = rel_stack.max()
        out[l, COL_MEAN_REL] = rel_stack.mean()
        out[l, COL_P98_ELEM] = torch.quantile(all_elem, 0.98)
        out[l, COL_P99_ELEM] = torch.quantile(all_elem, 0.99)
        out[l, COL_MAX_ELEM] = all_elem.max()

    return out


def print_table(table: Tensor, label: str = "") -> None:
    """Pretty-print a precision table with per-column means appended."""
    L = table.shape[0]
    header = f"{'L':>3} " + " ".join(f"{c:>11}" for c in COL_NAMES)
    if label:
        print(f"\n[{label}]")
    print(header)
    for l in range(L):
        row = " ".join(f"{table[l, i].item():>11.5f}" for i in range(table.shape[1]))
        print(f"{l:>3} {row}")
    mean_row = " ".join(f"{table[:, i].mean().item():>11.5f}" for i in range(table.shape[1]))
    print(f"{'MEAN':>3} {mean_row}")
