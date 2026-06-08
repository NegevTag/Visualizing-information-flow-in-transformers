"""Print + plot helpers for PrecisionResults tensors of shape (Layer, 5).

Columns: max_norm_rel, mean_norm_rel, p98_elm, p99_elm, max_elm.
"""

import typing

import matplotlib.pyplot as plt
import numpy as np
import torch
from rich.console import Console
from rich.table import Column, Table

PrecisionResults = typing.NewType("PrecisionResults", torch.Tensor)

COLS = ["layer", "max_norm_rel", "mean_norm_rel", "p98_elm", "p99_elm", "max_elm"]


def pretty_print_precision(results: PrecisionResults, *, title: str | None = None) -> None:
    """Pretty-print a (Layer, 5) tensor + a no-header 'max' summary row aligned beneath."""
    if title:
        Console().print(f"[bold]{title}[/bold]")

    table = Table(*[Column(c, header_style="none") for c in COLS], show_lines=False)
    for layer_idx, row in enumerate(results):
        table.add_row(str(layer_idx), *[f"{v.item():.5f}" for v in row])
    Console().print(table)

    max_per_col = results.amax(dim=0)
    summary = Table(
        *[Column(c, header_style="none", min_width=len(c)) for c in COLS],
        show_lines=False,
        show_header=False,
        header_style="none",
    )
    summary.add_row("max", *[f"{v.item():.5f}" for v in max_per_col])
    Console().print(summary)


def _pretty_print_diff(diff: torch.Tensor, blue_name: str, red_name: str) -> None:
    """Print signed diff = red - blue (positive => red bigger, negative => blue bigger)."""
    Console().print(f"[bold]diff = {red_name} - {blue_name}[/bold]  (+ {blue_name} wins, - {red_name} wins; smaller is better)")
    table = Table(*[Column(c, header_style="none") for c in COLS], show_lines=False)
    for layer_idx, row in enumerate(diff):
        table.add_row(str(layer_idx), *[f"{v.item():+.5f}" for v in row])
    Console().print(table)

    abs_max = diff.abs().amax(dim=0)
    summary = Table(
        *[Column(c, header_style="none", min_width=len(c)) for c in COLS],
        show_lines=False,
        show_header=False,
        header_style="none",
    )
    summary.add_row("max|d|", *[f"{v.item():.5f}" for v in abs_max])
    Console().print(summary)


def compare_percision(
    blue: PrecisionResults,
    red: PrecisionResults,
    *,
    blue_name: str = "blue",
    red_name: str = "red",
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Print both tables + their signed diff, and render a (Layer x 5) heatmap.

    These are *error* metrics, so smaller = better. Cell color encodes the winner:
    redder where `red` has the smaller value (red wins), bluer where `blue` has
    the smaller value (blue wins). Symmetric color scale around 0.
    """
    assert blue.shape == red.shape, f"shape mismatch: {tuple(blue.shape)} vs {tuple(red.shape)}"

    pretty_print_precision(blue, title=blue_name)
    pretty_print_precision(red, title=red_name)
    diff = (red.float() - blue.float()).cpu()  # +ve => red larger => red loses
    _pretty_print_diff(diff, blue_name=blue_name, red_name=red_name)

    # Winner-margin: positive when red wins (red is smaller).
    winner = (blue.float() - red.float()).cpu().numpy()
    L, n_cols = winner.shape

    # Two color scales: the per-position norm columns (max_norm_rel, mean_norm_rel)
    # live on a different magnitude than the per-element columns (p98/p99/max_elm),
    # so saturating them together hides the smaller group. Normalize each group
    # independently to [-1, 1] for coloring; cell text still shows the raw value.
    NORM_COLS = [0, 1]
    ELEM_COLS = [2, 3, 4]
    norm_vmax = float(np.abs(winner[:, NORM_COLS]).max()) or 1.0
    elem_vmax = float(np.abs(winner[:, ELEM_COLS]).max()) or 1.0
    scaled = winner.copy()
    scaled[:, NORM_COLS] = winner[:, NORM_COLS] / norm_vmax
    scaled[:, ELEM_COLS] = winner[:, ELEM_COLS] / elem_vmax

    fig, ax = plt.subplots(figsize=(1.2 * n_cols + 1.5, 0.28 * L + 1.5))
    im = ax.imshow(scaled, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(COLS[1:], rotation=30, ha="right")
    ax.set_yticks(range(L))
    ax.set_yticklabels([str(i) for i in range(L)])
    ax.set_ylabel("layer")
    ax.set_title(f"winner margin  (red = {red_name} wins, blue = {blue_name} wins; smaller is better)")

    for i in range(L):
        for j in range(n_cols):
            ax.text(j, i, f"{winner[i, j]:+.4f}", ha="center", va="center", fontsize=7, color="black")

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, ticks=[-1, 0, 1])
    cbar.set_label(
        f"+ve: {red_name} wins  |  norm-col scale ±{norm_vmax:.4f}, elem-col scale ±{elem_vmax:.4f}"
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
