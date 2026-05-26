"""Sanity checks for `calc_contribution_per_layer_per_residual`.

Runs locally on ToyLlama (no NDIF). Each check is independent and reports
PASS/FAIL with a short detail line. See `test_warmup_sanity.md` for what
each check is asserting and why.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests" / "scratchpad"))

from info_flow.math_warmap_cac_normal_res import calc_contribution_per_layer_per_residual  # noqa: E402
from toy_llama import ToyLlama  # noqa: E402

ATOL = 1e-2
RTOL = 1e-2
PROMPT = "The cat sat on the"


# ---------------------------------------------------------------------------
# small reporting helpers
# ---------------------------------------------------------------------------

@dataclass
class Result:
    name: str
    ok: bool
    detail: str = ""


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def print_results(results: list[Result]) -> None:
    for r in results:
        tag = "PASS" if r.ok else "FAIL"
        suffix = f" — {r.detail}" if r.detail else ""
        print(f"[{tag}] {r.name}{suffix}")


# ---------------------------------------------------------------------------
# data capture
# ---------------------------------------------------------------------------

@dataclass
class Reference:
    """Ground-truth attention outputs and patterns, one entry per layer."""
    attn_out: list[torch.Tensor]      # (Q, d_model)
    attn_pattern: list[torch.Tensor]  # (H_q, Q, K)


def capture_reference(model, prompt: str) -> Reference:
    with model.trace(prompt):
        attn_out = list().save()
        attn_pat = list().save()
        for layer in model.model.layers:
            attn_out.append(layer.self_attn.output[0][0])
            attn_pat.append(layer.self_attn.output[1][0])
    return Reference(attn_out=list(attn_out), attn_pattern=list(attn_pat))


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------

Check = Callable[[list[torch.Tensor], Reference, int], list[Result]]


def check_shapes(per_layer, ref, d_model) -> list[Result]:
    Q = ref.attn_out[0].shape[0]
    L = len(ref.attn_out)
    expected = (Q, Q, d_model)
    shapes = [tuple(per_layer[l].shape) for l in range(len(per_layer))]
    return [
        Result("len(per_layer) == LAYERS_NUM",
               len(per_layer) == L,
               f"{len(per_layer)} vs {L}"),
        Result("per_layer[l].shape == (Q, K, d_model) for all l",
               all(s == expected for s in shapes),
               f"got {shapes}"),
    ]


def check_pattern_is_causal_softmax(per_layer, ref, d_model) -> list[Result]:
    del per_layer, d_model  # this check only inspects the reference capture
    pat = ref.attn_pattern[0]
    Q = pat.shape[1]
    row_sums = pat.sum(dim=-1)
    row_ok = torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
    upper_max = max(
        (pat[:, q, q + 1:].abs().max().item() for q in range(Q - 1)),
        default=0.0,
    )
    return [
        Result("attn pattern row sums ~ 1", row_ok,
               f"max |row_sum - 1| = {(row_sums - 1).abs().max().item():.2e}"),
        Result("attn pattern strictly upper triangle is 0", upper_max < 1e-12,
               f"max upper-tri mass = {upper_max:.2e}"),
    ]


def check_reconstruction(per_layer, ref, d_model) -> list[Result]:
    del d_model
    out = []
    for l, (contrib, target) in enumerate(zip(per_layer, ref.attn_out)):
        recon = contrib.sum(dim=1)
        rel = ((recon - target).norm() / target.norm()).item()
        out.append(Result(
            f"layer {l}: sum_k contrib[q,k] == self_attn.output[q]",
            torch.allclose(recon, target, atol=ATOL, rtol=RTOL),
            f"rel_err={rel:.3e}",
        ))
    return out


def check_causal_contrib(per_layer, ref, d_model) -> list[Result]:
    del ref, d_model
    out = []
    for l, contrib in enumerate(per_layer):
        Q = contrib.shape[0]
        upper_max = max(
            (contrib[q, q + 1:].abs().max().item() for q in range(Q - 1)),
            default=0.0,
        )
        out.append(Result(
            f"layer {l}: contrib[q, k>q] is zero",
            upper_max < 1e-8,
            f"max |contrib[q, k>q]| = {upper_max:.2e}",
        ))
    return out


def report_per_query_error(per_layer, ref: Reference) -> None:
    """Diagnostic table: per-query reconstruction error, not a pass/fail check."""
    for l, (contrib, target) in enumerate(zip(per_layer, ref.attn_out)):
        recon = contrib.sum(dim=1)
        print(f"  layer {l}:")
        for q in range(target.shape[0]):
            ref_n = target[q].norm().item()
            err_n = (recon[q] - target[q]).norm().item()
            rel = err_n / ref_n if ref_n > 0 else float("inf")
            print(f"    q={q}: ||recon||={recon[q].norm().item():.3e}  "
                  f"||ref||={ref_n:.3e}  rel_err={rel:.3e}")


def report_diagonal_magnitudes(per_layer) -> None:
    """Diagnostic table: ||contrib[q, q]|| per (layer, q)."""
    for l, contrib in enumerate(per_layer):
        norms = [f"{contrib[q, q].norm().item():.3e}" for q in range(contrib.shape[0])]
        print(f"  layer {l}: {norms}")


# ---------------------------------------------------------------------------
# entrypoint
# ---------------------------------------------------------------------------

CHECKS: list[tuple[str, Check]] = [
    ("shapes",                         check_shapes),
    ("attn pattern is causal softmax", check_pattern_is_causal_softmax),
    ("reconstruction (main claim)",    check_reconstruction),
    ("contrib causal structure",       check_causal_contrib),
]


def main() -> None:
    torch.manual_seed(0)
    model = ToyLlama.build_nnsight_mode()
    d_model = model.model.config.hidden_size

    section("setup")
    ref = capture_reference(model, PROMPT)
    Q = ref.attn_out[0].shape[0]
    print(f"prompt={PROMPT!r}  Q={Q}  d_model={d_model}  layers={len(ref.attn_out)}")

    per_layer = calc_contribution_per_layer_per_residual(model, PROMPT, remote=False)

    for title, fn in CHECKS:
        section(f"check: {title}")
        print_results(fn(per_layer, ref, d_model))

    section("diagnostic: per-query reconstruction error")
    report_per_query_error(per_layer, ref)

    section("diagnostic: diagonal contribution norms ||contrib[q, q]||")
    report_diagonal_magnitudes(per_layer)


if __name__ == "__main__":
    main()
