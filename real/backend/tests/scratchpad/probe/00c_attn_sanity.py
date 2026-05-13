"""
Step 00c — is `self_attn.output[1]` actually post-softmax A^(ℓ,h)?

If it really is the attention probability matrix the math doc calls A, then:
  - rows sum to 1 (softmax invariant)
  - entries above the diagonal are exactly 0 (causal mask)
  - all entries are in [0, 1]

Captures self_attn.output for layer 0 of Llama-3.1-8B on NDIF, pulls out
slot [1], and runs the checks. If all three pass → confirmed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from _common import (  # noqa: E402
    PROMPT,
    REMOTE,
    section,
    ensure_out_dir,
    print_banner,
    configure_ndif,
    make_model,
    materialise,
)


def main() -> int:
    ensure_out_dir()
    print_banner()
    configure_ndif()

    section("LOAD MODEL")
    model = make_model()
    print("model loaded")

    section("CAPTURE self_attn.output[1] (layer 0)")
    with model.trace(PROMPT, remote=REMOTE):
        proxy = model.model.layers[0].self_attn.output.save()
    val = materialise(proxy)
    if not (isinstance(val, tuple) and len(val) >= 2 and isinstance(val[1], torch.Tensor)):
        print(f"unexpected shape: {type(val).__name__}: {val!r}")
        return 1

    A = val[1].to(torch.float32)
    print(f"  A.shape = {tuple(A.shape)}   dtype={val[1].dtype}")
    print(f"  A.min  = {A.min().item():+.4e}")
    print(f"  A.max  = {A.max().item():+.4e}")
    print(f"  A finite = {torch.isfinite(A).all().item()}")

    B, H, S_q, S_k = A.shape
    if S_q != S_k:
        print(f"  S_q={S_q} S_k={S_k}  (non-square — not the usual A^(p,s))")
        return 1

    section("CHECK 1 — row sums == 1 (softmax)")
    row_sums = A.sum(dim=-1)                          # [B, H, S]
    print(f"  row-sum  min={row_sums.min().item():.6f}  max={row_sums.max().item():.6f}")
    softmax_ok = torch.allclose(row_sums, torch.ones_like(row_sums), atol=2e-3)
    print(f"  rows ≈ 1 within 2e-3 ? {softmax_ok}")

    section("CHECK 2 — causal mask (upper triangle is 0)")
    upper = torch.triu(torch.ones(S_q, S_k, dtype=torch.bool), diagonal=1)
    upper_vals = A[..., upper]
    max_upper = upper_vals.abs().max().item()
    print(f"  max |A[p, s>p]|  = {max_upper:.3e}    (expect 0)")
    causal_ok = max_upper < 1e-6

    section("CHECK 3 — entries in [0, 1]")
    in_unit = (A.min().item() >= -1e-6) and (A.max().item() <= 1 + 1e-6)
    print(f"  all values in [0,1] (within 1e-6) ? {in_unit}")

    section("CHECK 4 — per-row monotonicity sanity")
    # For row p, the attention pattern should put nonzero weight on positions 0..p only.
    # Count nonzero entries per row.
    nonzero_per_row = (A > 1e-6).sum(dim=-1)          # [B, H, S]
    print(f"  nonzero columns per (B,H,row), should be ≤ row_index+1:")
    for p in range(S_q):
        nz = nonzero_per_row[:, :, p].float().mean().item()
        print(f"    row {p}: mean nonzero count = {nz:.2f}  (expected ≤ {p+1})")

    section("VERDICT")
    all_ok = softmax_ok and causal_ok and in_unit
    print(f"  softmax row-sum ≈ 1 : {softmax_ok}")
    print(f"  causal mask zero    : {causal_ok}")
    print(f"  values in [0, 1]    : {in_unit}")
    print()
    if all_ok:
        print("  → self_attn.output[1] IS the post-softmax causal-masked A^(l,h)_{p,s}.")
    else:
        print("  → some check failed; A may be raw scores or something else.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
