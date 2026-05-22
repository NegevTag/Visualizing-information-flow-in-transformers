"""
Step 03 — inspect: read whatever .pt files are on disk under
probe_output/<model>/ and print a structured shape/sanity report.

Does NOT load the model. Safe to run any time, even mid-capture or after a
partial run. Tells you exactly what was captured and whether the data passes
basic sanity (attention rows sum to 1, residual norm trends, no NaNs).

Usage:
    uv run python tests/scratchpad/probe/03_inspect.py
    INFO_FLOW_MODEL=Qwen/Qwen2.5-0.5B uv run python tests/scratchpad/probe/03_inspect.py
"""


import sys
from pathlib import Path

# Allow running directly:
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from _common import OUT_DIR, MANIFEST_PATH, section, read_json  # noqa: E402


def _load(name: str) -> torch.Tensor | None:
    p = OUT_DIR / f"{name}.pt"
    if not p.exists():
        return None
    return torch.load(p, map_location="cpu", weights_only=True)


def _fmt(t: torch.Tensor) -> str:
    t_f32 = t.detach().to(torch.float32)
    return (f"shape={tuple(t.shape):<22} dtype={str(t.dtype):<14} "
            f"min={t_f32.min().item():+.3e} max={t_f32.max().item():+.3e} "
            f"mean={t_f32.mean().item():+.3e} finite={torch.isfinite(t_f32).all().item()}")


def main() -> int:
    if not OUT_DIR.exists():
        print(f"no output directory yet at {OUT_DIR}")
        print("did you run 02_capture.py?", file=sys.stderr)
        return 1

    section(f"INSPECT  {OUT_DIR}")
    run_cfg = read_json("run_config")
    model_cfg = read_json("model_config")
    if run_cfg:
        print("[run_config]")
        for k, v in run_cfg.items():
            print(f"  {k:14s}: {v}")
    if model_cfg:
        print("[model_config]")
        for k, v in model_cfg.items():
            print(f"  {k:14s}: {v}")
    if not MANIFEST_PATH.exists():
        print("no manifest.json — capture did not produce any output yet")
        return 1

    # -- Walk the manifest ---------------------------------------------------
    section("MANIFEST FILES")
    pt_files = sorted(OUT_DIR.glob("*.pt"))
    print(f"{len(pt_files)} .pt files on disk")

    # Top-level one-offs
    section("EMBEDDINGS & FINAL NORM INPUT")
    for name in ("embed_tokens.output", "model.norm.input"):
        t = _load(name)
        if t is None:
            print(f"  MISSING: {name}.pt")
        else:
            print(f"  {name:30s} {_fmt(t)}")

    # Per-layer summary
    section("PER-LAYER SUMMARY")
    L = (model_cfg or {}).get("L", None)
    # If we don't know L, infer it from filenames
    if L is None:
        layer_ids = sorted({int(p.name.split(".")[0][1:]) for p in pt_files if p.name.startswith("L") and p.name[1:3].isdigit()})
        if layer_ids:
            L = max(layer_ids) + 1
        else:
            print("no per-layer files found")
            return 0

    # Print a compact per-layer overview for a few representative layers
    layers_to_show = [0, 1]
    if L > 4:
        layers_to_show.append(L // 2)
    if L > 2:
        layers_to_show.append(L - 1)
    layers_to_show = sorted(set(layers_to_show))

    suffixes = [
        "pre_attn_resid",
        "pre_mlp_resid",
        "post_layer_out",
        "attn_sublayer_out",
        "attn_weights",
        "gate_post_silu",
        "rmsnorm_r1",
        "rmsnorm_r2",
    ]
    for li in layers_to_show:
        print(f"--- layer {li:02d} ---")
        for sfx in suffixes:
            name = f"L{li:02d}.{sfx}"
            t = _load(name)
            if t is None:
                print(f"  MISSING: {name}")
            else:
                print(f"  {sfx:18s} {_fmt(t)}")

    # -- Sanity checks ------------------------------------------------------
    section("SANITY")

    # (1) attention row sum ≈ 1 at every layer that has attn_weights
    bad_attn = []
    attn_layers = []
    for li in range(L):
        w = _load(f"L{li:02d}.attn_weights")
        if w is None:
            continue
        rs = w.to(torch.float32).sum(dim=-1)
        attn_layers.append(li)
        if not (torch.allclose(rs, torch.ones_like(rs), atol=1e-3)):
            bad_attn.append((li, rs.min().item(), rs.max().item()))
    print(f"attn_weights present on {len(attn_layers)}/{L} layers")
    if bad_attn:
        print("attn rows NOT summing to ~1:")
        for li, lo, hi in bad_attn:
            print(f"  layer {li}: min={lo:.4f}  max={hi:.4f}")
    else:
        print("all captured attn_weights rows sum to ~1.0  (good)")

    # (2) residual norm trend over layers (sanity that pre-norm transformer
    #     residuals typically grow with depth)
    norms = []
    for li in range(L):
        t = _load(f"L{li:02d}.pre_attn_resid")
        if t is None:
            norms.append(None)
        else:
            norms.append(float(t.to(torch.float32).norm(dim=-1).mean().item()))
    present = [n for n in norms if n is not None]
    print(f"mean ‖x^(l)_p‖_2 captured on {len(present)}/{L} layers")
    if present:
        head = " ".join(f"{n:.2f}" for n in norms[:6] if n is not None)
        tail = " ".join(f"{n:.2f}" for n in norms[-3:] if n is not None)
        print(f"  first 6 : {head}")
        if L > 9:
            print(f"  last 3  : {tail}")

    # (3) causality: at layer 0, attention[:, p, s>p] should be 0
    w0 = _load("L00.attn_weights")
    if w0 is not None:
        # [B, H, p, s]
        B, Hd, S, S2 = w0.shape
        assert S == S2, f"attn weights shape {w0.shape} is not square in last two dims"
        mask_upper = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)
        upper = w0.to(torch.float32)[..., mask_upper]
        print(f"layer 0 attn upper-triangular max |a|: {upper.abs().max().item():.3e} "
              f"(expect 0 for causal mask)")

    # (4) RMSNorm scale positivity
    bad_r = []
    for li in range(L):
        for k in (1, 2):
            r = _load(f"L{li:02d}.rmsnorm_r{k}")
            if r is None:
                continue
            if not (r > 0).all():
                bad_r.append((li, k))
    if bad_r:
        print("RMSNorm scale has non-positive entries at:", bad_r)
    else:
        print("all RMSNorm scales captured are strictly positive  (good)")

    print("\nDONE (inspect).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
