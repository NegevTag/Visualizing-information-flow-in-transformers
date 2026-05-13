"""
Step 00b — does `output_attentions=True` give us attention weights on NDIF?

Single hypothesis test. Runs four isolated traces:

  1. trace WITHOUT output_attentions=True; save self_attn.output (baseline)
  2. trace WITH    output_attentions=True; save self_attn.output
  3. trace WITHOUT output_attentions=True; save layer.output (baseline)
  4. trace WITH    output_attentions=True; save layer.output

The relevant comparisons:
  - if (2) is a tuple with a 4-D attention tensor → HF honors it on NDIF;
    my earlier claim "NDIF drops it" is wrong.
  - if (2) is a Tensor identical to (1) → flag is silently dropped/ignored.
  - if (2) is a tuple but slot[1] is None → kernel is SDPA/FA, returns None.

This is the test I should have run before writing the Phase-1 caveat.

Run:
    PYTHONIOENCODING=utf-8 PYTHONUTF8=1 \
    INFO_FLOW_REMOTE=1 INFO_FLOW_MODEL=meta-llama/Meta-Llama-3.1-8B \
    uv run python tests/scratchpad/probe/00b_output_attentions.py
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


def describe(val) -> str:
    """One-line description of what nnsight returned."""
    if val is None:
        return "None"
    if isinstance(val, torch.Tensor):
        return f"Tensor{tuple(val.shape)} dtype={val.dtype}"
    if isinstance(val, tuple):
        parts = []
        for i, x in enumerate(val):
            if x is None:
                parts.append(f"[{i}]=None")
            elif isinstance(x, torch.Tensor):
                parts.append(f"[{i}]=Tensor{tuple(x.shape)} {x.dtype}")
            else:
                parts.append(f"[{i}]={type(x).__name__}")
        return f"tuple(len={len(val)}, {', '.join(parts)})"
    return f"{type(val).__name__}: {val!r}"


def run_one(model, label: str, target: str, **trace_kwargs) -> None:
    """Run one isolated trace with the given target ('self_attn' or 'layer')
    and the given trace kwargs (notably output_attentions=True/False)."""
    layer0 = model.model.layers[0]
    if target == "self_attn":
        getter = lambda: layer0.self_attn.output
    elif target == "layer":
        getter = lambda: layer0.output
    else:
        raise ValueError(target)
    try:
        with model.trace(PROMPT, remote=REMOTE, **trace_kwargs):
            proxy = getter().save()
        val = materialise(proxy)
        print(f"  {label:60s} → {describe(val)}")
    except Exception as e:
        msg = str(e).splitlines()[0] if str(e) else ""
        print(f"  {label:60s} ✗ {type(e).__name__}: {msg[:120]}")


def main() -> int:
    ensure_out_dir()
    print_banner()
    configure_ndif()

    section("LOAD MODEL")
    model = make_model()
    print("model loaded")

    section("OUTPUT_ATTENTIONS COMPARISON — 4 isolated traces")
    print("  what we expect to see:")
    print("    if HF honors the flag remotely → tuple with 4-D attn weights")
    print("    if silently dropped            → tensor (same as baseline)")
    print("    if SDPA fall-back              → tuple, but slot[1] is None")
    print()
    run_one(model, "(1) self_attn.output | output_attentions=False",
            target="self_attn")
    run_one(model, "(2) self_attn.output | output_attentions=True",
            target="self_attn", output_attentions=True)
    run_one(model, "(3) layer.output     | output_attentions=False",
            target="layer")
    run_one(model, "(4) layer.output     | output_attentions=True",
            target="layer", output_attentions=True)

    print()
    print("DONE (output_attentions probe).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
