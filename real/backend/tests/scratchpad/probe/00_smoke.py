"""
Step 00 — smoke probe: which `.save()` targets actually return a value?

We hit MissedProviderError on the first failing capture and then everything
downstream in the *same* trace cascades to `ValueError: ... outside of
interleaving`. The earlier multi-save smoke run could only tell us that the
first failing capture is `layers[i].output`; the cascade hid everything else.

This version runs ONE save per trace, in its own trace block, so each is
isolated. Slower (one NDIF round trip per probe target), but definitive.

Run:
    PYTHONIOENCODING=utf-8 PYTHONUTF8=1 \
    INFO_FLOW_REMOTE=1 INFO_FLOW_MODEL=meta-llama/Meta-Llama-3.1-8B \
    uv run python tests/scratchpad/probe/00_smoke.py
"""

from __future__ import annotations

import sys
import traceback
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
    unwrap_first,
)


# Each candidate is a label and a callable that, given the model, returns the
# nnsight proxy to call `.save()` on inside the trace.
CANDIDATES: list[tuple[str, callable]] = [
    ("embed_tokens.output",        lambda m: m.model.embed_tokens.output),
    ("norm.input",                 lambda m: m.model.norm.input),
    ("L0.input_layernorm.input",   lambda m: m.model.layers[0].input_layernorm.input),
    ("L1.input_layernorm.input",   lambda m: m.model.layers[1].input_layernorm.input),
    ("L0.post_attn_norm.input",    lambda m: m.model.layers[0].post_attention_layernorm.input),
    ("L0.layer.output",            lambda m: m.model.layers[0].output),
    ("Llast.output",               lambda m: m.model.layers[-1].output),
    ("L0.mlp.act_fn.output",       lambda m: m.model.layers[0].mlp.act_fn.output),
    ("L0.mlp.gate_proj.output",    lambda m: m.model.layers[0].mlp.gate_proj.output),
    ("L0.mlp.up_proj.output",      lambda m: m.model.layers[0].mlp.up_proj.output),
    ("L0.mlp.down_proj.output",    lambda m: m.model.layers[0].mlp.down_proj.output),
    ("L0.self_attn.q_proj.output", lambda m: m.model.layers[0].self_attn.q_proj.output),
    ("L0.self_attn.k_proj.output", lambda m: m.model.layers[0].self_attn.k_proj.output),
    ("L0.self_attn.v_proj.output", lambda m: m.model.layers[0].self_attn.v_proj.output),
    ("L0.self_attn.o_proj.output", lambda m: m.model.layers[0].self_attn.o_proj.output),
    ("L0.self_attn.output",        lambda m: m.model.layers[0].self_attn.output),
    ("L0.mlp.output",              lambda m: m.model.layers[0].mlp.output),
]


def try_one(model, label: str, getter) -> tuple[str, str]:
    """Run one tiny trace with just this single save; return (status, detail)."""
    try:
        with model.trace(PROMPT, remote=REMOTE):
            proxy = getter(model).save()
        val = unwrap_first(materialise(proxy))
        if isinstance(val, torch.Tensor):
            return ("OK", f"Tensor{tuple(val.shape)} dtype={val.dtype}")
        if isinstance(val, tuple):
            inner = ", ".join(
                f"Tensor{tuple(t.shape)}" if isinstance(t, torch.Tensor) else type(t).__name__
                for t in val
            )
            return ("OK", f"tuple({inner})")
        return ("OK", f"{type(val).__name__}: {val!r}")
    except Exception as e:
        # Extract a short signature of the failure
        msg = str(e).strip().splitlines()[0] if str(e) else ""
        # Truncate cruft
        return ("FAIL", f"{type(e).__name__}: {msg[:160]}")


def main() -> int:
    ensure_out_dir()
    print_banner()
    configure_ndif()

    section("LOAD MODEL")
    model = make_model()
    print("model loaded")

    section(f"ISOLATED PROBE — {len(CANDIDATES)} candidates, one trace each")
    rows: list[tuple[str, str, str]] = []
    for label, getter in CANDIDATES:
        status, detail = try_one(model, label, getter)
        rows.append((label, status, detail))
        marker = "✓" if status == "OK" else "✗"
        # Pad label so columns line up; truncate detail to one line.
        print(f"  {marker} {label:38s} {status:4s}  {detail}")
        sys.stdout.flush()

    # Summary
    section("SUMMARY")
    ok = [r for r in rows if r[1] == "OK"]
    fail = [r for r in rows if r[1] == "FAIL"]
    print(f"  {len(ok)}/{len(rows)} captures succeeded")
    print("  OK   :", ", ".join(r[0] for r in ok) or "(none)")
    print("  FAIL :", ", ".join(r[0] for r in fail) or "(none)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
