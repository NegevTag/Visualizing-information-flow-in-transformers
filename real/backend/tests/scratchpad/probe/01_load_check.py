"""
Step 01 — load-check: verify the model loads with the chosen device/dtype.

Cheap smoke test. Confirms HF auth, that nnsight can wrap the model, and that
`attn_implementation="eager"` is honored (required so we can later capture
attention probabilities as an explicit tensor). Prints the architectural
constants we'll rely on in the math (L, H, H_kv, d, d_ff).

Writes:
    probe_output/<model>/run_config.json
    probe_output/<model>/model_config.json

Does NOT run a forward pass. Run this first if you're not sure auth is set up
or whether the model fits in memory.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Allow running directly (`uv run python tests/scratchpad/probe/01_load_check.py`):
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from nnsight import LanguageModel  # noqa: E402

from _common import (  # noqa: E402
    MODEL_NAME,
    pick_device,
    pick_dtype,
    section,
    ensure_out_dir,
    print_banner,
    write_json,
)


def main() -> int:
    ensure_out_dir()
    banner = print_banner()

    section("LOAD MODEL (no forward pass)")
    device = pick_device()
    dtype = pick_dtype(device)
    t0 = time.time()
    try:
        model = LanguageModel(
            MODEL_NAME,
            device_map=device,
            torch_dtype=dtype,
            attn_implementation="eager",
        )
    except Exception as e:
        print(f"FAILED to load model: {type(e).__name__}: {e}", file=sys.stderr)
        print(
            "\nIf gated (Llama, Mistral, …): accept the license on HuggingFace "
            "and `huggingface-cli login`, OR override INFO_FLOW_MODEL=<non-gated>.",
            file=sys.stderr,
        )
        return 1
    load_time = time.time() - t0
    print(f"loaded in {load_time:.1f}s")

    cfg = model.config
    L = cfg.num_hidden_layers
    H = cfg.num_attention_heads
    H_KV = getattr(cfg, "num_key_value_heads", H)
    d = cfg.hidden_size
    d_ff = cfg.intermediate_size
    print(f"L={L}  H_q={H}  H_kv={H_KV}  d={d}  d_ff={d_ff}")

    # Quick architecture sanity — what we expect for any Llama-family model:
    print("\narchitecture inspection (first layer):")
    layer0 = model.model.layers[0]
    print(f"  input_layernorm        : {type(layer0.input_layernorm).__name__}")
    print(f"  post_attention_layernorm: {type(layer0.post_attention_layernorm).__name__}")
    print(f"  self_attn              : {type(layer0.self_attn).__name__}")
    print(f"  mlp                    : {type(layer0.mlp).__name__}")
    print(f"  mlp.act_fn             : {type(layer0.mlp.act_fn).__name__}")
    has_gate = hasattr(layer0.mlp, "gate_proj")
    has_up = hasattr(layer0.mlp, "up_proj")
    has_down = hasattr(layer0.mlp, "down_proj")
    print(f"  mlp has gate_proj      : {has_gate}")
    print(f"  mlp has up_proj        : {has_up}")
    print(f"  mlp has down_proj      : {has_down}")
    if not (has_gate and has_up and has_down):
        print("  WARNING: this MLP isn't SwiGLU-shaped. Frozen-gate math won't apply directly.")

    write_json("run_config", banner)
    write_json(
        "model_config",
        {"L": L, "H": H, "H_KV": H_KV, "d": d, "d_ff": d_ff,
         "load_time_sec": load_time,
         "input_layernorm_class": type(layer0.input_layernorm).__name__,
         "self_attn_class": type(layer0.self_attn).__name__,
         "mlp_class": type(layer0.mlp).__name__,
         "act_fn_class": type(layer0.mlp.act_fn).__name__,
         "is_swiglu_shaped": has_gate and has_up and has_down},
    )

    print("\nDONE (load-check).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
