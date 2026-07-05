"""Compile-check `calc_contribution_per_layer_per_residual` against a tiny
2-layer Llama built from scratch (no model download).

Why: catches Python-level bugs in the trace body without paying the NDIF
roundtrip and without downloading the 8B model. Uses `remote='local'` so the
recording AND execution both run on this machine — errors raise immediately
with normal tracebacks.

Run:
    uv run python tests/scratchpad/toy_compile_check.py
"""

from api_checks.model_calculator import calc_contribution_per_layer_per_residual
from tests.scratchpad.toy_llama import ToyLlama




if __name__ == "__main__":
    # NOTE: for this script to surface trace-body errors locally, you must
    # temporarily change `remote=True` -> `remote='local'` inside
    # calc_contribution_per_layer_per_residual. Otherwise it'll try to ship
    # to NDIF and you'll see MissedProviderError instead of the real cause.
    model = ToyLlama.build_nnsight_mode()
    prompt = "The cat sat on the"
    result = calc_contribution_per_layer_per_residual(model, prompt,remote = 'local')
    print(f"OK. layers returned: {len(result)}")
    for i, t1 in enumerate(result):
        t, _ = t1
        print(f"  layer {i}: shape={tuple(t.shape)}, dtype={t.dtype}")
