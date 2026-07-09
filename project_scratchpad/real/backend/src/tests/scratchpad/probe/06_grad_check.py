"""
Step 06 — is autograd even active server-side? (Would `no_grad` save memory?)

The user doesn't need backward passes. Disabling grad only helps if the remote
forward currently RUNS with grad enabled and the weights require grad — then every
matmul in the per-layer math stashes inputs "for backward", inflating the transient
peak. If NDIF already runs inference in no_grad/inference_mode, there's nothing to
save. This reads both flags server-side and returns them.

Run: INFO_FLOW_REMOTE=1 NDIF_API_KEY=... HF_TOKEN=... PYTHONIOENCODING=utf-8 \
        uv run python tests/scratchpad/probe/06_grad_check.py
CLAUDE_WRITTEN
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from _common import REMOTE, section, print_banner, configure_ndif, make_model  # noqa: E402


def main() -> int:
    if not REMOTE:
        print("Set INFO_FLOW_REMOTE=1.", file=sys.stderr)
        return 1
    print_banner()
    configure_ndif()
    import nnsight
    nnsight.CONFIG.APP.REMOTE_LOGGING = False
    model = make_model()

    section("GRAD STATE (server-side)")
    with model.trace("The cat sat", remote=REMOTE):
        w = model.model.layers[0].self_attn.v_proj.weight
        flags = torch.tensor(
            [
                float(torch.is_grad_enabled()),        # 1.0 => autograd tracking ON
                float(w.requires_grad),                # 1.0 => weights carry grad
            ]
        ).save()
    grad_enabled, w_requires_grad = (bool(x) for x in flags.tolist())
    print(f"  torch.is_grad_enabled()      : {grad_enabled}")
    print(f"  v_proj.weight.requires_grad  : {w_requires_grad}")

    section("VERDICT")
    if grad_enabled and w_requires_grad:
        print("  Grad IS active -> wrapping the compute in torch.no_grad() (or")
        print("  requires_grad_(False)) would drop the saved-for-backward buffers.")
    elif grad_enabled and not w_requires_grad:
        print("  Grad enabled but weights don't require grad. Your own .float() copies")
        print("  start grad-free, so matmuls off them won't build a graph. Minimal win,")
        print("  but no_grad is still a cheap safety net.")
    else:
        print("  Grad already OFF server-side -> no_grad saves nothing here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
