"""
Step 05 — leak-timing bracket: how long does the post-OOM leak persist, and is it
consistent across trials?

04_memory_probe.py showed a failed remote trace leaves +2.25 GiB that is gone by
+30s. But that was a single observation with a loose bracket (~3s..30s). Here we
poll at short intervals right after each OOM to find WHEN it clears, and repeat
over several trials to see whether "self-heals" is reliable or load-dependent.

Same client/session throughout (no reconnect) — so any recovery is server-side.

Run with (remote):
    INFO_FLOW_REMOTE=1 NDIF_API_KEY=... HF_TOKEN=... PYTHONIOENCODING=utf-8 \
        uv run python tests/scratchpad/probe/05_leak_timing.py
CLAUDE_WRITTEN
"""


import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from _common import (  # noqa: E402
    PROMPT,
    REMOTE,
    section,
    print_banner,
    configure_ndif,
    make_model,
)

GIB = 2**30
SHORT_PROMPT = "Zephyr Holt was born in"
FAILING_PROMPT = PROMPT
CLEAR_THRESHOLD_GIB = 16.1  # midway between clean (15.0) and leaked (~17.25)

POLL_INTERVAL_S = 3.0
MAX_WAIT_S = 45.0
N_TRIALS = 3


def read_allocated(model, prompt: str = SHORT_PROMPT) -> float:
    """Minimal trace: return the replica's allocated GiB (allocates nothing big)."""
    with model.trace(prompt, remote=REMOTE):
        mem = torch.tensor([float(torch.cuda.memory_allocated())]).save()
    return mem.item() / GIB


def trigger_oom(model, prompt: str) -> None:
    from api_checks.model_calculator import calc_contribution_per_layer_per_residual

    try:
        calc_contribution_per_layer_per_residual(model, prompt, remote=REMOTE)
        print("    (heavy run unexpectedly SUCCEEDED — no leak to time)")
    except Exception as e:  # noqa: BLE001
        print(f"    OOM triggered: {type(e).__name__}")
    sys.stdout.flush()


def main() -> int:
    if not REMOTE:
        print("Set INFO_FLOW_REMOTE=1.", file=sys.stderr)
        return 1
    print_banner()
    configure_ndif()
    import nnsight
    nnsight.CONFIG.APP.REMOTE_LOGGING = False
    model = make_model()

    section("LEAK-TIMING BRACKET (no reconnect between reads)")
    baseline = read_allocated(model)
    print(f"clean baseline: {baseline:.2f} GiB\n")

    clear_times: list[float | None] = []
    for trial in range(N_TRIALS):
        print(f"--- trial {trial + 1}/{N_TRIALS} ---")
        trigger_oom(model, FAILING_PROMPT)
        t0 = time.time()
        cleared_at: float | None = None
        # Poll from ~0s onward; the first read is essentially "immediately after".
        while time.time() - t0 < MAX_WAIT_S:
            elapsed = time.time() - t0
            alloc = read_allocated(model)
            state = "LEAKED" if alloc > CLEAR_THRESHOLD_GIB else "clear"
            print(f"    t={elapsed:5.1f}s  allocated={alloc:6.2f} GiB  [{state}]")
            if alloc <= CLEAR_THRESHOLD_GIB:
                cleared_at = elapsed
                break
            time.sleep(POLL_INTERVAL_S)
        clear_times.append(cleared_at)
        if cleared_at is None:
            print(f"    still leaked after {MAX_WAIT_S:.0f}s")
        else:
            print(f"    -> cleared by t={cleared_at:.1f}s")
        # Ensure we start the next trial from a clean state.
        while read_allocated(model) > CLEAR_THRESHOLD_GIB and time.time() - t0 < MAX_WAIT_S + 30:
            time.sleep(POLL_INTERVAL_S)
        print()

    section("SUMMARY")
    for i, ct in enumerate(clear_times):
        print(f"  trial {i + 1}: {'never (>%.0fs)' % MAX_WAIT_S if ct is None else f'cleared by {ct:.1f}s'}")
    observed = [ct for ct in clear_times if ct is not None]
    if len(observed) == len(clear_times) and observed:
        print(f"\n  self-heals every trial; clear time in "
              f"[{min(observed):.1f}s, {max(observed):.1f}s] (poll step {POLL_INTERVAL_S:.0f}s).")
    elif observed:
        print(f"\n  INCONSISTENT: cleared in {len(observed)}/{len(clear_times)} trials.")
    else:
        print(f"\n  never observed clearing within {MAX_WAIT_S:.0f}s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
