"""
Step 04 — server-side GPU memory probe: test the "poisoned-replica" hypothesis.

Why
---
`calc_contribution_per_layer_per_residual` OOMs on NDIF. Across three tracebacks
the *failure* state is always `17.52 GiB allocated / 17.69 GiB allowed`, but the
*pre-allocation baseline* differed:
  * run 1 reached layer 0  -> lines 62-63 had already allocated ~1.6 GiB of
    contribution stacks, so its baseline BEFORE our code was ~15.9 GiB;
  * later runs died at the very first allocation (line 62) with the baseline
    already at ~17.5 GiB.

Hypothesis: a prior trace OOM'd and its ~1.6 GiB of stacks were never freed
(PyTorch keeps the tensors alive through the server-side exception's references;
see https://discuss.pytorch.org/t/gpu-memory-not-freed-after-caught-error/24363),
so NDIF's long-lived Ray Serve replica stays "poisoned" and every later request
starts ~1.6 GiB in the hole and dies immediately.

How this tests it
-----------------
A trace that only *reads* `torch.cuda.memory_allocated()` allocates nothing large,
so it survives even on a wedged replica and reports that replica's baseline. The
number executes server-side (nnsight 0.7 compiles the `with` block into a function
that runs on the replica for remote=True) and comes back via `.save()`.

Decisive sequence (default when run as a script):
    1. probe()            -> B_fresh   (baseline right now)
    2. heavy_oom()        -> run the real calc, swallow the OOM
    3. probe()            -> B_after
Interpretation:
    B_after - B_fresh ~= +1.6 GiB  -> poisoning CONFIRMED (leaked stacks).
    B_after ~= B_fresh              -> no leak; genuine capacity limit -> go shrink.

Caveat: NDIF load-balances across replicas, so the three traces may not all land
on the same replica. They run back-to-back to maximise co-location; repeat a few
times before trusting the delta. Also run once right after a fresh reconnect to
capture a truly-clean B_fresh.

Run with (remote):
    INFO_FLOW_REMOTE=1 NDIF_API_KEY=... HF_TOKEN=... \
        uv run python tests/scratchpad/probe/04_memory_probe.py
CLAUDE_WRITTEN
"""


import sys
from pathlib import Path

# Allow running directly:
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

# A short, cheap prompt for the *probe* (we only read memory, allocate nothing big).
# The heavy run uses the real failing prompt so it actually OOMs like production.
SHORT_PROMPT = "Zephyr Holt was born in"
FAILING_PROMPT = PROMPT  # override via INFO_FLOW_PROMPT to your OOM sentence


def probe(model, prompt: str, label: str) -> dict[str, float]:
    """Minimal trace: read the replica's CUDA memory + PID and ship them back.

    Allocates nothing large, so it succeeds even when the replica is nearly full.
    `os.getpid()` runs server-side too, so it identifies WHICH replica process we
    hit — essential for distinguishing a real leak (same PID, memory grew) from
    NDIF load-balancing us onto a different, busier replica (different PID).
    """
    # NOTE: NDIF's sandbox whitelists `torch` but blocks `import os` etc., so we
    # can't read a PID for replica identity. Instead we lean on repetition: a clean
    # baseline that NEVER reads high, vs a post-error reading that STAYS high.
    with model.trace(prompt, remote=REMOTE):
        mem = torch.tensor(
            [
                float(torch.cuda.memory_allocated()),
                float(torch.cuda.memory_reserved()),
            ]
        ).save()
    allocated, reserved = (v / GIB for v in mem.tolist())
    print(f"  [{label:8s}] allocated={allocated:6.2f} GiB   reserved={reserved:6.2f} GiB")
    sys.stdout.flush()
    return {"allocated": allocated, "reserved": reserved}


def heavy_oom(model, prompt: str) -> None:
    """Run the real calculation and swallow any OOM, so we can probe again after."""
    # Imported lazily: pulls in the production math module only when needed.
    from api_checks.model_calculator import calc_contribution_per_layer_per_residual

    try:
        calc_contribution_per_layer_per_residual(model, prompt, remote=REMOTE)
        print("  heavy run SUCCEEDED (no OOM this time)")
    except Exception as e:  # noqa: BLE001 - we deliberately continue past the OOM
        print(f"  heavy run failed as expected: {type(e).__name__}: {str(e)[:100]}")
    sys.stdout.flush()


def main() -> int:
    if not REMOTE:
        print("Set INFO_FLOW_REMOTE=1 — this probe is about NDIF replica memory.", file=sys.stderr)
        return 1
    print_banner()
    configure_ndif()
    # Suppress the server-side "deepcopy of weight" LOG flood (as the notebook does).
    import nnsight
    nnsight.CONFIG.APP.REMOTE_LOGGING = False
    model = make_model()

    import time

    section("MEMORY PROBE battery: causally isolate 'error leaks memory'")
    print("Design (no replica-id available -> use repetition):")
    print("  * clean x5   : control. Each probe is itself a *successful* trace")
    print("                 (forward pass). If successful traces don't leak, this")
    print("                 must stay ~15.00 GiB and NEVER spontaneously read high.")
    print("  * heavy OOM  : run the real calc, let it error.")
    print("  * after x5   : immediately after the error.")
    print("  * delayed x3 : after a 30s wait — tests durability (GC/recycle?).\n")

    print("-- control: 5 clean probes (no error yet) --")
    cleans = [probe(model, SHORT_PROMPT, f"clean{i+1}")["allocated"] for i in range(5)]

    print("\n-- trigger the error --")
    heavy_oom(model, FAILING_PROMPT)

    print("\n-- immediately after the error --")
    afters = [probe(model, SHORT_PROMPT, f"after{i+1}")["allocated"] for i in range(5)]

    print("\n-- 30s later (durability) --")
    time.sleep(30)
    delayed = [probe(model, SHORT_PROMPT, f"delay{i+1}")["allocated"] for i in range(3)]

    section("VERDICT")
    clean_lo, clean_hi = min(cleans), max(cleans)
    print(f"  clean   : {[round(x, 2) for x in cleans]}  (range {clean_lo:.2f}-{clean_hi:.2f})")
    print(f"  after   : {[round(x, 2) for x in afters]}")
    print(f"  delayed : {[round(x, 2) for x in delayed]}")
    elevated = [x for x in afters + delayed if x - clean_hi > 0.8]
    if not elevated:
        print("\n  -> NO LEAK: post-error readings match the clean baseline.")
    elif clean_hi - clean_lo > 0.8:
        print("\n  -> BASELINE ITSELF VARIES: clean probes already bimodal — can't")
        print("     attribute the jump to the error. Investigate replica routing.")
    elif all(x - clean_hi > 0.8 for x in delayed):
        print("\n  -> CONFIRMED & DURABLE: clean baseline is stable and low, yet every")
        print("     post-error probe (incl. 30s later) reads high. The error leaks")
        print("     memory that persists on the replica.")
    else:
        print("\n  -> LEAKS BUT TRANSIENT / load-balanced: some post-error probes")
        print("     dropped back to baseline (recycled replica or different one).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
