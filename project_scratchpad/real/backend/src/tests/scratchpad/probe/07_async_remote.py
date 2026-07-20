"""Smoke test: are NDIF remote traces genuinely runnable concurrently via asyncio?

Background
----------
nnsight 0.7.0 *appears* to support `async with model.trace(..., remote=True)`
(Tracer.__aenter__ sets asynchronous=True -> RemoteBackend.async_request). But
its frame-capture helper `get_entered_frame` only walks past `__enter__` frames,
NOT the extra `__aenter__` frame the async path introduces, so the native
`async with` raises WithBlockNotFoundError. Verified by running it (see RECORD).

This script tests two ways to get concurrent remote traces that "look async":

  A) PATCHED native async: monkeypatch get_entered_frame to also skip
     `__aenter__`, then use `async with model.trace(..., remote=True)`. Truly
     single-threaded asyncio (one event loop, N in-flight WebSockets).

  B) THREAD-based async: run the ordinary *blocking* `with model.trace(...)`
     inside `asyncio.to_thread`. Reuses the battle-tested blocking code path;
     concurrency comes from threads (network I/O releases the GIL). This is the
     robust fallback and what we'd recommend if (A) is fragile.

Proof of concurrency
--------------------
Each coroutine records submit/return timestamps against a shared clock. If jobs
overlap, max-in-flight > 1 and gather wall-time << sum of per-job durations.

Run (from real/backend/):
    PYTHONIOENCODING=utf-8 uv run \
      project_scratchpad/real/backend/src/tests/scratchpad/probe/07_async_remote.py

Needs NDIF_API_KEY already in nnsight CONFIG (it is on this machine).

CLAUDE_WRITTEN
"""

import asyncio
import inspect
import os
import time

import torch
from nnsight import CONFIG, LanguageModel

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"  # NDIF-hosted; Llama-3.2 is not.

PROMPTS = [
    "The capital of France is",
    "Water is made of hydrogen and",
    "The opposite of hot is",
    "Two plus two equals",
]

t0 = time.perf_counter()


def now() -> float:
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Approach A: patch get_entered_frame so native `async with` can find the block.
# ---------------------------------------------------------------------------
def patch_async_frame_capture() -> None:
    """Make get_entered_frame skip `__aenter__` frames too (not just `__enter__`).

    Root cause: nnsight walks the stack past a chain of `__enter__` frames to
    reach the user's frame, but the async entry adds one `__aenter__` frame on
    top of `__enter__`, which the original loop does not skip -> it returns the
    `__aenter__` frame and parses the wrong source. We add `__aenter__` to the
    skip set. Backward compatible: plain `with` never has an `__aenter__` frame.
    """
    import nnsight.intervention.tracing.base as base

    def patched() -> object:
        frame = inspect.currentframe().f_back  # the __enter__ that called us
        while frame is not None and frame.f_back is not None:
            if frame.f_back.f_code.co_name in ("__enter__", "__aenter__"):
                frame = frame.f_back
                continue
            break
        return frame.f_back if frame is not None else None

    base.get_entered_frame = patched


def decode_next(model: LanguageModel, logits) -> tuple[str, tuple, bool]:
    val = getattr(logits, "value", logits)
    next_id = int(val[0, -1].argmax().item())
    tok = model.tokenizer.decode([next_id])
    finite = bool(torch.isfinite(val[0, -1].float()).all().item())
    return tok, tuple(val.shape), finite


async def run_native(model: LanguageModel, idx: int, prompt: str) -> dict:
    """Approach A: single-threaded async via patched `async with`."""
    start = now()
    async with model.trace(prompt, remote=True):
        logits = model.lm_head.output.save()
    end = now()
    tok, shape, finite = decode_next(model, logits)
    return dict(idx=idx, prompt=prompt, start=start, end=end, dur=end - start,
                next_tok=tok, logits_shape=shape, finite=finite)


async def run_thread(model: LanguageModel, idx: int, prompt: str) -> dict:
    """Approach B: blocking trace off-loaded to a worker thread."""
    start = now()

    def blocking() -> object:
        with model.trace(prompt, remote=True):
            logits = model.lm_head.output.save()
        return logits

    logits = await asyncio.to_thread(blocking)
    end = now()
    tok, shape, finite = decode_next(model, logits)
    return dict(idx=idx, prompt=prompt, start=start, end=end, dur=end - start,
                next_tok=tok, logits_shape=shape, finite=finite)


def summarize(label: str, results: list[dict], wall: float, n: int) -> bool:
    print(f"\n=== {label}: per-prompt results (s from origin) ===")
    for r in sorted(results, key=lambda r: r["idx"]):
        print(f"[{r['idx']}] {r['start']:6.2f} -> {r['end']:6.2f} "
              f"(dur {r['dur']:5.2f})  next={r['next_tok']!r:12s} "
              f"shape={r['logits_shape']} finite={r['finite']}  | {r['prompt']!r}")

    seq_equiv = sum(r["dur"] for r in results)
    events = sorted([(r["start"], +1) for r in results]
                    + [(r["end"], -1) for r in results])
    cur = peak = 0
    for _, d in events:
        cur += d
        peak = max(peak, cur)

    print(f"  wall-clock (gather)          : {wall:6.2f} s")
    print(f"  sum of durations (seq equiv) : {seq_equiv:6.2f} s")
    print(f"  speedup vs sequential        : {seq_equiv / wall:6.2f}x")
    print(f"  max concurrent in-flight     : {peak} / {n}")
    ok = all(r["finite"] for r in results) and peak >= 2
    print(f"  RESULT: {'PASS (valid + overlapped)' if ok else 'FAIL'}")
    return ok


async def gather_run(coro_fn, model) -> tuple[list[dict], float]:
    w = now()
    results = await asyncio.gather(*(coro_fn(model, i, p)
                                     for i, p in enumerate(PROMPTS)))
    return results, now() - w


async def main() -> None:
    key = CONFIG.API.APIKEY
    assert key, "No NDIF API key in CONFIG."
    print(f"NDIF host: {CONFIG.API.HOST}  key_len={len(key)}")
    print(f"Model: {MODEL_NAME}   prompts: {len(PROMPTS)}")

    model = LanguageModel(MODEL_NAME)  # meta model; weights live on NDIF.
    n = len(PROMPTS)
    verdicts = {}

    # --- Approach A: patched native async ---
    print("\n" + "#" * 70 + "\n# A) patched native `async with` (single-thread asyncio)\n" + "#" * 70)
    patch_async_frame_capture()
    try:
        res_a, wall_a = await gather_run(run_native, model)
        verdicts["A native"] = summarize("A native", res_a, wall_a, n)
    except Exception as e:
        print(f"  A FAILED to run: {type(e).__name__}: {e}")
        verdicts["A native"] = False

    # --- Approach B: thread-based async ---
    print("\n" + "#" * 70 + "\n# B) blocking trace via asyncio.to_thread (thread concurrency)\n" + "#" * 70)
    try:
        res_b, wall_b = await gather_run(run_thread, model)
        verdicts["B thread"] = summarize("B thread", res_b, wall_b, n)
    except Exception as e:
        print(f"  B FAILED to run: {type(e).__name__}: {e}")
        verdicts["B thread"] = False

    print("\n" + "=" * 70)
    for k, v in verdicts.items():
        print(f"  {k:10s}: {'PASS' if v else 'FAIL'}")
    print("=" * 70)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    asyncio.run(main())
