"""Minimal repro for the two bugs blocking native `async with model.trace(remote=True)`
in nnsight 0.7.0. Prints full tracebacks for a maintainer bug report.

Bug 1: fails at __aenter__ (no network needed).
Bug 2: with frame-capture monkeypatched, fails inside async_request (one NDIF job).

Run (from real/backend/):
    PYTHONIOENCODING=utf-8 uv run \
      project_scratchpad/real/backend/src/tests/scratchpad/probe/08_async_repro.py

CLAUDE_WRITTEN
"""

import asyncio
import inspect
import os
import platform
import traceback

import nnsight
import torch
import transformers
from nnsight import CONFIG, LanguageModel

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"
PROMPT = "The Eiffel Tower is in"


def banner() -> None:
    print("=== environment ===")
    print(f"  nnsight      : {nnsight.__version__}")
    print(f"  torch        : {torch.__version__}")
    print(f"  transformers : {transformers.__version__}")
    print(f"  python       : {platform.python_version()}")
    print(f"  platform     : {platform.platform()}")
    print(f"  ndif host    : {CONFIG.API.HOST}")


async def bug1(model: LanguageModel) -> None:
    print("\n=== BUG 1: native `async with` (unpatched) ===")
    try:
        async with model.trace(PROMPT, remote=True):
            _ = model.lm_head.output.save()
    except Exception:
        traceback.print_exc()


def patch_async_frame_capture() -> None:
    """Skip `__aenter__` frames too (workaround for bug 1) so we can reach bug 2."""
    import nnsight.intervention.tracing.base as base

    def patched():
        frame = inspect.currentframe().f_back
        while frame is not None and frame.f_back is not None:
            if frame.f_back.f_code.co_name in ("__enter__", "__aenter__"):
                frame = frame.f_back
                continue
            break
        return frame.f_back if frame is not None else None

    base.get_entered_frame = patched


async def bug2(model: LanguageModel) -> None:
    print("\n=== BUG 2: native `async with` (frame capture patched) ===")
    patch_async_frame_capture()
    try:
        async with model.trace(PROMPT, remote=True):
            _ = model.lm_head.output.save()
    except Exception:
        traceback.print_exc()


async def main() -> None:
    banner()
    model = LanguageModel(MODEL_NAME)
    await bug1(model)
    await bug2(model)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    asyncio.run(main())
