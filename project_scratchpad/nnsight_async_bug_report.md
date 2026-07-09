# Bug report: native `async with model.trace(..., remote=True)` is broken in nnsight 0.7.0

**Summary.** The async tracing entry point (`async with model.trace(...)`) is
unusable in nnsight 0.7.0. It fails with two independent bugs. The *synchronous*
blocking path (`with model.trace(..., remote=True)`) with the exact same body
works perfectly against NDIF — so both bugs are specific to the `async` entry,
not to remote execution or the intervention itself.

## Environment

| | |
|---|---|
| nnsight | 0.7.0 |
| torch | 2.12.0+cpu |
| transformers | 5.8.1 |
| python | 3.14.3 |
| platform | Windows-10-10.0.19045-SP0 |
| NDIF host | https://api.ndif.us |
| model | meta-llama/Meta-Llama-3.1-8B (NDIF-hosted) |

## Minimal reproduction

```python
import asyncio
from nnsight import LanguageModel

model = LanguageModel("meta-llama/Meta-Llama-3.1-8B")

async def main():
    async with model.trace("The Eiffel Tower is in", remote=True):
        logits = model.lm_head.output.save()
    print(logits)

asyncio.run(main())
```

(NDIF API key configured via `CONFIG.set_default_api_key(...)`.)

---

## Bug 1 — `WithBlockNotFoundError`: frame capture doesn't skip `__aenter__`

The unpatched repro above raises immediately, before any network activity:

```
File ".../nnsight/intervention/tracing/base.py", line 731, in __aenter__
    return self.__enter__()
File ".../nnsight/intervention/tracing/base.py", line 640, in __enter__
    self.capture(frame=get_entered_frame())
File ".../nnsight/intervention/tracing/tracer.py", line 390, in capture
    super().capture(frame)
File ".../nnsight/intervention/tracing/base.py", line 351, in capture
    start_line, source_lines, node = self.parse(source_lines, start_line)
File ".../nnsight/intervention/tracing/base.py", line 449, in parse
    raise WithBlockNotFoundError(message)
nnsight.intervention.tracing.base.WithBlockNotFoundError: With block not found at line 5
We looked here:
async def __aenter__(self):
    self.asynchronous = True
    return self.__enter__() <--- HERE
```

**Root cause.** `Tracer.__aenter__` (base.py:727) delegates to `__enter__`, which
calls `get_entered_frame()` to locate the user's frame by walking *past*
`__enter__` frames. But `get_entered_frame` (intervention/tracing/util.py:341)
only skips frames named `__enter__`:

```python
while frame is not None and frame.f_back is not None:
    back = frame.f_back
    if back.f_code.co_name == "__enter__":   # <-- does not include "__aenter__"
        frame = back
        continue
    break
return frame.f_back if frame is not None else None
```

Through the async path the stack has an extra `__aenter__` frame
(`user frame → __aenter__ → __enter__ → get_entered_frame`), which is not
skipped, so the helper returns the `__aenter__` frame and `parse()` tries to find
the user's `with` block in `__aenter__`'s source.

**Candidate fix.** Include `__aenter__` in the skip set:

```python
if back.f_code.co_name in ("__enter__", "__aenter__"):
```

This is backward compatible (a plain `with` never has an `__aenter__` frame).

---

## Bug 2 — `AttributeError: 'RemoteInterleavingTracer' object has no attribute 'model'`

Applying the Bug-1 fix (monkeypatching `get_entered_frame` to also skip
`__aenter__`) lets capture succeed, but the trace then fails during compile on
context exit. One real NDIF trace produces:

```
File ".../nnsight/intervention/tracing/base.py", line 735, in __aexit__
    await self.__exit__(exc_type, exc_val, exc_tb)
File ".../nnsight/intervention/backends/remote.py", line 942, in async_request
    data, headers = self.request(tracer)
File ".../nnsight/intervention/backends/remote.py", line 526, in request
    interventions = super().__call__(tracer)
File ".../nnsight/intervention/backends/base.py", line 56, in __call__
    tracer.compile()
File ".../nnsight/intervention/tracing/tracer.py", line 414, in compile
    if self.model._default_mediators:
       ^^^^^^^^^^
AttributeError: 'RemoteInterleavingTracer' object has no attribute 'model'
```

**Observation.** `compile()` (tracer.py:405) expects `self.model` (set in the
tracer `__init__` at tracer.py:366 and restored in `__setstate__` at
tracer.py:685). The tracer object reaching `compile()` via the async path lacks
that attribute. The **blocking** path exercises the same `request → base.__call__
→ tracer.compile()` chain with an identical trace body and does **not** hit this
— so something about how the async path constructs/serializes the
`RemoteInterleavingTracer` drops `.model`. (I did not diagnose further; flagging
for you.)

---

## What works (control)

The synchronous blocking path with the identical body runs fine, and multiple
such traces launched concurrently via `asyncio.to_thread` overlap correctly on
NDIF (verified: 4 concurrent jobs, ~3.9x speedup over sequential, all logits
finite):

```python
with model.trace("The Eiffel Tower is in", remote=True):
    logits = model.lm_head.output.save()   # works
```

So the ask is specifically: is `async with model.trace(remote=True)` intended to
be supported in 0.7.0, and if so, are Bugs 1 & 2 known? Happy to test a patch.
