"""Test: can nnsight `model.trace` modify a tensor that was created OUTSIDE the trace?

Allocates `arr` with torch.zeros before the `with model.trace(...)` block,
writes a proxy-derived value into one slot inside the trace, then checks
whether `arr` actually changed after the trace exits.

Three cases tried — each isolated so one failure doesn't mask the others:
  A) write a plain torch scalar into outside-arr from inside trace
  B) write the SUM of a saved proxy into outside-arr (proxy → scalar value)
  C) write an unsaved proxy expression directly into outside-arr
"""

import torch
from toy_llama import ToyLlama


def case_A_plain_scalar(model):
    arr = torch.zeros(3)
    print(f"[A] before:   arr = {arr.tolist()}")
    with model.trace("hi", remote="local"):
        arr[1] = 7.0     # plain torch op, no proxy
    print(f"[A] after :   arr = {arr.tolist()}     (expect [0, 7, 0])")


def case_B_saved_proxy_sum(model):
    arr = torch.zeros(3)
    print(f"[B] before:   arr = {arr.tolist()}")
    with model.trace("hi", remote="local"):
        embed = model.model.embed_tokens.output.save()
        # We will use embed AFTER the trace, by reading .value or just `embed`.
    s = embed.sum().item()   # post-trace: embed is now a real tensor
    arr[1] = s
    print(f"[B] after :   arr = {arr.tolist()}     (expect non-zero middle slot)")


def case_C_proxy_setitem(model):
    arr = torch.zeros(3)
    print(f"[C] before:   arr = {arr.tolist()}")
    try:
        with model.trace("hi", remote="local"):
            embed = model.model.embed_tokens.output.save()
            arr[1] = embed.sum()   # setitem with a proxy RHS into outside-allocated arr
        print(f"[C] after :   arr = {arr.tolist()}    (does the proxy get written back?)")
    except Exception as e:
        print(f"[C] raised:  {type(e).__name__}: {e}")


def case_D_list_append_proxy(model):
    items = []                 # outside-allocated plain list
    print(f"[D] before:   len(items) = {len(items)}")
    try:
        with model.trace("hi", remote="local"):
            embed = model.model.embed_tokens.output.save()
            items.append(embed.sum())          # append the proxy itself
            items.append(embed.mean())         # another proxy
        print(f"[D] after :   len(items) = {len(items)}")
        for i, x in enumerate(items):
            kind = type(x).__name__
            try:
                print(f"           items[{i}]: type={kind}, value={float(x):.6f}")
            except Exception as e:
                print(f"           items[{i}]: type={kind}, value=<{type(e).__name__}: {e}>")
    except Exception as e:
        print(f"[D] raised:  {type(e).__name__}: {e}")


if __name__ == "__main__":
    model = ToyLlama.build_nnsight_mode()
    print("=" * 60)
    case_A_plain_scalar(model)
    print("-" * 60)
    case_B_saved_proxy_sum(model)
    print("-" * 60)
    case_C_proxy_setitem(model)
    print("-" * 60)
    case_D_list_append_proxy(model)
    print("=" * 60)
