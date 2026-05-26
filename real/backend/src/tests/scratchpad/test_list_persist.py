"""Does a list survive `with model.trace(...)` when populated with
torch.zeros tensors that were modified via proxy-RHS setitem?

Compares against the "append proxy directly" case (which works — case D in
test_outside_init.py). The real `calc_contribution_per_layer_per_residual`
uses the first pattern, and the user reports the list comes out empty.
"""

import torch
from toy_llama import ToyLlama


def case_E_append_real_tensor(model):
    arr = []
    with model.trace("hi", remote="local"):
        for l in range(2):
            t = torch.zeros(3)
            arr.append(t)
        print(f"[E] inside trace : len(arr) = {len(arr)}, id(arr) = {id(arr)}")
    print(f"[E] outside trace: len(arr) = {len(arr)}, id(arr) = {id(arr)}")


def case_F_append_zeros_with_setitem(model):
    arr = []
    with model.trace("hi", remote="local"):
        embed = model.model.embed_tokens.output.save()
        for l in range(2):
            t = torch.zeros(3)
            t[0] = embed.sum()        # setitem with proxy RHS
            arr.append(t)
        print(f"[F] inside trace : len(arr) = {len(arr)}, id(arr) = {id(arr)}")
    print(f"[F] outside trace: len(arr) = {len(arr)}, id(arr) = {id(arr)}")
    for i, t in enumerate(arr):
        print(f"           arr[{i}] = {t.tolist()}")


def case_G_via_function(model):
    """Mirror the real code shape — list created inside a function, returned out."""
    def inner():
        arr = []
        with model.trace("hi", remote="local"):
            embed = model.model.embed_tokens.output.save()
            for l in range(2):
                t = torch.zeros(3)
                t[0] = embed.sum()
                arr.append(t)
            print(f"[G] inside trace : len(arr) = {len(arr)}")
        print(f"[G] after with   : len(arr) = {len(arr)}")
        return arr

    result = inner()
    print(f"[G] after return : len(result) = {len(result)}")


if __name__ == "__main__":
    model = ToyLlama.build_nnsight_mode()
    print("=" * 60)
    case_E_append_real_tensor(model)
    print("-" * 60)
    case_F_append_zeros_with_setitem(model)
    print("-" * 60)
    case_G_via_function(model)
    print("=" * 60)
