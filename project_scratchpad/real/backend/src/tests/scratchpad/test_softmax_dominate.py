"""Test what softmax([SOFTMAX_DOMNATE, 0]) returns."""

import torch

SOFTMAX_DOMNATE = float(10000000000000000000)

# Test 1: softmax with the giant value (bfloat16)
logits = torch.tensor([SOFTMAX_DOMNATE, 100000], dtype=torch.bfloat16)
result = torch.softmax(logits, dim=-1)
print(f"softmax([{SOFTMAX_DOMNATE}, 0]) = {result}")
print(f"  → first element: {result[0]}")
print(f"  → second element: {result[1]}")
print(f"  → sum: {result.sum()}")

# Test 2: What about in bfloat16