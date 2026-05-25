# pyright: ignore-all-errors
# mypy: ignore-errors
"""
Toy shape check for the down_proj step in ex2_calc_mlp_as_well.py.

Shapes (from the log):
    mlp_proj_weight: (4096, 14336)            # down_proj.weight  (d_model, d_mlp)
    pointwise_mlp  : (14336, 6, 6)            # (d_mlp, query, key)

Goal: try `pointwise_mlp.T @ mlp_proj_weight` and inspect the resulting shape.
Note: in torch, `.T` on a >2D tensor reverses all dims, so
    pointwise_mlp.T -> (6, 6, 14336)
"""

import torch

mlp_proj_weight = torch.randn(4096, 14336)
pointwise_mlp = torch.randn(14336, 6, 6)

print(f"mlp_proj_weight.shape = {mlp_proj_weight.shape}")
print(f"pointwise_mlp.shape   = {pointwise_mlp.shape}")
print(f"pointwise_mlp.T.shape = {pointwise_mlp.T.shape}")

out = pointwise_mlp.T @ mlp_proj_weight.T  # expected to fail or broadcast unexpectedly
print(f"out.shape = {out.shape}")
