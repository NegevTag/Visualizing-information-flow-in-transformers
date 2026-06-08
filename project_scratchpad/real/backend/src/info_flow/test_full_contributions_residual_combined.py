"""Compute and print the precision table for the cached bf16 reconstruction.

The metric and per-column meaning live in precision_metric.py.
Cache-only — run test_full_contributions.py first to populate .cache files.
"""

from pathlib import Path
import hashlib

import torch
from info_flow.config import Config
from info_flow.precision_metric import compute_precision_table, print_table

config = Config()
prompt = "The cat sat on the mat, and then afterword, he decided that "
model_name = config.info_flow_model

key = hashlib.sha1(f"{model_name}||{prompt}".encode()).hexdigest()[:16]
cache_dir = Path(__file__).resolve().parent / ".cache"
real = torch.load(cache_dir / f"real_{key}.pt", weights_only=False)
mine = torch.load(cache_dir / f"mine_{key}.pt", weights_only=False)


def get_cached_contributions():
    """Return (post_mlp_contributions, post_attn_contributions) from disk cache."""
    return mine["post_mlp"], mine["post_attn"]


table = compute_precision_table(
    get_cached_contributions,
    real_mlp=real["mlp"],
    real_attn=real["attn"],
    pos_denom_floor=1.0,
    elem_denom_floor=1.0,
)

print_table(table, label="bf16 reconstruction (cached)")

# Save for later comparison runs.
out_path = cache_dir / "precision_table_bf16.pt"
torch.save(table, out_path)
print(f"\n[saved] {out_path.name}  shape={tuple(table.shape)}")
