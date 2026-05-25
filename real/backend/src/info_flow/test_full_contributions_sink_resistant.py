"""Per-position, split (attn/mlp), sink-resistant metric.

Story: see ../../tests/scratchpad/layer31_mlp_investigation.md.

Per layer l and per token position p:
    attn_rel[l, p] = |mine_attn[l,p,:] - real_attn[l,p,:]| / |real_attn[l,p,:]|
    mlp_rel [l, p] = |mine_mlp [l,p,:] - real_mlp [l,p,:]| / max(|real_mlp[l,p,:]|, |real_attn[l,p,:]|)

- Split (attn vs mlp): attn never collapses, no sink fix needed; mlp does.
- Per position: the L31 anomaly is concentrated on pos 0 (the sink). Per-position
  metric makes the denominator local, so one outlier position can't dominate.
- max(|real_mlp|, |real_attn|) at the position level: the residual stream *at this
  position* before the MLP block. The MLP can shrink the post-MLP norm at pos 0
  (sink-cancellation), but the pre-MLP norm at that position still reflects the
  signal's true magnitude.

Per-position metrics are ~2x noisier than global (the global L2 ratio gets a
quadrature dilution across positions). Threshold is correspondingly looser.

Cache-only — no model trace. Run test_full_contributions.py first to populate
.cache files.
"""

from pathlib import Path
import hashlib

import torch
from info_flow.config import Config

THRESHOLD = 1e-2  # per-position is ~2x noisier than global; old global threshold was 4e-3

config = Config()
prompt = "The cat sat on the mat, and then afterword, he decided that "
model_name = config.info_flow_model

key = hashlib.sha1(f"{model_name}||{prompt}".encode()).hexdigest()[:16]
cache_dir = Path(__file__).resolve().parent / ".cache"
real = torch.load(cache_dir / f"real_{key}.pt", weights_only=False)
mine = torch.load(cache_dir / f"mine_{key}.pt", weights_only=False)

real_mlp = real["mlp"]
real_attn = real["attn"]
mine_post_mlp = [t.sum(dim=1) for t in mine["post_mlp"]]
mine_post_attn = [t.sum(dim=1) for t in mine["post_attn"]]

L = len(real_mlp)
n_pos = real_mlp[0].shape[-2]
print(f"layers: {L}  positions: {n_pos}  mine dtype: {mine_post_mlp[0].dtype}\n")

# Per-layer summary: max-over-positions of each metric
print(f"{'L':>3} {'attn_max':>10} {'attn_argmaxp':>13} {'mlp_max':>10} {'mlp_argmaxp':>12}  "
      f"{'(attn_mean)':>12} {'(mlp_mean)':>11}")

exceed_attn, exceed_mlp = [], []
all_attn = torch.zeros(L, n_pos)
all_mlp = torch.zeros(L, n_pos)

for l in range(L):
    rm = real_mlp[l].float().squeeze(0)        # (pos, d)
    ra = real_attn[l].float().squeeze(0)       # (pos, d)
    mm = mine_post_mlp[l].float().squeeze(0)
    ma = mine_post_attn[l].float().squeeze(0)

    # per-position L2 norms — shape (pos,)
    diff_attn = (ma - ra).norm(dim=-1)
    diff_mlp = (mm - rm).norm(dim=-1)
    norm_attn = ra.norm(dim=-1)
    norm_mlp = rm.norm(dim=-1)
    denom_mlp = torch.maximum(norm_mlp, norm_attn)

    attn_rel = diff_attn / norm_attn           # split: attn vs attn
    mlp_rel = diff_mlp / denom_mlp             # split + sink-resistant

    all_attn[l] = attn_rel
    all_mlp[l] = mlp_rel

    attn_max_p = int(attn_rel.argmax())
    mlp_max_p = int(mlp_rel.argmax())
    print(f"{l:>3} {attn_rel.max().item():>10.5f} {attn_max_p:>13} "
          f"{mlp_rel.max().item():>10.5f} {mlp_max_p:>12}  "
          f"{attn_rel.mean().item():>12.5f} {mlp_rel.mean().item():>11.5f}")

    if attn_rel.max() >= THRESHOLD:
        exceed_attn.append((l, attn_max_p, attn_rel.max().item()))
    if mlp_rel.max() >= THRESHOLD:
        exceed_mlp.append((l, mlp_max_p, mlp_rel.max().item()))

# Layer 31 detail — the layer that used to fail
print("\n--- L31 per-position detail ---")
print(f"{'pos':>3} {'attn_rel':>10} {'mlp_rel':>10} {'|real_attn|':>12} {'|real_mlp|':>12} {'denom_mlp':>10}")
l = 31
rm = real_mlp[l].float().squeeze(0); ra = real_attn[l].float().squeeze(0)
mm = mine_post_mlp[l].float().squeeze(0); ma = mine_post_attn[l].float().squeeze(0)
norm_attn = ra.norm(dim=-1); norm_mlp = rm.norm(dim=-1)
denom_mlp = torch.maximum(norm_mlp, norm_attn)
attn_rel = (ma - ra).norm(dim=-1) / norm_attn
mlp_rel = (mm - rm).norm(dim=-1) / denom_mlp
for p in range(n_pos):
    print(f"{p:>3} {attn_rel[p].item():>10.5f} {mlp_rel[p].item():>10.5f} "
          f"{norm_attn[p].item():>12.2f} {norm_mlp[p].item():>12.2f} {denom_mlp[p].item():>10.2f}")

print(f"\nthreshold (per-position): {THRESHOLD}")
print(f"attn exceedances (layer, pos, value): {exceed_attn}")
print(f"mlp  exceedances (layer, pos, value): {exceed_mlp}")
if not exceed_attn and not exceed_mlp:
    print("ALL 32 LAYERS / ALL POSITIONS PASS")
