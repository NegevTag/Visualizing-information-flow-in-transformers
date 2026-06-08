"""Investigation of the layer-31 MLP discrepancy.

See `layer31_mlp_investigation.md` (same directory) for hypotheses and the
experimental design. This script runs E1, E2, E3 from cached tensors only —
no model reload — and prints results we will paste back into the md.

  E1. Per-layer magnitudes of real_mlp / mine_mlp / real_attn.
  E2. bf16-metric vs fp32-metric for the rel-norm of the diff.
  E3. At L30 and L31, locate the diff by token-position and hidden-dim.
"""

from pathlib import Path

import torch

CACHE = Path(__file__).resolve().parents[2] / "src" / "info_flow" / ".cache"
REAL = CACHE / "real_2fdfffed30ff09b7.pt"
MINE = CACHE / "mine_2fdfffed30ff09b7.pt"

real_blob = torch.load(REAL, weights_only=False)
mine_blob = torch.load(MINE, weights_only=False)

real_mlp = real_blob["mlp"]
real_attn = real_blob["attn"]
mine_post_mlp = [t.sum(dim=1) for t in mine_blob["post_mlp"]]
mine_post_attn = [t.sum(dim=1) for t in mine_blob["post_attn"]]

L = len(real_mlp)
print(f"layers: {L}")
print(f"real_mlp[0] shape={tuple(real_mlp[0].shape)} dtype={real_mlp[0].dtype}")
print(f"mine post_mlp[0] shape={tuple(mine_post_mlp[0].shape)} dtype={mine_post_mlp[0].dtype}")


# ---- E1. magnitudes ------------------------------------------------------
print("\n=== E1: residual magnitudes by layer ===")
print(f"{'L':>3} {'|real_mlp|_max':>16} {'|real_mlp|_mean':>16} "
      f"{'|mine_mlp|_max':>16} {'|real_attn|_max':>16}")
for l in range(L):
    r = real_mlp[l].float()
    m = mine_post_mlp[l].float()
    a = real_attn[l].float()
    print(f"{l:>3} {r.abs().max().item():>16.4f} {r.abs().mean().item():>16.4f} "
          f"{m.abs().max().item():>16.4f} {a.abs().max().item():>16.4f}")


# ---- E2. bf16 vs fp32 metric --------------------------------------------
print("\n=== E2: bf16-metric vs fp32-metric ===")
print(f"{'L':>3} {'mlp_bf16':>12} {'mlp_fp32':>12} {'attn_bf16':>12} {'attn_fp32':>12}")
for l in range(L):
    rm, mm = real_mlp[l], mine_post_mlp[l]
    ra, ma = real_attn[l], mine_post_attn[l]
    bf_mlp = ((mm - rm).norm() / rm.norm()).item()
    bf_att = ((ma - ra).norm() / ra.norm()).item()
    f_mlp = ((mm.float() - rm.float()).norm() / rm.float().norm()).item()
    f_att = ((ma.float() - ra.float()).norm() / ra.float().norm()).item()
    print(f"{l:>3} {bf_mlp:>12.5f} {f_mlp:>12.5f} {bf_att:>12.5f} {f_att:>12.5f}")


# ---- E3. localize the diff at L30 / L31 ----------------------------------
print("\n=== E3: where the diff lives, L30 vs L31 ===")
for l in (30, 31):
    rm = real_mlp[l].float().squeeze(0)
    mm = mine_post_mlp[l].float().squeeze(0)
    diff = mm - rm
    pos_l2 = diff.norm(dim=-1)
    dim_l2 = diff.norm(dim=0)
    real_pos_l2 = rm.norm(dim=-1)
    real_dim_l2 = rm.norm(dim=0)
    print(f"\n-- layer {l} --")
    print("per-position L2 of diff (vs real):")
    for p in range(diff.shape[0]):
        print(f"  pos {p:>2}: diff={pos_l2[p].item():.4f}  real={real_pos_l2[p].item():.4f}  "
              f"ratio={pos_l2[p].item()/real_pos_l2[p].item():.4f}")
    top_dims = torch.topk(dim_l2, 5)
    print("top-5 hidden-dims by diff L2:")
    for v, i in zip(top_dims.values.tolist(), top_dims.indices.tolist()):
        print(f"  dim {i:>5}: diff_L2={v:.4f}  real_L2={real_dim_l2[i].item():.4f}  "
              f"real_abs_max={rm[:, i].abs().max().item():.4f}")
    pos_share_top1 = (pos_l2.max() ** 2 / (diff ** 2).sum()).item()
    dim_share_top10 = (torch.topk(dim_l2, 10).values.pow(2).sum() / (diff ** 2).sum()).item()
    print(f"share of squared diff in top-1 position: {pos_share_top1:.3f}")
    print(f"share of squared diff in top-10 dims:    {dim_share_top10:.3f}")
