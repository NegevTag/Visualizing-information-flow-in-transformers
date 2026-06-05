# pyright: ignore-all-errors
# mypy: ignore-errors
"""
Trivial-by-inspection tests for `calc_post_rmsnorm2`.

Contract under test:
    calc_post_rmsnorm2(layer, per_res) -> Tensor
        per_res:  (query, key, d_model)  contributions of each key to each query
                  (pre-rmsnorm2 residual stream, decomposed over source keys)
        returns:  (query, key, d_model)  post-rmsnorm2 contribution of each key to each query
        Property: per_res.sum(dim=1) is the actual pre-rmsnorm2 residual at each query,
                  and output.sum(dim=1) is the actual post-rmsnorm2 residual at each query.

Each test below uses a case where the answer is obvious without ever computing rmsnorm itself —
we only rely on linearity, zero-preservation, permutation-equivariance, and weight-scaling.

NOTE: this assumes `calc_post_rmsnorm2` has been lifted from its nested scope in
`calc_contribution_per_layer_per_residual` to module-level in `info_flow.ex2_calc_mlp_as_well`.
"""

import math
import einops as ein
from types import SimpleNamespace


import torch
from icecream import ic


def _calculate_mlp_contribution(rms_eps, rms_weight, contribution):  # contribution: (position,source,d_model)
    contribution_f32 = contribution.to(torch.float32)
    residual_vectors_reconstructed = contribution_f32.sum(dim=1)  # (prompt_len,d_model)
    ms = residual_vectors_reconstructed.pow(2).mean(dim=-1)
    rms_factor = torch.rsqrt(ms + rms_eps)  # (prompt_len)
    rms_factor_reshaped = ein.rearrange(rms_factor, "pl -> pl 1 1")  # (prompt_len,1 ,1)
    return rms_weight * (rms_factor_reshaped * contribution_f32).to(residual_vectors_reconstructed.dtype)


def calc_post_rmsnorm2(layer, post_attention_contibutions) -> torch.Tensor:  # per_residual_contribution  (position,source,d_model)
    rms_weight = layer.post_attention_layernorm.weight  # (d_model)
    rms_eps = 1e-5
    return _calculate_mlp_contribution(rms_eps, rms_weight, post_attention_contibutions)


def make_fake_layer(weight: torch.Tensor) -> SimpleNamespace:
    """Minimal stand-in for a Llama layer — only `.post_attention_layernorm.weight` is read."""
    return SimpleNamespace(post_attention_layernorm=SimpleNamespace(weight=weight))


def make_per_res(prompt_len: int, d_model: int, seed: int = 0) -> torch.Tensor:
    """Causal lower-triangular contributions with nonzero rows (so sum-over-keys ≠ 0)."""
    g = torch.Generator().manual_seed(seed)
    per_res = torch.zeros(prompt_len, prompt_len, d_model)
    for q in range(prompt_len):
        for k in range(q + 1):
            per_res[q, k] = torch.randn(d_model, generator=g)
    return per_res


# ---------------------------------------------------------------------------
# 1. shape preservation
# ---------------------------------------------------------------------------
def test_shape_preserved():
    P, D = 5, 8
    per_res = make_per_res(P, D)
    layer = make_fake_layer(torch.ones(D))
    out = calc_post_rmsnorm2(layer, per_res)
    assert out.shape == per_res.shape, f"expected {per_res.shape}, got {out.shape}"


# ---------------------------------------------------------------------------
# 2. per-key zero stays zero
#    if per_res[q, k*, :] = 0 then out[q, k*, :] must be 0,
#    because the operation is a (query-dependent) scalar * weight applied per-key
# ---------------------------------------------------------------------------
def test_per_key_zero_stays_zero():
    P, D = 4, 6
    per_res = make_per_res(P, D)
    per_res[2, 1, :] = 0.0  # wipe one (q,k) slot
    layer = make_fake_layer(torch.ones(D))
    out = calc_post_rmsnorm2(layer, per_res)
    assert torch.equal(out[2, 1], torch.zeros(D)), f"expected zero, got {out[2, 1]}"


# ---------------------------------------------------------------------------
# 3. scale invariance under positive input scaling
#    scaling per_res by c>0 must leave the output UNCHANGED — this is what
#    normalization is for. Numerator gets a factor c, denominator (norm of the
#    sum) gets the same c, they cancel. Holds exactly in the limit eps→0;
#    with eps>0 the cancellation is approximate, so tolerance is looser.
# ---------------------------------------------------------------------------
def test_positive_scale_invariance():
    P, D = 4, 6
    per_res = make_per_res(P, D)
    layer = make_fake_layer(torch.ones(D))
    out_1 = calc_post_rmsnorm2(layer, per_res)
    out_c = calc_post_rmsnorm2(layer, 3.0 * per_res)
    assert torch.allclose(out_c, out_1, atol=1e-4, rtol=1e-4), f"rmsnorm should be scale-invariant; max|Δ|={(out_c - out_1).abs().max()}"


# ---------------------------------------------------------------------------
# 4. weight elementwise scaling
#    doubling the rmsnorm weight must double the output (weight is a final elementwise mult)
# ---------------------------------------------------------------------------
def test_weight_scaling():
    P, D = 4, 6
    per_res = make_per_res(P, D)
    w = torch.linspace(0.5, 1.5, D)
    out_w = calc_post_rmsnorm2(make_fake_layer(w), per_res)
    out_2w = calc_post_rmsnorm2(make_fake_layer(2.0 * w), per_res)
    assert torch.allclose(out_2w, 2.0 * out_w, atol=1e-5, rtol=1e-5), f"max|Δ|={(out_2w - 2.0 * out_w).abs().max()}"


# ---------------------------------------------------------------------------
# 5. permutation equivariance on the key axis
#    sum-over-keys is permutation-invariant, so the normalising scalar at each query
#    is unchanged; therefore permuting the key axis must permute the output identically
# ---------------------------------------------------------------------------
def test_key_axis_permutation_equivariance():
    P, D = 4, 6
    per_res = make_per_res(P, D)
    perm = torch.tensor([3, 0, 2, 1])  # any permutation of range(P)
    layer = make_fake_layer(torch.ones(D))
    out = calc_post_rmsnorm2(layer, per_res)
    out_permuted_input = calc_post_rmsnorm2(layer, per_res[:, perm, :])
    assert torch.allclose(out_permuted_input, out[:, perm, :], atol=1e-6, rtol=1e-6), f"max|Δ|={(out_permuted_input - out[:, perm, :]).abs().max()}"


# ---------------------------------------------------------------------------
# 6. per-query independence
#    changing per_res at query q1 must not change output at query q2 (rmsnorm is positionwise)
# ---------------------------------------------------------------------------
def test_per_query_independent():
    P, D = 4, 6
    per_res_a = make_per_res(P, D, seed=0)
    per_res_b = per_res_a.clone()
    per_res_b[0] = torch.randn(P, D, generator=torch.Generator().manual_seed(42))  # mutate only q=0
    layer = make_fake_layer(torch.ones(D))
    out_a = calc_post_rmsnorm2(layer, per_res_a)
    out_b = calc_post_rmsnorm2(layer, per_res_b)
    assert torch.allclose(out_a[1:], out_b[1:], atol=1e-6, rtol=1e-6), "queries q>=1 should be untouched by changes at q=0"


if __name__ == "__main__":
    for fn in [
        test_shape_preserved,
        test_per_key_zero_stays_zero,
        test_positive_scale_invariance,
        test_weight_scaling,
        test_key_axis_permutation_equivariance,
        test_per_query_independent,
    ]:
        fn()
        print(f"OK  {fn.__name__}")
    print("all trivial-property checks passed")
