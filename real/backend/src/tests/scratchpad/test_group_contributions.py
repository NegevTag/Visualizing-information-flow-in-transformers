"""Black-box unit tests for ``api_checks.utils.group_contributions``.

Spec under test (agreed with supervisor, not read from the implementation):
- Signature: ``group_contributions(token_group_idnetifier: list[Any], contributions) -> Contributions``.
- The ``source`` axis (dim 2) of each ``(layer, position, source, d_model)`` tensor is
  collapsed: sources sharing a label are summed together. Both
  ``post_mlp_contribution`` and ``post_attention_contribution`` are grouped identically.
- Output group order follows *first appearance* of each label in the list.
- ``len(labels) != n_sources`` raises.

Tests are intentionally written against the spec only, via an independent reference
implementation, so they stay valid regardless of how the function is coded internally.

CLAUDE_WRITTEN
"""

from typing import Any

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from api_checks.full_run_result import Contributions
from api_checks.utils import get_tokenpacks_indexed_contributions


# --- helpers ---------------------------------------------------------------


def make_contributions(mlp: torch.Tensor, attn: torch.Tensor) -> Contributions:
    return Contributions(post_mlp_contribution=mlp, post_attention_contribution=attn)


def reference_group(labels: list[Any], tensor: torch.Tensor) -> torch.Tensor:
    """Reference: sum the source axis (dim 2) per label, groups in first-appearance order.

    ``tensor`` is ``(layer, position, source, d_model)``; returns ``(layer, position, group, d_model)``.
    """
    order: list[Any] = []
    for lab in labels:  # first-appearance order, equality-based (labels are ``Any``)
        if lab not in order:
            order.append(lab)
    grouped = [
        tensor[:, :, [i for i, lab in enumerate(labels) if lab == g], :].sum(dim=2)
        for g in order
    ]
    return torch.stack(grouped, dim=2)


def rand_contributions(layers: int, positions: int, sources: int, d_model: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    mlp = torch.randn(layers, positions, sources, d_model, generator=g)
    attn = torch.randn(layers, positions, sources, d_model, generator=g)
    return mlp, attn


# --- correctness -----------------------------------------------------------


def test_basic_grouping_matches_reference():
    # 5 sources -> 2 groups; labels chosen so group "a" = {0,1,3}, "b" = {2,4}.
    labels = ["a", "a", "b", "a", "b"]
    mlp, attn = rand_contributions(layers=2, positions=3, sources=5, d_model=4)
    out = get_tokenpacks_indexed_contributions(labels, make_contributions(mlp, attn)).tokenpacks_contributions

    assert out.post_mlp_contribution.shape == (2, 3, 2, 4)
    assert out.post_attention_contribution.shape == (2, 3, 2, 4)
    torch.testing.assert_close(out.post_mlp_contribution, reference_group(labels, mlp))
    torch.testing.assert_close(out.post_attention_contribution, reference_group(labels, attn))


def test_first_appearance_order():
    # "b" appears before "a", so output group 0 must be "b" (sources 0,3), group 1 "a" (1,2).
    labels = ["b", "a", "a", "b"]
    mlp, attn = rand_contributions(layers=1, positions=2, sources=4, d_model=3, seed=1)
    out = get_tokenpacks_indexed_contributions(labels, make_contributions(mlp, attn)).tokenpacks_contributions

    expected_b = mlp[:, :, [0, 3], :].sum(dim=2)
    expected_a = mlp[:, :, [1, 2], :].sum(dim=2)
    torch.testing.assert_close(out.post_mlp_contribution[:, :, 0, :], expected_b)
    torch.testing.assert_close(out.post_mlp_contribution[:, :, 1, :], expected_a)


def test_all_distinct_labels_is_identity():
    # Every source its own group, in order -> tensors returned unchanged.
    sources = 6
    labels = list(range(sources))
    mlp, attn = rand_contributions(layers=2, positions=2, sources=sources, d_model=3, seed=2)
    out = get_tokenpacks_indexed_contributions(labels, make_contributions(mlp, attn)).tokenpacks_contributions

    torch.testing.assert_close(out.post_mlp_contribution, mlp)
    torch.testing.assert_close(out.post_attention_contribution, attn)


def test_single_group_sums_all_sources():
    labels = ["only"] * 5
    mlp, attn = rand_contributions(layers=1, positions=3, sources=5, d_model=4, seed=3)
    out = get_tokenpacks_indexed_contributions(labels, make_contributions(mlp, attn)).tokenpacks_contributions

    assert out.post_mlp_contribution.shape == (1, 3, 1, 4)
    torch.testing.assert_close(out.post_mlp_contribution[:, :, 0, :], mlp.sum(dim=2))
    torch.testing.assert_close(out.post_attention_contribution[:, :, 0, :], attn.sum(dim=2))


def test_non_int_labels_supported():
    # Labels are ``Any`` -> strings must work exactly like any other hashable label.
    labels = ["subject", "verb", "subject"]
    mlp, attn = rand_contributions(layers=1, positions=2, sources=3, d_model=2, seed=4)
    out = get_tokenpacks_indexed_contributions(labels, make_contributions(mlp, attn)).tokenpacks_contributions

    assert out.post_mlp_contribution.shape[2] == 2  # two distinct labels
    torch.testing.assert_close(out.post_mlp_contribution, reference_group(labels, mlp))


def test_grouping_conserves_total_mass():
    # Summing all output groups must equal summing all input sources (no source dropped/double-counted).
    labels = ["x", "y", "x", "z", "y", "x"]
    mlp, attn = rand_contributions(layers=2, positions=2, sources=6, d_model=3, seed=5)
    out = get_tokenpacks_indexed_contributions(labels, make_contributions(mlp, attn)).tokenpacks_contributions

    torch.testing.assert_close(out.post_mlp_contribution.sum(dim=2), mlp.sum(dim=2))
    torch.testing.assert_close(out.post_attention_contribution.sum(dim=2), attn.sum(dim=2))


# --- error handling --------------------------------------------------------


@pytest.mark.parametrize("labels_len", [4, 6])  # one too short, one too long vs 5 sources
def test_length_mismatch_raises(labels_len: int):
    mlp, attn = rand_contributions(layers=1, positions=1, sources=5, d_model=2, seed=6)
    labels = ["a"] * labels_len
    with pytest.raises(Exception):
        get_tokenpacks_indexed_contributions(labels, make_contributions(mlp, attn)).tokenpacks_contributions


# --- property-based --------------------------------------------------------


@settings(max_examples=50, deadline=None)
@given(
    labels=st.lists(st.sampled_from(["a", "b", "c", "d"]), min_size=1, max_size=8),
    layers=st.integers(1, 3),
    positions=st.integers(1, 3),
    d_model=st.integers(1, 4),
    seed=st.integers(0, 10_000),
)
def test_property_matches_reference(labels, layers, positions, d_model, seed):
    sources = len(labels)
    mlp, attn = rand_contributions(layers, positions, sources, d_model, seed=seed)
    out = get_tokenpacks_indexed_contributions(labels, make_contributions(mlp, attn)).tokenpacks_contributions

    n_groups = len(dict.fromkeys(labels))  # unique count, first-appearance order
    assert out.post_mlp_contribution.shape == (layers, positions, n_groups, d_model)
    torch.testing.assert_close(out.post_mlp_contribution, reference_group(labels, mlp))
    torch.testing.assert_close(out.post_attention_contribution, reference_group(labels, attn))
