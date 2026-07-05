"""Generate a random `FullRunResults` for a given prompt length in tokens.

Useful for testing / prototyping downstream code (frontend serialization,
API caching, visualization) without paying for a real model forward pass.

The random result is *shape-faithful* to what `ModelInformationCalculatorF32.calc`
produces (see `real/backend/src/api_checks/model_calculator.py`):

    contributions.post_mlp_contribution        (layers, position, source, d_model)
    contributions.post_attention_contribution  (layers, position, source, d_model)
    precise.mlp_residual                        (layers, position, d_model)
    precise.attention_residual                  (layers, position, d_model)

We also respect *causality*: a source token at position `s > q` cannot
contribute to the residual at query position `q`, so those entries are zeroed
(matching the attention causal mask in a real run).

CLAUDE_WRITTEN
"""

import sys
from pathlib import Path

import torch

# Make `api_checks` importable (same trick the backend uses internally).
BACKEND_SRC = Path(__file__).resolve().parent.parent / "real" / "backend" / "src"
sys.path.insert(0, str(BACKEND_SRC))

from api_checks.full_run_result import (  # noqa: E402
    Contributions,
    FullRunResults,
    ResidualStream,
    ResultsDimentions,
)


def random_full_run_result(
    prompt_len: int,
    *,
    layers: int = 32,  # Llama-3.1-8B
    d_model: int = 4096,  # Llama-3.1-8B
    causal: bool = True,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    seed: int | None = None,
) -> FullRunResults:
    """Return a `FullRunResults` filled with random values.

    Args:
        prompt_len: number of tokens in the (fake) prompt.
        layers: number of transformer layers.
        d_model: residual stream width.
        causal: if True, zero out contributions where source > query position.
        dtype / device: tensor dtype and device.
        seed: optional RNG seed for reproducibility.
    """
    assert prompt_len > 0, "prompt_len must be positive"
    assert layers > 0 and d_model > 0

    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)

    def contrib() -> torch.Tensor:  # (layers, position, source, d_model)
        t = torch.randn(
            layers, prompt_len, prompt_len, d_model,
            generator=gen, device=device, dtype=dtype,
        )
        if causal:
            # keep only source <= query:  mask[q, s] = 1 iff s <= q
            mask = torch.tril(torch.ones(prompt_len, prompt_len, device=device, dtype=dtype))
            t = t * mask[None, :, :, None]
        return t

    def residual() -> torch.Tensor:  # (layers, position, d_model)
        return torch.randn(
            layers, prompt_len, d_model,
            generator=gen, device=device, dtype=dtype,
        )

    contributions = Contributions(
        post_mlp_contribution=contrib(),
        post_attention_contribution=contrib(),
    )
    precise = ResidualStream(
        mlp_residual=residual(),
        attention_residual=residual(),
    )
    dimentions = ResultsDimentions(layers=layers, prompt_len=prompt_len, d_model=d_model)

    return FullRunResults(contributions=contributions, precise=precise, dimentions=dimentions)


if __name__ == "__main__":
    # Quick smoke test: small dims so it's instant, assert shapes + causality.
    r = random_full_run_result(prompt_len=5, layers=3, d_model=8, seed=0)

    c = r.contributions.post_mlp_contribution
    assert c.shape == (3, 5, 5, 8), c.shape
    assert r.contributions.post_attention_contribution.shape == (3, 5, 5, 8)
    assert r.precise.mlp_residual.shape == (3, 5, 8)
    assert r.precise.attention_residual.shape == (3, 5, 8)
    assert (r.dimentions.layers, r.dimentions.prompt_len, r.dimentions.d_model) == (3, 5, 8)

    # Causality: at query position q=1, source s=3 (>1) must be zero.
    assert torch.all(c[:, 1, 3, :] == 0), "causal mask not applied"
    assert torch.any(c[:, 3, 1, :] != 0), "lower-triangular entries should be nonzero"

    print("OK: shapes and causality verified ->", tuple(c.shape))
