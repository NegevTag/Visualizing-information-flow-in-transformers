"""SAE feature lens for per-source residual-stream contributions.

This is the SAE analog of the logit lens used in `cosine_seem_to_self.ipynb`. There we
decode a residual-stream vector into its top *logit* predictions (last RMSNorm + unembedding).
Here we decode the same kind of vector into its top *SAE features*, and look up a human-readable
description for each feature on Neuronpedia.

Which SAE
---------
We use the **Llama Scope** 8x / 32,768-feature *residual-stream* SAEs for Llama-3.1-8B
(`release="llama_scope_lxr_8x"`, HF `fnlp/Llama3_1-8B-Base-LXR-8x`). These are the exact SAEs
Neuronpedia indexes as `llama3.1-8b/{L}-llamascope-res-32k`, so the local feature indices line up
1:1 with Neuronpedia for description lookup. The `R` (post-block residual) position matches our
`is_mlp=True` contributions, and SAE layer `L` aligns with
`LLMResidualPosition(layer=L, ..., is_mlp=True)`. (All verified from the sae_lens registry.)

Gating is JumpReLU (verified), not a hard TopK count
----------------------------------------------------
Verified by loading the SAE: `sae_lens` represents these as a **JumpReLU** SAE
(`cfg` is `JumpReLUSAEConfig`; there is a per-feature `threshold` tensor of shape `(d_sae,)`,
mean ~1.80; `cfg.k` is `None`). Per the sae_lens JumpReLU `encode` source, a feature activation
is:
    pre  = W^{enc} x + b^{enc}
    f(x) = ReLU(pre) * 1[pre > threshold]      # elementwise, per-feature learned threshold
So a feature is "active" iff its pre-activation clears its *own* threshold, and the number of
active features is **input-dependent** (~tens on a real residual; 37 at layer 31 / last pos here),
**not** a fixed K. NOTE: Llama Scope was *trained* with (Multi-)TopK (paper arXiv 2410.20526) and
released in this threshold/JumpReLU inference form -- so it *is* the TopK-trained Llama-3.1-8B SAE,
just gated by per-feature thresholds at inference rather than a hard top-k mask. (The "trained-TopK"
claim is from the paper; what is independently verified here is the loaded JumpReLU gating.)

Why we normalize the magnitude
------------------------------
Because `threshold` is calibrated for full-strength residuals, gating is *scale-dependent*: a
per-source contribution is only a fraction of a real residual, so at its native magnitude almost
nothing clears threshold (verified: the `hate` contribution fires 1 feature raw vs 43 when its
direction is rescaled to the residual norm). So before encoding we rescale each contribution's
*direction* to the real residual norm at that position (`normalize_to=`) -- the SAE analog of the
RMSNorm the logit lens applies before the unembedding: "what features would fire if this
contribution's direction were the full-strength residual." This is also robust to whether sae_lens
folds Llama Scope's `norm_scaling_factor` into the weights, since we simply feed a vector of the
magnitude a real residual would have.

CLAUDE_WRITTEN
"""

from __future__ import annotations

import json
import os
import urllib.request
from functools import lru_cache
from pathlib import Path

import torch
from sae_lens import SAE

D_MODEL = 4096  # Llama-3.1-8B hidden size; asserted against the loaded SAE.


class SAEFeatureLens:
    """Decode residual vectors into top Llama Scope SAE features + Neuronpedia descriptions.

    CLAUDE_WRITTEN
    """

    NEURONPEDIA_MODEL = "llama3.1-8b"
    RELEASE = "llama_scope_lxr_8x"

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._api_key = os.environ.get("NEURONPEDIA_API_KEY")  # optional; public GET works without
        # Persist Neuronpedia descriptions across runs (they're stable): {"model/sae_id/index": desc}.
        self._desc_cache_path = Path(__file__).resolve().parent / ".neuronpedia_desc_cache.json"
        try:
            self._desc_cache: dict[str, str] = json.loads(self._desc_cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._desc_cache = {}

    # --- SAE loading / encoding -------------------------------------------------

    @lru_cache(maxsize=8)
    def load_sae(self, layer: int) -> SAE:
        """Load (and cache) the 8x residual SAE for `layer`. CLAUDE_WRITTEN"""
        sae_id = f"l{layer}r_8x"
        loaded = SAE.from_pretrained(self.RELEASE, sae_id, device=self.device)
        sae = loaded[0] if isinstance(loaded, tuple) else loaded  # older sae_lens returns a tuple
        assert sae.cfg.d_in == D_MODEL, f"expected d_in={D_MODEL}, got {sae.cfg.d_in}"
        return sae

    def neuronpedia_sae_id(self, layer: int) -> str:
        return f"{layer}-llamascope-res-32k"

    def encode(self, vec: torch.Tensor, layer: int, normalize_to: float | None = None) -> torch.Tensor:
        """Encode a (d_model,) residual vector into SAE feature activations (n_features,).

        If `normalize_to` is given, the vector's *direction* is rescaled to that L2 norm first
        (pass the real residual norm at the position; see module docstring). CLAUDE_WRITTEN
        """
        sae = self.load_sae(layer)
        x = vec.detach().to(self.device, torch.float32)
        assert x.shape == (D_MODEL,), f"expected ({D_MODEL},), got {tuple(x.shape)}"
        if normalize_to is not None:
            norm = x.norm()
            assert norm > 0, "cannot normalize a zero vector"
            x = x / norm * normalize_to
        with torch.no_grad():
            acts = sae.encode(x.unsqueeze(0)).squeeze(0)  # (n_features,)
        return acts

    def top_features(
        self, vec: torch.Tensor, layer: int, k: int = 5, normalize_to: float | None = None
    ) -> list[tuple[int, float]]:
        """Top-k (feature_index, activation) for a residual vector. CLAUDE_WRITTEN"""
        acts = self.encode(vec, layer, normalize_to=normalize_to)
        vals, idx = torch.topk(acts, k)
        return [(int(i), float(v)) for v, i in zip(vals, idx)]

    # --- Neuronpedia descriptions -----------------------------------------------

    @lru_cache(maxsize=4096)
    def describe(self, layer: int, index: int) -> str:
        """Short human description of feature `index` from Neuronpedia. Cached in memory + on disk.

        Public GET endpoint:
        https://www.neuronpedia.org/api/feature/{model}/{L}-llamascope-res-32k/{index}
        Returns the top explanation string, or a short placeholder if none is available. CLAUDE_WRITTEN
        """
        key = f"{self.NEURONPEDIA_MODEL}/{self.neuronpedia_sae_id(layer)}/{index}"
        if key in self._desc_cache:  # disk-persisted hit (survives process restarts)
            return self._desc_cache[key]
        url = (
            f"https://www.neuronpedia.org/api/feature/"
            f"{self.NEURONPEDIA_MODEL}/{self.neuronpedia_sae_id(layer)}/{index}"
        )
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        if self._api_key:
            req.add_header("X-Api-Key", self._api_key)
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode())
        except Exception as e:  # network / 404 / rate-limit -> don't cache, don't kill the table
            return f"(fetch error: {type(e).__name__})"
        desc = self._extract_description(data)
        self._desc_cache[key] = desc  # persist stable results (incl. "(no description)")
        try:
            self._desc_cache_path.write_text(json.dumps(self._desc_cache), encoding="utf-8")
        except OSError:
            pass
        return desc

    @staticmethod
    def _extract_description(data: dict) -> str:
        """Pull the explanation text out of a Neuronpedia feature JSON. CLAUDE_WRITTEN.

        NOTE: confirm the exact field path against a real response (see check_sae_lens.py /
        the notebook debug cell). Expected shape: {"explanations": [{"description": "..."}], ...}.
        """
        explanations = data.get("explanations") or []
        if explanations and isinstance(explanations[0], dict):
            desc = explanations[0].get("description")
            if desc:
                return desc.strip()
        return "(no description)"

    def fetch_raw(self, layer: int, index: int) -> dict:
        """Raw Neuronpedia JSON for one feature -- for inspecting the field path. CLAUDE_WRITTEN"""
        url = (
            f"https://www.neuronpedia.org/api/feature/"
            f"{self.NEURONPEDIA_MODEL}/{self.neuronpedia_sae_id(layer)}/{index}"
        )
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        if self._api_key:
            req.add_header("X-Api-Key", self._api_key)
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode())
