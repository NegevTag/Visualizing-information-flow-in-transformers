"""Fail-fast checks for the SAE feature lens, in the order most likely to kill the project.

Run with:  uv run real/backend/src/tests/scratchpad/check_sae_lens.py
(from the repo root, inside the backend env). Read-only except for downloading SAE weights
and a few Neuronpedia GETs. CLAUDE_WRITTEN
"""

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[5]  # .../VisulaiztionInfoFlowDemo
BACKEND_SRC = REPO / "real" / "backend" / "src"
SCRATCHPAD = REPO / "project_scratchpad"
sys.path.insert(0, str(BACKEND_SRC))
sys.path.insert(0, str(SCRATCHPAD))

from api_checks.api_cache import APICache  # noqa: E402
from api_checks.position import LLMResidualPosition  # noqa: E402
from info_flow.config import Config  # noqa: E402
from sae_feature_lens import SAEFeatureLens  # noqa: E402

PROMPT = "I hate this person. I think he is so"
config = Config()
MODEL = config.info_flow_model
api_cache = APICache(hf_token=config.hf_token, cache_path=Path(config.result_cache_path))
calculator = api_cache.get_infomration_calculator(MODEL)
run = api_cache.get_full_run_results(MODEL, PROMPT)

tokens = calculator.calc_tokens(PROMPT)
LAST_POS = run.dimentions.prompt_len - 1
LAYER = run.dimentions.layers - 1
pos = LLMResidualPosition(layer=LAYER, token_position=LAST_POS, is_mlp=True)

lens = SAEFeatureLens(device="cpu")

# 1. SAE loads with the right shape (kills the project if the release/sae_id is wrong).
sae = lens.load_sae(LAYER)
print(f"[1] loaded SAE layer {LAYER}: d_in={sae.cfg.d_in}, d_sae={getattr(sae.cfg, 'd_sae', '?')}")
assert sae.cfg.d_in == 4096

# 2. Sanity on a REAL residual (the SAE was trained on these). Should give sensible nonzero acts.
real_resid = run.precise[pos].float()
resid_norm = real_resid.norm().item()
real_top = lens.top_features(real_resid, LAYER, k=5)
print(f"[2] real full residual @ ({LAYER},{LAST_POS}) norm={resid_norm:.1f} -> top5 {real_top}")
assert real_top[0][1] > 0, "real residual produced no active features -- normalization is wrong"

# 3. Neuronpedia describe: print one raw JSON to confirm the field path, then the parsed string.
top_idx = real_top[0][0]
raw = lens.fetch_raw(LAYER, top_idx)
print(f"[3] neuronpedia raw keys for feature {top_idx}: {list(raw.keys())}")
print(f"    parsed description: {lens.describe(LAYER, top_idx)!r}")

# 4. Normalization effect: raw (tiny) contribution vs normalized, for the 'hate' source token.
contribs = run.contributions[pos]  # (S, d_model)
hate_i = next(i for i, t in enumerate(tokens) if "hate" in t.lower())
c = contribs[hate_i].float()
print(f"[4] 'hate' contribution norm={c.norm():.3f} (vs residual norm {resid_norm:.1f})")
print(f"    raw        top5: {lens.top_features(c, LAYER, k=5)}")
print(f"    normalized top5: {lens.top_features(c, LAYER, k=5, normalize_to=resid_norm)}")
print("    -> expect raw to collapse toward generic/bias features; normalized to reflect content.")

print("\nAll checks ran.")
