"""Compare top-10 next-token predictions between the model's cached `logits`
(`original_logi`) and the re-derived `new_logits` (from summed final-layer
contributions -> calc_logits) across several *fresh* prompts.

Both quantities come from the same contribution decomposition; this is a
self-consistency check that the float32 `calc_logits` path reproduces the
in-trace bf16 logits at the level of the top-10 token *list* (order included).

Run: uv run project_scratchpad/compare_top10_new_prompts.py
CLAUDE_WRITTEN
"""

from pathlib import Path

from api_checks.api_cache import APICache
from info_flow.config import Config

# Fresh-ish prompts (distinct from "First test, lets see") to force real runs.
PROMPTS = [
    "The capital of France is",
    "Water is composed of hydrogen and",
    "The opposite of hot is",
    "In 1969, the first humans landed on the",
    "The mitochondria is the powerhouse of the",
    "Roses are red, violets are",
    "The square root of sixteen is",
    "She opened the door and saw a",
    "Python is a popular programming",
    "The sun rises in the",
]

TOP_K = 10


def main() -> None:
    config = Config()
    api_cache = APICache(hf_token=config.hf_token, cache_path=Path(config.result_cache_path))
    calculator = api_cache.get_infomration_calculator(config.info_flow_model)

    all_match = True
    for i, prompt in enumerate(PROMPTS):
        full = api_cache.get_full_run_results(config.info_flow_model, prompt)
        original_logi = full.logits  # (p_len, vocab) bf16 in-trace logits
        last_mlp_output = full.contributions.post_mlp_contribution[-1].sum(dim=1)  # (p_len, d_model)
        new_logits = calculator.calc_logits(last_mlp_output)  # (p_len, vocab) float32 path

        orig_top = list(calculator.calc_top_probabilities_from_logits(original_logi[-1], TOP_K).keys())
        new_top = list(calculator.calc_top_probabilities_from_logits(new_logits[-1], TOP_K).keys())

        ordered_match = orig_top == new_top
        set_match = set(orig_top) == set(new_top)
        all_match &= ordered_match

        status = "OK " if ordered_match else ("SET" if set_match else "DIFF")
        print(f"\n[{i:2d}] {status}  {prompt!r}")
        if not ordered_match:
            print(f"     orig: {orig_top}")
            print(f"     new : {new_top}")
        else:
            print(f"     top10: {orig_top}")

    print("\n" + ("ALL PROMPTS: top-10 lists match (order included)." if all_match else "MISMATCH found — see above."))


if __name__ == "__main__":
    main()
