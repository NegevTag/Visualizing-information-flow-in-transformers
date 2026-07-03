"""Rigorously measure how deep the top-k agreement between `original_logi`
(bf16 in-trace) and `new_logits` (float32 calc_logits path) actually goes.

For each prompt (all cached now -> no remote), rank the vocab by each logit
vector and report, at the LAST position:
  - ordered_k : largest k s.t. the ordered top-k id lists are identical
  - set_k     : largest k s.t. the top-k id *sets* are identical
Compared by token id (not decoded string) to avoid decode collisions.

Run: uv run project_scratchpad/prove_topk_agreement.py
CLAUDE_WRITTEN
"""

from pathlib import Path

import torch
from api_checks.api_cache import ModelAPICache
from info_flow.config import Config

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

MAXK = 20


def agreement_depth(a_ids: torch.Tensor, b_ids: torch.Tensor, maxk: int) -> tuple[int, int]:
    """Largest k with ordered-equal prefixes, and largest k with set-equal prefixes."""
    ordered_k = 0
    for k in range(1, maxk + 1):
        if torch.equal(a_ids[:k], b_ids[:k]):
            ordered_k = k
        else:
            break
    set_k = 0
    for k in range(1, maxk + 1):
        if set(a_ids[:k].tolist()) == set(b_ids[:k].tolist()):
            set_k = k
    return ordered_k, set_k


def main() -> None:
    config = Config()
    api_cache = ModelAPICache(hf_token=config.hf_token, cache_path=Path(config.result_cache_path))
    calculator = api_cache.get_infomration_calculator(config.info_flow_model)

    print(f"{'ordK':>4} {'setK':>4} {'top1':>5}  prompt")
    top1_hits = 0
    top8_set_all = True
    for prompt in PROMPTS:
        full = api_cache.get_full_run_results(config.info_flow_model, prompt)
        orig = full.logits[-1].float()  # bf16 -> f32 just for argsort
        last_mlp_output = full.contributions.post_mlp_contribution[-1].sum(dim=1)
        new = calculator.calc_logits(last_mlp_output)[-1]

        orig_ids = orig.argsort(descending=True)
        new_ids = new.argsort(descending=True)

        ordered_k, set_k = agreement_depth(orig_ids, new_ids, MAXK)
        top1 = bool(orig_ids[0] == new_ids[0])
        top1_hits += top1
        top8_set_all &= set_k >= 8
        print(f"{ordered_k:>4} {set_k:>4} {str(top1):>5}  {prompt!r}")

    print()
    print(f"top-1 agreement: {top1_hits}/{len(PROMPTS)}")
    print(f"top-8 SET agreement on ALL prompts: {top8_set_all}")


if __name__ == "__main__":
    main()
