import typing

from info_flow.config import Config
from info_flow.ex3_calc_full_contributions import FullRunResults
import torch
from rich.console import Console
from rich.table import Table
from info_flow.ex3_calc_full_contributions import ModelInformationCalculator, _get_model

NORM_FLOOR = 1
BENCHMARK_PROMPT = "The cat sat on the mat, and then afterword, he decided that "
PRECINTILES = (98, 99)

PrecisionResults = typing.NewType("PrecisionResults", torch.Tensor)


def calc_percision():
    config = Config()
    # information_calculator = ModelInformationCalculator(model_name=config.info_flow_model, hf_token=config.hf_token)
    # information = information_calculator.calc(BENCHMARK_PROMPT)
    information = FullRunResults.load("first_test")
    percision = percision_test(information)
    pretty_print_precision(percision)

def percision_test(result: FullRunResults) -> PrecisionResults:  # (Layer, (max_norm_rel),(mean_norm_rel) p98_elm,p_99_elm,max_norm)

    diff_mlp = result.contributions.post_mlp_contribution.sum(dim=2) - result.precise.mlp_residual
    diff_attention = result.contributions.post_attention_contribution.sum(dim=2) - result.precise.attention_residual

    max_residual_norms_clamped = torch.maximum(result.precise.mlp_residual.norm(dim=-1), result.precise.attention_residual.norm(dim=-1)).clamp(min=1)

    rel_diff_mlp_norms = diff_mlp.norm(dim=-1) / max_residual_norms_clamped
    rel_diff_attention_norms = diff_attention.norm(dim=-1) / max_residual_norms_clamped

    max_diff_norm_rel = torch.maximum(rel_diff_mlp_norms.amax(dim=-1), rel_diff_attention_norms.amax(dim=-1))
    mean_diff_norms_rel = torch.stack([rel_diff_mlp_norms.mean(dim=-1), rel_diff_attention_norms.mean(dim=-1)]).mean(dim=0)


    mlp_residual_sizes = result.precise.mlp_residual.abs()
    attention_residual_sizes = result.precise.attention_residual.abs()
    
    relative_diff_mlp = diff_mlp.abs() / torch.clamp(mlp_residual_sizes, float(NORM_FLOOR))
    relative_diff_attention = diff_attention.abs() / torch.clamp(attention_residual_sizes, float(NORM_FLOOR))

    relative_diff_full = torch.stack([relative_diff_attention, relative_diff_mlp], dim=-1)

    realtive_dif_p98 = torch.quantile(relative_diff_full.flatten(-3).float(), PRECINTILES[0]/100, dim=-1)
    realtive_dif_p99 = torch.quantile(relative_diff_full.flatten(-3).float(), PRECINTILES[1]/100, dim=-1)
    relaitive_diff_max = torch.amax(relative_diff_full.flatten(-3).float(),dim=-1,keepdim=False)
    print(relaitive_diff_max.shape)

    return torch.stack([max_diff_norm_rel, mean_diff_norms_rel, realtive_dif_p98, realtive_dif_p99, relaitive_diff_max],dim=-1)


def pretty_print_precision(results: PrecisionResults) -> None:
    """Pretty-print a (Layer, 5) precision tensor with columns:
    max_norm_rel, mean_norm_rel, p98_elm, p99_elm, max_elm."""

    from rich.table import Column
    cols = ["layer", "max_norm_rel", "mean_norm_rel", "p98_elm", "p99_elm", "max_elm"]
    table = Table(*[Column(c, header_style="none") for c in cols], show_lines=False)
    for layer_idx, row in enumerate(results):
        table.add_row(str(layer_idx), *[f"{v.item():.5f}" for v in row])
    Console().print(table)
if __name__ == '__main__':
    calc_percision()