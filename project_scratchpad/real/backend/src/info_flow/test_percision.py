from api_checks.model import ModelInformationCalculatorF32
from info_flow.config import Config
from info_flow.ex6_better_percision_key_in_mat_f32 import FullRunResults
from info_flow.ex4_models_norms_percisions import ModelInformationCalculatorRealNorms
from info_flow.ex5_better_percision_key_in_mat_calc import ModelInformationCalculatorNotPerKey
from info_flow.ex6_better_percision_key_in_mat_f32 import ModelInformationCalculatorF32Ex6
import torch
from info_flow.precision_visualization import PrecisionResults, compare_percision, pretty_print_precision
from tests.scratchpad.toy_llama_atten_one_back import ToyLllamaAttenOneBack

NORM_FLOOR = 1
BENCHMARK_PROMPT = "The cat sat on the mat, and then afterword, he decided that"
PRECINTILES = (98, 99)

# "first_test"
#"real_norms_calc"
# not_per_key_calc
#f32_calc
#f32_and_mat
#f32_and_mat_logits
#f32_and_mat_logits_no_trailing_space
def calc_percision():
    config = Config()
    information_calculator = ModelInformationCalculatorF32Ex6(model_name=config.info_flow_model, hf_token=config.hf_token)
    information = information_calculator.calc("The cat sat on the mat but he didnt")
    # information.dump("f32_and_mat_logits_no_trailing_space")
    percision = percision_test(information)
    pretty_print_precision(percision)

def percision_test(result: FullRunResults) -> PrecisionResults:  # (Layer, (max_norm_rel),(mean_norm_rel) p98_elm,p_99_elm,max_norm)
    result = result.get_f64()

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


if __name__ == '__main__':
    # calc_percision()
    
    config = Config()
    calculator = ModelInformationCalculatorF32(config.info_flow_model, config.hf_token)
    information = calculator.calc(prompt="My favorite food is pizza and")
    precision = percision_test(information)
    pretty_print_precision(precision)
    print(information.logits.shape)
    print(calculator.tokens_probabilities_from_logits(information.logits[-1]))


    # config = Config()
    # calculator = ModelInformationCalculatorF32(model_name=config.info_flow_model, hf_token=config.hf_token)
    # information = FullRunResults.load("f32_and_mat_logits_no_trailing_space")
    # tokens = calculator.calc_tokens(BENCHMARK_PROMPT)
    # print(tokens)
    # index = -1
    # print(tokens[index])
    # print(calculator.tokens_probabilities_from_logits(information.logits[-1]))
    
    # old_method = percision_test(FullRunResults.load('f32_calc'))
    # new_method = percision_test(FullRunResults.load('f32_and_mat'))
    # print( (old_method[:][1]-new_method[:][1]).sum())
    # compare_percision(blue=old_method,red=new_method,blue_name='32',red_name='32_and_mat')
    
    
    # config = Config()
    # information_calculator = ModelInformationCalculatorNotPerKey(model_name=config.info_flow_model, hf_token=config.hf_token)
    # information = FullRunResults.load("first_test")
    # calc_f32 = calc_contribution_per_layer_per_residual(model= information_calculator.model,prompt= BENCHMARK_PROMPT)
    # information.contributions.post_mlp_contribution = calc_f32[0].float()
    # information.contributions.post_attention_contribution = calc_f32[1].float()
    # information.precise.mlp_residual = information.precise.mlp_residual.float()
    # information.precise.attention_residual = information.precise.attention_residual.float()
    # information.dump("f32_calc")
    # f32_precision = percision_test(information)
    # pretty_print_precision(f32_precision)