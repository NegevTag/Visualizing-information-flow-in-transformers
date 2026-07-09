# pyright: ignore-all-errors
# mypy: ignore-errors
from enum import Enum
import math
from typing import OrderedDict

import nnsight
from pathlib import Path
from pydantic_settings import SettingsConfigDict
from api_checks.full_run_result import Contributions, FullRunResults, ResidualStream, ResultsDimentions
from torch import Tensor
import torch.nn as nn


from pydantic import BaseModel
import torch  # noqa: E402
import einops as ein
import heapq
from api_checks.model_parmeters import ModelParameters
from api_checks.position import LLMResidualPosition


def calc_contribution_per_layer_per_residual(model: nnsight.LanguageModel, prompt: str, remote: bool | str = True):  # ->(layer,position,source,d_model), (layer,position,source,d_model)
    LAYERS_NUM = len(model.model.layers)
    D_MODEL = model.model.config.hidden_size
    D_V = model.model.config.head_dim
    heads_ratio = model.model.config.num_attention_heads // model.model.config.num_key_value_heads

    with model.trace(prompt, remote=remote):

        def allclose(t1, t2, *, atol=(1e-2) * 2, rtol=(1e-2) * 2):
            return torch.allclose(t1, t2, atol=atol, rtol=rtol)

        def _calculate_mlp_contribution(rms_eps, rms_weight, contribution):  # contribution: (position,source,d_model)
            contribution_f32 = contribution.to(torch.float32)
            residual_vectors_reconstructed = contribution_f32.sum(dim=1)  # (prompt_len,d_model)
            ms = residual_vectors_reconstructed.pow(2).mean(dim=-1)
            rms_factor = torch.rsqrt(ms + rms_eps)  # (prompt_len)
            rms_factor_reshaped = ein.rearrange(rms_factor, "pl -> pl 1 1")  # (prompt_len,1 ,1)
            return rms_weight * (rms_factor_reshaped * contribution_f32).to(contribution.dtype)

        def caclulate_post_rmsnorm1_contribution(layer, post_mlp_contribution) -> torch.Tensor:  # post_mlp_layer_contribution: (position,source,d_model)
            rms_weight = layer.input_layernorm.weight.float()
            rms_eps = model.model.config.rms_norm_eps
            return _calculate_mlp_contribution(rms_eps, rms_weight, post_mlp_contribution)

        def calc_post_rmsnorm2(layer, post_attention_contibutions) -> torch.Tensor:  # per_residual_contribution  (position,source,d_model)
            rms_weight = layer.post_attention_layernorm.weight.float()  # (d_model)
            rms_eps = model.model.config.rms_norm_eps
            return _calculate_mlp_contribution(rms_eps, rms_weight, post_attention_contibutions)

        embed = model.model.embed_tokens.output[0]  # (seq_len, d_model)
        PROMPT_LEN = len(embed)

        # ouput init
        device, dtype = embed.device, torch.float32
        post_mlp_contribution = torch.zeros((LAYERS_NUM + 1, PROMPT_LEN, PROMPT_LEN, D_MODEL), device=device, dtype=dtype).save()  # (layer+1 (zero layer means nothing),position,source,d_model)
        post_attention_contribution = torch.zeros((LAYERS_NUM, PROMPT_LEN, PROMPT_LEN, D_MODEL), device=device, dtype=dtype).save()  # (layer,position,source,d_model)

        real_attention_residual = torch.zeros(LAYERS_NUM, PROMPT_LEN, D_MODEL, device=device, dtype=dtype).save()  # (layer,positon,d_model) for percision calculation
        real_mlp_residual = torch.zeros(LAYERS_NUM, PROMPT_LEN, D_MODEL, device=device, dtype=dtype).save()  # (layer,positon,d_model) for percision calculation
        PROMPT_LEN = len(embed)
        for p in range(PROMPT_LEN):
            post_mlp_contribution[0][p][p] = embed[p]
        del embed

        # per layer loop
        for l in range(LAYERS_NUM):
            layer = model.model.layers[l]
            W_V = layer.self_attn.v_proj.weight.float()  # (Hkv*d_v,d_model)
            W_O = layer.self_attn.o_proj.weight.float()  # (d, H_q* d_v)
            
            if W_V.device != post_mlp_contribution.device:
                print(f"Switching device to {W_V.device}")
                post_mlp_contribution = post_mlp_contribution.to(W_V.device)
                post_attention_contribution = post_attention_contribution.to(W_V.device)
                
            
            post_rmssnorm1_contribution = caclulate_post_rmsnorm1_contribution(layer, post_mlp_contribution[l])  # (position,source,d_model)

            attention_ouput_per_source = torch.zeros((PROMPT_LEN, PROMPT_LEN, D_MODEL), device=device, dtype=dtype) # (position(query),source,d_model)
            attn_pattern = layer.self_attn.output[1][0].float()  # (H_q,prompt_len(query),prompt_len(key)) post softmax
            for q_residual in range(PROMPT_LEN):
                per_head_key_v = post_rmssnorm1_contribution @ W_V.T  # (key,prompt_len (source),H_kv * d_v) =  (key,prompt_len(source),d_model) x (d_model,Hkv*dv)
                reshped_to_per_query_v = ein.repeat(per_head_key_v, "key pl (H_kv d_v)-> d_v (H_kv r) key pl", d_v=D_V, r=heads_ratio)  # (d_v,H_q,p_len(key),p_len(source))

                post_attention_per_query_v = reshped_to_per_query_v * attn_pattern[:, q_residual, :].unsqueeze(-1)  # (d_v,H_q,p_len(key),p_len(source)= (keyd_v, H_q , prompt_len(key),P_len(source))* (H_q ,p_len(key), 1)
                post_attention_per_query_v = post_attention_per_query_v.sum(dim=-2)  # (d_v,H_q,p_len(source))
                flatten_post_attention_per_query_v = ein.rearrange(post_attention_per_query_v, "d_v H_q pl -> (H_q d_v) pl")  # (H_q * d_v, p_len(source))
                attention_ouput_per_source[q_residual] = (W_O @ flatten_post_attention_per_query_v).T  # (p_len(source),d) = ((d, H_q* d_v) x (H_q * d_v, p_len(source)))^T

            post_attention_contribution[l] = attention_ouput_per_source + post_mlp_contribution[l]  # (layer,position,source,d_model)
            if not isinstance(layer.input_layernorm._module, nn.Identity):
                post_rms_norm2_contribution = calc_post_rmsnorm2(layer, post_attention_contribution[l])  # (position,source,d_model)
            else:
                post_rms_norm2_contribution = post_attention_contribution[l]

            real_attention_residual[l] = layer.post_attention_layernorm.input[0].save()  # (layer,p_len,d_model) for percision calcuations

            # assert allclose(post_rms_norm2_contribution.sum(dim=1), layer.post_attention_layernorm.output[0]), f"{post_rms_norm2_contribution.shape} {layer.post_attention_layernorm.output[0].shape}"
            # print("Rmsnorm2 ok")
            W_up = layer.mlp.up_proj.weight.float()
            upscale_contribution = post_rms_norm2_contribution @ W_up.T  # (position,source,d_mlp) = (position,source,d_model) x (d_model,d_mlp)
            g_l = ein.rearrange(layer.mlp.act_fn.output[0], "p_len d_mlp -> p_len 1 d_mlp").float()  # (prompt_len,1,d_mlp)
            upscale_g_contribution = g_l * upscale_contribution  # (position,source,d_mlp)
            W_down = layer.mlp.down_proj.weight.float()  # (d_model,d_mlp)
            mlp_contribution = upscale_g_contribution @ W_down.T  # (position,source,d_model) = (position,source,d_mlp) x (d_mlp,d_model)
            post_mlp_contribution[l + 1] = mlp_contribution + post_attention_contribution[l]  # (layer,position,source,d_model)

            real_mlp_residual[l] = layer.output[0].save()  # (layer,p_len,d_model)
            del W_up,W_down,W_O,W_V


    return (post_mlp_contribution[1:], post_attention_contribution), (real_mlp_residual, real_attention_residual)  # ((layer,position,source,d_model), (layer,position,source,d_model)),(#(layer,p_len,d_model),#(layer,p_len,d_model)) for percision calcuations


class ModelInformationCalculatorF32:
    def __init__(self, model: nnsight.LanguageModel,model_parameters:ModelParameters) -> None:
        self.model = model
        self.model_parmeters = model_parameters
        self.tokenizer = self.model.tokenizer
        self.tokenizer.clean_up_tokenization_spaces = False

    def calc(self, prompt: str, remote: bool | str = True) -> FullRunResults:
        (post_mlp_contribution, post_attention_contribution), (real_mlp_residual, real_attention_residual) = calc_contribution_per_layer_per_residual(self.model, prompt, remote=remote)
        contributiutions = Contributions(post_mlp_contribution=post_mlp_contribution, post_attention_contribution=post_attention_contribution)
        precise = ResidualStream(attention_residual=real_attention_residual, mlp_residual=real_mlp_residual)
        info_dimentions = ResultsDimentions(layers=post_mlp_contribution.shape[0], prompt_len=real_attention_residual.shape[1], d_model=real_attention_residual.shape[2])
        return FullRunResults(contributions=contributiutions, precise=precise, dimentions=info_dimentions)

    def calc_tokens(self, prompt: str) -> list[str]:
        tokens_ids = self.tokenizer(prompt)["input_ids"]
        return [self.tokenizer.decode([id]) for id in tokens_ids]

    def tokens_probabilities_from_logits(self, single_logits: torch.Tensor, max_results_amount: int = 5, min_prob=0.02) -> dict[str, float]:  # logits: (vocab_size) return dict[token->prob]
        probabilities = torch.softmax(single_logits, dim=-1)
        ids_probabilities = sorted(list(enumerate(probabilities.tolist())), key=lambda p_id: p_id[1], reverse=True)
        filtered_probabilities = [i_p for i_p in ids_probabilities if i_p[1] >= min_prob]
        filtered_probabilities_trimmed = filtered_probabilities[:max_results_amount]
        return OrderedDict({self.tokenizer.decode([id]): probability for id, probability in filtered_probabilities_trimmed})

    def calc_top_probabilities_from_logits(self, single_logits: torch.Tensor, number_of_points: int):  # logits: (vocab_size) return dict[token->prob]
        probabilities = torch.softmax(single_logits, dim=-1)
        ids_probabilities = sorted(list(enumerate(probabilities.tolist())), key=lambda p_id: p_id[1], reverse=True)
        filtered_probabilities = ids_probabilities[:number_of_points]
        return OrderedDict({self.tokenizer.decode([id]): probability for id, probability in filtered_probabilities})



    def calc_top_perdictions_from_vector(self, vector: torch.Tensor, prediction_num: int = 5, with_last_rms=True) -> dict[float, int]:
        if with_last_rms:
            vector2 = self._calc_last_rms(vector.unsqueeze(0))[0]
        assert not torch.allclose(vector2, vector)
        logits = self.model_parmeters.unembedding_matrix @ vector2
        return self.calc_top_probabilities_from_logits(logits, prediction_num)

    def calc_logits(self,last_mlp_output:torch.Tensor)->torch.Tensor: # (p_len,d_model) , (vocab_size,d_model) -> (p_len,vocab_size)
        post_rms = self._calc_last_rms(last_mlp_output)
        return post_rms @ self.model_parmeters.unembedding_matrix.T
        
    def _calc_last_rms(self, tensor: torch.Tensor) -> torch.Tensor:  # (p_len,d_model)
        float_vector = tensor.float()  # (p_len,d_model)
        rms_weight_float = self.model_parmeters.last_rms_weight.float()  # (d_model)
        rms_epsilon = self.model_parmeters.rms_norm_eps # scalar
        ms = float_vector.pow(2).mean(dim=-1) # (p_len)
        rms_factor = torch.rsqrt(ms + rms_epsilon).unsqueeze(dim=1)  # (p_len,1)
        print(float_vector.shape)
        print(rms_factor.shape)
        return (rms_factor * float_vector).to(tensor.dtype)* rms_weight_float  #((p_len,1) *(p_len,d_model) ) *d_model

