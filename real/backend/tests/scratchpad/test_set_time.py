# pyright: ignore-all-errors
# mypy: ignore-errors

from enum import Enum

import nnsight
from pathlib import Path
from pydantic_settings import SettingsConfigDict
from torch import Tensor
import sys
import os

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pydantic import BaseModel
import torch  # noqa: E402
import einops as ein
from icecream import ic

HF_TOKEN: str | None = os.environ.get("HF_TOKEN")


def _get_model(model_name: str, hf_token: str) -> nnsight.LanguageModel:
    model_kwargs_dict = {"token": hf_token}
    return nnsight.LanguageModel(model_name, **model_kwargs_dict)  # type: ignore[arg-type]


def caclulate_post_rmsnorm1_contribution(layer, post_mlp_layer_contribution) -> torch.Tensor:  # post_mlp_layer_contribution: (position,source,d_model)
    pre_rmsnorm1_residual = layer.input_layernorm.input.save()[0]  # (prompt_len,d_model)
    inv_residual_norms = (torch.norms(pre_rmsnorm1_residual, -1)) ** -1  # (prompt_len)
    inv_residual_norms_reshaped = ein.rearrange(inv_residual_norms, "pl 1 1")  # (prompt_len,1,1)
    rms_weight = layer.input_layernorm.weight  # (d_model)
    return (post_mlp_layer_contribution * inv_residual_norms_reshaped) * rms_weight  # (position, source ,d_model)


class VectorSavingMode(str, Enum):
    FULL_VECTOR = "full_vector"
    L2 = "l2"
    L_INF = "l_inf"
    L_0 = "l0"



class ContributionPerResidual(BaseModel):
    per_layer_contribution_per_residual :list[torch.Tensor] # (layer,(query,key,d_model))
    
    model_config = SettingsConfigDict(arbitrary_types_allowed=True)


def calc_contribution_per_layer_per_residual(model:nnsight.LanguageModel,prompt:str):
        LAYERS_NUM = len(model.model.layers)
        D_MODEL = model.model.config.hidden_size
        D_V = model.model.config.head_dim
        heads_ratio = model.model.config.num_attention_heads // model.model.config.num_key_value_heads
        ic("entering trace")
        per_layer_contribution_per_residual = []
        with model.trace(prompt, remote='local'):
            ic("inside trace, about to save embed")
            embed = model.model.embed_tokens.output.save()[0]  # (seq_len, d_model)
            ic("embed saved")
            PROMPT_LEN = len(embed)
            ic(PROMPT_LEN)
            for l in range(LAYERS_NUM):
                ic(l)
                layer = model.model.layers[l]
                W_V = layer.self_attn.v_proj.weight  # (Hkv*d_v,d_model)
                W_O = layer.self_attn.o_proj.weight  # (d, H_q* d_v)
                post_rms_norm_residual = layer.input_layernorm.output.save()[0]  # (prompt_len,d_model)
                ic("saved input_layernorm.output")
                attn_pattern = layer.self_attn.output[1][0]  # (H_q,prompt_len(query),prompt_len(key)) post softmax
                ic("got attn_pattern proxy")
                contriubution_per_residual = torch.zeros(PROMPT_LEN, PROMPT_LEN, D_MODEL)  # (query,key,d_model)
                ic("zeros allocated")
                # post_rmsnosrm1_contribution = caclulate_post_rmsnorm1_contribution(layer,post_mlp_contribution[0])#(position,source,d_model)
                for q_residual in range(PROMPT_LEN):
                    ic(q_residual)
                    for k_residual in range(q_residual):
                        ic(k_residual)
                        per_head_key_v = W_V  @ post_rms_norm_residual[k_residual]  # (H_kv * d_v))
                        ic("matmul W_V @ x done")
                        reshped_to_per_query_v = ein.repeat(per_head_key_v, "(H_kv d_v)  -> d_v (H_kv r)",d_v=D_V, r=heads_ratio)
                        ic("repeat done")
                        #(d_v,H_q)
                        post_attention_per_query_v = reshped_to_per_query_v * attn_pattern[:, q_residual,k_residual] # (d_v,H_q)
                        ic("post_attention_per_query_v computed")
                        flatten_post_attention_per_query_v = ein.rearrange(post_attention_per_query_v,'d_v H_q -> (H_q d_v)')# (H_q * d_v)
                        ic("rearrange done")
                        contriubution_per_residual[q_residual][k_residual] = W_O @ flatten_post_attention_per_query_v
                        ic("contrib assigned")
                    break
                per_layer_contribution_per_residual.append(contriubution_per_residual)
                ic(f"layer {l} done")
                break
        ic("trace body complete")
        return per_layer_contribution_per_residual
# def calc_contribution_per_layer_per_residual(model, prompt):
#     ic("entering")
#     with model.trace(prompt, remote='local'):
#         embed = model.model.embed_tokens.output.save()
#         layer = model.model.layers[0]
#         post_rms = layer.input_layernorm.output.save()
#         attn_pattern = layer.self_attn.output[1].save()    # ← save it too; bare access might not even capture it
#     ic("exited")
#     print(embed.shape, post_rms.shape, attn_pattern.shape)
#     return []

class ModelInformationCalculator:
    def __init__(self, model: nnsight.LanguageModel) -> None:
        self.model = model

    def calc(self, prompt: str) -> ContributionPerResidual:
        return ContributionPerResidual(cper_layer_contribution_per_residual = calc_contribution_per_layer_per_residual(self.model,prompt))