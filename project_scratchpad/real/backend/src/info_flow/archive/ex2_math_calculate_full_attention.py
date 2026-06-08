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

HF_TOKEN: str | None = os.environ.get("HF_TOKEN")


def _get_model(model_name: str, hf_token: str) -> nnsight.LanguageModel:
    model_kwargs_dict = {"token": hf_token}
    return nnsight.LanguageModel(model_name, **model_kwargs_dict)  # type: ignore[arg-type]


def caclulate_post_rmsnorm1_contribution(layer, post_mlp_layer_contribution) -> torch.Tensor:  # post_mlp_layer_contribution: (position,source,d_model)
    pre_rmsnorm1_residual = layer.input_layernorm.input.save()[0]  # (prompt_len,d_model)
    inv_residual_norms = (torch.norm(pre_rmsnorm1_residual, -1)) ** -1  # (prompt_len)
    inv_residual_norms_reshaped = ein.rearrange(inv_residual_norms, "pl-> pl 1 1")  # (prompt_len,1,1)
    rms_weight = layer.input_layernorm.weight  # (d_model)
    return (post_mlp_layer_contribution * inv_residual_norms_reshaped) * rms_weight  # (position, source ,d_model)


class VectorSavingMode(str, Enum):
    FULL_VECTOR = "full_vector"
    L2 = "l2"
    L_INF = "l_inf"
    L_0 = "l0"


class ContributionPerResidual(BaseModel):
    per_layer_contribution_per_residual: list[torch.Tensor]  # (layer,(query,key,d_model))

    model_config = SettingsConfigDict(arbitrary_types_allowed=True)


def calc_contribution_per_layer_per_residual(model: nnsight.LanguageModel, prompt: str, remote: bool | str = True):
    LAYERS_NUM = len(model.model.layers)
    D_MODEL = model.model.config.hidden_size
    D_V = model.model.config.head_dim
    heads_ratio = model.model.config.num_attention_heads // model.model.config.num_key_value_heads
    with model.trace(prompt, remote=remote):
        #ouput init
        post_mlp_contribution = torch.zeros((LAYERS_NUM, PROMPT_LEN, PROMPT_LEN, D_MODEL)).save()  # (layer,position,source,d_model)
        post_attention_contribution = torch.zeros((LAYERS_NUM, PROMPT_LEN, PROMPT_LEN, D_MODEL)).save()  # (layer,position,source,d_model)
        per_layer_contribution_per_residual = list().save()
        embed = model.model.embed_tokens.output.save()[0]  # (seq_len, d_model)
        PROMPT_LEN = len(embed)
        for p in range(PROMPT_LEN):
            post_mlp_contribution[0][p][p] = embed[p]
        
        #per layer loop
        for l in range(LAYERS_NUM):
            layer = model.model.layers[l]
            W_V = layer.self_attn.v_proj.weight  # (Hkv*d_v,d_model)
            W_O = layer.self_attn.o_proj.weight  # (d, H_q* d_v)
            post_rmssnorm1_contribution = caclulate_post_rmsnorm1_contribution(layer,post_mlp_contribution[l]) #(position,source,d_model)
            attn_pattern = layer.self_attn.output[1][0]  # (H_q,prompt_len(query),prompt_len(key)) post softmax
            for q_residual in range(PROMPT_LEN):
                for k_residual in range(q_residual+1):
                    per_head_key_v = W_V @ post_rmssnorm1_contribution[k_residual]  # (H_kv * d_v,prompt_len (source))
                    reshped_to_per_query_v = ein.repeat(per_head_key_v, "(H_kv d_v) pl  -> d_v (H_kv r) pl", d_v=D_V, r=heads_ratio) #(d_v, H_q , prompt_len)
                    # (d_v,H_q,p_len)
                    post_attention_per_query_v = reshped_to_per_query_v * attn_pattern[:, q_residual, k_residual].unsqueeze(-1)  # (d_v,H_q,p_len)= ((d_v, H_q , prompt_len)* (H_q ,1)
                    flatten_post_attention_per_query_v = ein.rearrange(post_attention_per_query_v, "d_v H_q pl -> (H_q d_v) pl")  # (H_q * d_v, p_len)
                    post_attention_contribution[l][q_residual][k_residual] = W_O @ flatten_post_attention_per_query_v # (d,p_len)
            post_mlp_contribution[l] = post_attention_contribution[l]
    return post_attention_contribution


class ModelInformationCalculator:
    def __init__(self, model: nnsight.LanguageModel, remote: bool | str = True) -> None:
        self.model = model
        self.remote = remote

    def calc(self, prompt: str) -> ContributionPerResidual:
        return ContributionPerResidual(per_layer_contribution_per_residual=calc_contribution_per_layer_per_residual(self.model, prompt))
