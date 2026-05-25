# pyright: ignore-all-errors
# mypy: ignore-errors

from enum import Enum
import math

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


class VectorSavingMode(str, Enum):
    FULL_VECTOR = "full_vector"
    L2 = "l2"
    L_INF = "l_inf"
    L_0 = "l0"


class PostMlpContributionPerResidual(BaseModel):
    post_mlp_per_layer_contribution_per_residual: list[torch.Tensor]  # (layer,(query,key,d_model))

    model_config = SettingsConfigDict(arbitrary_types_allowed=True)


def calc_contribution_per_layer_per_residual(model: nnsight.LanguageModel, prompt: str, remote: bool | str = True):
        LAYERS_NUM = len(model.model.layers)
        D_MODEL = model.model.config.hidden_size
        D_V = model.model.config.head_dim
        heads_ratio = model.model.config.num_attention_heads // model.model.config.num_key_value_heads
        with model.trace(prompt, remote=remote):
            per_layer_contribution_per_residual = list().save()
            embed = model.model.embed_tokens.output.save()[0]  # (seq_len, d_model)
            PROMPT_LEN = len(embed)
            for l in range(LAYERS_NUM):
                layer = model.model.layers[l]
                W_V = layer.self_attn.v_proj.weight  # (Hkv*d_v,d_model)
                W_O = layer.self_attn.o_proj.weight  # (d, H_q* d_v)
                post_rms_norm_residual = layer.input_layernorm.output.save()[0]  # (prompt_len,d_model)
                attn_pattern = layer.self_attn.output[1][0]  # (H_q,prompt_len(query),prompt_len(key)) post softmax
                contriubution_per_residual = torch.zeros(PROMPT_LEN, PROMPT_LEN, D_MODEL, device=embed.device, dtype=embed.dtype)  # (query,key,d_model)
                for q_residual in range(PROMPT_LEN):
                    for k_residual in range(q_residual+1):
                        per_head_key_v = W_V @ post_rms_norm_residual[k_residual]  # (H_kv * d_v))
                        reshped_to_per_query_v = ein.repeat(per_head_key_v, "(H_kv d_v)  -> d_v (H_kv r)", d_v=D_V, r=heads_ratio)
                        # (d_v,H_q)
                        post_attention_per_query_v = reshped_to_per_query_v * attn_pattern[:, q_residual, k_residual]  # (d_v,H_q)
                        flatten_post_attention_per_query_v = ein.rearrange(post_attention_per_query_v, "d_v H_q -> (H_q d_v)")  # (H_q * d_v)
                        contriubution_per_residual[q_residual][k_residual] = W_O @ flatten_post_attention_per_query_v
                per_layer_contribution_per_residual.append(contriubution_per_residual.save())
            print(per_layer_contribution_per_residual)
            per_layer_contribution_per_residual.save()

            # mlp
            assert torch.is_same_size(contriubution_per_residual.sum(dim=1),layer.self_attn.output[0][0].save())
            print("Same dim")
            assert torch.allclose(contriubution_per_residual.sum(dim=1),layer.self_attn.output[0][0].save(),rtol=1e-2,atol=1e-2)
            print("Attention equal")
            # residul = torch.zeros_like(contriubution_per_residual)  # (query,key,d_model)
            # for p in range(PROMPT_LEN):
            #     residul[p][p] = input_pre_rms_norm[p]  # (query,key,d_model)
            # contriubution_per_residual = contriubution_per_residual + residul
            # print(f"contribution_per_resshape = {contriubution_per_residual.shape}")
            # my_post_attention = contriubution_per_residual.sum(dim=1)
            # post_attention = layer.post_attention_layernorm.input[0]
            # assert torch.is_same_size(my_post_attention, post_attention)
            # assert torch.allclose(my_post_attention, post_attention)
            # print("POST ATTENTION PRE rmsnorm Pass")
            # post_rms_contribituion_per_residual = calc_post_rmsnorm2(layer, contriubution_per_residual)  # (query,key,d_model)

            # post_rms_residual = post_rms_contribituion_per_residual.sum(dim=1)  # (p_len,d_model)

            # real_post_layer_norm = layer.post_attention_layernorm.output[0]  # (p_len,d_model)
            # assert torch.is_same_size(real_post_layer_norm, post_rms_residual)
            # print(f"Ratio {torch.norm(post_rms_residual)/torch.norm(real_post_layer_norm)}")
            # assert torch.allclose(post_rms_residual, real_post_layer_norm, atol=1e-2, rtol=1e-2)

            # print(f"post_rms_residua.shape ={post_rms_residual.shape}")
            # # SILU
            # silu_gate_per_pos = torch.nn.functional.silu(layer.mlp.gate_proj.weight @ post_rms_residual.T)  # (d_mlp,p_len) =(d_mpl,d_model) x (d_model,p_len)
            # silu_gate_per_pos_reshaped = ein.rearrange(silu_gate_per_pos, "d_mlp pl -> d_mlp pl 1")  # (d_mlp,p_len,1)

            # # Up proj
            # up_proj_per_key = post_rms_contribituion_per_residual @ layer.mlp.up_proj.weight.T  # (p_len (query),p_len (key),d_mlp) = (p_len (query),p_len (key),d_model) x (d_model,d_mlp)
            # up_proj_per_key_reshaped = ein.rearrange(up_proj_per_key, "q k d_mlp -> d_mlp q k")  # (d_mlp,q,k)

            # pointwise_mlp = up_proj_per_key_reshaped * silu_gate_per_pos_reshaped  # (d_mlp,query,key)
            # print(f"mlp_proj_weight = {layer.mlp.down_proj.weight.shape}")
            # print(f"pointwise mlp shape = {pointwise_mlp.shape}")
            # down_proj = pointwise_mlp.T @ layer.mlp.down_proj.weight.T  # (key,query,d_model) = (key,query,d_mlp) x  (d_mlp,d_model)
            # down_proj_reahsped = ein.rearrange(down_proj, "k q d_mlp -> q k d_mlp")  # (query,key,d_model)

            # post_mlp_residual = post_rms_residual + down_proj_reahsped  # (query,key,d_model)



class ModelInformationCalculator:
    def __init__(self, model: nnsight.LanguageModel, remote: bool | str = True) -> None:
        self.model = model
        self.remote = remote

    def calc(self, prompt: str) -> PostMlpContributionPerResidual:
        return PostMlpContributionPerResidual(post_mlp_per_layer_contribution_per_residual=calc_contribution_per_layer_per_residual(self.model, prompt))
