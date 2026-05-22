# pyright: ignore-all-errors
# mypy: ignore-errors

from enum import Enum

import nnsight
from pathlib import Path
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


class SaveConfig(BaseModel):
    vector_saving_mode: VectorSavingMode


class ContributionPerResidual(BaseModel):
    per_layer_contribution_per_residual: list[torch.Tensor]  # (layer,(query,key,d_model))


class ModelInformationCalculator:
    def __init__(self, model: nnsight.LanguageModel, save_config: SaveConfig) -> None:
        self.model = model
        self.save_config = save_config

    def calc(self, prompt: str) -> ContributionPerResidual:

        with self.model.trace(prompt, remote=True):
            PROMPT_LEN = len(embed)
            LAYERS_NUM = len(self.model.layers)
            D_MODEL = self.model.model.config.hidden_size
            embed = self.model.model.embed_tokens.output.save()[0]  # (seq_len, d_model)
            per_layer_contribution_per_residual = []
            # post_mlp_contribution = torch.zeros((LAYERS_NUM, PROMPT_LEN, PROMPT_LEN, D_MODEL))  # (layer,position,source,d_model)
            # post_attention_contribution = torch.zeros((LAYERS_NUM, PROMPT_LEN, PROMPT_LEN, D_MODEL))  # (layer,position,source,d_model)
            for p in range(PROMPT_LEN):
                self.post_mlp[0][p][p] = embed[p]

            for l in LAYERS_NUM:
                layer = self.model.model.layers[l]
                W_V = layer.self_attn.v_proj.weight  # (Hkv*d_v,d_model)
                W_O = layer.self_attn.o_proj.weight  # (d, H_q* d_v)
                attn_pattern = layer.self_attn.output[1][0]  # (H_q,prompt_len(query),prompt_len(key)) post softmax
                contriubution_per_residual = torch.zeros(PROMPT_LEN, PROMPT_LEN, D_MODEL)  # (query,key,d_model)
                post_rms_norm_residual = layer.input_layernorm.output.save()[0]  # (prompt_len,d_model)
                # post_rmsnosrm1_contribution = self.caclulate_post_rmsnorm1_contribution(layer,post_mlp_contribution[0])#(position,source,d_model)
                for q_residual in range(len(prompt)):
                    for k_residual in len(q_residual):
                        per_head_key_v = W_V @ post_rms_norm_residual[k_residual]  # (H_kv * d_v))
                        reshped_to_per_query_v = ein.repeat(per_head_key_v, "(H_kv d_v)  -> d_v (H_kv r)", r=self.model.config.num_attention_heads // self.model.config.num_key_value_heads)
                        # (d_v,H_q)
                        post_attention_per_query_v = reshped_to_per_query_v * attn_pattern[:, q_residual, k_residual]  # (d_v,H_q)
                        flatten_post_attention_per_query_v = ein.reshape(post_attention_per_query_v, "d_v H_q-> (H_q d_v)")  # (H_q * d_v)
                        contriubution_per_residual[q_residual][k_residual] = W_O @ flatten_post_attention_per_query_v
                per_layer_contribution_per_residual.append(contriubution_per_residual)

    def caclulate_post_rmsnorm1_contribution(self, layer, post_mlp_layer_contribution) -> torch.Tensor:  # post_mlp_layer_contribution: (position,source,d_model)
        pre_rmsnorm1_residual = layer.input_layernorm.input.save()[0]  # (prompt_len,d_model)
        inv_residual_norms = (torch.norms(pre_rmsnorm1_residual, -1)) ** -1  # (prompt_len)
        inv_residual_norms_reshaped = ein.rearrange(inv_residual_norms, "pl 1 1")  # (prompt_len,1,1)
        rms_weight = layer.input_layernorm.weight  # (d_model)
        return (post_mlp_layer_contribution * inv_residual_norms_reshaped) * rms_weight  # (position, source ,d_model)
