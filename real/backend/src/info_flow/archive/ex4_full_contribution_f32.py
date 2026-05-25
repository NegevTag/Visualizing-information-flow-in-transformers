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


class Contributions(BaseModel):
    post_mlp_contribution: list = list[torch.Tensor]  # (layer+1 (zero layer is embeddings),position,source,d_model)
    post_attention_contribution: list = list[torch.Tensor]  # (layer,position,source,d_model)

    model_config = SettingsConfigDict(arbitrary_types_allowed=True)


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
            return rms_weight * (rms_factor_reshaped * contribution_f32)

        def caclulate_post_rmsnorm1_contribution(layer, post_mlp_contribution) -> torch.Tensor:  # post_mlp_layer_contribution: (position,source,d_model)
            rms_weight = layer.input_layernorm.weight.float()
            rms_eps = model.model.config.rms_norm_eps
            return _calculate_mlp_contribution(rms_eps, rms_weight, post_mlp_contribution)

        def calc_post_rmsnorm2(layer, post_attention_contibutions) -> torch.Tensor:  # per_residual_contribution  (position,source,d_model)
            rms_weight = layer.post_attention_layernorm.weight.float()  # (d_model)
            rms_eps = model.model.config.rms_norm_eps
            return _calculate_mlp_contribution(rms_eps, rms_weight, post_attention_contibutions)

        embed = model.model.embed_tokens.output.save()[0]  # (seq_len, d_model)
        PROMPT_LEN = len(embed)

        # ouput init
        device, dtype = embed.device, torch.float32
        post_mlp_contribution = torch.zeros((LAYERS_NUM + 1, PROMPT_LEN, PROMPT_LEN, D_MODEL), device=device, dtype=dtype).save()  # (layer+1 (zero layer means nothing),position,source,d_model)
        post_attention_contribution = torch.zeros((LAYERS_NUM, PROMPT_LEN, PROMPT_LEN, D_MODEL), device=device, dtype=dtype).save()  # (layer,position,source,d_model)
        PROMPT_LEN = len(embed)
        for p in range(PROMPT_LEN):
            post_mlp_contribution[0][p][p] = embed[p]

        # per layer loop
        for l in range(LAYERS_NUM):
            layer = model.model.layers[l]
            W_V = layer.self_attn.v_proj.weight.float()  # (Hkv*d_v,d_model)
            W_O = layer.self_attn.o_proj.weight.float()  # (d, H_q* d_v)
            post_rmssnorm1_contribution = caclulate_post_rmsnorm1_contribution(layer, post_mlp_contribution[l])  # (position,source,d_model)

            _real_post_rms1 = layer.input_layernorm.output[0]
            # assert allclose(post_rmssnorm1_contribution.sum(dim=1), _real_post_rms1)
            del _real_post_rms1

            print("RMS_NORM1 ok")
            print(f"post Rms1 shape {post_rmssnorm1_contribution.shape}")
            attention_ouput_per_source = torch.zeros((PROMPT_LEN, PROMPT_LEN, PROMPT_LEN, D_MODEL), device=device, dtype=dtype).save()  # (position(query),key,source,d_model)
            attn_pattern = layer.self_attn.output[1][0].float()  # (H_q,prompt_len(query),prompt_len(key)) post softmax
            for q_residual in range(PROMPT_LEN):
                for k_residual in range(q_residual + 1):
                    per_head_key_v = W_V @ post_rmssnorm1_contribution[k_residual].T  # (H_kv * d_v,prompt_len (source)) = (Hkv*d_v,d_model) x (d_mode,prompt_len(source))
                    reshped_to_per_query_v = ein.repeat(per_head_key_v, "(H_kv d_v) pl  -> d_v (H_kv r) pl", d_v=D_V, r=heads_ratio)  # (d_v,H_q, prompt_len)
                    # (d_v,H_q,p_len)
                    post_attention_per_query_v = reshped_to_per_query_v * attn_pattern[:, q_residual, k_residual].unsqueeze(-1)  ## (d_v,H_q,p_len)= ((d_v, H_q , prompt_len)* (H_q ,1)
                    flatten_post_attention_per_query_v = ein.rearrange(post_attention_per_query_v, "d_v H_q pl -> (H_q d_v) pl")  # (H_q * d_v, p_len)
                    attention_ouput_per_source[q_residual][k_residual] = (W_O @ flatten_post_attention_per_query_v).T  # (p_len,d) = ((d, H_q* d_v) x (H_q * d_v, p_len))^T

            post_attention_contribution[l] = attention_ouput_per_source.sum(dim=1) + post_mlp_contribution[l]  # (layer,position,source,d_model)
            post_rms_norm2_contribution = calc_post_rmsnorm2(layer, post_attention_contribution[l])  # (position,source,d_model)

            # assert allclose(post_rms_norm2_contribution.sum(dim=1), layer.post_attention_layernorm.output[0]), f"{post_rms_norm2_contribution.shape} {layer.post_attention_layernorm.output[0].shape}"
            print("Rmsnorm2 ok")
            W_up = layer.mlp.up_proj.weight.float()
            upscale_contribution = post_rms_norm2_contribution @ W_up.T  # (position,source,d_mlp) = (position,source,d_model) x (d_model,d_mlp)
            g_l = ein.rearrange(layer.mlp.act_fn.output[0].float(), "p_len d_mlp -> p_len 1 d_mlp")  # (prompt_len,1,d_mlp)
            upscale_g_contribution = g_l * upscale_contribution  # (position,source,d_mlp)

            # assert allclose(layer.mlp.down_proj.input[0], upscale_g_contribution.sum(dim=1)), f"{layer.mlp.down_proj.input[0].shape} ,{upscale_g_contribution.sum(dim=1).shape}"
            W_down = layer.mlp.down_proj.weight.float()  # (d_model,d_mlp)
            mlp_contribution = upscale_g_contribution @ W_down.T  # (position,source,d_model) = (position,source,d_mlp) x (d_mlp,d_model)

            post_mlp_contribution[l + 1] = mlp_contribution + post_attention_contribution[l]

    # Dump attention_ouput_per_source for offline inspection (layer 0 only — loop is range(1)).
    _dump_dir = Path(__file__).resolve().parent / ".cache"
    _dump_dir.mkdir(exist_ok=True)
    _dump_path = _dump_dir / "attention_output_per_source_L0.pt"
    torch.save(attention_ouput_per_source.detach().cpu(), _dump_path)
    print(f"[dump] attention_ouput_per_source -> {_dump_path}  shape={tuple(attention_ouput_per_source.shape)}")

    return post_mlp_contribution[1:], post_attention_contribution  # (layer,position,source,d_model), (layer,position,source,d_model)


class ModelInformationCalculator:
    def __init__(self, model: nnsight.LanguageModel, remote: bool | str = True) -> None:
        self.model = model
        self.remote = remote

    def calc(self, prompt: str) -> Contributions:
        post_mlp_contribution, post_attention_contribution = calc_contribution_per_layer_per_residual(self.model, prompt)
        return Contributions(post_mlp_contribution=post_mlp_contribution, post_attention_contribution=post_attention_contribution)
