# pyright: ignore-all-errors
# mypy: ignore-errors

from enum import Enum
import math
from typing import OrderedDict

import nnsight
from pathlib import Path
from pydantic_settings import SettingsConfigDict
from torch import Tensor
import torch.nn as nn
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
    post_mlp_contribution: Tensor  # (layer,position,source,d_model)
    post_attention_contribution: Tensor  # (layer,position,source,d_model)

    model_config = SettingsConfigDict(arbitrary_types_allowed=True)


class ResidualStream(BaseModel):
    mlp_residual: Tensor  # (layer,position,d_model)
    attention_residual: Tensor  # (layer,position,d_model)

    model_config = SettingsConfigDict(arbitrary_types_allowed=True)


class ResultsDimentions(BaseModel):
    layers: int
    prompt_len: int
    d_model: int

    model_config = SettingsConfigDict(arbitrary_types_allowed=True)


LOCAL_STORAGE_DIR = Path(__file__).resolve().parent / "local_storage"


class FullRunResults(BaseModel):
    logits: torch.Tensor  # (p_len,vocab_size)
    contributions: Contributions
    precise: ResidualStream
    dimentions: ResultsDimentions

    model_config = SettingsConfigDict(arbitrary_types_allowed=True)

    def dump(self, key: str) -> Path:
        # serialize tensors + scalars to a single .pt file keyed by `key`
        LOCAL_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        path = LOCAL_STORAGE_DIR / f"{key}.pt"
        payload = {
            "logits": self.logits,
            "post_mlp_contribution": self.contributions.post_mlp_contribution,
            "post_attention_contribution": self.contributions.post_attention_contribution,
            "mlp_residual": self.precise.mlp_residual,
            "attention_residual": self.precise.attention_residual,
            "layers": self.dimentions.layers,
            "prompt_len": self.dimentions.prompt_len,
            "d_model": self.dimentions.d_model,
        }
        torch.save(payload, path)
        return path

    @classmethod
    def load(cls, key: str) -> "FullRunResults":
        path = LOCAL_STORAGE_DIR / f"{key}.pt"
        payload = torch.load(path, weights_only=False)
        return cls(
            logits=payload["logits"],
            contributions=Contributions(
                post_mlp_contribution=payload["post_mlp_contribution"],
                post_attention_contribution=payload["post_attention_contribution"],
            ),
            precise=ResidualStream(
                mlp_residual=payload["mlp_residual"],
                attention_residual=payload["attention_residual"],
            ),
            dimentions=ResultsDimentions(
                layers=payload["layers"],
                prompt_len=payload["prompt_len"],
                d_model=payload["d_model"],
            ),
        )

    def get_f64(self) -> "FullRunResults":
        return FullRunResults(
            logits=self.logits.double(),
            contributions=Contributions(
                post_mlp_contribution=self.contributions.post_mlp_contribution.double(),
                post_attention_contribution=self.contributions.post_attention_contribution.double(),
            ),
            precise=ResidualStream(
                mlp_residual=self.precise.mlp_residual.double(),
                attention_residual=self.precise.attention_residual.double(),
            ),
            dimentions=self.dimentions,
        )


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

        def calc_post_rms_last(post_attention_contibutions):
            rms_weight = model.model.norm.weight.float()
            rms_eps = model.model.config.rms_norm_eps
            return _calculate_mlp_contribution(rms_eps, rms_weight, post_attention_contibutions)

        embed = model.model.embed_tokens.output.save()[0]  # (seq_len, d_model)
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

        # per layer loop
        for l in range(LAYERS_NUM):
            layer = model.model.layers[l]
            W_V = layer.self_attn.v_proj.weight.float()  # (Hkv*d_v,d_model)
            W_O = layer.self_attn.o_proj.weight.float()  # (d, H_q* d_v)
            if not isinstance(layer.input_layernorm._module,nn.Identity):
                print(type(layer.input_layernorm))
                post_rmssnorm1_contribution = caclulate_post_rmsnorm1_contribution(layer, post_mlp_contribution[l])  # (position,source,d_model)
            else:
                post_rmssnorm1_contribution = post_mlp_contribution[l]

            # _real_post_rms1 = layer.input_layernorm.output[0]
            # # assert allclose(post_rmssnorm1_contribution.sum(dim=1), _real_post_rms1)
            # del _real_post_rms1

            print("RMS_NORM1 ok")
            print(f"post Rms1 shape {post_rmssnorm1_contribution.shape}")
            attention_ouput_per_source = torch.zeros((PROMPT_LEN, PROMPT_LEN, D_MODEL), device=device, dtype=dtype).save()  # (position(query),source,d_model)
            # if l == 0:
            #     post_rms_res = layer.input_layernorm.output[0] # (prompt_len,d_model)
            #     print(f"post rms {post_rms_res}")
            #     key_inner =  post_rms_res @ layer.self_attn.k_proj.weight.T # (prompt_len,H_k * d_kv) = (prompt_len,d_model) x (d_model, H_k*d_kv)
            #     key_repeated = ein.repeat(key_inner,"p_len (H_k d_kv) -> p_len (H_k r) d_kv",d_kv = D_V, r=heads_ratio) # (p_len, H_q, d_kv)
            #     query = post_rms_res @ layer.self_attn.q_proj.weight.T # (prompt_len,H_q * d_kv) = (prompt_len,d_model) x (d_model, H_q*d_kv)
            #     query_repeated = ein.repeat(query, "p_len (H_q d_kv) -> p_len H_q d_kv",d_kv = D_V) # (prompt_len, H_q, d_kv)
            #     attention_pattern_pre_mask_pre_softmax = ein.einsum(query_repeated, key_repeated,"q_plen Hq dkv, k_plen Hq dkv -> Hq q_plen k_plen")
            #     print("Pre softmax attention")
            #     print(attention_pattern_pre_mask_pre_softmax)
            #     attn_pattern = layer.self_attn.output[1][0].float()  # (H_q,prompt_len(query),prompt_len(key)) post softmax
            #     print("Post mask softmax attention")
            #     print(attn_pattern)
            attn_pattern = layer.self_attn.output[1][0].float()  # (H_q,prompt_len(query),prompt_len(key)) post softmax
            for q_residual in range(PROMPT_LEN):
                per_head_key_v = post_rmssnorm1_contribution @ W_V.T  # (key,prompt_len (source),H_kv * d_v) =  (key,prompt_len(source),d_model) x (d_model,Hkv*dv)
                reshped_to_per_query_v = ein.repeat(per_head_key_v, "key pl (H_kv d_v)-> d_v (H_kv r) key pl", d_v=D_V, r=heads_ratio)  # (d_v,H_q,p_len(key),p_len(source))

                post_attention_per_query_v = reshped_to_per_query_v * attn_pattern[:, q_residual, :].unsqueeze(-1)  # (d_v,H_q,p_len(key),p_len(source)= (keyd_v, H_q , prompt_len(key),P_len(source))* (H_q ,p_len(key), 1)
                # print(f"post_attention_per_query_v {post_attention_per_query_v.shape}")
                post_attention_per_query_v = post_attention_per_query_v.sum(dim=-2)  # (d_v,H_q,p_len(source))
                # print(f"shape after sum {post_attention_per_query_v.shape}")
                flatten_post_attention_per_query_v = ein.rearrange(post_attention_per_query_v, "d_v H_q pl -> (H_q d_v) pl")  # (H_q * d_v, p_len(source))
                attention_ouput_per_source[q_residual] = (W_O @ flatten_post_attention_per_query_v).T  # (p_len(source),d) = ((d, H_q* d_v) x (H_q * d_v, p_len(source)))^T

            # print(f"addition shape{(attention_ouput_per_source + post_mlp_contribution[l]).shape}")
            post_attention_contribution[l] = attention_ouput_per_source + post_mlp_contribution[l]  # (layer,position,source,d_model)
            if not isinstance(layer.input_layernorm._module,nn.Identity):
                post_rms_norm2_contribution = calc_post_rmsnorm2(layer, post_attention_contribution[l])  # (position,source,d_model)
            else:
                post_rms_norm2_contribution = post_attention_contribution[l]

            real_attention_residual[l] = layer.post_attention_layernorm.input[0].save()  # (layer,p_len,d_model) for percision calcuations

            # assert allclose(post_rms_norm2_contribution.sum(dim=1), layer.post_attention_layernorm.output[0]), f"{post_rms_norm2_contribution.shape} {layer.post_attention_layernorm.output[0].shape}"
            # print("Rmsnorm2 ok")
            W_up = layer.mlp.up_proj.weight.float()
            upscale_contribution = post_rms_norm2_contribution @ W_up.T  # (position,source,d_mlp) = (position,source,d_model) x (d_model,d_mlp)
            if l==0:
                print("updacale contriubtuions")
                print(upscale_contribution)
            g_l = ein.rearrange(layer.mlp.act_fn.output[0], "p_len d_mlp -> p_len 1 d_mlp").float()  # (prompt_len,1,d_mlp)
            if l==0:
                print("gl")
                print(layer.mlp.act_fn.output[0])
            upscale_g_contribution = g_l * upscale_contribution  # (position,source,d_mlp)

            if l == 0:
                print(f"Upscale g contribution: {upscale_g_contribution}")
            # assert allclose(layer.mlp.down_proj.input[0], upscale_g_contribution.sum(dim=1)), f"{layer.mlp.down_proj.input[0].shape} ,{upscale_g_contribution.sum(dim=1).shape}"
            W_down = layer.mlp.down_proj.weight.float()  # (d_model,d_mlp)
            mlp_contribution = upscale_g_contribution @ W_down.T  # (position,source,d_model) = (position,source,d_mlp) x (d_mlp,d_model)
            if l == 0:
                print("mmlp contributions")
                print(mlp_contribution)
            post_mlp_contribution[l + 1] = mlp_contribution + post_attention_contribution[l]  # (layer,position,source,d_model)


            real_mlp_residual[l] = layer.output[0].save()  # (layer,p_len,d_model)
            

        # output
        post_last_rms = calc_post_rms_last(post_mlp_contribution[l + 1]).sum(dim=-2)  # (position,d_model)
        W_lm = model.lm_head.weight  # (vocab_size,d_model)
        logits = (W_lm @ post_last_rms.to(torch.bfloat16).T).save()  # (vocab_size,p_len)
        print(f"logits shape{logits.shape}")

    return (post_mlp_contribution[1:], post_attention_contribution), logits, (real_mlp_residual, real_attention_residual)  # ((layer,position,source,d_model), (layer,position,source,d_model)),(#(layer,p_len,d_model),#(layer,p_len,d_model)) for percision calcuations


class ModelInformationCalculatorF32Ex6:
    def __init__(self, model_name: str, hf_token: str, remote: bool | str = True) -> None:
        self.model = _get_model(model_name, hf_token)
        self.tokenizer = self.model.tokenizer
        self.tokenizer.clean_up_tokenization_spaces = False
        self.remote = remote

    def calc(self, prompt: str) -> FullRunResults:
        (post_mlp_contribution, post_attention_contribution), logits, (real_mlp_residual, real_attention_residual) = calc_contribution_per_layer_per_residual(self.model, prompt, remote=self.remote)
        contributiutions = Contributions(post_mlp_contribution=post_mlp_contribution, post_attention_contribution=post_attention_contribution)
        precise = ResidualStream(attention_residual=real_attention_residual, mlp_residual=real_mlp_residual)
        info_dimentions = ResultsDimentions(layers=post_mlp_contribution.shape[0], prompt_len=real_attention_residual.shape[1], d_model=real_attention_residual.shape[2])
        return FullRunResults(contributions=contributiutions, logits=logits, precise=precise, dimentions=info_dimentions)

    def calc_tokens(self, prompt: str) -> list[str]:
        tokens_ids = self.tokenizer(prompt)["input_ids"]
        return [self.tokenizer.decode([id]) for id in tokens_ids]

    def tokens_probabilities_from_logits(self, single_logits: torch.Tensor, min_prob=0.04) -> dict[str, float]:  # logits: (vocab_size) return dict[token->prob]
        probabilities = torch.softmax(single_logits, dim=-1)
        ids_probabilities = sorted(list(enumerate(probabilities.tolist())), key=lambda p_id: p_id[1], reverse=True)
        print(ids_probabilities[140])
        filtered_probabilities = [i_p for i_p in ids_probabilities if i_p[1] >= min_prob]
        return OrderedDict({self.tokenizer.decode([id]): probability for id, probability in filtered_probabilities})
