"""Tiny Llama toy model for sainity checks."""

import math

import torch
import torch.nn as nn
import einops as ein

from tests.scratchpad.toy_llama_unit_embedding import ToyLllamaUnitEmbedding

LlamaDType = torch.bfloat16

torch.set_printoptions(precision=2, threshold=torch.inf)  # Show 3 decimal places


class NoRope(torch.nn.Module):
    # recieve x, returns rotationsl cos and sin that will be multiplied late, cold ones
    def __init__(self, d_q):
        super().__init__()
        self.d_q = d_q

    def forward(self, x, position_ids):  # qs : [B,prompt_len,d_q] position_ids [B,prompt_len]
        return (torch.ones([x.shape[0], x.shape[1], self.d_q], dtype=LlamaDType), torch.zeros([x.shape[0], x.shape[1], self.d_q], dtype=LlamaDType))


MOVE_ONE_BACK_HEAD_INDEX = 2
SOFTMAX_DOMNATE = float(10000000000000000000)


def get_query_3_back_matrix(d_model: int, h_q: int):  # assumes d_model = d_k
    attend_one_back = torch.zeros([h_q * d_model, d_model], dtype=LlamaDType)
    attend_one_back[d_model * MOVE_ONE_BACK_HEAD_INDEX][0] = 1.0
    for i in range(d_model - 1):
        attend_one_back[d_model * MOVE_ONE_BACK_HEAD_INDEX + i][i + 1] = SOFTMAX_DOMNATE
    return attend_one_back


def get_key_identity_matrix(d_model: int, h_k: int):  # second matrix is identity
    identity = torch.eye(d_model, dtype=LlamaDType)
    return torch.cat(
        [
            torch.zeros([d_model, d_model], dtype=LlamaDType),
            identity,
            # ein.repeat(torch.zeros([d_model, d_model], dtype=LlamaDType), "d_model_a d_model_b -> (h_k_remaning d_model_a) d_model_b", h_k_remaning=h_k - 2),
        ]
    )


def get_v_matrix_to_last_dim(d_model: int, h_k: int):  # second matrix is having effect
    to_last_dim = torch.zeros([d_model, d_model], dtype=LlamaDType)
    for i in range(d_model):
        to_last_dim[d_model - 1][i] = 1/(i+1)
    return torch.cat(
        [
            torch.zeros([d_model, d_model], dtype=LlamaDType),
            to_last_dim,
            ein.repeat(torch.zeros([d_model, d_model], dtype=LlamaDType), "d_model_a d_model_b -> (h_k_remaining d_model_a) d_model_b", h_k_remaining=h_k - 2),
        ]
    )


def get_o_matrix_zero_with_identity_at_index(d_model: int, h_q: int):
    o_matrix = torch.zeros([d_model, h_q * d_model], dtype=LlamaDType)
    for i in range(d_model):
        o_matrix[i][d_model * MOVE_ONE_BACK_HEAD_INDEX + i] = 1
    return o_matrix


def get_down_project_last_dim_minus(d_model: int):
    down_project_last_dim_min = torch.zeros([d_model, d_model], dtype=LlamaDType)
    down_project_last_dim_min[d_model - 1][d_model - 1] = -1
    return down_project_last_dim_min


D_MODEL = 32
INV_1_SILU = 1.2784645428  # SiLU^-1(1)


# Toy llama move delete
class ToyLllamaAttenOne_Over_N_Back(ToyLllamaUnitEmbedding):
    DEFAULT_CONFIG = dict(
        hidden_size=D_MODEL,
        head_dim=D_MODEL,
        intermediate_size=D_MODEL,
        num_hidden_layers=10,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        attn_implementation="eager",
    )
    """Builder for a tiny Llama wrapped in nnsight.LanguageModel."""

    @classmethod
    def build_hf_model(cls, config_overrides, tokenizer):
        hf_model = super().build_hf_model(config_overrides, tokenizer)

        with torch.no_grad():
            config = hf_model.config
            hf_model.model.rotary_emb = NoRope(config.head_dim)
            for layer in hf_model.model.layers:
                rsqrt_d_model = torch.rsqrt(torch.tensor([float(config.hidden_size)]))
                # attention
                layer.input_layernorm = nn.Identity()
                layer.self_attn.k_proj.weight.copy_(get_key_identity_matrix(config.hidden_size, config.num_key_value_heads))
                layer.self_attn.q_proj.weight.copy_(get_query_3_back_matrix(config.hidden_size, config.num_attention_heads))
                # layer.self_attn.k_proj.weight.copy_(ein.repeat(torch.eye(config.hidden_size, dtype=LlamaDType), "d_model_a d_model_b -> (h_k d_model_a) d_model_b", h_k=config.num_key_value_heads))
                # layer.self_attn.q_proj.weight.copy_(ein.repeat(torch.eye(config.hidden_size, dtype=LlamaDType), "d_model_a d_model_b -> (h_q d_model_a) d_model_b", h_q=config.num_attention_heads))

                layer.self_attn.v_proj.weight.copy_(get_v_matrix_to_last_dim(config.hidden_size, config.num_key_value_heads))
                layer.self_attn.o_proj.weight.copy_(get_o_matrix_zero_with_identity_at_index(config.hidden_size, config.num_attention_heads))

                # # mlp
                layer.post_attention_layernorm = nn.Identity()
                layer.mlp.act_fn.forward = lambda self,x: torch.ones_like(x)
                layer.mlp.gate_proj.weight.copy_(torch.eye(config.hidden_size, dtype=LlamaDType))
                layer.mlp.up_proj.weight.copy_(torch.eye(config.hidden_size, dtype=LlamaDType))
                layer.mlp.down_proj.weight.copy_(get_down_project_last_dim_minus(config.hidden_size))

                # layer.self_attn.v_proj.weight.zero_()
                # layer.self_attn.o_proj.weight.zero_()
                # layer.mlp.gate_proj.weight.zero_()
                # layer.mlp.up_proj.weight.zero_()
                # layer.mlp.down_proj.weight.zero_()

        return hf_model
