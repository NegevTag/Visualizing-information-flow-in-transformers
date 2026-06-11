from enum import Enum
import math
from typing import OrderedDict
from warnings import deprecated

import nnsight
from pathlib import Path
from pydantic_settings import SettingsConfigDict
from torch import Tensor
import torch.nn as nn
import sys
import os

from api_checks.position import LLMResidualPosition

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pydantic import BaseModel
import torch  # noqa: E402
import einops as ein

HF_TOKEN: str | None = os.environ.get("HF_TOKEN")


class VectorSavingMode(str, Enum):
    FULL_VECTOR = "full_vector"
    L2 = "l2"
    L_INF = "l_inf"
    L_0 = "l0"


class Contributions(BaseModel):
    post_mlp_contribution: Tensor  # (layer,position,source,d_model)
    post_attention_contribution: Tensor  # (layer,position,source,d_model)

    def __getitem__(self, position: LLMResidualPosition):
        return self.get_by_position(position)

    def get_by_position(self, position: LLMResidualPosition):
        if position.is_mlp:
            return self.post_mlp_contribution[position.layer][position.token_position]
        else:
            return self.post_attention_contribution[position.layer][position.token_position]

    model_config = SettingsConfigDict(arbitrary_types_allowed=True)


class ResidualStream(BaseModel):
    mlp_residual: Tensor  # (layer,position,d_model)
    attention_residual: Tensor  # (layer,position,d_model)

    def __getitem__(self, position: LLMResidualPosition):
        return self.get_by_position(position)

    def get_by_position(self, position: LLMResidualPosition):
        if position.is_mlp:
            return self.mlp_residual[position.layer][position.token_position]
        else:
            return self.attention_residual[position.layer][position.token_position]

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

    @deprecated("Use Only for quick tests")
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
    @deprecated("Use only for quick tests")
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
