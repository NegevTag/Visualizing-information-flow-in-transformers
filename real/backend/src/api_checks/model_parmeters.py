

import nnsight
import torch
from functools import lru_cache
from urllib.parse import quote
from functools import cached_property

class ModelParameters:
    def __init__(self, cache_path: str, model: nnsight.LanguageModel):
        self._cache_path = cache_path
        self._unembedding_matricies_cache_path = self._cache_path / "unembedding_matracies"
        self.model = model
        self.model_name = model.model.config._name_or_path
        self.rms_norm_eps = model.model.config.rms_norm_eps

    def load(self):
        self.unembedding_matrix()
        self.last_rms_weight()

    @cached_property
    def unembedding_matrix(self) -> torch.Tensor:
        unembedding_matrix_path = self._unembedding_matricies_cache_path / self._get_unembedding_key_name(self.model_name)
        self._unembedding_matricies_cache_path.mkdir(parents=True, exist_ok=True)
        try:
            return torch.load(unembedding_matrix_path)
        except FileNotFoundError:
            model = self.model
            with model.trace("", remote=True):
                unembedding_matrix = model.lm_head.weight.save()
            unembedding_matrix = unembedding_matrix.detach().float().contiguous()
            torch.save(unembedding_matrix, unembedding_matrix_path)
            return unembedding_matrix

    @cached_property
    def last_rms_weight(self) -> torch.Tensor:
        model = self.model
        with model.trace("", remote=True):
            rms_weight = model.model.norm.weight.save()
        return rms_weight

    @classmethod
    def _get_unembedding_key_name(
        cls,
        model_name: str,
    ) -> str:
        return quote(f"{model_name}", safe="")
