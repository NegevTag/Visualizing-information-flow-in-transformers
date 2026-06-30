import datetime
from pathlib import Path

from api_checks.full_run_result import Contributions, FullRunResults, ResidualStream, ResultsDimentions
import torch
from api_checks.model_calculator import ModelInformationCalculatorF32
import api_checks.utils as utils
from functools import lru_cache
from urllib.parse import quote


class APICache:
    # Two levels cache, memory (managed by functools.lru_cach, both for large results and no need to connect each time), and disk
    def __init__(self, cache_path: Path, hf_token):
        self.cache_path = cache_path
        self.results_cache_path = self.cache_path / "run_results"
        self.unembedding_matricies_cache_path = self.cache_path / "unembedding_matracies"
        self.hf_token = hf_token
        self.latest_model_name: str

    @lru_cache(maxsize=1)
    def get_full_run_results(self, model_name: str, prompt: str) -> FullRunResults:
        try:
            result, time = self._load(model_name, prompt)
            print(f"CACHE HIT {model_name} {prompt} {time}")
            return result
        except FileNotFoundError:
            information_calculator = self.get_infomration_calculator(model_name)
            result = information_calculator.calc(prompt)
            self._dump(model_name=model_name, prompt=prompt, result=result)
            print("Result saved in cache sucessfully")
            return result

    @lru_cache(maxsize=10)
    def get_infomration_calculator(self, model_name: str) -> ModelInformationCalculatorF32:
        model = utils.get_model(model_name, self.hf_token)
        return ModelInformationCalculatorF32(model)

    @lru_cache(maxsize=1)
    def get_unembedding_matrix(self, model_name: str) -> torch.Tensor:
        unembedding_matrix_path = self.unembedding_matricies_cache_path / self._get_unembedding_key_name(model_name)
        self.unembedding_matricies_cache_path.mkdir(parents=True, exist_ok=True)
        try:
            return torch.load(unembedding_matrix_path)
        except FileNotFoundError:
            model = self.get_infomration_calculator(model_name).model
            with model.trace("", remote=True):
                unembedding_matrix = model.lm_head.weight.save()
            unembedding_matrix = unembedding_matrix.detach().float().contiguous()
            torch.save(unembedding_matrix, unembedding_matrix_path)
            return unembedding_matrix
    @lru_cache(maxsize=1)
    def get_last_rms_weight(self, model_name: str) -> torch.Tensor:
        model = self.get_infomration_calculator(model_name).model
        with model.trace("", remote=True):
            rms_weight = model.model.norm.weight.save()
        return rms_weight
    
    @classmethod
    def _get_result_key_name(cls, model_name: str, prompt: str) -> str:
        return quote(f"{model_name}|||{prompt}", safe="")

    @classmethod
    def _get_unembedding_key_name(
        cls,
        model_name: str,
    ) -> str:
        return quote(f"{model_name}", safe="")

    def _dump(self, result: FullRunResults, model_name: str, prompt: str) -> Path:
        # serialize tensors + scalars to a single .pt file keyed by `key`
        self.results_cache_path.mkdir(parents=True, exist_ok=True)
        path = self.results_cache_path / f"{self._get_result_key_name(model_name,prompt)}.pt"
        payload = {
            "logits": result.logits,
            "post_mlp_contribution": result.contributions.post_mlp_contribution,
            "post_attention_contribution": result.contributions.post_attention_contribution,
            "mlp_residual": result.precise.mlp_residual,
            "attention_residual": result.precise.attention_residual,
            "layers": result.dimentions.layers,
            "prompt_len": result.dimentions.prompt_len,
            "d_model": result.dimentions.d_model,
        }
        torch.save(payload, path)
        return path

    def _load(self, model_name: str, prompt: str) -> tuple[FullRunResults, datetime.datetime]:
        path = self.results_cache_path / f"{self._get_result_key_name(model_name,prompt)}.pt"
        creation_datetime = utils.get_creation_datetime(path)
        payload = torch.load(path, weights_only=False)
        return (
            FullRunResults(
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
            ),
            creation_datetime,
        )
