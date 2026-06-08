import datetime
from pathlib import Path

from api_checks.full_run_result import Contributions, FullRunResults, ResidualStream, ResultsDimentions
import torch
from api_checks.model import ModelInformationCalculatorF32
import api_checks.utils as utils
from urllib.parse import quote

class APICache:
    def __init__(self, cache_path: Path, hf_token):
        self.cache_path = cache_path
        self.information_calculator_dict: dict[str, ModelInformationCalculatorF32] = {}
        self.hf_token = hf_token

    def get_full_run_results(self, model_name: str, prompt: str) -> FullRunResults:
        try:
            result, time = self._load(model_name, prompt)
            print(f"CACHE HIT {model_name} {prompt} {time}")
            return result
        except FileNotFoundError:
            if model_name not in self.information_calculator_dict:
                model = utils.get_model(model_name, self.hf_token)
                self.information_calculator_dict[model_name] = ModelInformationCalculatorF32(model)
            result = self.information_calculator_dict[model_name].calc(prompt)
            self._dump(model_name=model_name,prompt=prompt,result=result)
            print("Result saved in cache sucessfully")
            return result

    def get_infomration_calculator(self, model_name: str):
        if model_name not in self.information_calculator_dict:
            model = utils.get_model(model_name, self.hf_token)
            self.information_calculator_dict[model_name] = ModelInformationCalculatorF32(model)
        return self.information_calculator_dict[model_name]

    @classmethod
    def _get_key_name(cls, model_name: str, prompt: str) -> str:
        return quote(f"{model_name}|||{prompt}",safe='')

    def _dump(self, result: FullRunResults, model_name: str, prompt: str) -> Path:
        # serialize tensors + scalars to a single .pt file keyed by `key`
        self.cache_path.mkdir(parents=True, exist_ok=True)
        path = self.cache_path / f"{self._get_key_name(model_name,prompt)}.pt"
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
        path = self.cache_path / f"{self._get_key_name(model_name,prompt)}.pt"
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
