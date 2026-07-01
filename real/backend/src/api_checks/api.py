from pathlib import Path
from typing import Any

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from api_checks.api_cache import APICache
from api_checks.position import LLMResidualPosition
import api_checks.utils as utils
from info_flow.config import Config
from pydantic import BaseModel
import safetensors.torch
import uvicorn

app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
config = Config()
api_cache = APICache(hf_token=config.hf_token, cache_path=Path(config.result_cache_path))


class ReturnInfo(BaseModel):
    attention_norms: list[list[list[float]]]  # (layer,position,source)
    mlp_norms: list[list[list[float]]]  # (layer,position,source)
    tokens: list[str]
    top_perdictions: dict[str, float]

class ContributionsReturnInfo(BaseModel):
    attention_norms: list[list[list[float]]]  # (layer,position,source)
    mlp_norms: list[list[list[float]]]  #

class Args(BaseModel):
    model: str = config.info_flow_model
    prompt: str | None
    mask: list | None


app.state.args = Args(prompt=None, mask=None)


@app.get("/")
def calc_norms(prompt: str):
    app.state.args.prompt = prompt
    app.state.args.model = config.info_flow_model
    information = api_cache.get_full_run_results(prompt=prompt, model_name=config.info_flow_model)
    calculator = api_cache.get_infomration_calculator(config.info_flow_model)
    tokens = calculator.calc_tokens(prompt)
    mlp_norms = information.contributions.post_mlp_contribution.norm(dim=-1)
    attention_norms = information.contributions.post_attention_contribution.norm(dim=-1)
    logits = calculator.calc_logits(information.contributions.post_mlp_contribution[-1].sum(dim=1))
    top_perdictions = calculator.tokens_probabilities_from_logits(logits[-1])
    return ReturnInfo(attention_norms=attention_norms, mlp_norms=mlp_norms, tokens=tokens, top_perdictions=top_perdictions)


@app.post("/load_unembedding")
def load_unembeddings() -> None:
    api_cache.load_unembedding_matrix(app.state.args.model)


@app.post("/apply_mask")
def get_contributions_grouped_by_mask(mask: list[Any]):
    app.state.args.mask = mask
    masked_contributions =  api_cache.get_contributions(app.state.args.model, app.state.args.prompt, tuple(mask))
    return ContributionsReturnInfo(attention_norms=masked_contributions.post_attention_contribution.norm(dim=-1),
                                   mlp_norms = masked_contributions.post_mlp_contribution.norm(dim=-1))

@app.post("/group_by_words")
def get_contributions_grouped_by_words():
    tokens = api_cache.get_infomration_calculator(app.state.args.model).calc_tokens(app.state.args.prompt)
    mask = utils.get_group_by_words_mask(tokens)
    return get_contributions_grouped_by_mask(mask)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
