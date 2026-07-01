from pathlib import Path

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from api_checks.api_cache import APICache
from api_checks.position import LLMResidualPosition
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


class Args(BaseModel):
    model: str = config.info_flow_model
    prompt: str | None


app.state.args = Args(prompt=None)


@app.get("/")
def calc_norms(prompt: str):
    app.state.args.prompt = prompt
    app.state.args.model = config.info_flow_model
    information = api_cache.get_full_run_results(prompt=prompt, model_name=config.info_flow_model)
    calculator = api_cache.get_infomration_calculator(config.info_flow_model)
    tokens = calculator.calc_tokens(prompt)
    mlp_norms = information.contributions.post_mlp_contribution.norm(dim=-1)
    attention_norms = information.contributions.post_attention_contribution.norm(dim=-1)
    top_perdictions = calculator.tokens_probabilities_from_logits(information.logits[-1])
    return ReturnInfo(attention_norms=attention_norms, mlp_norms=mlp_norms, tokens=tokens, top_perdictions=top_perdictions)

@app.post("/load_unembedding")
def load_unembeddings() -> None:
    api_cache.load_unembedding_matrix(app.state.args.model)



if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
