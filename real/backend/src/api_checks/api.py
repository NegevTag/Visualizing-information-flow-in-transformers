from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api_checks.model import ModelInformationCalculatorF32
from info_flow.config import Config
from pydantic import BaseModel
import uvicorn


app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ReturnInfo(BaseModel):
    attention_norms: list[list[list[float]]]  # (layer,position,source)
    mlp_norms: list[list[list[float]]]  # (layer,position,source)
    tokens: list[str]


@app.get("/")
def calc_norms(prompt: str):
    config = Config()
    calculator = ModelInformationCalculatorF32(config.info_flow_model, config.hf_token)
    tokens = calculator.calc_tokens(prompt)
    information = calculator.calc(prompt)
    mlp_norms = information.contributions.post_mlp_contribution.norm(dim=-1)
    attention_norms = information.contributions.post_attention_contribution.norm(dim=-1)
    return ReturnInfo(attention_norms=attention_norms, mlp_norms=mlp_norms, tokens=tokens)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
