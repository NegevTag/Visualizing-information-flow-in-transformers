
from fastapi import FastAPI
from info_flow.config import Config
from tests.scratchpad.toy_llama_no_attention_no_ov import ToyLlamaNoAttentionNoOV
from info_flow.ex6_better_percision_key_in_mat_f32 import ModelInformationCalculatorF32
from pydantic import BaseModel
import uvicorn

app = FastAPI()


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


@app.get("/toy_model")
def calc_norms(prompt: str):
    config = Config()
    calculator = ModelInformationCalculatorF32(config.info_flow_model, config.hf_token)
    calculator.model = ToyLlamaNoAttentionNoOV.build_nnsight_mode()
    tokens = calculator.calc_tokens(prompt)
    information = calculator.calc(prompt)
    mlp_norms = information.contributions.post_mlp_contribution.norm(dim=-1)
    attention_norms = information.contributions.post_attention_contribution.norm(dim=-1)
    return ReturnInfo(attention_norms=attention_norms, mlp_norms=mlp_norms, tokens=tokens)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
