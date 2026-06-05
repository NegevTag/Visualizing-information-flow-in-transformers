from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from info_flow.config import Config
from info_flow.ex6_better_percision_key_in_mat_f32 import ModelInformationCalculatorF32
from pydantic import BaseModel
import uvicorn
app = FastAPI()

# Allow the vite dev server (a different origin) to read responses.
# Without this the browser blocks the JSON even though the request returns 200.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ReturnInfo(BaseModel):
    attention_norms:list[list[list[float]]] #(layer,position,source)
    mlp_norms: list[list[list[float]]] #(layer,position,source)
    tokens: list[str]
    

@app.get("/")
def calc_norms(prompt:str):
    config = Config()
    calculator = ModelInformationCalculatorF32(config.info_flow_model,config.hf_token)
    tokens = calculator.calc_tokens(prompt)
    information = calculator.calc(prompt)
    mlp_norms = information.contributions.post_mlp_contribution.norm(dim=-1)
    attention_norms = information.contributions.post_attention_contribution.norm(dim=-1)
    return ReturnInfo(attention_norms=attention_norms,mlp_norms=mlp_norms,tokens=tokens)


@app.get("/toy_model")
def calc_norms(prompt:str):
    config = Config()
    calculator = ModelInformationCalculatorF32(config.info_flow_model,config.hf_token)
    calculator.model =
    tokens = calculator.calc_tokens(prompt)
    information = calculator.calc(prompt)
    mlp_norms = information.contributions.post_mlp_contribution.norm(dim=-1)
    attention_norms = information.contributions.post_attention_contribution.norm(dim=-1)
    return ReturnInfo(attention_norms=attention_norms,mlp_norms=mlp_norms,tokens=tokens)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)