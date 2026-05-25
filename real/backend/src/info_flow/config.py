
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
ENV_FILE = Path(__file__).resolve().parents[2] / ".env.local"   # adjust depth to repo root
model_config = SettingsConfigDict(env_file=ENV_FILE,env_file_encoding="utf-8",extra="ignore",)



class Config(BaseSettings):
    hf_token: str
    ndif_api_key: str
    info_flow_model: str

    default_atol: float
    default_rtol: float

    model_config = model_config
    