from pydantic import BaseModel


class LLMResidualPosition(BaseModel):
    layer:int
    token_position:int
    is_mlp:bool = False