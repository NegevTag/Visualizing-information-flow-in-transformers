import datetime
from pathlib import Path
from typing import Any

import nnsight
import safetensors
import torch

from api_checks.full_run_result import FullRunResults


def get_model(model_name: str, hf_token: str) -> nnsight.LanguageModel:
    model_kwargs_dict = {"token": hf_token}
    return nnsight.LanguageModel(model_name, **model_kwargs_dict)  # type: ignore[arg-type]


def get_creation_datetime(path: Path) -> datetime.datetime:
    stat = path.stat()
    try:
        time_stamp_file_creation = stat.st_birthtime
    except AttributeError:
        time_stamp_file_creation = stat.st_mtime
    except AttributeError:
        time_stamp_file_creation = stat.st_ctime
    return datetime.datetime.fromtimestamp(time_stamp_file_creation)


# def group_full_run_result(token_group_idnetifier: dict[str, Any], full_run_result: FullRunResults) -> FullRunResults:
#     group_identifiers = []
#     mask = []
#     for token, token_group_idnetifier in token_group_idnetifier.items():
#         if token_group_idnetifier not in group_identifiers:
#             group_identifiers.append(token_group_idnetifier)
#         mask += group_identifiers.index(token_group_idnetifier)
#     full_run_result = 
