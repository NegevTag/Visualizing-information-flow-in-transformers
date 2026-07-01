import datetime
from pathlib import Path
from typing import Any

import nnsight
import torch
from api_checks.full_run_result import Contributions


class NotEveryTokenIsAssigned(Exception):
    pass


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


def get_group_by_words_mask(tokens: list[str]) -> list[int]:
    group_ind = 1
    groups = [0,1]
    for token in tokens[2:]:
        if not token[0].isalpha():
            group_ind += 1
        groups.append(group_ind)
    return groups


def group_contributions(mask: list[Any], contributions: Contributions) -> Contributions:
    if len(mask) != contributions.post_mlp_contribution.shape[2]:
        raise NotEveryTokenIsAssigned(f"No every token is assigned: group_len {len(mask)}, number of tokens {contributions.post_mlp_contribution.shape[2]}")
    group_identifiers = []
    indexed_mask = []
    for a in mask:
        if a not in group_identifiers:
            group_identifiers.append(a)
        indexed_mask.append(group_identifiers.index(a))
    mask_matrix = torch.nn.functional.one_hot(torch.tensor(mask, dtype=torch.long)).float()
    grouped_post_mlp_contributions = torch.einsum("lpsd,sg -> lpgd ", contributions.post_mlp_contribution, mask_matrix)
    grouped_post_attetnion_contributions = torch.einsum("lpsd,sg -> lpgd", contributions.post_attention_contribution, mask_matrix)
    return Contributions(post_mlp_contribution=grouped_post_mlp_contributions, post_attention_contribution=grouped_post_attetnion_contributions)
