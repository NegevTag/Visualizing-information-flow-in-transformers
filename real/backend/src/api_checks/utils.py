import datetime
from pathlib import Path

import nnsight


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
