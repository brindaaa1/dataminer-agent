"""数据访问层：切分读取 + holdout 隔离（§4.0-1）。

holdout 只能由 evaluation.final_gate 读取：final_gate 调用 issue_holdout_token() 换取 capability token，
其他调用方拿不到 token（发放时检查调用者模块名），load("holdout") 一律抛 HoldoutAccessError。
"""
import hmac
import secrets
import sys
from pathlib import Path

import pandas as pd

_SECRET = secrets.token_hex(16)
_ALLOWED_ISSUER = "evaluation.final_gate"
TOOL_SPLITS = ("train", "valid", "oot_dev")


class HoldoutAccessError(PermissionError):
    pass


def issue_holdout_token() -> str:
    caller = sys._getframe(1).f_globals.get("__name__")
    if caller != _ALLOWED_ISSUER:
        raise HoldoutAccessError(f"{caller} 无权获取 holdout token")
    return _SECRET


class DataAccess:
    def __init__(self, dataset: str, artifacts_root: str = "artifacts"):
        self.dataset = dataset
        self.root = Path(artifacts_root)

    def load(self, split: str, token: str | None = None) -> pd.DataFrame:
        if split == "holdout":
            if token is None or not hmac.compare_digest(str(token), _SECRET):
                raise HoldoutAccessError("holdout 只能由 final_gate 读取")
            return pd.read_parquet(self.root / "holdout" / self.dataset / "holdout.parquet")
        if split not in TOOL_SPLITS:
            raise ValueError(f"未知切分 {split}")
        return pd.read_parquet(self.root / "splits" / self.dataset / f"{split}.parquet")
