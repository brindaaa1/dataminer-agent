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


def numpy_types(df: pd.DataFrame) -> dict[str, str]:
    """pandas 可空类型（boolean、Int64 等）就地转成 float64，缺失变 NaN：pd.NA 让 CatBoost、IV 分箱等下游报错（v5 评测 home_credit）。
    返回 {列名: 原类型}。写切分文件时调用一次（data/splits.py）；特征工程挂上来的列在 modeling/prep.py 里再调一次。"""
    out = {}
    for c in df.columns:
        if isinstance(df[c].dtype, pd.api.extensions.ExtensionDtype) and (pd.api.types.is_bool_dtype(df[c]) or pd.api.types.is_numeric_dtype(df[c])):
            out[c] = str(df[c].dtype)
            df[c] = df[c].astype("float64")
    return out


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
