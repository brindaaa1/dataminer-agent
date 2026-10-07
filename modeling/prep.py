"""特征预处理：date 类型字段（从 data_dictionary 的 type: date 读取）转为"距观察时间的月数"，object 转 category。"""
import pandas as pd

from data.access import numpy_types

META = ("_row_id", "_label", "_split_time", "_obs_time")


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META]


def prepare(df: pd.DataFrame, date_fields: list[str], cats: dict | None = None, date_fmt: str | None = None):
    """返回 (X, cats)。cats 记录训练集的类别集合，valid/oot 复用，避免类别不一致。"""
    X = df[feature_cols(df)].copy()
    for c in date_fields:
        if c in X:
            d = pd.to_datetime(X[c], format=date_fmt, errors="coerce")
            X[c] = (df["_obs_time"].dt.year - d.dt.year) * 12 + (df["_obs_time"].dt.month - d.dt.month)
    for c in X.columns:                                       # 真正的日期类型列 → 相对观察时间的天数（无观察时间则用 epoch 天数）
        if pd.api.types.is_datetime64_any_dtype(X[c]):
            ref = df["_obs_time"] if "_obs_time" in df else pd.Timestamp("1970-01-01")
            X[c] = (X[c] - ref).dt.days.astype(float)
    numpy_types(X)                                            # 特征集挂上来的列也可能是可空类型
    fit = cats is None
    cats = {} if fit else cats
    for c in X.columns:
        if X[c].dtype == object or str(X[c].dtype) == "category":
            if fit:
                cats[c] = sorted(X[c].dropna().astype(str).unique())
            X[c] = pd.Categorical(X[c].astype(str).where(X[c].notna()), categories=cats[c])
    return X, cats
