"""在干净数据里人为放进已知的数据问题，同时记下标准答案。每个函数只改副本，返回（改过的表，答案）。"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Answer:
    kind: str                  # sentinel / duplicate_rows / duplicate_key / category_variants / missing_months / post_outcome / after_cutoff
    table: str
    column: str | None         # 整张表层面的问题（重复行）为 None
    detail: dict = field(default_factory=dict)
    must_find: bool = False    # 每次都必须找到
    native: bool = False       # 数据原本就有，不是人为放的
    optional: bool = False     # 找到算对、没找到不扣分（可得时间存疑的字段）


def _pick(cand: np.ndarray, frac: float, rng, at_least: int = 1) -> np.ndarray:
    return rng.choice(cand, size=max(at_least, int(len(cand) * frac)), replace=False)


def sentinel(df, table, col, value, frac, rng, only_missing=False):
    """把一部分值换成占位值。only_missing：只换原本缺失的行，模拟"系统把缺失填成 999"。"""
    assert not df[col].eq(value).any(), f"{table}.{col} 里本来就有 {value}，答案会分不清"
    out = df.copy()
    cand = np.flatnonzero(out[col].isna()) if only_missing else np.arange(len(out))
    idx = _pick(cand, frac, rng)
    out.iloc[idx, out.columns.get_loc(col)] = value
    return out, Answer("sentinel", table, col, {"value": value, "n_rows": len(idx)})


def duplicate_rows(df, table, frac, rng):
    idx = _pick(np.arange(len(df)), frac, rng)
    out = pd.concat([df, df.iloc[idx]], ignore_index=True)
    out = out.sample(frac=1, random_state=int(rng.integers(2**31))).reset_index(drop=True)
    return out, Answer("duplicate_rows", table, None, {"n_rows": len(idx)})


def duplicate_key(df, table, key, frac, rng):
    """追加几行：内容取自一些行，编号取自另一些行。同一个编号出现两次、内容不同；原有的行不动，明细表的关联不断。"""
    n = max(1, int(len(df) * frac))
    tgt, src = rng.choice(len(df), size=(2, n), replace=False)
    extra = df.iloc[tgt].copy()
    extra[key] = df[key].to_numpy()[src]
    out = pd.concat([df, extra], ignore_index=True).sample(frac=1, random_state=int(rng.integers(2**31))).reset_index(drop=True)
    return out, Answer("duplicate_key", table, key, {"n_rows": n})


def category_variants(df, table, col, value, variants, frac, rng):
    """同一个类别改成几种写法，比如 RENT → rent、" RENT"。"""
    assert not df[col].isin(variants).any(), f"{table}.{col} 里本来就有 {variants} 中的写法"
    out = df.copy()
    idx = _pick(np.flatnonzero(out[col].to_numpy() == value), frac, rng, at_least=len(variants))
    out.iloc[idx, out.columns.get_loc(col)] = np.resize(np.array(variants, dtype=object), len(idx))
    return out, Answer("category_variants", table, col, {"value": value, "variants": variants, "n_rows": len(idx)})


def missing_months(df, table, col, period, months):
    """去掉某几个月的数据。period 是每行所在的年月（如 "2016-03"），由调用方从各自的时间列算出。"""
    lo, hi = period.min(), period.max()
    assert all(lo < m < hi for m in months), "去掉的月份要在中间；在两头看起来只是数据起止时间不同"
    drop = period.isin(months).to_numpy()
    return df[~drop].reset_index(drop=True), Answer("missing_months", table, col, {"months": months, "n_rows": int(drop.sum())})


def post_outcome(df, table, col, y, noise, rng):
    """加一列结果出来之后才有的字段：约 (1 - noise) 与结果一致。"""
    out = df.copy()
    flip = rng.random(len(out)) < noise
    out[col] = np.where(flip, 1 - y, y) + rng.normal(0, 0.05, len(out))
    return out, Answer("post_outcome", table, col, {"noise": noise})


def after_cutoff(df, table, col, frac, rng, max_days=365):
    """明细表的事件日（相对申请日的天数，负数 = 申请之前）改成申请之后。"""
    assert (df[col] <= 0).all(), f"{table}.{col} 里本来就有申请之后的记录"
    out = df.copy()
    idx = _pick(np.arange(len(out)), frac, rng)
    out.iloc[idx, out.columns.get_loc(col)] = rng.integers(1, max_days, len(idx))
    return out, Answer("after_cutoff", table, col, {"n_rows": len(idx)})
