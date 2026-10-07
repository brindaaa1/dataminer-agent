"""INVESTIGATE 用的只读查询：只访问 train 切分，只返回聚合值，有行数上限（§4.6）。
指标都是任何二分类都成立的通用口径（不限于风控）：区分力（单特征 AUC、IV、互信息）、区分力在哪些取值上（分箱正类比例与提升度）、
时间稳定性（训练期前后两半的 PSI）、缺失是否有信息、特征冗余。"""
import numpy as np
import pandas as pd

from data.access import DataAccess
from evaluation.guardrails import psi, single_feature_auc
from features.screen import iv

QUESTIONS = ("single_feature_auc", "missing_rate", "iv", "mutual_info", "target_rate_by_bin",
             "stability_psi", "missing_vs_target", "redundant_pairs")
MAX_ROWS, TOP, BINS, DETAIL = 100_000, 10, 10, 3


def _numeric(x: pd.Series) -> pd.Series:
    """类别按各取值的出现频次编码，日期转成整数：只用于互信息、相关和 PSI 这类需要数值的统计。"""
    if pd.api.types.is_datetime64_any_dtype(x):
        return x.astype("int64").where(x.notna()).astype(float)
    if pd.api.types.is_numeric_dtype(x):
        return x.astype(float)
    return x.map(x.value_counts(dropna=True)).astype(float)


def _top(v: dict) -> dict:
    return dict(sorted(v.items(), key=lambda kv: -kv[1])[:TOP])


def _bins(x: pd.Series, y: pd.Series) -> list[dict]:
    """数值：分位数分箱（缺失单独一箱）；类别：取值数最多的若干个，其余归"其他"。提升度 = 箱内正类比例 / 整体。"""
    base = y.mean()
    if pd.api.types.is_numeric_dtype(x) and x.nunique() > BINS:
        b = pd.qcut(x, BINS, duplicates="drop").astype(str).where(x.notna(), "缺失")
    else:
        keep = x.astype(object).value_counts().index[:BINS]
        b = x.astype(object).where(x.isin(keep), "其他").where(x.notna(), "缺失").astype(str)
    g = pd.DataFrame({"b": b.values, "y": y.values}).groupby("b", sort=False)["y"].agg(["count", "mean"])
    if pd.api.types.is_numeric_dtype(x) and x.nunique() > BINS:          # 数值箱按取值顺序排
        order = sorted((k for k in g.index if k != "缺失"), key=lambda s: float(s.strip("([").split(",")[0]))
        g = g.reindex(order + (["缺失"] if "缺失" in g.index else []))
    return [{"bin": k, "n": int(r["count"]), "target_rate": round(float(r["mean"]), 4), "lift": round(float(r["mean"] / base), 2) if base else None}
            for k, r in g.iterrows()]


def answer(df: pd.DataFrame, question: str, features: list[str]) -> dict | list:
    """df：train 切分（含 _label；有 _obs_time 时可算时间稳定性）。返回聚合值，按问题取最相关的前几个。"""
    fs, y = [f for f in features if f in df], df["_label"]
    if question == "single_feature_auc":
        return _top({f: round(single_feature_auc(df[f], y), 4) for f in fs})
    if question == "missing_rate":
        return _top({f: round(float(df[f].isna().mean()), 4) for f in fs})
    if question == "iv":
        return _top({f: round(iv(df[f], y), 4) for f in fs})
    if question == "mutual_info":
        from sklearn.feature_selection import mutual_info_classif
        X = pd.DataFrame({f: _numeric(df[f]) for f in fs})
        X = X.fillna(X.median())
        mi = mutual_info_classif(X.values, y.values, random_state=0)
        return _top({f: round(float(v), 4) for f, v in zip(fs, mi)})
    if question == "target_rate_by_bin":
        pick = fs[:DETAIL] if len(fs) <= DETAIL else list(answer(df, "iv", fs))[:DETAIL]   # 太多时只看 IV 最高的几个
        return {f: _bins(df[f], y) for f in pick}
    if question == "stability_psi":
        t = df["_obs_time"] if "_obs_time" in df else pd.Series(np.arange(len(df)), index=df.index)
        early = t <= t.quantile(0.5)
        out = {}
        for f in fs:
            x = _numeric(df[f])
            a, b = x[early].dropna(), x[~early].dropna()
            if len(a) > 50 and len(b) > 50:
                out[f] = round(float(psi(a.values, b.values)), 4)
        return _top(out)
    if question == "missing_vs_target":
        out = {}
        for f in fs:
            m = df[f].isna()
            if 0.01 < m.mean() < 0.99:
                out[f] = {"missing_rate": round(float(m.mean()), 4), "target_rate_missing": round(float(y[m].mean()), 4),
                          "target_rate_present": round(float(y[~m].mean()), 4)}
        return dict(sorted(out.items(), key=lambda kv: -abs(kv[1]["target_rate_missing"] - kv[1]["target_rate_present"]))[:TOP])
    if question == "redundant_pairs":
        X = pd.DataFrame({f: _numeric(df[f]) for f in fs}).corr(method="spearman").abs()
        pairs = [{"pair": [a, b], "corr": round(float(X.loc[a, b]), 4)} for i, a in enumerate(fs) for b in fs[i + 1:]
                 if pd.notna(X.loc[a, b]) and X.loc[a, b] > 0.9]
        return sorted(pairs, key=lambda p: -p["corr"])[:TOP]
    raise ValueError(question)


def investigate(dataset: str, cfg: dict, question: str, features: list[str]) -> dict | list:
    df = DataAccess(dataset, cfg["paths"]["artifacts_root"]).load("train")
    return answer(df.sample(min(len(df), MAX_ROWS), random_state=0), question, features)
