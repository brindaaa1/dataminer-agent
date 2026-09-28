"""筛选漏斗（§4.11）：缺失率 → IV → 单特征 AUC 报警 → 相关性去重 → PSI。全部确定性代码，阈值来自 config.screen。"""
import numpy as np
import pandas as pd

from evaluation.guardrails import psi, single_feature_auc


def iv(x: pd.Series, y: pd.Series, bins: int = 10) -> float:
    """分位数分箱 + 缺失单独一箱的 Information Value。"""
    if pd.api.types.is_datetime64_any_dtype(x):
        x = x.astype("int64").where(x.notna())
    if not pd.api.types.is_numeric_dtype(x):                # 类别特征：每个取值（含缺失）一箱
        b = x.astype(object).where(x.notna(), "__na__").astype(str)
        g = pd.DataFrame({"b": b.values, "y": y.values}).groupby("b")["y"].agg(["sum", "count"])
        bad, good = g["sum"] + 0.5, g["count"] - g["sum"] + 0.5
        pb, pg = bad / bad.sum(), good / good.sum()
        return float(((pb - pg) * np.log(pb / pg)).sum())
    if x.notna().sum() < 50:
        return 0.0
    edges = np.unique(np.nanquantile(x, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        b = pd.Series(np.where(x.isna(), -1, 0), index=x.index)
    else:
        b = pd.Series(pd.cut(x, edges, labels=False, include_lowest=True), index=x.index).fillna(-1)
    g = pd.DataFrame({"b": b, "y": y.values}).groupby("b")["y"].agg(["sum", "count"])
    bad, good = g["sum"] + 0.5, g["count"] - g["sum"] + 0.5
    pb, pg = bad / bad.sum(), good / good.sum()
    return float(((pb - pg) * np.log(pb / pg)).sum())


def screen(X: pd.DataFrame, y: pd.Series, X_oot: pd.DataFrame, cfg: dict) -> dict:
    """X/X_oot：只含候选特征列。返回 kept / dropped(原因) / leak_suspect / 每个特征的指标。"""
    c, thr = cfg["screen"], cfg["evaluator"]["leak_single_auc"]
    dropped, leak, metrics = {}, {}, {}
    cand = list(X.columns)
    for f in cand:
        metrics[f] = {"missing": float(X[f].isna().mean())}
    for f in [f for f in cand if metrics[f]["missing"] > c["missing_max"]]:
        dropped[f] = "missing"
    cand = [f for f in cand if f not in dropped]
    for f in cand:
        metrics[f]["iv"] = iv(X[f], y, c["iv_bins"])
    for f in [f for f in cand if metrics[f]["iv"] < c["iv_min"]]:
        dropped[f] = "low_iv"
    cand = [f for f in cand if f not in dropped]
    for f in cand:
        metrics[f]["auc"] = single_feature_auc(X[f], y)
    for f in [f for f in cand if metrics[f]["auc"] > thr]:
        leak[f] = metrics[f]["auc"]                       # 不直接丢弃：交给 guardrail / 人工，不进入 kept
    cand = [f for f in cand if f not in leak]
    # 相关性去重：按 IV 从高到低，与已保留特征相关 > 阈值则丢弃
    if cand:
        sample = X[cand].sample(min(len(X), 50_000), random_state=0)
        corr = sample.apply(pd.to_numeric, errors="coerce").fillna(sample.median()).corr().abs()
        kept = []
        for f in sorted(cand, key=lambda f: -metrics[f]["iv"]):
            hit = next((k for k in kept if corr.loc[f, k] > c["corr_max"]), None)
            if hit:
                dropped[f] = f"corr>{c['corr_max']} with {hit}"
            else:
                kept.append(f)
        cand = kept
    for f in list(cand):
        metrics[f]["psi"] = psi(X[f], X_oot[f], cfg["evaluator"]["psi_bins"])
        if metrics[f]["psi"] > c["psi_max"]:
            dropped[f] = "psi"
    kept = [f for f in cand if f not in dropped]
    return {"kept": kept, "dropped": dropped, "leak_suspect": leak, "metrics": metrics,
            "funnel": {"candidates": len(X.columns), "after_missing": len(X.columns) - sum(v == "missing" for v in dropped.values()),
                       "kept": len(kept), "leak_suspect": len(leak),
                       "dropped_by": {r: sum(1 for v in dropped.values() if v.split(">")[0].split(" ")[0] == r) for r in ("missing", "low_iv", "corr", "psi")}}}
