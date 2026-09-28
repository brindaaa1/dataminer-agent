"""配对 bootstrap：同一批 OOT-dev 样本上比较两个预测的 AUC 差，置信区间下界 > 0 才算显著提升（§4.9）。"""
import numpy as np


def _auc(y, p):
    """秩和法 AUC，向量化，处理并列。"""
    from scipy.stats import rankdata
    r = rankdata(p)
    n1 = y.sum()
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def paired_bootstrap(y, p_new, p_base, alpha: float, n: int = 1000, seed: int = 0) -> dict:
    """返回 delta、单侧 (1-alpha) 置信区间下界/上界。用同一批重采样索引，保证配对。"""
    y, p_new, p_base = map(np.asarray, (y, p_new, p_base))
    rng = np.random.default_rng(seed)
    idx_all = np.arange(len(y))
    deltas = np.empty(n)
    for i in range(n):
        idx = rng.choice(idx_all, len(y))
        deltas[i] = _auc(y[idx], p_new[idx]) - _auc(y[idx], p_base[idx])
    lo = float(np.quantile(deltas, alpha))          # 单侧：下界取 alpha 分位
    hi = float(np.quantile(deltas, 1 - alpha))
    delta = float(_auc(y, p_new) - _auc(y, p_base))
    return {"delta": delta, "ci_low": lo, "ci_high": hi, "alpha": alpha, "significant": lo > 0}


def monthly_auc(times, y, p, min_n: int = 200) -> dict:
    """按月份分组的 AUC；样本太少或只有单一类别的月份跳过。"""
    import pandas as pd
    df = pd.DataFrame({"m": pd.to_datetime(times).dt.strftime("%Y-%m"), "y": np.asarray(y), "p": np.asarray(p)})
    out = {}
    for m, g in df.groupby("m"):
        if len(g) >= min_n and g.y.nunique() == 2:
            out[m] = {"auc": float(_auc(g.y.values, g.p.values)), "n": int(len(g)), "bad_rate": float(g.y.mean())}
    return out
