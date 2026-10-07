"""解释：特征贡献。lgbm 用 LightGBM 自带的 TreeSHAP（pred_contrib）；lr_scorecard 用 |系数| 近似；
其他模型（catboost、random_forest …）用置换近似：把一列打乱后预测的变化。不依赖 shap 包，模型库新加的模型也能用。"""
import numpy as np
import pandas as pd

from data.access import DataAccess
from modeling.inner_loop import fit_model


def contributions(m, model: str, X: pd.DataFrame) -> np.ndarray:
    """每行每个特征的贡献（n × p）。"""
    if model == "lgbm":
        return m.model.predict(X, pred_contrib=True, num_iteration=m.best_iter)[:, :-1]
    if model == "lr_scorecard":
        return m._transform(X).values * m.lr.coef_[0]
    base, rng, out = m.predict(X), np.random.default_rng(0), np.empty(X.shape)
    for j, c in enumerate(X.columns):
        Xp = X.copy()
        Xp[c] = X[c].values[rng.permutation(len(X))]
        out[:, j] = base - m.predict(Xp)
    return out


def explain(dataset, cfg, result: dict, n: int = 5000) -> dict:
    """返回 {feature: {"share": 平均|SHAP|占比, "corr": 特征值与 SHAP 的秩相关（方向）}}。只使用 oot_dev 样本。"""
    m, prep = fit_model(dataset, cfg, result["config"], result["best_params"])
    oo = DataAccess(dataset, cfg["paths"]["artifacts_root"]).load("oot_dev")
    oo = oo.sample(min(n, len(oo)), random_state=0)
    X = prep(oo, "oot_dev")
    sv = contributions(m, result["config"]["model"], X)
    imp = np.abs(sv).mean(0)
    out = {}
    for j, f in enumerate(X.columns):
        x = pd.to_numeric(X[f], errors="coerce") if str(X[f].dtype) != "category" else pd.Series(X[f].cat.codes, index=X.index).astype(float)
        c = pd.Series(sv[:, j], index=X.index).corr(x, method="spearman") if x.nunique() > 1 else 0.0
        out[f] = {"share": float(imp[j] / imp.sum()) if imp.sum() else 0.0, "corr": float(c) if pd.notna(c) else 0.0}
    return out
