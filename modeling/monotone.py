"""单调约束（§4.12）：方向来自 knowhow/<数据集>/priors.yaml 的 target_direction / 模板的 monotone_prior，不是 LLM 临场生成。
只给"有先验且方向明确"的数值特征加约束；先验与数据明显相反的特征跳过并写进报告（数据一致性守卫）。"""

import pandas as pd

from data.knowhow import target_direction
from features.registry import Registry


def priors_for(dataset: str, cfg: dict, config: dict) -> dict:
    """{特征名: 目标方向 ±1/0}。优先级：模板 agg 上的 prior > 数据集的 priors.yaml（含别名）。"""
    base = target_direction(dataset, cfg)
    alias = cfg["task_specs"].get(dataset, {}).get("prior_aliases", {})          # 预处理后名字对不上先验时的映射
    out = {f: base[a] for f, a in alias.items() if a in base}
    out.update({k: v for k, v in base.items() if isinstance(v, int)})
    for fs in config.get("feature_sets") or []:
        for d in Registry(cfg).get(fs)["defs"]:
            if d.get("prior") is not None:
                out[d["name"]] = d["prior"]
    return out


def constraints_for(dataset: str, cfg: dict, config: dict, X: pd.DataFrame, y: pd.Series):
    """返回 (与 X.columns 对齐的 monotone_constraints 向量, 报告)。
    报告：constrained（已约束）、no_prior（没有先验，不约束）、categorical、disagree（先验与数据相反，跳过）。"""
    pri = priors_for(dataset, cfg, config)
    min_rho = cfg["monotone"]["disagree_min_abs_rho"]
    vec, rep = [], {"constrained": [], "no_prior": [], "zero_prior": [], "categorical": [], "disagree": {}}
    for c in X.columns:
        d = pri.get(c)
        if str(X[c].dtype) == "category" or not pd.api.types.is_numeric_dtype(X[c]):
            vec.append(0); rep["categorical"].append(c); continue
        if d is None:
            vec.append(0); rep["no_prior"].append(c); continue
        if d == 0:
            vec.append(0); rep["zero_prior"].append(c); continue
        rho = pd.Series(X[c]).corr(pd.Series(y.values, index=X.index), method="spearman")
        if pd.notna(rho) and abs(rho) >= min_rho and (rho > 0) != (d > 0):        # 数据明确说方向相反 → 不强加先验
            vec.append(0); rep["disagree"][c] = round(float(rho), 4); continue
        vec.append(int(d)); rep["constrained"].append(c)
    return vec, rep
