"""确定性诊断码（§4.8）。输入实验指标与特征信息，输出诊断码列表 + 明细。"""
import numpy as np
import pandas as pd

from evaluation.compare import _auc


def psi(expected, actual, bins: int = 10) -> float:
    e, a = np.asarray(expected, float), np.asarray(actual, float)
    e, a = e[~np.isnan(e)], a[~np.isnan(a)]
    if len(e) == 0 or len(a) == 0:
        return float("nan")
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    pe = np.histogram(e, edges)[0] / len(e)
    pa = np.histogram(a, edges)[0] / len(a)
    pe, pa = np.clip(pe, 1e-6, None), np.clip(pa, 1e-6, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def single_feature_auc(x: pd.Series, y: pd.Series) -> float:
    """缺失值填成极小值（保留缺失与 label 的关系）；方向无关，取 max(auc, 1-auc)。"""
    if pd.api.types.is_datetime64_any_dtype(x):               # 日期列：按时间先后排序即可（NaT 保持缺失）
        x = x.astype("int64").where(x.notna())
    if x.dtype == object or str(x.dtype) == "category":
        x = x.astype("category").cat.codes.astype(float).where(x.notna())
    v = x.astype(float)
    v = v.fillna(v.min() - 1 if v.notna().any() else 0)
    a = _auc(y.values, v.values)
    return float(max(a, 1 - a))


def diagnose(metrics: dict, cfg: dict, *, n_features: int, score_psi: float | None = None,
             feature_aucs: dict | None = None, unavailable_fields: list | None = None,
             leak_pattern_hits: list | None = None, history_verdicts: list | None = None,
             remaining_budget_frac: float | None = None, best_iter: int | None = None,
             n_rounds_max: int | None = None, data_fatal: bool = False, oom: bool = False,
             model: str = "lgbm") -> dict:
    """返回 {"codes": [...], "details": {code: ...}}。所有阈值来自 config.evaluator。"""
    c = cfg["evaluator"]
    codes, det = [], {}

    if metrics["valid_auc"] - metrics["oot_dev_auc"] > c["gap_max"]:
        codes.append("OVERFIT_GAP"); det["OVERFIT_GAP"] = {"gap": metrics["valid_auc"] - metrics["oot_dev_auc"]}

    leak = {}
    for f, a in (feature_aucs or {}).items():
        if a > c["leak_single_auc"]:
            leak[f] = {"reason": "single_feature_auc", "auc": a}
    for f in unavailable_fields or []:
        leak.setdefault(f, {"reason": "availability_unknown"})
    for f in leak_pattern_hits or []:
        leak.setdefault(f, {"reason": "name_pattern"})
    if leak:
        codes.append("LEAK_SUSPECT"); det["LEAK_SUSPECT"] = leak

    if score_psi is not None and score_psi > c["psi_warn"]:
        codes.append("PSI_DRIFT")
        det["PSI_DRIFT"] = {"score_psi": score_psi, "severity": "severe" if score_psi > c["psi_severe"] else "warn"}

    if history_verdicts:
        n = c["no_progress_rounds"]
        if len(history_verdicts) >= n and all(v in ("REJECT", "INCONCLUSIVE") for v in history_verdicts[-n:]):
            codes.append("NO_PROGRESS")
    if remaining_budget_frac is not None and remaining_budget_frac < c["budget_low_frac"]:
        codes.append("BUDGET_LOW")
    if model == "lgbm" and best_iter is not None and n_rounds_max and best_iter < n_rounds_max * c["early_stop_min_frac"]:
        codes.append("NO_CONVERGE"); det["NO_CONVERGE"] = {"best_iter": best_iter}
    if data_fatal:
        codes.append("DATA_FATAL")
    if oom:
        codes.append("OOM")
    if n_features > c["max_features"]:
        det["TOO_MANY_FEATURES"] = {"n_features": n_features}
    return {"codes": codes, "details": det}


# 诊断码 → 候选动作（策略先验，§4.8；DECIDE 默认从中选择）
POLICY_PRIOR = {
    "OVERFIT_GAP": ["PRUNE_FEATURES", "TUNE", "SWITCH_MODEL"],
    "LEAK_SUSPECT": ["ESCALATE_HUMAN"],
    "PSI_DRIFT": ["PRUNE_FEATURES"],
    "PLATEAU": ["EXPAND_FEATURES", "TUNE"],
    "NO_CONVERGE": ["TUNE"],
    "NO_PROGRESS": ["BACKTRACK", "EXPAND_FEATURES", "STOP"],
    "BUDGET_LOW": ["PROMOTE_FIDELITY", "STOP"],
    "DATA_FATAL": ["ESCALATE_HUMAN"],
    "OOM": ["TUNE", "SWITCH_MODEL"],
    "NO_FINAL_MODEL": ["PROMOTE_FIDELITY"],
}


def promotable(recs) -> list[dict]:
    """可以升到全量复验的低保真实验（含模型赛跑）：没被 guardrail 拒掉（PROMISING 或 INCONCLUSIVE）、没失效、没升过。
    INCONCLUSIVE 也算：模型赛跑里 AUC 略低但更稳的模型（如评分卡）常被判不确定，只收 PROMISING 会把它们排除在外，
    最终模型就只能在同一个模型上打转（hotel_deepseek2：lightgbm 三次全量复验都被 OVERFIT_GAP 拒）。
    每条带上 OOT-dev AUC 和 gap（valid − OOT-dev），让 LLM 同时看到效果和稳定性。"""
    done = {r.diff.get("exp_id") for r in recs if r.action_type == "PROMOTE_FIDELITY"}
    out = []
    for r in recs:
        if r.fidelity == "low" and r.verdict in ("PROMISING", "INCONCLUSIVE") and not r.invalidated and r.exp_id not in done:
            m = r.metrics
            out.append({"exp_id": r.exp_id, "model": r.config.get("model"), "verdict": r.verdict,
                        "oot_dev_auc": round(m["oot_dev_auc"], 4) if "oot_dev_auc" in m else None,
                        "gap": round(m["valid_auc"] - m["oot_dev_auc"], 4) if {"valid_auc", "oot_dev_auc"} <= m.keys() else None})
    return out


def decision_codes(p: dict, recs) -> list[str]:
    """DECIDE 看到的诊断码 = 上一个实验的诊断 + 任务级状态。
    NO_FINAL_MODEL：还没有全保真度的最优模型，但已有可升级的低保真实验。只有全保真度实验能被 ACCEPT，
    不升保真度就不会有最终模型（hotel_deepseek1 连跑 5 轮低保真调参，最后没有模型）。上下文和校验都用这个函数，保证一致。"""
    codes = list(p.get("last_codes", []))
    if not p.get("best_exp_id") and promotable(recs):
        codes.append("NO_FINAL_MODEL")
    return codes
