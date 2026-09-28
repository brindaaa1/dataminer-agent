"""结构化查询组装 DECIDE 上下文（§4.13）：不用滑动窗口，按当前分支路径、全局最优、最近被拒来查。"""
from evaluation.guardrails import POLICY_PRIOR, decision_codes, promotable
from modeling.zoo import ZOO


def _line(r):
    m = r.metrics or {}
    return {"exp_id": r.exp_id, "action": r.action_type, "model": r.config.get("model"), "fidelity": r.fidelity,
            "oot_dev_auc": round(m["oot_dev_auc"], 4) if "oot_dev_auc" in m else None,
            "gap": round(m["gap"], 4) if "gap" in m else None, "verdict": r.verdict, "codes": r.diagnosis_codes}


def build_context(store, p: dict, cfg: dict, spec, oot_budget, remaining: dict, models: list[str]) -> dict:
    k = cfg["decide"]["topk"]
    tid = p["task_id"]
    recs = {r.exp_id: r for r in store.all(tid)}
    codes = decision_codes(p, list(recs.values()))
    prior = {c: POLICY_PRIOR[c] for c in codes if c in POLICY_PRIOR}
    node = recs.get(p["current_node"])
    return {
        "mode": "DECIDE",
        "spec": {"dataset": spec.dataset, "label_def": spec.label_def, "target": [spec.target_metric, spec.target_value],
                 "constraints": spec.constraints.model_dump(), "autonomy": spec.autonomy},
        "round_no": p["round_no"] + 1, "max_rounds": cfg["run"]["max_rounds"],
        "budget_remaining": remaining,
        "best": _line(recs[p["best_exp_id"]]) if p["best_exp_id"] else None,
        "current_node": p["current_node"],
        "current_config": {**node.config, "features": f"{len(node.config['features'])} 个"} if node else None,
        "path": [_line(r) for r in store.path_to_root(p["current_node"])],
        "top_k": [_line(r) for r in store.top_k(tid, k)],
        "recent_rejected": [_line(r) for r in store.recent_rejected(tid, k)],
        "race": [_line(recs[e]) for e in p["race"] if e in recs],
        "promoted": [r.diff.get("exp_id") for r in recs.values() if r.action_type == "PROMOTE_FIDELITY"],
        "diagnosis": {"codes": codes, "policy_prior": prior},
        # 最终模型只能来自全保真度实验；把这条规则和当前状态直接写出来，不让 LLM 自己从 best/promoted 推断
        "final_model": {"exists": bool(p["best_exp_id"]), "promotable": promotable(list(recs.values())),
                        "rounds_left": cfg["run"]["max_rounds"] - p["round_no"],
                        "rule": "只有全保真度（full）实验能被 ACCEPT 成为最终模型；PROMOTE_FIDELITY 把 promotable 里的低保真实验在全量数据上复验。"
                                "全量复验同样要过 OVERFIT_GAP（gap = valid − OOT-dev 超过阈值即拒），选谁时要同时看 AUC 和 gap"},
        "unhandled_leaks": p["pending_leaks"],
        "oot_dev": {"used": oot_budget.used, "max": oot_budget.K, "next_alpha": round(oot_budget.current_alpha(), 4)},
        "taboo_count": len(store.taboo_hashes(tid)),
        "models_available": models, "templates": p.get("templates", []),
        # TUNE 只能改这些参数的上下限（当前节点模型的搜索空间）；不在此列的参数名会被校验拒绝
        "tunable": {k: {kk: vv for kk, vv in v.items() if kk in ("low", "high", "log", "type", "choices")}
                    for k, v in ((node.config.get("space") or ZOO[node.config["model"]].default_space) if node else {}).items()},
        "monotone": {"on": bool(node and node.config.get("monotone")),
                     "supported": bool(node and ZOO[node.config["model"]].supports["monotone_constraints"]),
                     "rule": "TUNE 带 monotone:true 后，若 AUC 不显著变差且分数 PSI 更优则保留；方向来自领域先验，没有先验的特征不加约束"},
        "fidelity_options": {k: {"sample_frac": v["sample_frac"], "n_trials": v["n_trials"]} for k, v in cfg["fidelity"].items()},
        "directions": p.get("directions", []),
        "memory": p.get("memory_ctx"),                    # 相似历史任务的 prior + ACTIVE Lesson；冷启动时为 None
        "investigations": p.get("investigations", {}),
    }


# ---------- 跨任务检索（§4.13）：结构化指纹距离，不用向量库 ----------
import math


def _vec(fp: dict, scales: dict) -> dict:
    return {"log10_n_samples": math.log10(max(fp["n_samples"], 1)), "n_features": fp["n_features"], "bad_rate": fp["bad_rate"],
            "missing_rate": fp["missing_rate"], "time_span_months": fp["time_span_months"]}


def fingerprint_distance(a: dict, b: dict, cfg: dict) -> float:
    sc = cfg["memory"]["scales"]
    va, vb = _vec(a, sc), _vec(b, sc)
    return math.sqrt(sum(((va[k] - vb[k]) / sc[k]) ** 2 for k in sc))


def nearest_task(mem, fp: dict, cfg: dict):
    """返回 (任务记录, 距离)。没有历史任务，或最近的也超过阈值 → (None, 距离)：冷启动，不复用任何经验。"""
    best = None
    for t in mem.tasks():
        if t["fingerprint"]["task_id"] == fp["task_id"]:
            continue
        d = fingerprint_distance(fp, t["fingerprint"], cfg)
        if best is None or d < best[1]:
            best = (t, d)
    if best is None:
        return None, None
    return (best[0] if best[1] <= cfg["memory"]["max_distance"] else None), best[1]


def active_lessons(mem, fp: dict, cfg: dict) -> list[dict]:
    """只检索 ACTIVE 且 scope 与当前指纹匹配的 Lesson（含负向经验）。"""
    return [l for l in mem.lessons() if l["status"] == "ACTIVE"
            and fingerprint_distance(fp, l["scope"], cfg) <= cfg["memory"]["max_distance"]]
