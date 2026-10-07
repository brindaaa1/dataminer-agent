"""「评测」tab 的三个视图（spec 2026-10-05 §8）。纯函数：输入由 app.py 从 sources / examples / eval_tasks 收集。"""
from eval.process import process_metrics

LABELS = {"rounds": "轮数", "wasted_rounds": "无效轮次", "rejected_decisions": "被拒决策", "decide_failed": "决策失败收尾",
          "upgrade_hit_rate": "升全量命中率", "upgrade_success_rate": "升全量成功率", "final_lineage": "最终模型来源",
          "fe_funnel": "特征工程进最终模型", "recovery": "失败后换方向", "human": "问人次数", "oot": "OOT-dev 比较次数",
          "cost": "花费（美元）"}


METRIC_LABELS = {"holdout": "holdout", "delta_b1": "相对 B1 的差", "wasted_rate": "无效轮次率", "rejected_decisions": "被拒决策",
                 "upgrade_hit_rate": "升全量命中率", "fe_in_final": "特征工程进最终模型"}
FLAG_LABELS = {"failed_run": "运行失败或没有最终模型", "decide_failed": "因决策失败收尾", "new_reject_kind": "新出现的被拒原因",
               "scenario_regressed": "情景从通过变不通过", "scenario_failed": "情景不通过", "overfit": "新出现的 OOT-dev 过拟合",
               "cost_up": "用时或花费上升"}
COST_LABELS = {"wall_sec": "用时（秒）", "cost_usd": "花费（美元）"}


def segments(events: list[dict]) -> dict[int, str]:
    """与 web/src/lib/trace.ts 的 segmentOf 同一规则：进 DECIDE 开始新一轮，进 FINAL_GATE 之后是收尾。"""
    n, finale, out = 0, False, {}
    for e in events:
        if e["type"].startswith("intake."):
            out[e["id"]] = "intake"
            continue
        out[e["id"]] = "finale" if finale else "setup" if n == 0 else f"r{n}"
        if e["type"] == "state_transition":
            to = (e["payload"] or {}).get("to")
            n += to == "DECIDE"
            finale = finale or to == "FINAL_GATE"
    return out


def _evidence(refs: list, events: list[dict], seg: dict[int, str]) -> list[dict]:
    at = {}                       # 实验 → 它的 recorded 事件；赛跑实验没有 recorded，用 evaluated
    for t in ("recorded", "evaluated"):
        for e in events:
            if e["type"] == t and e.get("exp_id"):
                at.setdefault(e["exp_id"], e["id"])
    return [{"ref": str(r), "round": seg.get(r) if isinstance(r, int) else seg.get(at.get(r))} for r in refs]


def process_view(events: list[dict], records: list[dict], max_rounds: int | None, cost: float | None = None) -> list[dict]:
    """cost：评测运行已经按价格算好的美元数；用户任务不知道单价，花费只显示用量。"""
    pm, seg = process_metrics(events, records, max_rounds or 0), segments(events)
    if cost is not None:
        pm["cost"]["value"] = cost
    return [{"key": k, "label": lab, "value": pm[k]["value"], "detail": {x: y for x, y in pm[k].items() if x not in ("value", "evidence")},
             "evidence": _evidence(pm[k].get("evidence") or [], events, seg)} for k, lab in LABELS.items() if k in pm]


def _b1(v: dict, ds: str):
    return ((v["datasets"][ds]["baselines"].get("b1") or {}).get("holdout_auc"))


def version_view(v: dict, reg: dict | None) -> dict:
    ev = lambda t: f"ev-{t}" if t else None
    runs = [{"task": f"ev-{r['task_id']}", "dataset": ds, "seed": r["seed"], "status": r["status"], "holdout": r.get("holdout"),
             "delta_b1": None if r.get("holdout") is None or _b1(v, ds) is None else r["holdout"] - _b1(v, ds)}
            for ds, d in v["datasets"].items() for r in d["runs"]]
    running = v.get("status") == "running"
    summary = [{"dataset": ds, **d["summary"], "worst_task": ev(d["summary"].get("worst_task"))}
               for ds, d in v["datasets"].items() if d.get("summary")]
    if reg is None or running:
        why = "运行中：跑完后和同档上一版本对比" if running else "没有回归对比（没有可比的同档上一版本）"
        return {"label": v["label"], "tier": v["tier"], "running": running, "against": None, "conclusion": why,
                "noise_calibrated": False, "table": [], "flags": [], "cost": [], "runs": runs, "summary": summary}
    return {"label": v["label"], "tier": v["tier"], "running": False, "against": reg["old"], "conclusion": reg["conclusion"],
            "noise_calibrated": reg.get("noise_calibrated", False),
            "table": [{k: t[k] for k in ("dataset", "metric", "new", "old", "delta", "n", "verdict")}
                      | {"label": METRIC_LABELS.get(t["metric"], t["metric"]), "task": ev(t.get("worst_task"))} for t in reg.get("table", [])],
            "flags": [{"kind": f["kind"], "label": FLAG_LABELS.get(f["kind"], f["kind"]), "detail": f["detail"], "task": ev(f.get("task_id"))}
                      for f in reg.get("flags", [])],
            "cost": [{"dataset": c["dataset"], "label": COST_LABELS[c["key"]], "new": c["new"], "old": c["old"], "ratio": c["ratio"],
                      "threshold": c["threshold"]} for c in reg.get("cost", [])],
            "runs": runs, "summary": summary}


def scenarios_view(v: dict) -> list[dict]:
    return [{"scenario": s["scenario"], "base": s["base"], "passed": bool(s.get("passed")), "reason": s.get("reason") or s.get("error") or "",
             "task": f"ev-{s['task_id']}"} for s in v.get("scenarios", [])]
