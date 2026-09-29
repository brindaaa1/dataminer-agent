"""事件日志 → 按轮次的 trace（纯函数，不读库、不依赖状态机对象）。

分段规则只看 state_transition 事件（它在每个状态处理完之后才写，所以状态内部的事件都排在它前面）：
- 第一次转入 DECIDE 之前：准备阶段（round=0，含 PLAN 和模型赛跑）
- 每次转入 DECIDE：开始新的一轮（round=1,2,…）；人工回复后回到 DECIDE 不写 state_transition，仍算同一轮
- 转入 FINAL_GATE：收尾阶段（round=None）
"""

PHASE_SETUP, PHASE_ROUND, PHASE_FINALE = "setup", "round", "finale"


def _new(kind: str, no: int | None) -> dict:
    return {"kind": kind, "round": no, "events": [], "exp_ids": [], "states": [], "llm_calls": [], "decision": None,
            "rejected": [], "evaluations": [], "recorded": None, "stop_reason": None, "final": None, "awaited": [],
            "tokens": 0, "llm_latency_s": 0.0, "ts_start": None, "ts_end": None}


def _add(seg: dict, e: dict):
    t, pl = e["type"], e.get("payload") or {}
    seg["events"].append(e)
    seg["ts_start"] = e["ts"] if seg["ts_start"] is None else seg["ts_start"]
    seg["ts_end"] = e["ts"]
    if e.get("exp_id") and e["exp_id"] not in seg["exp_ids"]:
        seg["exp_ids"].append(e["exp_id"])
    if t == "state_transition":
        seg["states"].append(pl.get("from"))
    elif t == "llm_call":
        seg["llm_calls"].append(pl)
        seg["tokens"] += pl.get("tokens") or 0
        seg["llm_latency_s"] = round(seg["llm_latency_s"] + (pl.get("latency_s") or 0), 3)
    elif t == "decision":
        seg["decision"] = pl.get("decision")
    elif t == "decision_rejected":
        seg["rejected"].append(pl)
    elif t == "evaluated":
        seg["evaluations"].append({"exp_id": e.get("exp_id"), **pl})
    elif t == "recorded":
        seg["recorded"] = {"exp_id": e.get("exp_id"), **pl}
    elif t == "stop":
        seg["stop_reason"] = pl.get("reason")
    elif t == "final_gate":                       # holdout 最终验收结果
        seg["final"] = pl
    elif t in ("await_human", "USER_INSTRUCTION", "resumed"):
        seg["awaited"].append({"type": t, **pl})


def build_trace(events: list[dict]) -> list[dict]:
    """events：EventLog.query() 的输出（按 id 升序）。返回段列表，每段汇总该轮的决策、LLM 调用、评估与记录。"""
    segs = [_new(PHASE_SETUP, 0)]
    n = 0
    for e in sorted(events, key=lambda x: x["id"]):
        _add(segs[-1], e)
        if e["type"] != "state_transition":
            continue
        to = (e.get("payload") or {}).get("to")
        if to == "DECIDE":
            n += 1
            segs.append(_new(PHASE_ROUND, n))
        elif to == "FINAL_GATE" and segs[-1]["kind"] != PHASE_FINALE:
            segs.append(_new(PHASE_FINALE, None))
    return [s for s in segs if s["events"]]


def summarize(trace: list[dict]) -> dict:
    """整个任务的汇总：轮数、LLM 调用数、token、各结论数量、停止原因。"""
    calls = [c for s in trace for c in s["llm_calls"]]
    verdicts = [s["recorded"]["verdict"] for s in trace if s["recorded"]]
    stop = next((s["stop_reason"] for s in reversed(trace) if s["stop_reason"]), None)
    if stop is None:                              # 早期日志里 LLM 主动 STOP 不写 stop 事件，取最后一次 STOP 决策的理由
        d = next((s["decision"] for s in reversed(trace) if s["decision"]), None)
        if d and d.get("action") == "STOP":
            stop = "LLM_STOP: " + str((d.get("params") or {}).get("reason", ""))
    state = None                                  # 最后一次 state_transition / resumed 指向的状态
    for e in (e for s in trace for e in s["events"] if e["type"] in ("state_transition", "resumed")):
        state = (e.get("payload") or {}).get("to", state)
    return {"n_rounds": sum(s["kind"] == PHASE_ROUND for s in trace), "n_llm_calls": len(calls),
            "n_llm_errors": sum("error" in c for c in calls), "tokens": sum(c.get("tokens") or 0 for c in calls),
            "n_rejected": sum(len(s["rejected"]) for s in trace),
            "verdicts": {v: verdicts.count(v) for v in dict.fromkeys(verdicts)}, "stop_reason": stop,
            "final_state": state, "final": next((s["final"] for s in trace if s["final"]), None)}


def segment_evaluations(seg: dict, records: list[dict] | None = None) -> list[dict]:
    """该段的评估结果。优先用 evaluated 事件；早期日志没有这类事件时，退回到实验记录（ExperimentRecord 的 dict）：
    轮次段取本轮记录的实验，准备段取模型赛跑（action_type=RACE）的实验。退回的条目标 source=record，没有配对比较结果。"""
    if seg["evaluations"] or not records:
        return seg["evaluations"]
    if seg["kind"] == PHASE_SETUP:
        pick = [r for r in records if r.get("action_type") == "RACE"]
    else:
        pick = [r for r in records if seg["recorded"] and r.get("exp_id") == seg["recorded"]["exp_id"]]
    return [{"exp_id": r["exp_id"], "verdict": r.get("verdict"), "codes": r.get("diagnosis_codes") or [],
             "metrics": {k: v for k, v in (r.get("metrics") or {}).items() if k.startswith(("valid_", "oot_dev_")) or k == "score_psi"},
             "model": (r.get("config") or {}).get("model"), "fidelity": r.get("fidelity"), "compare": None,
             "source": "record"} for r in pick if r.get("verdict") != "NONE"]


def round_rows(trace: list[dict], records: list[dict] | None = None, pm: str = "auc") -> list[dict]:
    """每段一行的表格视图（给 UI 用）。records 可选，见 segment_evaluations。pm：主指标（旧运行只有 AUC）。"""
    label = {"auc": "AUC", "pr_auc": "PR-AUC", "ks": "KS"}[pm]
    out = []
    for s in trace:
        d, r = s["decision"] or {}, s["recorded"] or {}
        evs = segment_evaluations(s, records)
        if r:                                     # 本轮记录的实验对应的评估
            ev = next((x for x in reversed(evs) if x["exp_id"] == r.get("exp_id")), {})
        else:                                     # 准备阶段（模型赛跑）：取 OOT-dev 主指标最高的一个
            ev = max(evs, key=lambda x: (x.get("metrics") or {}).get(f"oot_dev_{pm}") or -1, default={})
        cmp = ev.get("compare") or {}
        out.append({"段": {"setup": "准备", "round": f"第 {s['round']} 轮", "finale": "收尾"}[s["kind"]],
                    "动作": d.get("action") or ("RACE" if s["kind"] == PHASE_SETUP and evs else None),
                    "实验": r.get("exp_id") or ", ".join(x["exp_id"] for x in evs if x["exp_id"]) or None,
                    "结论": r.get("verdict") or (ev.get("verdict") if s["kind"] == PHASE_SETUP else None),
                    f"OOT-dev {label}": (ev.get("metrics") or {}).get(f"oot_dev_{pm}"), f"Δ{label}": cmp.get("delta"),
                    "诊断码": ", ".join(r.get("codes") or ev.get("codes") or []) or None,
                    "LLM 调用": len(s["llm_calls"]), "被拒": len(s["rejected"]), "tokens": s["tokens"],
                    "耗时(s)": round(s["ts_end"] - s["ts_start"], 2) if s["ts_start"] is not None else None})
    return out


def load_trials(optuna_db: str, exp_ids: list[str]) -> dict:
    """每个实验一个 Optuna study（study 名 = exp_id）。value 是 valid 上的主指标；被剪枝的 trial 也保留，页面画成灰色。"""
    import optuna
    storage = f"sqlite:///{optuna_db}"
    names = {s.study_name for s in optuna.get_all_study_summaries(storage)}
    return {e: [{"n": t.number, "value": t.value, "state": t.state.name, "params": t.params}
                for t in optuna.load_study(study_name=e, storage=storage).trials]
            for e in exp_ids if e in names}
