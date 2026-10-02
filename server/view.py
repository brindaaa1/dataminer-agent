"""事件 + 实验记录 → 前端快照。纯函数：不读库、不读文件，输入由 sources.py / examples.py 收集。
真实任务和示例任务走同一个 snapshot()，前端用同一套渲染。轮次划分复用 runtime/trace.py（已有测试）。"""
import json

from runtime.trace import PHASE_FINALE, PHASE_SETUP, build_trace, round_rows, segment_evaluations, summarize

METRIC = {"auc": "AUC", "pr_auc": "PR-AUC", "ks": "KS"}
VERDICT = {"ACCEPT": "采纳", "PROMISING": "有希望", "INCONCLUSIVE": "不确定", "REJECT": "拒绝", "NONE": "完成"}
TONE = {"ACCEPT": "ok", "PROMISING": "warn", "INCONCLUSIVE": "neutral", "REJECT": "bad", "NONE": "neutral"}
ACTIONS = {"RACE": "模型赛跑", "TUNE": "调参", "SWITCH_MODEL": "换模型", "PROMOTE_FIDELITY": "全量复验",
           "EXPAND_FEATURES": "加特征", "PRUNE_FEATURES": "删特征", "BACKTRACK": "回退", "INVESTIGATE": "查数据",
           "ESCALATE_HUMAN": "请求人工", "STOP": "停止"}
STOP_REASONS = {"MAX_ROUNDS": "达到最大轮数", "OOT_BUDGET_EXHAUSTED": "OOT-dev 比较次数用完", "TARGET_REACHED": "达到目标指标"}
STAGE = {"intake": "intake", "intake_failed": "intake", "config_ready": "confirm", "running": "modeling",
         "stopped": "modeling", "failed": "modeling", "done": "report", "accepted": "report"}
REQUIRED = ("label", "observation_time_expr", "windows", "metric")
WINDOWS = {"train_valid": "训练/验证", "oot_dev": "OOT-dev", "holdout": "holdout"}
DRAFT_KEYS = ("label", "observation_time_expr", "split_time_expr", "windows", "metric", "leakage", "quarantine")
ARTIFACTS = {"model_card.md": "模型卡", "experiment_tree.mmd": "实验树", "data_dictionary.yaml": "数据字典",
             "blacklist.yaml": "剔除与隔离", "task_spec.yaml": "任务配置", "lessons.json": "本次经验", "events.json": "完整事件"}


def _f(x) -> str:
    return "—" if x is None else f"{x:.3f}"


def _cut(s: str | None, n: int = 160) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n] + "…"


# ---------- 接入 ----------
def _label_text(v: dict) -> str:
    return f"{v['col']} = {v['positive']}：{v.get('definition', '')}"


def _metric_text(m: dict) -> str:
    g = m.get("guards") or {}
    return f"主指标 {METRIC.get(m['primary'], m['primary'])}" + \
        ("，护栏 " + "、".join(f"{METRIC.get(k, k)} 跌幅 ≤ {v}" for k, v in g.items()) if g else "")


def display(key: str, recommended: str) -> str:
    """确认卡上显示的说法：必答项按结构翻成人话；预测时点是表达式，原样显示（中文说明留待内部逻辑补字段）。"""
    try:
        v = json.loads(recommended)
    except ValueError:
        return recommended
    if key == "label" and isinstance(v, dict):
        return _label_text(v)
    if key == "windows" and isinstance(v, dict):
        return " · ".join(f"{WINDOWS[k]} {a}~{b}" for k, (a, b) in v.items())
    if key == "metric" and isinstance(v, dict):
        return _metric_text(v)
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def _group(name: str, f: dict, d: dict) -> str:
    if name in d["leakage"] or f["availability"] in ("at_checkin", "post_outcome"):
        return "leak"
    if f["availability"] == "meta":
        return "meta"
    if name in d["quarantine"] or f["availability"] not in ("at_booking", "history"):
        return "quarantine"
    return "ok"


def _draft(d: dict, confirmed: dict) -> dict:
    return {"label": _label_text(d["label"]) if d["label"] else None, "observation_time_expr": d["observation_time_expr"],
            "windows": d["windows"], "metric": _metric_text(d["metric"]) if d["metric"] else None,
            "fields": [{"name": k, "type": f["type"], "availability": f["availability"], "reason": f.get("reason", ""),
                        "group": _group(k, f, d)} for k, f in d["fields"].items()],
            "confirmed": sorted(confirmed)}


def _changed(prev: dict | None, cur: dict) -> list[str]:
    if not prev:
        return []
    return [k for k in DRAFT_KEYS if prev.get(k) != cur.get(k)] + \
        [f"field:{k}" for k, v in cur["fields"].items() if prev["fields"].get(k) != v]


def _intake(ev: list[dict], state: dict | None, status: str, busy: bool) -> dict:
    state = state or {}
    drafts = [e["payload"] for e in ev if e["type"] == "draft"] or ([state["last_draft"]] if state.get("last_draft") else [])
    qs = next((e["payload"] for e in reversed(ev) if e["type"] == "questions"), [])
    show = status == "intake" and not busy
    return {"round": state.get("round", 0),
            "questions": [{**q, "display": display(q["key"], q["recommended"]), "required": q["key"] in REQUIRED}
                          for q in qs] if show else [],
            "draft": _draft(drafts[-1], state.get("confirmed", {})) if drafts else None,
            "changed": _changed(drafts[-2] if len(drafts) > 1 else None, drafts[-1]) if drafts else [],
            "yaml": (state.get("written") or {}).get("yaml", {})}


# ---------- 建模 ----------
def _round(s: dict, row: dict, recs: dict, trials: dict, pm: str, running: bool) -> dict:
    d = s["decision"] or {}
    evs = list({e["exp_id"]: e for e in segment_evaluations(s, list(recs.values()))}.values())   # 续跑重做的状态会再评估一次：同一实验只留最后一条
    m =(recs.get((s["recorded"] or {}).get("exp_id")) or {}).get("metrics") or {}
    if not m and evs:                     # 准备阶段（模型赛跑）：取时间外主指标最高的一个
        m = max(evs, key=lambda x: (x.get("metrics") or {}).get(f"oot_dev_{pm}") or -1).get("metrics") or {}
    action = row["动作"]
    return {"key": "setup" if s["kind"] == PHASE_SETUP else f"r{s['round']}", "title": row["段"], "action": action,
            "action_label": ACTIONS.get(action, action or "决策中"), "verdict": row["结论"],
            "codes": row["诊断码"].split(", ") if row["诊断码"] else [],
            "metrics": {"train": m.get(f"train_{pm}"), "valid": m.get(f"valid_{pm}"), "oot_dev": m.get(f"oot_dev_{pm}"),
                        "oot_dev_ks": m.get("oot_dev_ks"), "gap": m.get("gap")},
            "hypothesis": d.get("hypothesis"), "alternatives": d.get("alternatives_considered") or [],
            "stop_reason": (d.get("params") or {}).get("reason") if action == "STOP" else None,
            "evaluations": [{"exp_id": e["exp_id"], "model": e.get("model"), "fidelity": e.get("fidelity"),
                             "verdict": e.get("verdict"), "codes": e.get("codes") or [],
                             "oot_dev": (e.get("metrics") or {}).get(f"oot_dev_{pm}")} for e in evs],
            "llm_calls": len(s["llm_calls"]), "running": running and not s["recorded"] and action != "STOP",
            "trials": {e: trials.get(e, []) for e in s["exp_ids"]}}


def _say(rd: dict, kind: str, L: str) -> dict:
    """每轮在对话里播报一句。模板拼接，不调 LLM。"""
    if kind == PHASE_SETUP:
        parts = [f"{e['model']} 时间外 {L} {_f(e['oot_dev'])}（{VERDICT.get(e['verdict'], '—')}）" for e in rd["evaluations"]]
        return {"key": rd["key"], "text": "模型赛跑：" + ("；".join(parts) or "运行中…"),
                "tone": "running" if rd["running"] else "neutral"}
    head = f"{rd['title']} · {rd['action_label']}"
    if rd["action"] == "STOP":
        return {"key": rd["key"], "text": f"{rd['title']} · 停止：{_cut(rd['stop_reason'])}", "tone": "neutral"}
    if rd["running"]:
        return {"key": rd["key"], "text": f"{head}：运行中…" if rd["action"] else f"{rd['title']}：LLM 决策中…", "tone": "running"}
    if not rd["verdict"]:
        return {"key": rd["key"], "text": f"{head}：未完成", "tone": "neutral"}
    m, v = rd["metrics"], rd["verdict"]
    text = f"{head}：{VERDICT.get(v, v)}" + (f" · 时间外 {L} {_f(m['oot_dev'])}" if m["oot_dev"] is not None else "") + \
        (f"，gap {_f(m['gap'])}" if m["gap"] is not None else "") + (f"（{'、'.join(rd['codes'])}）" if rd["codes"] else "")
    return {"key": rd["key"], "text": text, "tone": TONE.get(v, "neutral")}


def _closing(ev: list[dict], sm: dict, pm: str, L: str) -> list[dict]:
    out = []
    r = sm["stop_reason"]
    if r and not r.startswith("LLM_STOP"):            # LLM 主动停止已在 STOP 那一轮播报过
        out.append({"key": "stop", "text": f"停止：{STOP_REASONS.get(r, r)}", "tone": "neutral"})
    if f := sm["final"]:
        out.append({"key": "final", "tone": "bad" if f.get("overfit_to_oot_dev") else "ok",
                    "text": f"最终检验：holdout {L} {_f(f.get(f'holdout_{pm}'))}，比 OOT-dev 低 {_f(f.get('drop'))}"
                            + ("，疑似对 OOT-dev 过拟合" if f.get("overfit_to_oot_dev") else "")})
    for e in ev:
        if e["type"] == "final_gate_skipped":
            out.append({"key": "final", "text": "最终检验跳过：没有通过评估的全量模型", "tone": "warn"})
        elif e["type"] == "consolidated":
            out.append({"key": "lessons", "text": f"本次沉淀 {e['payload']['admitted']} 条经验", "tone": "neutral"})
        elif e["type"] == "worker_error":
            out.append({"key": f"error-{e['id']}", "text": f"运行出错：{e['payload']['error']}", "tone": "bad"})
    return out


def risks(model_card: str) -> list[str]:
    sec = model_card.split("## 8. 风险提示", 1)
    return [ln[2:].strip().replace("**", "") for ln in sec[1].splitlines() if ln.startswith("- ")] if len(sec) > 1 else []


def _final(f: dict | None, recs: dict, pm: str, files: dict) -> dict | None:
    if not f:
        return None
    rec = recs.get(f["exp_id"]) or {}
    return {"exp_id": f["exp_id"], "model": (rec.get("config") or {}).get("model"), "oot_dev": f.get(f"oot_dev_{pm}"),
            "holdout": f.get(f"holdout_{pm}"), "holdout_ks": f.get("holdout_ks"), "drop": f.get("drop"),
            "overfit": bool(f.get("overfit_to_oot_dev")), "n_experiments": len(recs),
            "n_accepted": sum(r.get("verdict") == "ACCEPT" for r in recs.values()),
            "risks": risks(files.get("model_card.md", ""))}


def _run(ev: list[dict], records: list[dict], trials: dict, status: str, settings: dict, files: dict) -> dict | None:
    if not ev and status != "running":
        return None
    pm = next((e["payload"]["metric"]["primary"] for e in ev if e["type"] == "spec_locked"), "auc")
    L = METRIC.get(pm, pm)
    trace = build_trace(ev)
    recs = {r["exp_id"]: r for r in records}
    rounds, narration, chart = [], [], []
    for s, row in zip(trace, round_rows(trace, records, pm)):
        if s["kind"] == PHASE_FINALE:
            continue
        rd = _round(s, row, recs, trials, pm, running=s is trace[-1] and status == "running")
        rounds.append(rd)
        narration.append(_say(rd, s["kind"], L))
        if rd["metrics"]["oot_dev"] is not None:
            chart.append({"label": "赛跑" if s["kind"] == PHASE_SETUP else f"R{s['round']}",
                          "value": rd["metrics"]["oot_dev"], "verdict": rd["verdict"]})
    sm = summarize(trace)
    return {"metric": pm, "max_rounds": settings.get("max_rounds"), "rounds": rounds, "chart": chart,
            "narration": narration + _closing(ev, sm, pm, L), "final": _final(sm["final"], recs, pm, files),
            "tree": [{"exp_id": r["exp_id"], "parent": r["parent_exp_id"], "verdict": r["verdict"],
                      "label": f"{ACTIONS.get(r['action_type'], r['action_type'])} · {(r.get('config') or {}).get('model', '')} · {r['fidelity']}"}
                     for r in records]}


def snapshot(src: dict) -> dict:
    meta, events = src["meta"], src["events"]
    intake_ev = [{**e, "type": e["type"][len("intake."):]} for e in events if e["type"].startswith("intake.")]
    run_ev = [e for e in events if not e["type"].startswith("intake.") and e["type"] != "accepted"]
    status, busy = meta["status"], meta.get("busy", False)
    prof = next((e["payload"] for e in intake_ev if e["type"] == "profile"), None) or (src["intake"] or {}).get("profile")
    return {"id": meta["id"], "title": meta["title"], "status": status, "stage": STAGE[status],
            "example": meta.get("example", False), "readonly": meta.get("example", False) or status == "accepted",
            "busy": busy, "error": meta.get("error"), "provider": meta["provider"], "langfuse_url": meta.get("langfuse_url"),
            "intake_langfuse_url": meta.get("intake_langfuse_url"), "note": meta.get("note"),
            "data": {**src["data"], "n_rows": prof["n_rows"] if prof else None, "n_cols": len(prof["columns"]) if prof else None},
            "intake": _intake(intake_ev, src["intake"], status, busy) if (intake_ev or src["intake"]) else None,
            "run": _run(run_ev, src["records"], src["trials"], status, meta.get("settings") or {}, src["files"]),
            "artifacts": [{"name": n, "label": ARTIFACTS.get(n, n)} for n in src["files"]],
            "last_event_id": max((e["id"] for e in events), default=0)}
