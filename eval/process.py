"""一次运行的过程指标（spec 2026-10-05 §4）：只读事件和实验记录，每项带证据（实验 id 或事件 id），供界面跳到 trace。
指标针对本轮评测里实际碰到的问题：加特征一个没留 / 模板重复（无效轮次）、撞禁忌表死循环（被拒决策、决策失败收尾）、
有候选不升（升全量命中率）、特征工程有没有进最终模型（漏斗）。"""
from eval.suite.metrics import cost_usd, process_stats, usage_totals

FAIL_CODES = {"EXPAND_FAILED", "EXPAND_EMPTY", "BUDGET_EXCEEDED"}
REJECT_KINDS = (("duplicate", ("禁忌表", "已经用")), ("template", ("模板",)), ("budget", ("预算",)),
                ("prior", ("偏离策略先验",)), ("format", ("JSON",)))


def _kind(err: str) -> str:
    return next((k for k, keys in REJECT_KINDS if any(x in err for x in keys)), "other")


def _rate(a: int, b: int):
    return round(a / b, 3) if b else None


def _failed(r: dict) -> bool:
    return r["verdict"] == "REJECT" or bool(set(r["diagnosis_codes"]) & FAIL_CODES)


def _params(r: dict) -> dict:
    return {k: v for k, v in r["diff"].items() if k not in ("expand", "result")}


def _step(r: dict) -> str:
    a = r["action_type"]
    if a == "RACE":
        return f"赛跑 {r['config'].get('model')}"
    if a == "EXPAND_FEATURES":
        return f"加特征 {r['diff'].get('template_id') or '代码'}"
    return {"PROMOTE_FIDELITY": "升全量", "TUNE": "调参", "SWITCH_MODEL": f"换模型 {r['config'].get('model')}",
            "PRUNE_FEATURES": "删特征"}.get(a, a)


def process_metrics(events: list[dict], records: list[dict], max_rounds: int, price: dict | None = None) -> dict:
    recs = [r for r in records if r["action_type"] != "RACE"]
    byid = {r["exp_id"]: r for r in records}
    evals = [e for e in events if e["type"] == "evaluated"]
    delta = {e["exp_id"]: (e["payload"].get("compare") or {}).get("delta") for e in evals}
    rounds = sum(e["type"] == "recorded" for e in events)
    st = process_stats(events)
    out = {"rounds": {"value": rounds, "max": max_rounds, "stop_reason": st["stop_reason"], "evidence": []}}

    seen, wasted = set(), []                 # 无效轮次：实验失败，或配置与之前某个实验完全相同
    for r in records:
        h = r["config_hash"] if r["fidelity"] != "none" else ""
        if r["action_type"] != "RACE" and (set(r["diagnosis_codes"]) & FAIL_CODES or (h and h in seen)):
            wasted.append(r["exp_id"])
        if h:
            seen.add(h)
    out["wasted_rounds"] = {"value": _rate(len(wasted), len(recs)), "n": len(wasted), "evidence": wasted}

    rej = [e for e in events if e["type"] == "decision_rejected"]
    kinds = {}
    for e in rej:
        for err in e["payload"].get("errors", []):
            kinds[_kind(err)] = kinds.get(_kind(err), 0) + 1
    out["rejected_decisions"] = {"value": len(rej), "by_kind": kinds, "evidence": [e["id"] for e in rej]}

    df = [e for e in events if e["type"] == "decide_failed"]
    out["decide_failed"] = {"value": bool(df), "rounds_lost": max(max_rounds - rounds, 0) if df else 0,
                            "evidence": [e["id"] for e in df]}

    first_accept = next((i for i, r in enumerate(records) if r["verdict"] == "ACCEPT"), None)

    mark = {e["exp_id"]: e["payload"]["mark"] for e in events if e["type"] == "recorded" and "mark" in e["payload"]}
    worthy = {e["exp_id"]: e["payload"]["worth"] for e in events if e["type"] == "recorded" and "worth" in e["payload"]}

    def up(r):                               # 低保真、比对照好：以 agent 记下的 worth（差距 ≥ MDE）/ 标记为准；旧运行都没有，按差 > 0
        if r["exp_id"] in worthy:
            return worthy[r["exp_id"]]
        if r["exp_id"] in mark:
            return r["fidelity"] == "low" and mark[r["exp_id"]] in ("PROMISING", "INCONCLUSIVE_UP")
        d = delta.get(r["exp_id"])
        return r["fidelity"] == "low" and (r["verdict"] == "PROMISING" or (r["verdict"] == "INCONCLUSIVE" and d is not None and d > 0))
    cands = [r["exp_id"] for i, r in enumerate(records)
             if first_accept is not None and i > first_accept and r["action_type"] != "RACE" and up(r)]
    rec_ids = [e["id"] for e in events if e["type"] == "recorded"]
    rec_at = {e["exp_id"]: e["id"] for e in events if e["type"] == "recorded"}
    # 记下之后再没有任何一轮：不可能被升，不算没升（终审 Important 3：情景只有 5 轮时很常见，会混进 A/A 噪声带）
    unreachable = [e for e in cands if not any(i > rec_at.get(e, float("inf")) for i in rec_ids)]
    worth = [e for e in cands if e not in unreachable]
    promoted = {r["diff"].get("exp_id"): r for r in records if r["action_type"] == "PROMOTE_FIDELITY"}
    out["upgrade_hit_rate"] = {"value": _rate(sum(e in promoted for e in worth), len(worth)), "worth": worth,
                               "unreachable": unreachable, "evidence": [e for e in worth if e not in promoted]}
    # 第一次升全量（还没有最终模型）几乎总被采纳，不算：只看已有最终模型之后想换更好的那些
    ups = [r for i, r in enumerate(records) if r["action_type"] == "PROMOTE_FIDELITY" and first_accept is not None and i > first_accept]
    out["upgrade_success_rate"] = {"value": _rate(sum(r["verdict"] == "ACCEPT" for r in ups), len(ups)),
                                   "evidence": [r["exp_id"] for r in ups]}

    fg = next((e["payload"] for e in events if e["type"] == "final_gate"), None)
    chain, cur = [], fg and fg.get("exp_id")
    while cur in byid:
        chain.append(cur)
        cur = byid[cur]["parent_exp_id"]
    chain.reverse()
    out["final_lineage"] = {"value": " → ".join(_step(byid[e]) for e in chain) or None, "evidence": chain}

    ex = [r for r in records if r["action_type"] == "EXPAND_FEATURES"]
    kept = [r for r in ex if (r["diff"].get("expand") or {}).get("kept")]
    promising = [r["exp_id"] for r in kept if up(r)]
    up_fe = [e for e in promising if e in promoted]
    out["fe_funnel"] = {"value": any(byid[e]["action_type"] == "EXPAND_FEATURES" for e in chain),
                        "attempts": len(ex), "kept": len(kept), "promising": len(promising), "promoted": len(up_fe),
                        "accepted": sum(promoted[e]["verdict"] == "ACCEPT" for e in up_fe), "evidence": [r["exp_id"] for r in ex]}

    pairs = [(a["exp_id"], a["action_type"] != b["action_type"] or _params(a) != _params(b))
             for a, b in zip(recs, recs[1:]) if _failed(a)]
    out["recovery"] = {"value": _rate(sum(c for _, c in pairs), len(pairs)), "evidence": [e for e, c in pairs if not c]}

    waits = [e for e in events if e["type"] == "await_human"]
    out["human"] = {"value": len(waits), "confirmed": sum(e["type"] == "spec_confirmed" for e in events),
                    "evidence": [e["id"] for e in waits]}

    cmps = [e["payload"]["compare"] for e in evals if e["payload"].get("compare")]
    pw = next((e["payload"] for e in events if e["type"] == "oot_power"), {})
    out["oot"] = {"value": max((c.get("k") or 0 for c in cmps), default=0), "last_alpha": cmps[-1].get("alpha") if cmps else None,
                  "mde": pw.get("mde"), "underpowered": pw.get("underpowered"), "evidence": []}

    u = usage_totals(events)
    out["cost"] = {"value": cost_usd(u, price), "wall_sec": st["wall_sec"],
                   "tokens": u["input_cache_hit"] + u["input_cache_miss"] + u["output"],
                   "sec_per_round": round(st["wall_sec"] / rounds, 1) if rounds else None, "evidence": []}
    return out
