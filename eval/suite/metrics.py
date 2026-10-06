"""评测指标：全部从事件日志算（纯函数，不读库、不调 LLM）。"""
from runtime.trace import build_trace, summarize

KEYS = ("input_cache_hit", "input_cache_miss", "output", "reasoning")


def usage_totals(events: list[dict]) -> dict:
    """LLM 用量合计。没有分项 usage 的旧事件，总数记进 input_cache_miss（只用于粗看，不精确）。"""
    out = dict.fromkeys(KEYS, 0) | {"calls": 0}
    for e in events:
        if e["type"] != "llm_call":
            continue
        out["calls"] += 1
        u = e["payload"].get("usage")
        if u:
            for k in KEYS:
                out[k] += u[k]
        else:
            out["input_cache_miss"] += e["payload"].get("tokens") or 0
    return out


def cost_usd(usage: dict, price: dict | None) -> float | None:
    if price is None:
        return None
    return sum(usage[k] * price[k] for k in ("input_cache_hit", "input_cache_miss", "output")) / 1e6


def process_stats(events: list[dict]) -> dict:
    sm = summarize(build_trace(events))
    f = sm["final"] or {}
    pm = f.get("metric", "auc")
    return {"n_rounds": sm["n_rounds"], "n_rejected": sm["n_rejected"], "n_llm_errors": sm["n_llm_errors"],
            "stop_reason": sm["stop_reason"], "final_exp": f.get("exp_id"), "holdout": f.get(f"holdout_{pm}"),
            "oot_dev": f.get(f"oot_dev_{pm}"), "overfit": bool(f.get("overfit_to_oot_dev")),
            "no_final_model": any(e["type"] == "final_gate_skipped" for e in events),
            # 各状态耗时之和：跨续跑累加，不含等人回复（v6 评测：续跑后只算续跑那段，B1 的时间预算只给了 30 秒）
            "wall_sec": round(sum(e["payload"].get("sec", 0) for e in events if e["type"] == "state_transition"), 1)}


def _f(x, n=4):
    return "—" if x is None else f"{x:.{n}f}"


def summary_md(label: str, settings: dict, rows: list[dict], baselines: dict) -> str:
    """每个数据集一段：每个种子一行（agent 对 B0 的 Δ），末尾是 B0/B1 与合计。"""
    L = [f"# 评测 {label}", "", "设置：" + "，".join(f"{k}={v}" for k, v in settings.items()), ""]
    for ds in dict.fromkeys(r["dataset"] for r in rows):
        rs, b = sorted((r for r in rows if r["dataset"] == ds), key=lambda r: r["seed"]), baselines.get(ds, {})
        b0 = (b.get("b0") or {}).get("holdout_auc")
        L += [f"## {ds}", "", "| 种子 | 状态 | 最终模型 | holdout | Δ vs B0 | OOT-dev | 过拟合 OOT-dev | 轮数 | 被拒决策 | LLM 失败 | 停止原因 | 输入 token（命中/未命中） | 输出（推理） | 美元 | 耗时(s) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        wins = 0
        for r in rs:
            u = r["usage"]
            if r["status"] != "ok":
                L.append(f"| {r['seed']} | 失败：{(r['error'] or '')[:60]} | | | | | | | | | | {u['input_cache_hit']}/{u['input_cache_miss']} | {u['output']}（{u['reasoning']}） | {_f(r['cost_usd'])} | |")
                continue
            d = None if r["holdout"] is None or b0 is None else r["holdout"] - b0
            wins += d is not None and d > 0
            L.append(f"| {r['seed']} | ok | {r['final_model'] or '无最终模型'} | {_f(r['holdout'])} | {'—' if d is None else f'{d:+.4f}'} | "
                     f"{_f(r['oot_dev'])} | {'是' if r['overfit'] else ''} | {r['n_rounds']} | {r['n_rejected']} | {r['n_llm_errors']} | "
                     f"{(r['stop_reason'] or '')[:30]} | {u['input_cache_hit']}/{u['input_cache_miss']} | {u['output']}（{u['reasoning']}） | "
                     f"{_f(r['cost_usd'])} | {r['wall_sec']} |")
        cost = sum(r["cost_usd"] or 0 for r in rs)
        L += ["", f"- 赢 B0 的种子：{wins}/{len(rs)}；本数据集花费合计 {cost:.4f} 美元",
              *(f"- {k.upper()}：holdout {_f(v.get('holdout_auc'))}，OOT-dev {_f(v.get('oot_dev_auc'))}，耗时 {_f(v.get('wall_sec'), 1)}s"
                for k, v in b.items()), ""]
    return "\n".join(L)


def _pct(x):
    return "—" if x is None else f"{x:.0%}"


def process_md(rows: list[dict]) -> str:
    """过程指标（spec §4）：每个数据集一张表，每个种子一行。"""
    L = ["## 过程指标", ""]
    for ds in dict.fromkeys(r["dataset"] for r in rows):
        L += [f"### {ds}", "", "| 种子 | 无效轮次 | 被拒决策 | 决策失败收尾 | 升全量命中 | 特征工程进最终模型 | 最终模型来源 |",
              "|---|---|---|---|---|---|---|"]
        for r in sorted((r for r in rows if r["dataset"] == ds), key=lambda r: r["seed"]):
            p = r.get("process") or {}
            g = lambda k: (p.get(k) or {}).get("value")
            L.append(f"| {r['seed']} | {_pct(g('wasted_rounds'))} | {g('rejected_decisions') if g('rejected_decisions') is not None else '—'} | "
                     f"{'是' if g('decide_failed') else ''} | {_pct(g('upgrade_hit_rate'))} | {'是' if g('fe_funnel') else ''} | {g('final_lineage') or '—'} |")
        L.append("")
    return "\n".join(L)


def scenarios_md(scenarios: list[dict]) -> str:
    L = ["## 情景用例", "", "| 情景 | 底座 | 结果 | 原因 | 任务 |", "|---|---|---|---|---|"]
    for s in scenarios:
        L.append(f"| {s['scenario']} | {s['base']} | {'通过' if s.get('passed') else '不通过'} | {s.get('reason') or s.get('error') or ''} | {s.get('task_id', '')} |")
    return "\n".join(L + [""])