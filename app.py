"""DataMiner Agent 演示页：看 agent 一轮一轮怎么决策、代码怎么判定。

    streamlit run app.py

不需要账号、不需要下载数据：
- 历史回放：examples/traces/ 里导出的真实 LLM 运行，按原顺序重新播放（不是实时调用）；
- 现场运行：在 data_sample/ 的酒店样本上从头跑一遍。默认用确定性的 mock LLM，不需要 key；
  也可以填自己的 key 调真实模型。key 只留在本次页面会话的内存里，不写盘、不进日志。
"""
import json
import os
from html import escape
import time
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)                                           # config.yaml、knowhow/ 等路径都相对仓库根目录

from agent.llm import extract_json, redact  # noqa: E402
from app_helpers import REQUIRED_KEYS, draft_rows, intake_rounds, metric_label, primary_of, q_label, trial_chart  # noqa: E402
from data.sample import SAMPLE_CSV, sample_art  # noqa: E402
from runtime.env import load_env  # noqa: E402
from runtime.trace import PHASE_ROUND, PHASE_SETUP, build_trace, load_trials, round_rows, segment_evaluations, summarize  # noqa: E402

EXAMPLES = Path(os.environ.get("DATAMINER_EXAMPLES") or ROOT / "examples" / "traces")     # 测试可指向临时目录
load_env()                                               # .env 里的 LLM key 和 LANGFUSE_*（不覆盖已有环境变量）


def esc(x) -> str:
    return escape(str(x)) if x is not None else ""

# 判定：图标 + 文字 + 状态色（图中再加形状），颜色不单独承载含义
VERDICTS = {
    "ACCEPT": ("✅", "接受", "#0ca30c", "circle", "全量训练下显著优于当前最优，成为新的最优模型"),
    "PROMISING": ("🟡", "有希望", "#fab219", "diamond", "抽样试跑通过了检查（已有最优模型时还要显著更优），值得拿全量数据复验"),
    "INCONCLUSIVE": ("⬜", "不确定", "#8c8c88", "square", "与当前最优差异不显著"),
    "REJECT": ("❌", "拒绝", "#d03b3b", "cross", "未通过 guardrail（过拟合、泄漏、分数漂移等），或显著变差"),
    "NONE": ("➖", "无判定", None, None, "本轮没有训练模型（如 INVESTIGATE、BACKTRACK，或执行失败）"),
}
CODES = {
    "OVERFIT_GAP": "验证集与 OOT-dev 的主指标差距超过阈值：过拟合或跨时间不稳定",
    "GUARD_FAIL": "护栏指标比当前最优变差超过容差：主指标再好也不接受",
    "LEAK_SUSPECT": "疑似泄漏：单特征 AUC 过高、字段可得时间不明或字段名命中泄漏模式",
    "PSI_DRIFT": "模型分数在 OOT-dev 上的分布漂移（PSI）超过告警线",
    "SCORE_PSI": "分数 PSI 超过严重阈值，直接拒绝",
    "TOO_MANY_FEATURES": "特征数超过上限",
    "PLATEAU": "与当前最优的配对 bootstrap 比较不显著：进入平台期",
    "NO_PROGRESS": "连续多轮只有拒绝或不确定",
    "NO_CONVERGE": "最佳迭代数过少，模型几乎没有学到东西",
    "BUDGET_LOW": "剩余预算低于阈值",
    "NO_FINAL_MODEL": "还没有全量训练出来的最优模型，但已有可以拿去全量复验的试跑结果：该复验了",
    "OOT_BUDGET_EXHAUSTED": "OOT-dev 比较次数已用完",
    "DATA_FATAL": "数据问题（如标签成熟度不足），无法继续",
    "OOM": "内存不足",
    "BUDGET_EXCEEDED": "估算成本超过剩余预算，实验没有运行",
    "EXPAND_FAILED": "特征生成失败",
}
ACTIONS = {"TUNE": "调参", "SWITCH_MODEL": "换模型", "PROMOTE_FIDELITY": "全量复验",
           "EXPAND_FEATURES": "加特征", "PRUNE_FEATURES": "删特征", "BACKTRACK": "回退到之前的实验",
           "INVESTIGATE": "查数据", "ESCALATE_HUMAN": "请求人工", "STOP": "主动停止"}
STATES = {"INTAKE": "检查任务信息", "CLARIFY": "向人确认任务信息", "PROFILE": "切分数据、生成数据画像",
          "PLAN": "LLM 规划方向，代码跑模型赛跑", "DECIDE": "LLM 决定下一步", "EXECUTE": "执行：训练 / 生成特征 / 查数据",
          "EVALUATE": "代码评估与诊断", "RECORD": "写实验记录", "STOP_CHECK": "检查是否该停止",
          "FINAL_GATE": "holdout 最终验收", "REPORT": "生成模型卡", "CONSOLIDATE": "沉淀跨任务经验",
          "DONE": "完成", "FAILED": "失败", "AWAIT_HUMAN": "等待人工"}
AVAILABILITY = {"at_booking": "预订时已知", "history": "预订时已知的客户历史", "updated_until_event": "到入住前一直在变 → 按不明处理",
                "at_checkin": "入住时才确定 → 泄漏", "post_outcome": "结果发生后才有 → 泄漏", "unknown": "可得时间不明 → 默认隔离"}
PROVIDERS = {"mock": "mock（确定性策略，无需 key）", "deepseek": "DeepSeek", "anthropic": "Anthropic", "kimi": "Kimi（Moonshot）"}
FIDELITY = {"low": "抽样试跑", "full": "全量训练"}     # 代码里的 fidelity: low / full
STOP_REASONS = {"MAX_ROUNDS": "达到最大轮数", "OOT_BUDGET_EXHAUSTED": "OOT-dev 比较次数用完", "TARGET_REACHED": "达到目标指标"}


# ---------- 数据 ----------
@st.cache_data
def load_sample() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_CSV, keep_default_na=False, na_values=["NA"])


@st.cache_data
def load_fields() -> pd.DataFrame:
    import yaml
    d = yaml.safe_load((ROOT / "knowhow" / "hotel_bookings" / "data_dictionary.yaml").read_text())
    return pd.DataFrame([{"字段": k, "可得时间": AVAILABILITY.get(v.get("availability"), v.get("availability") or "—"),
                          "说明": v.get("note", "") or ("标签校验用，读完即剔除" if v.get("role") == "label_check_only" else "")}
                         for k, v in d["fields"].items()])


@st.cache_data
def load_example(name: str) -> dict:
    return json.loads((EXAMPLES / f"{name}.json").read_text())


def example_names() -> list[str]:
    """有样本数据的（酒店）排在前面。"""
    names = [p.stem for p in EXAMPLES.glob("*.json")]
    return sorted(names, key=lambda n: (not load_example(n).get("meta", {}).get("has_sample_data"), n))


# ---------- 文字化 ----------
def verdict_txt(v: str | None) -> str:
    if v not in VERDICTS:
        return v or "—"
    icon, zh, *_ = VERDICTS[v]
    return f"{icon} {zh}（{v}）"


def codes_md(codes: list[str]) -> str:
    return "；".join(f"`{c}` {CODES.get(c, '')}" for c in codes)


def stop_txt(r: str | None) -> str:
    if not r:
        return "—"
    if r.startswith("LLM_STOP"):
        return "LLM 判断继续实验不划算，主动停止：" + r[len("LLM_STOP: "):]
    return STOP_REASONS.get(r, r)


def name_story(task_id: str, meta: dict) -> str:
    return (f"`{task_id}` 是启动运行时用 `--task-id` 起的名字，习惯写成「数据集_模型序号」。"
            f"这次运行用的数据是 **{meta.get('dataset', '—')}**，模型是 **{meta.get('llm', '—')}**。")


def plan_directions(seg: dict) -> list[str]:
    for c in seg["llm_calls"]:
        if c.get("stage") == "PLAN" and c.get("response"):
            try:
                return [d.get("direction", "") for d in extract_json(c["response"]).get("directions", [])]
            except (ValueError, AttributeError):
                return []
    return []


def seg_title(s: dict) -> str:
    return {"setup": "准备", "round": f"第 {s['round']} 轮", "finale": "收尾"}[s["kind"]]


def _num(x) -> str:
    return "—" if x is None else f"{x:.4f}"


def pill(v: str | None) -> str:
    if v not in VERDICTS:
        return ""
    icon, zh, *_ = VERDICTS[v]
    return f'<span class="dm-pill dm-v-{v}">{icon} {zh}<span class="dm-en">{v}</span></span>'


def codes_html(codes: list[str]) -> str:
    return "<br>".join(f"<code>{esc(c)}</code> {esc(CODES.get(c, ''))}" for c in codes)


def card(head: str, rows: list[tuple[str, str]], verdict: str | None = None, notes: list[str] = (), pending: bool = False) -> str:
    """一张轮次卡片的 HTML。rows 里的文字调用前已转义。"""
    body = "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in rows if v)
    notes_html = "".join(f'<div class="dm-note">{n}</div>' for n in notes)
    tail = '<div class="dm-pending"><span class="dm-dot"></span>进行中…</div>' if pending else ""
    return (f'<div class="dm-card dm-v-{verdict or "none"}"><div class="dm-head">{head}{pill(verdict)}</div>'
            f'{notes_html}<dl class="dm-rows">{body}</dl>{tail}</div>')


def round_card(s: dict, records: list[dict], live: bool = False, pm: str = "auc", trials: dict | None = None):
    """一段的可读摘要：做了什么、为什么、结果如何、代码怎么判。pm：本次运行锁定的主指标。"""
    evs = segment_evaluations(s, records)
    ml, key = metric_label(pm), f"oot_dev_{pm}"
    if s["kind"] == PHASE_SETUP:
        dirs = plan_directions(s)
        race = "<br>".join(f"<b>{esc(e.get('model'))}</b>　OOT-dev {ml} {_num((e.get('metrics') or {}).get(key))}　{pill(e.get('verdict'))}"
                           for e in evs)
        html_ = card('<span class="dm-step">准备</span><span class="dm-action">切分数据 · 规划方向 · 模型赛跑</span>',
                     [("规划方向", "<br>".join(f"{i}. {esc(d)}" for i, d in enumerate(dirs, 1))),
                      ("模型赛跑", race + ('<div class="dm-sub">同样的特征，每个模型先抽样试跑一个小 study，最好的一个进入决策循环</div>' if race else ""))],
                     pending=live and not evs)
    elif s["kind"] != PHASE_ROUND:
        f = s["final"]
        fm = f.get("metric", "auc") if f else pm                       # 旧运行的验收结果没有 metric 字段：当时只有 AUC
        final = (f"<b>{esc(f['exp_id'])}</b>　holdout {metric_label(fm)} <b>{f[f'holdout_{fm}']:.4f}</b>"
                 f"（OOT-dev {f[f'oot_dev_{fm}']:.4f}，下降 {f['drop']:+.4f}）"
                 + ("<br>⚠️ 疑似对 OOT-dev 过拟合" if f.get("overfit_to_oot_dev") else "")
                 + '<div class="dm-sub">holdout 只在这里读一次，决策过程中从未用到</div>') if f else \
            ("没有通过评估的全量训练模型，跳过验收" if any(e["type"] == "final_gate_skipped" for e in s["events"]) else "")
        html_ = card('<span class="dm-step">收尾</span><span class="dm-action">最终验收 · 模型卡 · 沉淀经验</span>',
                     [("停止原因", esc(stop_txt(s["stop_reason"])) if s["stop_reason"] else ""), ("最终验收", final)],
                     pending=live and not f)
    else:
        d, rec = s["decision"] or {}, s["recorded"]
        a = d.get("action")
        head = (f'<span class="dm-step">第 {s["round"]} 轮</span><span class="dm-action">'
                + (f'{esc(ACTIONS.get(a, a))}<code>{esc(a)}</code>' if a else "等待 LLM 决策…") + "</span>")
        notes = [f"⚠️ 第 {r.get('attempt', 0) + 1} 次输出被代码校验拒绝，错误已回灌给 LLM：{esc('；'.join(r.get('errors', []))[:200])}"
                 for r in s["rejected"]]
        ev = next((e for e in evs if rec and e["exp_id"] == rec["exp_id"]), None)
        result = ""
        if ev:
            cmp = ev.get("compare")
            result = f"<b>{esc(ev['exp_id'])}</b>　OOT-dev {ml} <b>{_num((ev.get('metrics') or {}).get(key))}</b>"
            if cmp:
                result += f"，比当前最优 {cmp['delta']:+.4f}（{'显著' if cmp['significant'] else '不显著'}）"
        elif rec and rec.get("verdict") == "NONE":
            result = "本轮没有训练模型"
        html_ = card(head, [("假设", esc(d.get("hypothesis", ""))), ("理由", esc(d.get("rationale", ""))), ("结果", result),
                            ("诊断", codes_html(rec.get("codes") or []) if rec else ""),
                            ("停止", esc(stop_txt(s["stop_reason"])) if s["stop_reason"] else "")],
                     verdict=rec.get("verdict") if rec else None, notes=notes, pending=live and not rec)
    st.markdown(html_, unsafe_allow_html=True)
    ids = [e["exp_id"] for e in evs if e.get("exp_id")]
    if live or trials is None or not ids:
        return
    n = sum(len(trials.get(i, [])) for i in ids)
    m = sum(t["state"] == "PRUNED" for i in ids for t in trials.get(i, []))
    accepted = s["kind"] == PHASE_ROUND and (s["recorded"] or {}).get("verdict") == "ACCEPT"
    with st.expander(f"调参过程：{n} 个 trial，其中 {m} 个被剪枝" if n else "调参过程：未保存调参过程", expanded=accepted):
        names = {e["exp_id"]: e.get("model") for e in evs} if s["kind"] == PHASE_SETUP else None
        ch = trial_chart(trials, ids, pm, names)
        if ch is None:
            st.caption("未保存调参过程")
        else:
            st.altair_chart(ch, use_container_width=True)
            st.caption("Optuna 只看 valid；OOT-dev 在搜索结束后评估一次，结果见上方卡片。灰点是被剪枝的 trial。")


def show_progress(events: list[dict], records: list[dict], state: str | None, max_rounds: int | None, running: bool,
                  trials: dict | None = None):
    trace = build_trace(events)
    pm = primary_of(events)
    n = sum(s["kind"] == PHASE_ROUND for s in trace)
    if running:
        st.markdown(f'<div class="dm-status"><span class="dm-dot"></span>当前阶段　<b>{esc(STATES.get(state, state or "…"))}</b></div>',
                    unsafe_allow_html=True)
        if max_rounds:
            st.progress(min(n / max_rounds, 1.0), text=f"决策轮次 {n} / {max_rounds}")
    if not running:
        ch = metric_chart(trace, records, pm)
        if ch is not None:
            st.altair_chart(ch, use_container_width=True)
    for s in trace:
        round_card(s, records, live=running, pm=pm, trials=trials if not running else None)


def legend():
    st.markdown('<div class="dm-legend">' + "".join(f'<span title="{esc(v[4])}">{pill(k)}</span>' for k, v in VERDICTS.items()
                                                    if v[2]) + '<span class="dm-sub">判定全部由代码给出，鼠标悬停看含义。LLM 原文里的"低保真 / 全保真"就是抽样试跑 / 全量训练</span></div>',
                unsafe_allow_html=True)


# ---------- 完整 trace ----------
def metric_chart(trace: list[dict], records: list[dict], pm: str):
    """每个实验一个点，纵轴 OOT-dev 主指标，颜色和形状表示判定；被接受的点加粗描边，一眼看出走到最终模型的路径。"""
    y, d = f"OOT-dev {metric_label(pm)}", f"Δ{metric_label(pm)}"
    rows = [{"段": seg_title(s), "实验": e["exp_id"], "判定": e.get("verdict"), "模型": e.get("model"),
             "训练方式": FIDELITY.get(e.get("fidelity"), e.get("fidelity")), y: (e.get("metrics") or {}).get(f"oot_dev_{pm}"),
             d: (e.get("compare") or {}).get("delta"), "接受": e.get("verdict") == "ACCEPT"}
            for s in trace for e in segment_evaluations(s, records)]
    df = pd.DataFrame([r for r in rows if r[y] is not None and r["判定"] in VERDICTS and VERDICTS[r["判定"]][2]])
    if df.empty:
        return None
    shown = [v for v in VERDICTS if v in set(df["判定"])]
    labels = [f"{VERDICTS[v][1]} {v}" for v in shown]
    df["判定"] = df["判定"].map(lambda v: f"{VERDICTS[v][1]} {v}")
    return alt.Chart(df, title=f"每轮实验的 {y}（颜色和形状表示判定，粗边框为被接受）").mark_point(filled=True, size=110, opacity=1).encode(
        x=alt.X("段:N", sort=list(dict.fromkeys(df["段"])), title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y(f"{y}:Q", scale=alt.Scale(zero=False)),
        stroke=alt.condition(alt.datum.接受, alt.value("#1f1f1f"), alt.value(None)),
        strokeWidth=alt.condition(alt.datum.接受, alt.value(2.5), alt.value(0)),
        color=alt.Color("判定:N", scale=alt.Scale(domain=labels, range=[VERDICTS[v][2] for v in shown]),
                        legend=alt.Legend(orient="top", title=None)),
        shape=alt.Shape("判定:N", scale=alt.Scale(domain=labels, range=[VERDICTS[v][3] for v in shown]), legend=None),
        tooltip=["段", "实验", "判定", "模型", "训练方式", alt.Tooltip(f"{y}:Q", format=".4f"),
                 alt.Tooltip(f"{d}:Q", format="+.4f")],
    ).properties(height=260)


def show_decision(s: dict):
    d = s["decision"]
    if not d:
        st.caption("本段没有 LLM 决策。")
        return
    st.markdown(f"**动作** `{d['action']}`　**预期收益** {d.get('expected_gain') or '—'}")
    st.markdown(f"**假设**　{d.get('hypothesis') or '—'}")
    st.markdown(f"**理由**　{d.get('rationale') or '—'}")
    if d.get("alternatives_considered"):
        st.markdown("**考虑过的其他动作**\n" + "\n".join(f"- {a}" for a in d["alternatives_considered"]))
    st.markdown("**参数**")
    st.json(d.get("params") or {}, expanded=False)


def show_evaluations(s: dict, records: list[dict], pm: str):
    evs = segment_evaluations(s, records)
    ml = metric_label(pm)
    if not evs:
        st.caption("本段没有评估结果。")
    for e in evs:
        m, cmp = e.get("metrics") or {}, e.get("compare")
        fmt = lambda k, f: format(m[k], f) if m.get(k) is not None else "—"
        st.markdown(f"**{e['exp_id']}**（{e.get('model') or '?'} · {FIDELITY.get(e.get('fidelity'), e.get('fidelity') or '?')}）　{verdict_txt(e.get('verdict'))}")
        st.markdown(f"valid {ml} {fmt(f'valid_{pm}', '.4f')} · OOT-dev {ml} {fmt(f'oot_dev_{pm}', '.4f')} · 分数 PSI {fmt('score_psi', '.3f')}")
        if cmp:
            st.caption(f"配对 bootstrap：Δ{ml} {cmp['delta']:+.4f}，95% 区间 [{cmp['ci_low']:+.4f}, {cmp['ci_high']:+.4f}]，"
                       f"α={cmp['alpha']:.4f}（第 {cmp.get('k')} 次使用 OOT-dev），{'显著' if cmp['significant'] else '不显著'}；对照 {e.get('baseline')}")
        if e.get("codes"):
            st.markdown(codes_md(e["codes"]))
        if e.get("source") == "record":
            st.caption("早期日志没有 evaluated 事件，指标取自实验记录。")


def show_llm_calls(s: dict, has_llm_events: bool):
    if not s["llm_calls"]:
        st.caption("本段没有调用 LLM。" if has_llm_events else "这份日志早于 llm_call 事件，没有记录 prompt 与回复原文。")
    for i, c in enumerate(s["llm_calls"], 1):
        st.markdown(f"**#{i} · {c.get('stage')}** · {c.get('model')} · {c.get('latency_s', 0):.2f}s · {c.get('tokens', 0)} tokens")
        if c.get("error"):
            st.error(c["error"])
        left, right = st.columns(2)
        with left:
            st.caption("prompt（user 部分）")
            with st.container(height=260):
                st.code(c.get("prompt", ""), language="markdown")
        with right:
            st.caption("回复")
            with st.container(height=260):
                st.code(c.get("response", ""), language="json")


def show_full_trace(events: list[dict], records: list[dict]):
    trace = build_trace(events)
    pm = primary_of(events)
    y, d = f"OOT-dev {metric_label(pm)}", f"Δ{metric_label(pm)}"
    has_llm = any(e["type"] == "llm_call" for e in events)
    st.markdown("#### 完整 trace")
    st.dataframe(pd.DataFrame(round_rows(trace, records, pm)), hide_index=True, use_container_width=True,
                 column_config={y: st.column_config.NumberColumn(format="%.4f"), d: st.column_config.NumberColumn(format="%+.4f")})
    for s in trace:
        d, r = s["decision"] or {}, s["recorded"] or {}
        with st.expander(" · ".join(x for x in (seg_title(s), d.get("action"), r.get("verdict")) if x)):
            st.caption("状态路径：" + " → ".join(s["states"]))
            t = st.tabs(["决策与理由", "评估与诊断", "LLM 调用原文", "原始事件"])
            with t[0]:
                show_decision(s)
            with t[1]:
                show_evaluations(s, records, pm)
            with t[2]:
                show_llm_calls(s, has_llm)
            with t[3]:
                st.dataframe(pd.DataFrame([{"id": e["id"], "时间": time.strftime("%H:%M:%S", time.localtime(e["ts"])),
                                            "类型": e["type"], "实验": e["exp_id"],
                                            "payload": json.dumps(e["payload"], ensure_ascii=False)[:500]}
                                           for e in s["events"]]), hide_index=True, use_container_width=True)


# ---------- 现场运行 ----------
def friendly_error(e: Exception) -> str:
    m = str(e).lower()
    if "429" in m or "quota" in m or "balance" in m or "rate limit" in m:
        return "LLM 服务拒绝了请求：账户余额不足或触发限流。"
    if "401" in m or "auth" in m or "api key" in m or "未设置" in str(e):
        return "API key 无效或未提供。"
    if "connect" in m or "timeout" in m:
        return "连不上 LLM 服务（网络或超时）。"
    return "运行中出错。"


def live_run(provider: str, api_key: str | None, max_rounds: int, box, dataset: str = "hotel_bookings", prepare=None) -> dict:
    """prepare(cfg) -> cfg：建任务之前改 cfg（接入页用它注册生成的数据集）。"""
    from agent.llm import make_llm
    from agent.orchestrator import Orchestrator
    from agent.schemas import TaskSpec
    from agent.states import TERMINAL, State
    from data.knowhow import load_config
    from data.sample import use_sample
    from evaluation.final_gate import FinalGate
    from runtime.tracing import make_tracer

    cfg = use_sample(load_config())
    cfg["fidelity"]["full"]["n_trials"], cfg["run"]["max_rounds"] = 5, max_rounds
    cfg["memory"]["db"] = str(sample_art() / f"memory_{provider}.db")    # mock 的经验不混进真实模型的上下文
    Path(cfg["paths"]["artifacts_root"]).mkdir(parents=True, exist_ok=True)
    if prepare:
        cfg = prepare(cfg)
    tid = f"live_{provider}_{time.strftime('%m%d_%H%M%S')}"
    out = {"task_id": tid, "error": None, "state": "FAILED"}
    o = tracer = None
    try:
        llm = make_llm(provider, cfg, api_key=api_key)
        spec = TaskSpec.from_knowhow(dataset, cfg, autonomy="L0")    # 无人值守：页面上不处理人工审批
        tracer = make_tracer(tid, {"dataset": f"{dataset}（样本）", "llm": provider, "source": "app"})
        o = Orchestrator(spec, cfg, llm, FinalGate(dataset, cfg, tid), tid, tracer=tracer)
        while o.state not in TERMINAL and o.state != State.AWAIT_HUMAN:
            with box.container():
                show_progress(o.events.query(), [], o.state.value, max_rounds, running=True)
            o.step()
        out["state"] = o.state.value
    except Exception as e:                                                # 出错也保留已经跑出来的部分
        out["error"] = (friendly_error(e), redact(f"{type(e).__name__}: {str(e)[:500]}"))
        out["state"] = o.state.value if o else "FAILED"
    out["langfuse"] = (tracer.url() or "enabled") if tracer is not None and tracer.enabled else None
    out["events"] = o.events.query() if o else []
    out["records"] = [r.model_dump(mode="json") for r in o.store.all(tid)] if o else []
    db = Path(cfg["paths"]["artifacts_root"]) / "optuna.db"
    out["trials"] = load_trials(str(db), [r["exp_id"] for r in out["records"]]) if db.exists() else {}
    box.empty()
    return out


def show_live_result(res: dict, toggle_key: str):
    if res["error"]:
        st.error(f"{res['error'][0]}（已跑完的部分保留在下面）")
        with st.expander("错误详情"):
            st.code(res["error"][1])
    else:
        sm = summarize(build_trace(res["events"]))
        msg = f"{res['task_id']} 结束，最终状态 {res['state']}；{sm['n_rounds']} 轮决策，停止原因：{stop_txt(sm['stop_reason'])}"
        (st.success if res["state"] == "DONE" else st.warning)(msg)       # FAILED（如决策连续校验失败）不能显示成成功
    if res.get("langfuse"):
        st.markdown(f"已上报 Langfuse：[打开这次运行的 trace]({res['langfuse']})" if res["langfuse"].startswith("http")
                    else f"已上报 Langfuse，可按 session `{res['task_id']}` 搜索。")
    legend()
    show_progress(res["events"], res["records"], None, None, running=False, trials=res.get("trials") or {})
    if res["events"] and st.toggle("显示完整 trace（逐轮表、LLM 原文、原始事件）", key=toggle_key):
        show_full_trace(res["events"], res["records"])


# ---------- 接入新数据 ----------
INTAKE_REC = ROOT / "examples" / "traces" / "intake" / "intake_hotel_deepseek.json"


def show_intake_events(events: list[dict]):
    """按轮展示接入会话：画像 → 每轮草稿要点、校验错误、问题 → 用户回答。"""
    for rnd, evs in intake_rounds(events):
        by = {e["type"]: e["payload"] for e in evs}
        if rnd == 0:
            p = by["profile"]
            st.markdown(f"**数据画像**（代码计算，LLM 只看这份统计，不看原始数据行）：{p['n_rows']:,} 行 · {len(p['columns'])} 列 · "
                        f"候选日期列 {', '.join(p['date_candidates'])} · 候选标签列 {', '.join(p['label_candidates'])}")
            continue
        with st.container(border=True):
            st.markdown(f"**第 {rnd} 轮起草**")
            for e in evs:
                if e["type"] == "validation_failed":
                    st.warning("草稿未通过校验，错误已回灌给 LLM 重写：" + "；".join(x[:160] for x in e["payload"]["errors"]))
            if "draft" in by:
                st.dataframe(pd.DataFrame(draft_rows(by["draft"]), columns=["项", "草稿"]), hide_index=True, use_container_width=True)
            qs = by.get("questions") or []
            if qs:
                st.markdown("**追问**（" + "、".join(q_label(q["key"]) for q in qs) + "）")
                st.dataframe(pd.DataFrame([{"类型": "必答项" if q["key"] in REQUIRED_KEYS else "字段", "问题": q["question"],
                                            "LLM 推荐": q["recommended"]} for q in qs]), hide_index=True, use_container_width=True)
            elif "written" in by:
                st.success("没有待确认的问题，校验通过，写出配置。")
            calls = [e["payload"] for e in evs if e["type"] == "llm_call"]
            with st.expander(f"本轮 LLM 回复原文（{len(calls)} 次）"):
                for c in calls:
                    st.code(c["response"], language="json")
        ans = next((e["payload"] for e in events if e["type"] == "answers" and e["round"] == rnd), None)
        if ans:
            st.markdown("**用户回答**")
            st.dataframe(pd.DataFrame([{"问题": q_label(k), "回答": v} for k, v in ans.items()]), hide_index=True, use_container_width=True)


def show_compare(c: dict):
    st.markdown("**与手写配置对比**（两边都经过切分时同一套字段规则）")
    k = st.columns(3)
    k[0].metric("泄漏字段识别", f"{len(c['leak_found'])} / {len(c['leak_ref'])}")
    k[1].metric("存疑字段隔离", f"{len(c['quarantine_found'])} / {len(c['quarantine_ref'])}")
    k[2].metric("可得时间一致", f"{c['availability_agree']:.1%}")
    miss = c["leak_missed"] + c["quarantine_missed"] + c["meta_missed"]
    if miss:
        st.caption("没识别出来的：" + "、".join(miss) + "。这些字段会进入模型，所以真实接入时仍需要人复核可得时间。")
    if c["disagreements"]:
        st.dataframe(pd.DataFrame(c["disagreements"]).rename(columns={"field": "字段", "generated": "agent 判断", "reference": "手写配置"}),
                     hide_index=True, use_container_width=True)


def replay_prepare(rec: dict):
    """把录制里写出的 YAML 放到运行时目录（不动仓库的 knowhow/），再注册进 cfg。"""
    def prep(cfg):
        from intake.writer import register
        out = sample_art() / "intake" / "replay"
        kd = out / "knowhow" / rec["dataset_id"]
        kd.mkdir(parents=True, exist_ok=True)
        for n, t in rec["written"]["yaml"].items():
            (kd / n).write_text(t)
        return register(cfg, {**rec["written"], "knowhow_root": str(out / "knowhow"), "csv": str(SAMPLE_CSV)})
    return prep


# ---------- 页面 ----------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700&display=swap');
html, body, .stApp, .stMarkdown, p, li, dd, dt, label, h1, h2, h3, h4, h5, input, textarea, button,
[data-testid="stMetricValue"], [data-testid="stMetricLabel"], [data-baseweb="tab"] {
  font-family: "Noto Sans SC", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", system-ui, sans-serif;
}
.block-container { max-width: 1120px; padding-top: 2.2rem; padding-bottom: 4rem; }
.stMarkdown p, .stMarkdown li { line-height: 1.75; }
h1 { font-weight: 700; letter-spacing: -0.01em; margin-bottom: 0.2rem; }
[data-testid="stMetricValue"] { font-size: 1.45rem; font-weight: 600; }
[data-testid="stMetricLabel"] p { font-size: 0.82rem; opacity: 0.7; }
[data-baseweb="tab-list"] { gap: 0.5rem; }
[data-baseweb="tab"] p { font-size: 1.02rem; font-weight: 500; }

.dm-lede { font-size: 1.05rem; opacity: 0.85; margin: 0 0 1.1rem; line-height: 1.8; }
.dm-flow { display: flex; gap: 10px; flex-wrap: wrap; margin: 0 0 1.6rem; }
.dm-flow > div { flex: 1 1 200px; border: 1px solid rgba(128,128,128,.22); border-radius: 10px; padding: 10px 14px;
  background: rgba(128,128,128,.045); font-size: .92rem; line-height: 1.6; }
.dm-flow b { display: block; font-size: .78rem; letter-spacing: .04em; opacity: .6; font-weight: 500; }
.dm-section { font-size: 1.15rem; font-weight: 700; margin: 0.4rem 0 0.2rem; }
.dm-sub { font-size: .82rem; opacity: .62; margin-top: 4px; line-height: 1.6; }

.dm-card { --c: rgba(128,128,128,.45); border: 1px solid rgba(128,128,128,.22); border-left: 4px solid var(--c);
  border-radius: 10px; padding: 14px 18px 12px; margin: 0 0 12px; background: rgba(128,128,128,.035); }
.dm-head { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
.dm-step { font-weight: 700; font-size: 1.02rem; }
.dm-action { opacity: .85; }
.dm-action code, .dm-rows code { font-size: .78rem; padding: 1px 6px; border-radius: 5px; }
.dm-action code { margin-left: 6px; } .dm-rows code { margin-right: 4px; }
.dm-head .dm-pill { margin-left: auto; }
.dm-rows { display: grid; grid-template-columns: 4.2em 1fr; gap: 6px 14px; margin: 10px 0 2px; }
.dm-rows dt { opacity: .55; font-size: .88rem; padding-top: 2px; }
.dm-rows dd { margin: 0; line-height: 1.7; font-size: .95rem; }
.dm-note { margin-top: 8px; padding: 6px 10px; border-radius: 6px; font-size: .86rem; line-height: 1.6;
  background: color-mix(in srgb, #fab219 12%, transparent); }

.dm-pill { --c: rgba(128,128,128,.6); display: inline-flex; align-items: center; gap: 5px; white-space: nowrap;
  padding: 2px 10px; border-radius: 999px; font-size: .84rem; font-weight: 500;
  border: 1px solid color-mix(in srgb, var(--c) 55%, transparent); background: color-mix(in srgb, var(--c) 14%, transparent); }
.dm-en { font-size: .7rem; opacity: .6; letter-spacing: .03em; }
.dm-v-ACCEPT { --c: #0ca30c; } .dm-v-PROMISING { --c: #fab219; } .dm-v-INCONCLUSIVE { --c: #8c8c88; }
.dm-v-REJECT { --c: #d03b3b; } .dm-v-NONE { --c: rgba(128,128,128,.6); }
.dm-legend { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin: 4px 0 14px; }
.dm-legend .dm-sub { margin: 0 0 0 4px; }

.dm-status { display: flex; align-items: center; gap: 8px; margin: 6px 0 4px; }
.dm-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--primary-color, #3d6fd9);
  animation: dm-pulse 1.2s ease-in-out infinite; display: inline-block; }
.dm-pending { display: flex; align-items: center; gap: 8px; font-size: .86rem; opacity: .7; margin-top: 6px; }
@keyframes dm-pulse { 0%,100% { opacity: .25; } 50% { opacity: 1; } }
@media (prefers-reduced-motion: reduce) { .dm-dot { animation: none; } }
</style>
"""

st.set_page_config(page_title="DataMiner Agent 演示", page_icon="🔎", layout="wide", initial_sidebar_state="collapsed")
st.markdown(CSS, unsafe_allow_html=True)
st.title("DataMiner Agent")
st.markdown('<p class="dm-lede">给一份表格数据和建模目标，agent 自己一轮一轮做实验。'
            '下面可以回放一次真实 LLM 的运行，或者现场跑一次。</p>'
            '<div class="dm-flow">'
            '<div><b>① LLM</b>看诊断和历史，提出下一步动作，写下假设和理由</div>'
            '<div><b>② 代码</b>校验动作是否合法，执行训练、评估、诊断</div>'
            '<div><b>③ 代码</b>用配对 bootstrap 判定接受或拒绝，LLM 的意见不参与</div>'
            '</div>', unsafe_allow_html=True)

df = load_sample()
st.markdown('<div class="dm-section">数据：酒店预订，预测订单会不会被取消</div>', unsafe_allow_html=True)
c = st.columns(4)
c[0].metric("样本行数", f"{len(df):,}")
c[1].metric("取消率", f"{df['is_canceled'].mean():.1%}")
c[2].metric("到店时间", f"{df['arrival_date_year'].min()}-07 ~ {df['arrival_date_year'].max()}-08")
c[3].metric("原始字段", f"{df.shape[1]}")
st.dataframe(df.head(8), hide_index=True, use_container_width=True, height=318)
st.markdown('<div class="dm-sub">来源：Antonio, Almeida &amp; Nunes (2019) <i>Hotel booking demand datasets</i>，CC BY 4.0；'
            '仓库里是按到店月份 × 是否取消分层抽取的样本（data_sample/hotel_bookings/NOTICE.md）。'
            '按到店时间切成 train/valid（2015-07~2016-12）、OOT-dev（2017-01~04）、holdout（2017-05~08）。</div>',
            unsafe_allow_html=True)
with st.expander("哪些字段 agent 能用、哪些会被隔离（know-how 标注）"):
    st.dataframe(load_fields(), hide_index=True, use_container_width=True)
st.write("")

tab_replay, tab_live, tab_intake = st.tabs(["① 历史回放：真实 LLM 的一次运行", "② 现场运行：现在跑一次",
                                            "③ 接入新数据：从数据和说明生成配置"])

with tab_replay:
    names = example_names()
    if not names:
        st.info("examples/traces/ 下还没有示例。")
    else:
        labels = {" · ".join(x for x in (n, load_example(n).get("meta", {}).get("llm"),
                                          load_example(n).get("meta", {}).get("dataset")) if x): n for n in names}
        name = labels[st.selectbox("选择一次运行", list(labels), key="example")]
        ex = load_example(name)
        meta, events, records = ex.get("meta", {}), ex["events"], ex.get("records") or []
        sm = summarize(build_trace(events))
        left, right = st.columns([3, 2], gap="large")
        with left:
            with st.container(border=True):
                st.markdown("**运行档案**")
                st.markdown(name_story(name, meta))
                c = st.columns(4)
                c[0].metric("运行日期", time.strftime("%Y-%m-%d", time.localtime(events[0]["ts"])))
                c[1].metric("耗时", f"{(events[-1]['ts'] - events[0]['ts']) / 60:.1f} 分钟")
                c[2].metric("决策轮数", sm["n_rounds"])
                c[3].metric("被拒次数", sm["n_rejected"])
                st.caption(" · ".join(x for x in (meta.get("settings"), ex.get("note")) if x))
                if not meta.get("has_sample_data"):
                    st.caption("⚠️ 这次运行用的不是上面的酒店样本，原始数据也不在仓库里；这里只回放它的决策过程。")
        with right:
            st.info("这是**历史回放**，不是实时运行：播放的是当时真实调用 LLM 记下的事件日志，顺序不变。"
                    "想看实时效果，请用「② 现场运行」。")
            speed = st.select_slider("回放速度", ["慢", "中", "快"], value="中", key="speed")
            clicked = st.button("▶ 回放这次运行", type="primary", key="replay", use_container_width=True)
        done = st.session_state.setdefault("replayed", set())
        if clicked:
            legend()
            box, delay = st.empty(), {"慢": 0.8, "中": 0.35, "快": 0.1}[speed]
            for i, e in enumerate(events):
                if e["type"] not in ("state_transition", "decision", "recorded"):
                    continue
                with box.container():
                    show_progress(events[: i + 1], records, e["payload"].get("to") if e["type"] == "state_transition" else None,
                                  sm["n_rounds"], running=True)
                time.sleep(delay)
            box.empty()
            done.add(name)
            st.rerun()
        if name in done:
            st.success(f"回放结束：{sm['n_rounds']} 轮决策，停止原因：{stop_txt(sm['stop_reason'])}")
            legend()
            show_progress(events, records, None, None, running=False, trials=ex.get("trials") or {})
            if st.toggle("显示完整 trace（逐轮表、LLM 原文、原始事件）", key="full_replay"):
                show_full_trace(events, records)

with tab_live:
    left, right = st.columns([2, 3], gap="large")
    with left:
        with st.container(border=True):
            provider = {v: k for k, v in PROVIDERS.items()}[st.selectbox("用哪个 LLM 做决策", list(PROVIDERS.values()), key="provider")]
            api_key = None
            if provider != "mock":
                from data.knowhow import load_config
                pcfg = load_config()["llm"]["providers"][provider]
                api_key = st.text_input(f"API key（模型 {pcfg['model']}）", type="password", key="api_key",
                                        help="只在本次页面会话中使用：不写盘、不进日志、不放进 URL。") or None
                has_env = bool(os.environ.get(pcfg["key_env"]))
                if has_env:
                    st.caption(f"留空则使用 .env 里的 {pcfg['key_env']}。")
                ready = bool(api_key) or has_env
            else:
                ready = True
            rounds = st.slider("最多决策轮数", 1, 5, 3, key="max_rounds")
            run = st.button("▶ 开始运行", type="primary", key="run", disabled=not ready, use_container_width=True)
    with right:
        st.markdown("在上面的酒店样本上**从头真实跑一遍**：切分数据 → 模型赛跑 → LLM 决策、代码训练和判定循环 → holdout 验收。"
                    "每完成一步，下面的卡片就更新一次。")
        if provider == "mock":
            st.caption("mock 按固定的策略先验选动作，理由是模板文字，约 10 秒跑完，用来演示流程本身。"
                       "想看真实的决策理由，选一个真实模型，或者看「① 历史回放」。")
        else:
            st.caption("真实模型每次决策要等几秒到几十秒；一次 5 轮的运行约消耗 3 万 token（DeepSeek 实测），费用由该 key 的账户承担。")
        lf = bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))
        st.caption("Langfuse：已配置，本次运行会实时上报 trace（含 prompt 与回复）。" if lf else
                   "Langfuse：未配置。在 .env 里填 LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY 后，运行会同时上报到 Langfuse。")
    if run:
        legend()
        st.session_state["live"] = live_run(provider, api_key, rounds, st.empty())
        st.rerun()
    if st.session_state.get("live"):
        show_live_result(st.session_state["live"], "full_live")

with tab_intake:
    st.markdown("上传一张表和一段业务说明，agent 先看代码算出的数据画像，起草建模配置，再追问它不能自行假设的内容。"
                "标签、预测时点、切分窗口、指标四项是**必答项**：LLM 给了推荐也必须由人确认，由代码强制。"
                "时间表达式在 DuckDB 里实跑校验，切分后每段都要有正负样本；通过后写出与手写格式相同的配置，接着走同一套建模流程。")
    mode = st.radio("方式", ["回放一次真实的接入会话", "现场接入（需要 API key）"], horizontal=True, key="intake_mode",
                    label_visibility="collapsed")
    if mode.startswith("回放"):
        rec = json.loads(INTAKE_REC.read_text())
        st.caption(f"录制：{rec['model']} · 酒店样本 · 业务方的回答依据仓库里手写的配置如实给出，LLM 没问到的不主动透露。")
        with st.expander("业务说明原文"):
            st.markdown(rec["description"])
        if st.button("▶ 回放接入过程", type="primary", key="intake_replay"):
            st.session_state["intake_shown"] = True
        if st.session_state.get("intake_shown"):
            show_intake_events(rec["events"])
            for n, t in rec["written"]["yaml"].items():
                with st.expander(n):
                    st.code(t, language="yaml")
            show_compare(rec["compare"])
            st.markdown("**用这份配置建模**：生成的配置放到运行时目录，用 mock LLM 从头跑一遍（约 10 秒），证明它能直接用。")
            if st.button("▶ 用这份配置跑一次", key="intake_run"):
                legend()
                st.session_state["intake_live"] = live_run("mock", None, 2, st.empty(), dataset=rec["dataset_id"],
                                                           prepare=replay_prepare(rec))
                st.rerun()
            if st.session_state.get("intake_live"):
                show_live_result(st.session_state["intake_live"], "full_intake")
    else:
        from data.knowhow import load_config
        real = {v: k for k, v in PROVIDERS.items() if k != "mock"}
        prov = real[st.selectbox("用哪个 LLM 起草", list(real), key="intake_provider")]
        pcfg = load_config()["llm"]["providers"][prov]
        key = st.text_input(f"API key（模型 {pcfg['model']}）", type="password", key="intake_key",
                            help="只在本次页面会话中使用：不写盘、不进日志、不放进 URL。") or None
        up = st.file_uploader("数据文件（CSV）", type="csv", key="intake_csv")
        desc = st.text_area("业务说明", (SAMPLE_CSV.parent / "业务说明.md").read_text(), height=220, key="intake_desc")
        ready = bool(up) and (bool(key) or bool(os.environ.get(pcfg["key_env"])))
        if st.button("▶ 开始接入", type="primary", key="intake_start", disabled=not ready):
            from agent.llm import make_llm
            from intake.session import IntakeSession
            p = sample_art() / "intake" / "upload.csv"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(up.getvalue())
            for k in ("intake_live2", "intake_langfuse"):             # 新的接入会话：清掉上一次的结果
                st.session_state.pop(k, None)
            s = IntakeSession(make_llm(prov, load_config(), api_key=key), str(p), desc, "user_data", str(sample_art() / "intake" / "live"))
            with st.spinner("LLM 起草中…"):
                st.session_state["intake_qs"] = s.step()
            st.session_state["intake_session"] = s
        s = st.session_state.get("intake_session")
        if s:
            show_intake_events(s.events)
            qs = st.session_state.get("intake_qs") or []
            if qs:
                with st.form("intake_answers"):
                    st.markdown("**请回答**（默认填的是 LLM 的推荐，采纳就直接提交）")
                    replies = {q.key: st.text_input(("【必答项】" if q.key in REQUIRED_KEYS else "") + q.question, q.recommended,
                                                    key=f"ans_{q.key}") for q in qs}
                    if st.form_submit_button("提交回答"):
                        s.answer(replies)
                        with st.spinner("LLM 根据回答修改草稿…"):
                            st.session_state["intake_qs"] = s.step()
                        st.rerun()
            elif s.written:
                if "intake_langfuse" not in st.session_state:          # 写出配置后上报一次（配了 LANGFUSE_* 才会上报）
                    from intake.tracing import report_intake
                    from runtime.tracing import langfuse_client
                    lf = langfuse_client()
                    tid = report_intake({"dataset_id": s.dataset_id, "llm": prov, "model": pcfg["model"], "description": s.description,
                                         "events": s.events, "written": s.written}, lf)
                    st.session_state["intake_langfuse"] = lf.get_trace_url(trace_id=tid) if tid else None
                if st.session_state["intake_langfuse"]:
                    st.markdown(f"已上报 Langfuse：[打开这次接入的 trace]({st.session_state['intake_langfuse']})")
                for n, t in s.written["yaml"].items():
                    with st.expander(n):
                        st.code(t, language="yaml")
                if st.button("▶ 用这份配置跑一次", key="intake_live_run"):
                    legend()
                    st.session_state["intake_live2"] = live_run("mock", None, 2, st.empty(), dataset=s.dataset_id, prepare=s.register)
                    st.rerun()
                if st.session_state.get("intake_live2"):
                    show_live_result(st.session_state["intake_live2"], "full_intake_live")

with st.expander("判定与诊断码说明"):
    st.markdown("\n".join(f"- {v[0]} **{v[1]}** `{k}`：{v[4]}" for k, v in VERDICTS.items()))
    st.markdown("\n".join(f"- `{k}` {v}" for k, v in CODES.items()))
