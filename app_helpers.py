"""演示页的纯函数（不依赖 streamlit 会话），便于单测。"""
import altair as alt
import pandas as pd

LABELS = {"auc": "AUC", "pr_auc": "PR-AUC", "ks": "KS"}


def primary_of(events: list[dict]) -> str:
    """本次运行锁定的主指标；旧运行没有 spec_locked 事件，当时只有 AUC。"""
    return next((e["payload"]["metric"]["primary"] for e in events if e["type"] == "spec_locked"), "auc")


def metric_label(pm: str) -> str:
    return LABELS[pm]


def trial_rows(trials: dict, exp_ids: list[str], names: dict | None = None) -> pd.DataFrame:
    """完成的 trial 画得分；被剪枝的没有得分，画在当时的"至今最优"上；失败、运行中的不画。"""
    rows = []
    for e in exp_ids:
        best = None
        for t in trials.get(e, []):
            if t["state"] == "COMPLETE" and t["value"] is not None:
                best = t["value"] if best is None else max(best, t["value"])
                rows.append({"曲线": (names or {}).get(e, e), "trial": t["n"], "得分": t["value"], "状态": "完成", "至今最优": best})
            elif t["state"] == "PRUNED":
                rows.append({"曲线": (names or {}).get(e, e), "trial": t["n"], "得分": best, "状态": "剪枝", "至今最优": best})
    return pd.DataFrame(rows)


def trial_chart(trials: dict, exp_ids: list[str], pm: str, names: dict | None = None):
    """横轴 trial 序号，纵轴 valid 主指标；剪枝的 trial 画成灰色；阶梯线为至今最优。
    names：exp_id → 图例名（赛跑阶段用模型名区分几条曲线）。"""
    df = trial_rows(trials, exp_ids, names)
    if df.empty or df["得分"].isna().all():
        return None
    y = f"valid {metric_label(pm)}"
    base = alt.Chart(df).encode(x=alt.X("trial:Q", title="trial 序号", axis=alt.Axis(tickMinStep=1)))
    line = base.mark_line(interpolate="step-after", opacity=.55).encode(
        y=alt.Y("至今最优:Q", title=y, scale=alt.Scale(zero=False)), color=alt.Color("曲线:N", legend=None))
    done = base.transform_filter(alt.datum.状态 == "完成").mark_point(filled=True, size=55).encode(
        y="得分:Q", color=alt.Color("曲线:N", legend=alt.Legend(orient="top", title=None)),
        tooltip=["曲线", "trial", "状态", alt.Tooltip("得分:Q", format=".4f")])
    pruned = base.transform_filter(alt.datum.状态 == "剪枝").mark_point(size=55, color="#a0a0a0").encode(
        y="得分:Q", tooltip=["曲线", "trial", "状态"])
    return (line + done + pruned).properties(height=210)


# ---------- 接入新数据 ----------
REQUIRED_KEYS = ("label", "observation_time_expr", "windows", "metric")
_Q_NAMES = {"label": "标签", "observation_time_expr": "预测时点", "windows": "切分窗口", "metric": "指标"}


def q_label(key: str) -> str:
    return _Q_NAMES.get(key, key)


def intake_rounds(events: list[dict]) -> list[tuple[int, list[dict]]]:
    """按 round 分组；回答事件不在这里（页面单独展示在它所回答的那一轮之后）。"""
    out: dict[int, list] = {}
    for e in events:
        if e["type"] != "answers":
            out.setdefault(e["round"], []).append(e)
    return sorted(out.items())


def draft_rows(d: dict) -> list[tuple[str, str]]:
    """草稿要点：页面上一张两列表。"""
    lab, w, m = d["label"] or {}, d["windows"] or {}, d["metric"] or {}
    known = sum(v["availability"] in ("at_booking", "history") for v in d["fields"].values())
    return [("标签", f"{lab.get('col')}（正类 {lab.get('positive')}）：{lab.get('definition')}"),
            ("预测时点", d["observation_time_expr"] or "—"), ("切分时间", d["split_time_expr"] or "—"),
            ("切分窗口", " / ".join(f"{k} {v[0]}~{v[1]}" for k, v in w.items()) or "—"),
            ("剔除（泄漏）", "、".join(d["leakage"]) or "—"), ("隔离（存疑）", "、".join(d["quarantine"]) or "—"),
            ("预测时已知的字段", f"{known} 个"),
            ("指标", f"主指标 {m.get('primary')}，护栏 {m.get('guards')}" if m else "—")]
