"""最终交付物（§4.18）：模型卡 + 特征清单 + Mermaid 实验树 + 风险提示。
数据全部由代码填充；LLM 只写一段执行摘要，且摘要中的数字必须能在事实数据里找到，否则丢弃。"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data.access import DataAccess
from data.knowhow import field_table, load_knowhow, normalize
from evaluation.compare import monthly_auc
from evaluation.guardrails import psi
from features.registry import Registry
from features.screen import iv
from modeling.explain import explain

NARRATIVE_SYSTEM = ("你是报告撰写节点。根据给定的 facts（JSON）写 3–5 句中文执行摘要。"
                    "只能使用 facts 中出现的数字，不要自己计算或推断新数字，不要评价方法优劣。只输出摘要文字。")


def safe_narrative(text: str, facts: dict) -> str | None:
    """摘要里出现的每个数字都必须能在 facts 里找到（允许四舍五入），否则整段丢弃。"""
    nums = []

    def walk(x):
        if isinstance(x, dict):
            [walk(v) for v in x.values()]
        elif isinstance(x, (list, tuple)):
            [walk(v) for v in x]
        elif isinstance(x, (int, float)) and not isinstance(x, bool):
            nums.append(float(x))
    walk(facts)
    for tok in re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?![\w])", text):
        d = len(tok.split(".")[1]) if "." in tok else 0
        if not any(abs(f - float(tok)) <= 0.5 * 10 ** -d + 1e-9 for f in nums):
            return None
    return text.strip()


def mermaid_tree(records) -> str:
    """节点：动作 / 保真度 / OOT-dev AUC（相对父节点的变化）/ 判定；失效节点虚线。"""
    by = {r.exp_id: r for r in records}
    sid = lambda e: "n_" + re.sub(r"\W", "_", e)
    lines = ["flowchart TD", "  ROOT((start))"]
    for r in records:
        auc = r.metrics.get("oot_dev_auc")
        par = by.get(r.parent_exp_id)
        delta = ""
        if auc is not None and par is not None and par.metrics.get("oot_dev_auc") is not None:
            delta = f" ({auc - par.metrics['oot_dev_auc']:+.4f})"
        body = f"{r.exp_id}<br/>{r.action_type} [{r.fidelity}]<br/>" + (f"AUC {auc:.4f}{delta}<br/>" if auc is not None else "") + r.verdict
        if r.diagnosis_codes:
            body += "<br/>" + ",".join(r.diagnosis_codes)
        lines.append(f'  {sid(r.exp_id)}["{body}"]:::{r.verdict.lower()}{"_inv" if r.invalidated else ""}')
    for r in records:
        lines.append(f"  {sid(r.parent_exp_id) if r.parent_exp_id in by else 'ROOT'} --> {sid(r.exp_id)}")
    lines += ["  classDef accept fill:#c8e6c9,stroke:#2e7d32", "  classDef reject fill:#ffcdd2,stroke:#c62828",
              "  classDef inconclusive fill:#fff9c4,stroke:#f9a825", "  classDef promising fill:#bbdefb,stroke:#1565c0",
              "  classDef none fill:#eeeeee,stroke:#757575",
              "  classDef accept_inv,reject_inv,inconclusive_inv,promising_inv,none_inv fill:#f5f5f5,stroke:#9e9e9e,stroke-dasharray:4 3"]
    return "\n".join(lines)


def _feature_meta(dataset, cfg, config):
    kh = load_knowhow(dataset, cfg)
    d = kh["dictionary"]
    fields, b = normalize(kh, d.get("label", {}).get("source_table"))
    meta = {f: {"source": "原始字段", "definition": (s.get("note") or s.get("type") or ""), "prior": None} for f, s in fields.items()}
    _, dexpr, _ = field_table(kh, d.get("label", {}).get("source_table"), cfg["task_specs"].get(dataset, {}).get("derived_expr_overrides"))
    for n_, e_ in dexpr.items():                                  # know-how 里的 classic_derived
        meta[n_] = {"source": "派生特征（classic_derived）", "definition": e_, "prior": None}
    reg = Registry(cfg) if config.get("feature_sets") else None
    for fs in config.get("feature_sets") or []:
        for df_ in reg.get(fs)["defs"]:
            meta[df_["name"]] = {"source": f"模板 {df_['template']}", "definition": f"{df_['expr']} @ {df_['window']}（{df_.get('rationale', '')}）",
                                 "prior": df_.get("prior")}
    pp = Path(cfg["paths"]["knowhow_root"]) / "domain_priors.yaml"
    pri = (yaml.safe_load(open(pp)).get("risk_direction", {}).get(dataset, {})) if pp.exists() else {}   # 新数据集可以没有先验
    return meta, pri, set(b.get("incumbent_model", {}).get("fields", [])) if b.get("incumbent_model", {}).get("action") != "exclude_by_default" else set()


def build_report(cfg: dict, spec, store, p: dict, final: dict | None, oot_used: int, llm=None) -> dict:
    task, ds = p["task_id"], spec.dataset
    art, rd = Path(cfg["paths"]["artifacts_root"]), Path(cfg["paths"]["reports_root"])
    rd.mkdir(exist_ok=True)
    recs = store.all(task)
    best_id = p["best_exp_id"]
    da = DataAccess(ds, str(art))
    tr, va, oo = da.load("train"), da.load("valid"), da.load("oot_dev")
    risks, L = [], []
    best = json.loads((art / "experiments" / best_id / "result.json").read_text()) if best_id else None

    facts = {"dataset": ds, "best_exp": best_id, "n_experiments": len(recs), "n_accept": sum(r.verdict == "ACCEPT" for r in recs),
             "oot_used": oot_used, "stop_reason": p["stop_reason"]}
    if best:
        facts.update(model=best["config"]["model"], oot_dev_auc=round(best["metrics"]["oot_dev_auc"], 4))
    if final:
        facts["holdout_auc"] = round(final["holdout_auc"], 4)
    narrative = None
    if llm is not None:
        try:
            text, _ = llm.complete(NARRATIVE_SYSTEM, "```json\n" + json.dumps({"mode": "REPORT", "facts": facts}, ensure_ascii=False) + "\n```", 0.0)
            narrative = safe_narrative(text, facts)
        except Exception:
            narrative = None

    L += [f"# 模型卡：{task}", "", f"> 数据集 `{ds}`；本文档数据全部由代码生成。", ""]
    L += ["## 执行摘要", "", narrative or "（未生成摘要，或摘要中的数字无法在事实数据中核对，已丢弃。）", ""]
    s = spec
    L += ["## 1. 任务规格", "", "| 项 | 值 |", "|---|---|", f"| label 定义 | {s.label_def} |", f"| 观察时间列 | {s.observation_time_col} |",
          f"| OOT 窗口 | {json.dumps(s.oot_windows, ensure_ascii=False)} |", f"| 目标指标 | {s.target_metric}（目标值 {s.target_value}） |",
          f"| 约束 | {s.constraints.model_dump()} |", f"| 预算 | {s.budget.model_dump()} |", f"| 自治级别 | {s.autonomy} |", ""]
    L += ["## 2. 数据切分", "", "| 切分 | 样本数 | 坏样本率 |", "|---|---|---|"]
    for nm, df in (("train", tr), ("valid", va), ("oot_dev", oo)):
        L.append(f"| {nm} | {len(df):,} | {df['_label'].mean():.4f} |")
    if final:
        L.append(f"| holdout（仅 final_gate） | {final['n_holdout']:,} | {final['holdout_bad_rate']:.4f} |")
    has_time = "_obs_time" in oo
    if not has_time:
        L += ["", "> ⚠ 该数据集没有绝对时间：OOT-dev 与 holdout 都是**随机切分近似**，无法检验时间稳定性，PSI 类指标仅作参考。"]
    L.append("")

    if best:
        m = best["metrics"]
        L += ["## 3. 最终模型", "", f"- 实验：`{best_id}`；模型：`{best['config']['model']}`；保真度：{best['config']['fidelity']}；trial 数：{best['n_trials']}（剪枝 {best['n_pruned']}）",
              f"- 特征数：{best['n_features']}；最优参数：`{json.dumps(best['best_params'], ensure_ascii=False)}`", ""]
        L += ["## 4. 各段指标", "", "| 段 | AUC | KS |", "|---|---|---|", f"| train | {m['train_auc']:.4f} | |", f"| valid | {m['valid_auc']:.4f} | |",
              f"| OOT-dev | {m['oot_dev_auc']:.4f} | {m['oot_dev_ks']:.4f} |"]
        if final:
            L.append(f"| holdout | {final['holdout_auc']:.4f} | {final['holdout_ks']:.4f} |")
        L += ["", f"- valid − OOT-dev gap：{m['gap']:.4f}；分数 PSI（valid→OOT-dev）：{m.get('score_psi', float('nan')):.4f}；OOT-dev 使用次数：{oot_used}", ""]
        if final and final["overfit_to_oot_dev"]:
            risks.append(f"holdout AUC 比 OOT-dev 低 {final['drop']:.4f}，超过 δ={cfg['final_gate']['delta']}：**疑似对 OOT-dev 过拟合**。")
        if has_time:
            oot_pred = np.load(art / "experiments" / best_id / "oot_dev_pred.npy")
            val_pred = np.load(art / "experiments" / best_id / "valid_pred.npy")
            tcol = "_split_time" if "_split_time" in oo else "_obs_time"       # 与切分一致（如到店日）
            mo = monthly_auc(oo[tcol], oo["_label"], oot_pred, min_n=50)
            hist = pd.concat([tr, va])                                          # 去年同期坏样本率：从训练期（train+valid）取 12 个月前的同一月份
            hbr = hist.groupby(pd.to_datetime(hist[tcol]).dt.strftime("%Y-%m"))["_label"].mean().to_dict()
            prev = lambda k: (pd.Period(k) - 12).strftime("%Y-%m")
            L += ["## 5. 分月 AUC、坏样本率与分数 PSI", "", f"（月份按 `{tcol}` 划分）", "",
                  "| 月份 | 段 | n | AUC | 坏样本率 | 去年同期坏样本率 | 分数 PSI（相对 valid） |", "|---|---|---|---|---|---|---|"]
            months = pd.to_datetime(oo[tcol]).dt.strftime("%Y-%m")
            yoy = lambda k: f"{hbr[prev(k)]:.3f}" if prev(k) in hbr else "-"
            for k, v in mo.items():
                L.append(f"| {k} | OOT-dev | {v['n']} | {v['auc']:.4f} | {v['bad_rate']:.3f} | {yoy(k)} | {psi(val_pred, oot_pred[(months == k).values], cfg['evaluator']['psi_bins']):.4f} |")
            for k, v in (final or {}).get("holdout_monthly_auc", {}).items():
                L.append(f"| {k} | holdout | {v['n']} | {v['auc']:.4f} | {v.get('bad_rate', float('nan')):.3f} | {yoy(k)} | |")
            aucs = [v["auc"] for v in list(mo.values()) + list((final or {}).get("holdout_monthly_auc", {}).values())]
            if aucs and max(aucs) - min(aucs) > 0.05:
                risks.append(f"分月 AUC 极差 {max(aucs) - min(aucs):.3f} > 0.05（sanity_rules）。注意月度样本量差异较大。")
            cav = load_knowhow(ds, cfg)["dictionary"].get("suggested_split", {}).get("caveat")
            L += ([""] + [f"> {ln.strip()}" for ln in str(cav).strip().splitlines()]) if cav else []
            L.append("")

        # ---- 特征清单 ----
        ex = explain(ds, cfg, best)
        meta, prior, incumbent = _feature_meta(ds, cfg, best["config"])
        trf = tr.copy()
        if best["config"].get("feature_sets"):
            trf = Registry(cfg).attach(trf, best["config"]["feature_sets"], "train")
        rows = []
        for f, e in sorted(ex.items(), key=lambda x: -x[1]["share"]):
            mt = meta.get(f, {"source": "原始字段", "definition": "", "prior": None})
            pr = prior.get(f, mt["prior"]) if prior.get(f, None) is not None or mt["prior"] is None else mt["prior"]
            flag = ""
            if pr not in (None, 0) and abs(e["corr"]) > 0.1 and np.sign(e["corr"]) != np.sign(pr) and e["share"] >= cfg["report"]["sanity_min_share"]:
                flag = "⚠ 反直觉"
                risks.append(f"特征 `{f}` 的 SHAP 方向与先验风险方向相反（{flag}），交人工判断。")
            v_iv = iv(trf[f], trf["_label"]) if f in trf else float("nan")
            rows.append((f, mt["source"], mt["definition"][:70], v_iv, e["share"], flag))
        L += ["## 6. 特征清单（按平均 |SHAP| 排序）", "", "| 特征 | 来源 | 定义/说明 | IV | 贡献占比 | 备注 |", "|---|---|---|---|---|---|"]
        for f, src, dfn, v_iv, sh, flag in rows[:40]:
            L.append(f"| {f} | {src} | {dfn} | {v_iv:.3f} | {sh:.1%} | {flag} |")
        if len(rows) > 40:
            L.append(f"| …其余 {len(rows) - 40} 个 | | | | {sum(r[4] for r in rows[40:]):.1%} | |")
        top = rows[0]
        if top[4] > 0.4:
            risks.append(f"单个特征 `{top[0]}` 贡献占比 {top[4]:.0%} > 40%（sanity_rules）。")
        inc = [(f, sh) for f, _, _, _, sh, _ in rows if f in incumbent]
        if inc:
            L += ["", f"**在位模型/外部评分字段的合计贡献**（know-how 约定默认保留，单独列出）：{sum(s_ for _, s_ in inc):.1%}（" + "、".join(f"{f} {s_:.1%}" for f, s_ in inc) + "）"]
        L.append("")
    else:
        L += ["## 3. 最终模型", "", "没有通过 evaluator 的全保真度实验，未运行 holdout。", ""]
        risks.append("没有 ACCEPT 的全保真度实验，本次运行无最终模型。")

    tree = mermaid_tree(recs)
    (rd / f"experiment_tree_{task}.mmd").write_text(tree)
    L += ["## 7. 实验树", "", "```mermaid", tree, "```", "", "| 实验 | 父节点 | 动作 | 保真度 | OOT-dev AUC | 判定 | 诊断码 |", "|---|---|---|---|---|---|---|"]
    for r in recs:
        a = r.metrics.get("oot_dev_auc")
        L.append(f"| {r.exp_id}{' (失效)' if r.invalidated else ''} | {r.parent_exp_id or '-'} | {r.action_type} | {r.fidelity} | {f'{a:.4f}' if a is not None else '-'} | {r.verdict} | {','.join(r.diagnosis_codes)} |")
    L.append("")

    if p["quarantined"]:
        risks.append(f"被隔离的疑似泄漏特征（{len(p['quarantined'])} 个）：" + "、".join(f"`{f}`" for f in p["quarantined"][:15]) + (" …" if len(p["quarantined"]) > 15 else ""))
    inv = [r.exp_id for r in recs if r.invalidated]
    if inv:
        risks.append(f"因约束变更而失效的实验：{', '.join(inv)}。")
    prof = load_knowhow(ds, cfg)
    review = (normalize(prof, prof["dictionary"].get("label", {}).get("source_table"))[1].get("compliance", {}) or {}).get("review", [])
    if best and set(review) & set(best["config"]["features"]):
        risks.append("合规待审字段在最终特征中：" + "、".join(sorted(set(review) & set(best["config"]["features"]))) + "。")
    if p["stop_reason"]:
        risks.append(f"停止原因：{p['stop_reason']}。")
    L += ["## 8. 风险提示", ""] + [f"- {r}" for r in risks or ["无"]] + [""]

    path = rd / f"model_card_{task}.md"
    path.write_text("\n".join(L))
    return {"model_card": str(path), "tree": str(rd / f"experiment_tree_{task}.mmd"), "risks": risks, "narrative": narrative}
