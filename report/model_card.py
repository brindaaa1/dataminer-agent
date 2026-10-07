"""最终交付物（§4.18）：模型卡 + 特征清单 + Mermaid 实验树 + 风险提示。
数据全部由代码填充；LLM 只写一段执行摘要，且摘要中的数字必须能在事实数据里找到，否则丢弃。"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from data.access import DataAccess
from data.knowhow import field_table, load_knowhow, normalize, target_direction
from evaluation.compare import monthly_auc
from evaluation.metrics import METRICS, primary
from evaluation.guardrails import psi
from features.registry import Registry
from features.screen import iv
from modeling.explain import explain
from modeling.zoo import ZOO

NARRATIVE_SYSTEM = ("你是报告撰写节点。根据给定的 facts（JSON）给业务人员写 3–5 句中文执行摘要：预测的是什么、最终用了什么模型、"
                    "效果怎样（时间外验证和最终检验）、有什么要注意的。用平实的业务语言，不要出现字段名、实验 id、诊断码和缩写"
                    "（facts 里的 oot_dev 叫「时间外验证」，holdout 叫「最终检验」）。只能使用 facts 中出现的数字，不要自己计算或推断新数字。只输出摘要文字。")


def _relative(r):
    return f"相对{r['relative_to']}（{'、'.join(r['offset_cols'])}）" if r else None


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


def mermaid_tree(records, pm: str) -> str:
    """节点：动作 / 保真度 / OOT-dev 主指标（相对父节点的变化）/ 判定；失效节点虚线。"""
    by = {r.exp_id: r for r in records}
    sid = lambda e: "n_" + re.sub(r"\W", "_", e)
    lines = ["flowchart TD", "  ROOT((start))"]
    for r in records:
        k = f"oot_dev_{pm}"
        v = r.metrics.get(k)
        par = by.get(r.parent_exp_id)
        delta = ""
        if v is not None and par is not None and par.metrics.get(k) is not None:
            delta = f" ({v - par.metrics[k]:+.4f})"
        body = f"{r.exp_id}<br/>{r.action_type} [{r.fidelity}]<br/>" + (f"{pm.upper()} {v:.4f}{delta}<br/>" if v is not None else "") + r.verdict
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
    pri = target_direction(dataset, cfg)                                    # 新数据集可以没有先验
    return meta, pri, set(b.get("incumbent_model", {}).get("fields", [])) if b.get("incumbent_model", {}).get("action") != "exclude_by_default" else set()


def _narrative(llm, facts: dict) -> str | None:
    if llm is None:
        return None
    try:
        text, _ = llm.complete(NARRATIVE_SYSTEM, "```json\n" + json.dumps({"mode": "REPORT", "facts": facts}, ensure_ascii=False) + "\n```", 0.0)
        return safe_narrative(text, facts)
    except Exception:                       # 摘要只是锦上添花：LLM 出错时模型卡照常生成
        return None


def _spec_section(s) -> list[str]:
    return ["## 1. 任务规格", "", "| 项 | 值 |", "|---|---|", f"| label 定义 | {s.label_def} |",
            f"| 观察时间 | {s.observation_time_col or _relative(s.observation_time_relative)} |",
            f"| OOT 窗口 | {json.dumps(s.oot_windows, ensure_ascii=False)} |", f"| 目标指标 | {s.metric.primary}（护栏 {s.metric.guards}，目标值 {s.target_value}） |",
            f"| 约束 | {s.constraints.model_dump()} |", f"| 预算 | {s.budget.model_dump()} |", f"| 自治级别 | {s.autonomy} |", ""]


def _split_section(tr, va, oo, final) -> list[str]:
    L = ["## 2. 数据切分", "", "| 切分 | 样本数 | 正类比例 |", "|---|---|---|"]
    L += [f"| {nm} | {len(df):,} | {df['_label'].mean():.4f} |" for nm, df in (("train", tr), ("valid", va), ("oot_dev", oo))]
    if final:
        L.append(f"| holdout（仅 final_gate） | {final['n_holdout']:,} | {final['holdout_bad_rate']:.4f} |")
    if "_obs_time" not in oo:
        L += ["", "> ⚠ 该数据集没有绝对时间：OOT-dev 与 holdout 都是**随机切分近似**，无法检验时间稳定性，PSI 类指标仅作参考。"]
    return L + [""]


def _model_section(cfg, p, best, best_id, final, oot_used, pm, risks) -> list[str]:
    m = best["metrics"]
    L = ["## 3. 最终模型", "", f"- 实验：`{best_id}`；模型：`{best['config']['model']}`；保真度：{best['config']['fidelity']}；trial 数：{best['n_trials']}（剪枝 {best['n_pruned']}）",
         f"- 特征数：{best['n_features']}；最优参数：`{json.dumps(best['best_params'], ensure_ascii=False)}`"]
    tie = p.get("race_tie")
    if tie and len(tie["tied"]) > 1:
        L.append(f"- 赛跑打平：{'、'.join(tie['tied'])} 在 OOT-dev 上与最优差距小于可分辨的最小提升（MDE {tie['mde']:.4f}），效果相当；"
                 f"按偏好顺序（config race.tie_preference）推荐 {tie['recommended']}。")
    names = [pm] + [k for k in METRICS if k != pm]                    # 主指标放第一列并加粗
    bold = lambda k, t: f"**{t}**" if k == pm else t
    row = lambda label, src, pre: f"| {label} | " + " | ".join(bold(k, f"{src[f'{pre}_{k}']:.4f}") for k in names) + " |"
    L += ["", "## 4. 各段指标", "", "| 段 | " + " | ".join(bold(k, k.upper()) for k in names) + " |", "|---|" + "---|" * len(names),
          row("train", m, "train"), row("valid", m, "valid"), row("OOT-dev", m, "oot_dev")]
    if final:
        L.append(row("holdout", final, "holdout"))
    L += ["", f"- valid − OOT-dev gap：{m['gap']:.4f}；分数 PSI（valid→OOT-dev）：{m.get('score_psi', float('nan')):.4f}；OOT-dev 使用次数：{oot_used}"]
    if final:
        L.append(f"- 补充参考指标（holdout，只报告、不参与选模型）：分数最高的头部 10% 里正类浓度是整体的 {final['holdout_lift_top10']:.2f} 倍，"
                 f"覆盖 {final['holdout_recall_top10']:.0%} 的正类；Brier 分数 {final['holdout_brier']:.4f}（越小说明预测概率越接近实际）")
        if final["overfit_to_oot_dev"]:
            ref = final.get("reference_drop")
            risks.append(f"holdout {pm.upper()} 比 OOT-dev 低 {final['drop']:.4f}"
                         + (f"（默认参照模型自己低 {ref:.4f}），比默认参照模型多掉 {final['drop'] - ref:.4f}" if ref is not None else "")
                         + f"，超过 δ={cfg['final_gate']['delta']}：**疑似对 OOT-dev 过拟合**。")
    return L + [""]


def _monthly_section(cfg, ds, art, best_id, tr, va, oo, final, risks) -> list[str]:
    oot_pred = np.load(art / "experiments" / best_id / "oot_dev_pred.npy")
    val_pred = np.load(art / "experiments" / best_id / "valid_pred.npy")
    tcol = "_split_time" if "_split_time" in oo else "_obs_time"       # 与切分一致（如到店日）
    mo = monthly_auc(oo[tcol], oo["_label"], oot_pred, min_n=50)
    hist = pd.concat([tr, va])                                          # 去年同期正类比例：从训练期（train+valid）取 12 个月前的同一月份
    hbr = hist.groupby(pd.to_datetime(hist[tcol]).dt.strftime("%Y-%m"))["_label"].mean().to_dict()
    prev = lambda k: (pd.Period(k) - 12).strftime("%Y-%m")
    yoy = lambda k: f"{hbr[prev(k)]:.3f}" if prev(k) in hbr else "-"
    months = pd.to_datetime(oo[tcol]).dt.strftime("%Y-%m")
    hold = (final or {}).get("holdout_monthly_auc", {})
    L = ["## 5. 分月 AUC、正类比例与分数 PSI", "", f"（月份按 `{tcol}` 划分）", "",
         "| 月份 | 段 | n | AUC | 正类比例 | 去年同期正类比例 | 分数 PSI（相对 valid） |", "|---|---|---|---|---|---|---|"]
    L += [f"| {k} | OOT-dev | {v['n']} | {v['auc']:.4f} | {v['bad_rate']:.3f} | {yoy(k)} | {psi(val_pred, oot_pred[(months == k).values], cfg['evaluator']['psi_bins']):.4f} |"
          for k, v in mo.items()]
    L += [f"| {k} | holdout | {v['n']} | {v['auc']:.4f} | {v.get('bad_rate', float('nan')):.3f} | {yoy(k)} | |" for k, v in hold.items()]
    aucs = [v["auc"] for v in list(mo.values()) + list(hold.values())]
    if aucs and max(aucs) - min(aucs) > 0.05:
        risks.append(f"分月 AUC 极差 {max(aucs) - min(aucs):.3f} > 0.05（sanity_rules）。注意月度样本量差异较大。")
    cav = load_knowhow(ds, cfg)["dictionary"].get("suggested_split", {}).get("caveat")
    return L + (([""] + [f"> {ln.strip()}" for ln in str(cav).strip().splitlines()]) if cav else []) + [""]


def _features_section(cfg, ds, best, tr, risks) -> list[str]:
    try:
        ex, ex_err = explain(ds, cfg, best), None
    except Exception as e:                  # 最终检验已出结果：这一节算不出来只在模型卡里注明，不能让整次运行失败
        ex, ex_err = {}, f"{type(e).__name__}: {str(e)[:200]}"
        risks.append(f"特征贡献计算失败（{ex_err}），特征清单与 SHAP 方向检查缺失。")
    meta, prior, incumbent = _feature_meta(ds, cfg, best["config"])
    trf = Registry(cfg).attach(tr.copy(), best["config"]["feature_sets"], "train") if best["config"].get("feature_sets") else tr
    rows = []
    for f, e in sorted(ex.items(), key=lambda x: -x[1]["share"]):
        mt = meta.get(f, {"source": "原始字段", "definition": "", "prior": None})
        pr = prior.get(f, mt["prior"]) if prior.get(f, None) is not None or mt["prior"] is None else mt["prior"]
        flag = ""
        if pr not in (None, 0) and abs(e["corr"]) > 0.1 and np.sign(e["corr"]) != np.sign(pr) and e["share"] >= cfg["report"]["sanity_min_share"]:
            flag = "⚠ 反直觉"
            risks.append(f"特征 `{f}` 的 SHAP 方向与先验目标方向相反（{flag}），交人工判断。")
        rows.append((f, mt["source"], mt["definition"][:70], iv(trf[f], trf["_label"]) if f in trf else float("nan"), e["share"], flag))
    L = ["## 6. 特征清单（按平均 |SHAP| 排序）", "", "| 特征 | 来源 | 定义/说明 | IV | 贡献占比 | 备注 |", "|---|---|---|---|---|---|"]
    if ex_err:
        L.append(f"| 特征贡献计算失败：{ex_err} | | | | | |")
    L += [f"| {f} | {src} | {dfn} | {v_iv:.3f} | {sh:.1%} | {flag} |" for f, src, dfn, v_iv, sh, flag in rows[:40]]
    if len(rows) > 40:
        L.append(f"| …其余 {len(rows) - 40} 个 | | | | {sum(r[4] for r in rows[40:]):.1%} | |")
    if rows and rows[0][4] > 0.4:
        risks.append(f"单个特征 `{rows[0][0]}` 贡献占比 {rows[0][4]:.0%} > 40%（sanity_rules）。")
    inc = [(f, sh) for f, _, _, _, sh, _ in rows if f in incumbent]
    if inc:
        L += ["", f"**在位模型/外部评分字段的合计贡献**（know-how 约定默认保留，单独列出）：{sum(s_ for _, s_ in inc):.1%}（" + "、".join(f"{f} {s_:.1%}" for f, s_ in inc) + "）"]
    return L + [""]


def setup_section(cfg, spec_, best, pm) -> list[str]:
    """最终模型的设定：训练方式、参数取值和含义、特征清单。说明由代码按模板生成，用来复现。"""
    c, spec = best["config"], ZOO[best["config"]["model"]]
    train = spec.train_doc.format(es=cfg["fidelity"]["full"]["early_stopping_rounds"], max_trees=cfg["inner_loop"]["n_rounds_max"], best_iter=best["best_iter"])
    num = lambda v: f"{v:.6g}" if isinstance(v, float) else str(v)
    w = spec_.oot_windows
    split = (f"train 和 valid 来自同一时期，valid 是从中随机抽出的 {cfg['split']['valid_frac']:.0%}（种子 {cfg['split']['seed']}）；"
             f"时间外验证 {w['oot_dev'][0]} ~ {w['oot_dev'][1]}，最终检验 {w['holdout'][0]} ~ {w['holdout'][1]}"
             if isinstance(w["oot_dev"], list) else f"没有时间列，四段按比例分层随机切分（种子 {cfg['split']['seed']}）")
    L = ["## 9. 模型设定（复现用）", "", "用下面的设定，在同样的切分数据上可以重新训练出最终模型。", "",
         f"- 模型：{c['model']}。{spec.profile}",
         f"- 训练：{train}；随机种子 {c['seed']}。",
         f"- 训练数据：train 段 {best['n_rows']:,} 行。{split}。",
         f"- 参数怎么来的：Optuna（TPE）在 {best['n_trials']} 组参数里，选 valid 段 {pm.upper()} 最高的一组。",
         "", "| 参数 | 取值 | 含义 |", "|---|---|---|"]
    L += [f"| {k} | {num(v)} | {spec.param_doc.get(k, '')} |" for k, v in best["best_params"].items()]
    L += ["", f"- 特征（{best['n_features']} 个，按输入顺序）：" + "、".join(f"`{f}`" for f in c["features"]),
          "- 派生特征的定义见特征清单。" + ("带单调约束训练。" if c.get("monotone") else ""), ""]
    return L


def _tree_section(recs, pm, tree) -> list[str]:
    L = ["## 7. 实验树", "", "```mermaid", tree, "```", "", f"| 实验 | 父节点 | 动作 | 保真度 | OOT-dev {pm.upper()} | 判定 | 诊断码 |", "|---|---|---|---|---|---|---|"]
    for r in recs:
        a = r.metrics.get(f"oot_dev_{pm}")
        L.append(f"| {r.exp_id}{' (失效)' if r.invalidated else ''} | {r.parent_exp_id or '-'} | {r.action_type} | {r.fidelity} | {f'{a:.4f}' if a is not None else '-'} | {r.verdict} | {','.join(r.diagnosis_codes)} |")
    return L + [""]


def _run_risks(cfg, ds, p, recs, best, risks):
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


def build_report(cfg: dict, spec, store, p: dict, final: dict | None, oot_used: int, llm=None) -> dict:
    task, ds, pm = p["task_id"], spec.dataset, primary(cfg)
    art, rd = Path(cfg["paths"]["artifacts_root"]), Path(cfg["paths"]["reports_root"])
    rd.mkdir(exist_ok=True)
    recs, best_id = store.all(task), p["best_exp_id"]
    da = DataAccess(ds, str(art))
    tr, va, oo = da.load("train"), da.load("valid"), da.load("oot_dev")
    best = json.loads((art / "experiments" / best_id / "result.json").read_text()) if best_id else None
    facts = {"dataset": ds, "best_exp": best_id, "n_experiments": len(recs), "n_accept": sum(r.verdict == "ACCEPT" for r in recs),
             "oot_used": oot_used, "stop_reason": p["stop_reason"]}
    if best:
        facts.update({"model": best["config"]["model"], "metric": pm, f"oot_dev_{pm}": round(best["metrics"][f"oot_dev_{pm}"], 4)})
    if final:
        facts[f"holdout_{pm}"] = round(final[f"holdout_{pm}"], 4)
    narrative = _narrative(llm, facts)

    risks = []
    L = [f"# 模型卡：{task}", "", f"> 数据集 `{ds}`；本文档数据全部由代码生成。", "",
         "## 执行摘要", "", narrative or "（未生成摘要，或摘要中的数字无法在事实数据中核对，已丢弃。）", ""]
    L += _spec_section(spec) + _split_section(tr, va, oo, final)
    if best:
        L += _model_section(cfg, p, best, best_id, final, oot_used, pm, risks)
        if "_obs_time" in oo:
            L += _monthly_section(cfg, ds, art, best_id, tr, va, oo, final, risks)
        L += _features_section(cfg, ds, best, tr, risks)
    else:
        L += ["## 3. 最终模型", "", "没有通过 evaluator 的全保真度实验，未运行 holdout。", ""]
        risks.append("没有 ACCEPT 的全保真度实验，本次运行无最终模型。")
    tree = mermaid_tree(recs, pm)
    (rd / f"experiment_tree_{task}.mmd").write_text(tree)
    L += _tree_section(recs, pm, tree)
    _run_risks(cfg, ds, p, recs, best, risks)
    L += ["## 8. 风险提示", ""] + [f"- {r}" for r in risks or ["无"]] + [""]
    if best:
        L += setup_section(cfg, spec, best, pm)

    path = rd / f"model_card_{task}.md"
    path.write_text("\n".join(L))
    return {"model_card": str(path), "tree": str(rd / f"experiment_tree_{task}.mmd"), "risks": risks, "narrative": narrative}
