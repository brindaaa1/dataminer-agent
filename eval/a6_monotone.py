"""A6：单调约束 开 vs 关（Lending Club，同样的 lgbm、特征、Optuna 预算、种子）。
比较：OOT-dev/holdout AUC、分数 PSI、分月 AUC 波动、与先验方向相反的特征数（反直觉）；
并用 evaluator 的非劣效规则判定"是否保留约束"。python -m eval.a6_monotone [--trials 15]"""
import argparse
import json

import numpy as np

from data.access import DataAccess
from data.knowhow import load_config
from evaluation.compare import monthly_auc
from evaluation.evaluator import Evaluator
from evaluation.final_gate import FinalGate
from modeling.explain import explain
from modeling.inner_loop import run_study
from modeling.monotone import priors_for

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="lending_club")
ap.add_argument("--trials", type=int, default=15)
a = ap.parse_args()
DS, cfg = a.dataset, load_config()
cfg["fidelity"]["full"]["n_trials"] = a.trials
da = DataAccess(DS, cfg["paths"]["artifacts_root"])
tr, oo = da.load("train"), da.load("oot_dev")
feats = [c for c in tr.columns if not c.startswith("_")]

res, fg, viol = {}, {}, {}
for arm, mono in (("free", False), ("monotone", True)):
    r = run_study(f"a6_{arm}", DS, "lgbm", cfg, features=feats, fidelity="full", monotone=mono, seed=0)
    res[arm] = r
    fg[arm] = FinalGate(DS, cfg, f"a6_{arm}").run_once(r)
    pri = priors_for(DS, cfg, r["config"])
    ex = explain(DS, cfg, r)
    viol[arm] = sorted(f for f, e in ex.items() if pri.get(f, 0) != 0 and abs(e["corr"]) > 0.1 and np.sign(e["corr"]) != np.sign(pri[f]))
    p = np.load(f"{cfg['paths']['artifacts_root']}/experiments/a6_{arm}/oot_dev_pred.npy")
    tcol = "_split_time" if "_split_time" in oo else "_obs_time"
    mo = monthly_auc(oo[tcol], oo["_label"], p, min_n=50)
    res[arm]["monthly_auc_std"] = float(np.std([v["auc"] for v in mo.values()]))
    print(arm, round(r["metrics"]["oot_dev_auc"], 4), round(fg[arm]["holdout_auc"], 4), flush=True)

ev = Evaluator(cfg, DS, "a6", "artifacts/a6_state.db")
out = ev.evaluate(res["monotone"], res["free"])
rep = res["monotone"]["monotone_report"]
row = lambda k: (f"{res[k]['metrics']['oot_dev_auc']:.4f} | {fg[k]['holdout_auc']:.4f} | {res[k]['metrics']['score_psi']:.4f} | "
                 f"{res[k]['monthly_auc_std']:.4f} | {len(viol[k])}")
md = ["# A6：单调约束 开 vs 关（%s，lgbm，%d trial）" % (DS, a.trials), "",
      "| 配置 | OOT-dev AUC | holdout AUC | 分数 PSI | 分月 AUC 标准差 | 与先验方向相反的特征数 |", "|---|---|---|---|---|---|",
      f"| 不加约束 | {row('free')} |", f"| 加约束 | {row('monotone')} |", "",
      f"- 加约束 − 不加约束（OOT-dev，配对 bootstrap）：Δ={out['compare']['delta']:+.4f}，95% CI [{out['compare']['ci_low']:+.4f}, {out['compare']['ci_high']:+.4f}]",
      f"- evaluator 判定：**{out['verdict'].value}**" + ("（非劣效规则：AUC 未显著变差且分数 PSI 更优 → 保留约束）" if out.get("monotone_noninferior") else ""),
      f"- 与先验相反的特征（不加约束）：{viol['free'] or '无'}；（加约束）：{viol['monotone'] or '无'}", "",
      "## 约束的实际覆盖", "",
      f"- 已约束（{len(rep['constrained'])}）：{rep['constrained']}",
      f"- 先验为 0，不约束（{len(rep['zero_prior'])}）：{rep['zero_prior']}",
      f"- 先验与数据相反，跳过（{len(rep['disagree'])}）：{rep['disagree']}  ← 数据一致性守卫",
      f"- 没有先验，不约束：{len(rep['no_prior'])} 个；类别变量：{len(rep['categorical'])} 个", ""]
open("reports/a6_monotone.md", "w").write("\n".join(md))
json.dump({"free": {**res["free"]["metrics"], "holdout": fg["free"]["holdout_auc"], "violations": viol["free"]},
           "monotone": {**res["monotone"]["metrics"], "holdout": fg["monotone"]["holdout_auc"], "violations": viol["monotone"]},
           "compare": out["compare"], "verdict": out["verdict"].value, "report": rep}, open("reports/a6_monotone.json", "w"), indent=1, default=float)
print("\n".join(md))
