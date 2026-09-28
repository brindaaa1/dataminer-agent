"""运行 B0 / B1，并与 agent 的最优实验对比。
python -m eval.run_baselines --dataset lending_club --agent-exp lc_mock2_e03 --agent-task lc_mock2
python -m eval.run_baselines --dataset home_credit --agent-exp hcfull_factory --agent-task hcfull_factory --factory-features"""
import argparse
import json
import sqlite3

import numpy as np

from baselines import b0_default_lgbm, b1_automl
from data.access import DataAccess
from data.knowhow import load_config
from evaluation.compare import paired_bootstrap
from features.registry import Registry

TEMPLATES = ["hc_request_profile", "hc_prev_refusal", "hc_bureau_overdue", "hc_installment_behavior"]
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
ap.add_argument("--agent-exp", required=True, help="agent 最优实验 id（读取其 OOT-dev 预测与耗时）")
ap.add_argument("--agent-task", required=True, help="agent 的 task_id（读取 final_gate 里的 holdout 成绩）")
ap.add_argument("--extended-sec", type=float, default=300, help="额外的大预算 B1（看 AutoML 上限）；0 表示不跑")
ap.add_argument("--factory-features", action="store_true", help="额外跑：B0/B1 + 特征工厂特征（分离『特征』与『搜索』的贡献）")
ap.add_argument("--tag", default="")
a = ap.parse_args()

cfg = load_config()
root = cfg["paths"]["artifacts_root"]
ag = json.load(open(f"{root}/experiments/{a.agent_exp}/result.json"))
ag_wall = ag["cost"]["wall_minutes"] * 60
try:                                                    # 优先用整个任务的训练耗时（含赛跑）
    s = json.load(open(f"{cfg['paths']['reports_root']}/summary_{a.agent_task}.json"))
    ag_wall = s["budget"]["used"]["wall_minutes"] * 60
except FileNotFoundError:
    pass
ag_hold = json.loads(sqlite3.connect(f"{root}/state.db").execute("SELECT result FROM final_gate WHERE task_id=?", (a.agent_task,)).fetchone()[0])
y = DataAccess(a.dataset, root).load("oot_dev")["_label"].values
p_ag = np.load(f"{root}/experiments/{a.agent_exp}/oot_dev_pred.npy")
budget = max(ag_wall, 30.0)

rows = {"agent": {"oot_dev_auc": ag["metrics"]["oot_dev_auc"], "holdout_auc": ag_hold["holdout_auc"], "wall_sec": ag_wall,
                  "n_features": ag["n_features"], "model": ag["config"]["model"]}}
rows["B0 默认 LightGBM"] = b0_default_lgbm.run(a.dataset, cfg, name=f"b0_{a.dataset}{a.tag}")
rows[f"B1 FLAML（对齐算力 {budget:.0f}s）"] = b1_automl.run(a.dataset, cfg, budget, name=f"b1_{a.dataset}{a.tag}")
if a.extended_sec:
    rows[f"B1 FLAML（扩大预算 {a.extended_sec:.0f}s）"] = b1_automl.run(a.dataset, cfg, a.extended_sec, name=f"b1x_{a.dataset}{a.tag}")
if a.factory_features:
    fs = [Registry.fs_id(a.dataset, t, root) for t in TEMPLATES]
    rows["B0 + 工厂特征"] = b0_default_lgbm.run(a.dataset, cfg, fs, name=f"b0f_{a.dataset}{a.tag}")
    rows[f"B1 + 工厂特征（对齐算力 {budget:.0f}s）"] = b1_automl.run(a.dataset, cfg, budget, fs, name=f"b1f_{a.dataset}{a.tag}")

cmp = {}
for k, r in rows.items():
    if k == "agent":
        continue
    p = np.load(f"{root}/experiments/{r['name']}/oot_dev_pred.npy")
    b = paired_bootstrap(y, p_ag, p, 0.05, cfg["evaluator"]["bootstrap_n"])
    cmp[k] = {"agent_minus_baseline_oot_dev": b["delta"], "ci": [b["ci_low"], b["ci_high"]], "agent_significantly_better": b["significant"],
              "agent_minus_baseline_holdout": ag_hold["holdout_auc"] - r["holdout_auc"]}
out = {"dataset": a.dataset, "rows": rows, "agent_vs_baseline": cmp}
json.dump(out, open(f"{cfg['paths']['reports_root']}/baselines_{a.dataset}{a.tag}.json", "w"), ensure_ascii=False, indent=1, default=float)
print(f"\n| 方法 | OOT-dev AUC | holdout AUC | 耗时(s) | 特征数 | agent−基线 OOT-dev [95%CI] |\n|---|---|---|---|---|---|")
for k, r in rows.items():
    c = cmp.get(k)
    print(f"| {k} | {r['oot_dev_auc']:.4f} | {r['holdout_auc']:.4f} | {r['wall_sec']:.0f} | {r.get('n_features','')} | "
          + (f"{c['agent_minus_baseline_oot_dev']:+.4f} [{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}]" if c else "-") + " |")
