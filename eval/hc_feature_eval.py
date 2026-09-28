"""第 6 项评估：Home Credit 上，同样的 lgbm + 同样的 Optuna 预算，比较
  基线：只用申请表原始字段（去掉特征工厂 = 消融 A1）
  工厂：基线 + 4 个模板产出、经筛选漏斗保留的特征
指标：OOT-dev AUC（配对 bootstrap 95% CI）与 holdout AUC（final_gate 只读一次）。
注意：Home Credit 没有绝对时间，OOT/holdout 都是随机切分近似（报告须注明）。
用法：python -m eval.hc_feature_eval --dev（3 万样本）| 不带参数（全量）"""
import argparse
import copy
import json

import numpy as np

from data.access import DataAccess
from data.knowhow import load_config
from data.splits import build_splits
from evaluation.compare import paired_bootstrap
from evaluation.final_gate import FinalGate
from modeling.inner_loop import run_study
from tools.generate_features import generate_features

DS = "home_credit"
TEMPLATES = ["hc_request_profile", "hc_prev_refusal", "hc_bureau_overdue", "hc_installment_behavior"]
ap = argparse.ArgumentParser()
ap.add_argument("--dev", action="store_true")
ap.add_argument("--trials", type=int, default=20)
ap.add_argument("--per-template", action="store_true", help="额外跑：基线 + 单个模板")
a = ap.parse_args()

cfg = load_config()
cfg["fidelity"]["full"]["n_trials"] = a.trials
tag = "hcdev" if a.dev else "hcfull"
if a.dev:
    cfg["paths"]["artifacts_root"] = "artifacts_hcdev"
    build_splits(DS, cfg, sample_n=cfg["task_specs"][DS]["dev_sample_ids"])
root = cfg["paths"]["artifacts_root"]
tr = DataAccess(DS, root).load("train")
base_feats = [c for c in tr.columns if not c.startswith("_")]
print("基线特征数:", len(base_feats), flush=True)

fsets, gen = [], {}
for t in TEMPLATES:
    g = generate_features(t, DS, cfg)
    gen[t] = {"kept": len(g["kept"]), "funnel": g["funnel"], "leak_suspect": g["leak_suspect"]}
    print(t, g["funnel"], "leak:", g["leak_suspect"], flush=True)
    fsets.append(g["feature_set_id"])
from features.registry import Registry
reg = Registry(cfg)
kept_all = [f for fs in fsets for f in reg.get(fs)["kept"]]


def run(name, feats, sets):
    r = run_study(f"{tag}_{name}", DS, "lgbm", cfg, features=feats, fidelity="full", feature_sets=sets, seed=0)
    fg = FinalGate(DS, cfg, f"{tag}_{name}").run_once(r)
    print(name, "OOT-dev", round(r["metrics"]["oot_dev_auc"], 4), "holdout", round(fg["holdout_auc"], 4), flush=True)
    return r, fg


rb, fb = run("baseline", base_feats, [])
rf, ff = run("factory", base_feats + kept_all, fsets)
y = DataAccess(DS, root).load("oot_dev")["_label"].values
p_b = np.load(f"{root}/experiments/{tag}_baseline/oot_dev_pred.npy")
p_f = np.load(f"{root}/experiments/{tag}_factory/oot_dev_pred.npy")
cmp = paired_bootstrap(y, p_f, p_b, 0.05, cfg["evaluator"]["bootstrap_n"])
res = {"n_base_features": len(base_feats), "n_factory_features_kept": len(kept_all), "templates": gen,
       "baseline": {"oot_dev_auc": rb["metrics"]["oot_dev_auc"], "holdout_auc": fb["holdout_auc"]},
       "factory": {"oot_dev_auc": rf["metrics"]["oot_dev_auc"], "holdout_auc": ff["holdout_auc"]},
       "oot_dev_delta": cmp, "holdout_delta": ff["holdout_auc"] - fb["holdout_auc"], "trials": a.trials,
       "note": "随机切分近似 OOT；EXT_SOURCE 默认保留（know-how 约定）"}
if a.per_template:
    res["per_template"] = {}
    for t, fs in zip(TEMPLATES, fsets):
        r, f = run(f"only_{t}", base_feats + reg.get(fs)["kept"], [fs])
        res["per_template"][t] = {"n_kept": len(reg.get(fs)["kept"]), "oot_dev_auc": r["metrics"]["oot_dev_auc"], "holdout_auc": f["holdout_auc"]}
json.dump(res, open(f"reports/hc_feature_results_{tag}.json", "w"), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: res[k] for k in ("baseline", "factory", "oot_dev_delta", "holdout_delta")}, indent=1, default=float))
