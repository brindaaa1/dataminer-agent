"""A3 演示：把贷后泄漏字段放回数据（不告诉 agent），对比 有/无 guardrail 时的 OOT-dev AUC 虚高与泄漏检出。
无 guardrail = 关闭泄漏扫描、gap/PSI/特征数拦截，只保留 compare（cfg.ablation.no_guardrail）。
额外评估「上线视角」：这些字段在申请时不可得，把它们置空后重新对 holdout 打分。"""
import argparse
import copy
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agent.llm import PolicyMockLLM
from agent.orchestrator import Orchestrator
from agent.schemas import TaskSpec
from data.access import DataAccess
from data.knowhow import load_config, load_knowhow
from data.splits import build_splits
from evaluation.evaluator import scan_features
from evaluation.final_gate import FinalGate

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="lending_club")
ap.add_argument("--clean-task", default="lc_mock1", help="干净数据上的 agent 运行（作为基准柱）")
ap.add_argument("--include-suspects", action="store_true", help="可得时间存疑的字段也放回，并单独统计识别率（酒店 blacklist.eval_ground_truth: suspected）")
a = ap.parse_args()
DS = a.dataset
suffix = "" if DS == "lending_club" else f"_{DS}"
cfg = load_config()
ROOT = f"artifacts_a3{suffix}"
cfg["paths"]["artifacts_root"] = ROOT
cfg["fidelity"]["full"]["n_trials"] = 15
cfg["run"]["max_rounds"] = 5
out = Path("reports")

if not Path(ROOT, "splits", DS, "train.parquet").exists():
    rep = build_splits(DS, cfg, leak_back=True, suspects_back=a.include_suspects)
    json.dump({"leak": rep["policy"]["leak_truth"], "suspect": rep["policy"]["suspect_truth"] if a.include_suspects else []},
              open(f"{ROOT}/truth.json", "w"))
tj = json.load(open(f"{ROOT}/truth.json")) if Path(f"{ROOT}/truth.json").exists() else {"leak": json.load(open(f"{ROOT}/leak_truth.json")), "suspect": []}
tr = DataAccess(DS, ROOT).load("train")
present = [c for c in tj["leak"] if c in tr.columns]
present_suspect = [c for c in tj["suspect"] if c in tr.columns]
print(f"放回 {len(present)} 个泄漏字段，{len(present_suspect)} 个可得时间存疑字段")


class MaskedGate:                       # 评估专用包装：额外给出上线视角 AUC
    def __init__(self, fg): self.fg = fg
    def run_once(self, best): return self.fg.run_once(best, mask_fields=present + present_suspect)


results = {}
for arm, no_guard in (("guardrail_on", False), ("guardrail_off", True)):
    c = copy.deepcopy(cfg)
    c["ablation"] = {"no_guardrail": no_guard}
    tid = f"a3{suffix}_{arm}"
    spec = TaskSpec.from_knowhow(DS, c, autonomy="L0")
    o = Orchestrator(spec, c, PolicyMockLLM(), MaskedGate(FinalGate(DS, c, tid)), tid)
    print(arm, "->", o.run().value, flush=True)
    best = o._result(o.p["best_exp_id"])
    feats = set(best["config"]["features"])
    fg = o.p["final"]
    results[arm] = {"best_exp": o.p["best_exp_id"], "oot_dev_auc": best["metrics"]["oot_dev_auc"],
                    "holdout_auc": fg["holdout_auc"], "holdout_auc_deploy_view": fg["holdout_auc_masked"],
                    "leak_features_in_final_model": sorted(feats & set(present)), "suspect_features_in_final_model": sorted(feats & set(present_suspect)),
                    "quarantined": o.p["quarantined"], "stop_reason": o.p["stop_reason"]}

# ---- 泄漏检出：整体 + 分规则 ----
kh = load_knowhow(DS, cfg)
scan = scan_features(tr, [c for c in tr.columns if not c.startswith("_")], kh, cfg, DS)
thr = cfg["evaluator"]["leak_single_auc"]
rules = {"single_feature_auc>0.75（纯统计）": {f for f, a in scan["feature_aucs"].items() if a > thr},
         "命名模式（leakage_patterns）": set(scan["leak_pattern_hits"]),
         "可得时间标注（data_dictionary）": set(scan["unavailable_fields"])}
allflag = set().union(*rules.values())
tset = set(present)
sset = set(present_suspect)
ok = tset | sset                                     # 存疑字段被标为可疑是 know-how 认可的行为，不算误报
detect = {k: {"recall": len(v & tset) / len(tset), "precision": len(v & ok) / max(len(v), 1), "n_flagged": len(v)} for k, v in rules.items()}
detect["三条规则合并（guardrail 实际行为）"] = {"recall": len(allflag & tset) / len(tset), "precision": len(allflag & ok) / max(len(allflag), 1), "n_flagged": len(allflag)}
q = set(results["guardrail_on"]["quarantined"])
detect["agent 实际隔离（guardrail_on 运行结果）"] = {"recall": len(q & tset) / len(tset), "precision": len(q & ok) / max(len(q), 1), "n_flagged": len(q)}
missed = sorted(tset - allflag)
if sset:                                                     # 存疑字段：agent 是否识别为可疑（不算"确定泄漏"，单独统计）
    detect["【存疑字段】agent 实际隔离"] = {"recall": len(q & sset) / len(sset), "precision": 1.0, "n_flagged": len(q & sset)}
    detect["【存疑字段】可得时间规则"] = {"recall": len(set(scan["unavailable_fields"]) & sset) / len(sset), "precision": 1.0, "n_flagged": len(set(scan["unavailable_fields"]) & sset)}
top_auc = dict(sorted(((f, round(scan["feature_aucs"][f], 3)) for f in present if f in scan["feature_aucs"]), key=lambda x: -x[1])[:8])

clean = json.load(open(f"reports/summary_{a.clean_task}.json"))
ce = next(e for e in clean["experiments"] if e["exp_id"] == clean["best_exp_id"])
results["clean_reference"] = {"oot_dev_auc": ce["oot_dev_auc"], "holdout_auc": clean["final_gate"]["holdout_auc"],
                              "holdout_auc_deploy_view": clean["final_gate"]["holdout_auc"]}
json.dump({"n_leak_fields_present": len(present), "arms": results, "detection": detect, "missed_by_all_rules": missed,
           "top_single_feature_auc_of_leak_fields": top_auc}, open(out / f"a3_results{suffix}.json", "w"), ensure_ascii=False, indent=1)

# ---- 图 ----
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.3, 1]})
arms = [("clean_reference", "干净数据\n（基准）"), ("guardrail_on", "泄漏字段在数据里\nguardrail 开"), ("guardrail_off", "泄漏字段在数据里\nguardrail 关")]
metrics = [("oot_dev_auc", "OOT-dev AUC（agent 可见）", "#4C78A8"), ("holdout_auc", "holdout AUC（含泄漏字段）", "#F58518"),
           ("holdout_auc_deploy_view", "上线视角 AUC（泄漏字段不可得）", "#54A24B")]
w = 0.26
for j, (k, lab, col) in enumerate(metrics):
    vals = [results[a][k] for a, _ in arms]
    bars = ax[0].bar([i + (j - 1) * w for i in range(3)], vals, w, label=lab, color=col)
    for b, v in zip(bars, vals):
        ax[0].text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
ax[0].set_xticks(range(3)); ax[0].set_xticklabels([l for _, l in arms], fontsize=9)
ax[0].set_ylim(0.5, 1.05); ax[0].set_ylabel("AUC"); ax[0].legend(fontsize=8, loc="upper left"); ax[0].set_title(f"A3（{DS}）：去掉 guardrail 后 AUC 虚高，上线即塌陷")
names = list(detect)
ax[1].barh(range(len(names)), [detect[n]["recall"] for n in names], color="#4C78A8")
for i, n in enumerate(names):
    ax[1].text(detect[n]["recall"] + 0.01, i, f"R={detect[n]['recall']:.2f}  P={detect[n]['precision']:.2f}", va="center", fontsize=8)
ax[1].set_yticks(range(len(names))); ax[1].set_yticklabels(names, fontsize=8); ax[1].set_xlim(0, 1.4); ax[1].invert_yaxis()
ax[1].set_title(f"泄漏检出（{len(present)} 个真泄漏 + {len(present_suspect)} 个存疑）")
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC"]
plt.tight_layout(); plt.savefig(out / f"a3_comparison{suffix}.png", dpi=140)
print(json.dumps({"arms": results, "detection": detect, "missed": missed}, ensure_ascii=False, indent=1))
