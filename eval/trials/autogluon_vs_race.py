"""试验：AutoGluon 单模型（只选最好的一个，不融合）和融合模型，与现有 agent 的最终模型比较。
用 .venvs/ag 运行：caffeinate -i .venvs/ag/bin/python eval/trials/autogluon_vs_race.py"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from autogluon.tabular import TabularPredictor  # noqa: E402

from baselines.common import finish, load_wide  # noqa: E402
from data.splits import build_splits  # noqa: E402
from eval.suite.datasets import DATASETS  # noqa: E402
from eval.suite.runner import B1_BUDGET_SEC, build_cfg, prepare_run  # noqa: E402
from evaluation.mde import oot_power  # noqa: E402

LABEL = "trial_ag"
AGENT_RUNS = {"hotel_bookings": "v13", "lending_club": "v13", "home_credit": "v6"}
MODES = {"单模型": dict(presets="medium_quality", hyperparameters={"GBM": {}, "CAT": {}, "XGB": {}}, fit_weighted_ensemble=False),
         "融合": dict(presets="best_quality", use_bag_holdout=True)}


def agent_numbers(name: str) -> dict:
    d = ROOT / "eval" / "suite" / "results" / AGENT_RUNS[name]
    runs = [json.loads(l) for l in (d / "runs.jsonl").read_text().splitlines() if json.loads(l)["dataset"] == name]
    b1 = json.loads((d / "baselines.json").read_text())[name]["b1"]
    return {"version": AGENT_RUNS[name], "agent_oot": np.mean([r["oot_dev"] for r in runs]),
            "agent_holdout": np.mean([r["holdout"] for r in runs]), "b1_oot": b1["oot_dev_auc"], "b1_holdout": b1["holdout_auc"]}


def run(name: str, mode: str) -> dict:
    sub = f"ag_{mode}"
    prepare_run(LABEL, name, sub)
    cfg, ds = build_cfg(name, LABEL, 0, sub), DATASETS[name]["dataset"]
    build_splits(ds, cfg)
    Xtr, ytr, Xva, yva, Xoo, yoo, prep = load_wide(ds, cfg)
    path = Path(cfg["paths"]["artifacts_root"]) / "autogluon"
    t0 = time.time()
    pr = TabularPredictor(label="_y", eval_metric="roc_auc", path=str(path), verbosity=0).fit(
        Xtr.assign(_y=ytr.to_numpy()), tuning_data=Xva.assign(_y=yva.to_numpy()), time_limit=B1_BUDGET_SEC, **MODES[mode])
    wall = time.time() - t0
    predict = lambda X: pr.predict_proba(X)[1].to_numpy()
    r = finish(f"ag_{mode}_{name}", ds, cfg, predict, prep, Xoo, yoo, wall, {"best_model": pr.model_best})
    return {**r, "mde": oot_power(yoo, predict(Xoo), cfg)["mde"]}


def main():
    md = ["# AutoGluon 是否比现有建模好", "", f"每次训练限时 {B1_BUDGET_SEC:.0f} 秒（与 B1 相同）；切分种子 42。", "",
          "| 数据 | 方式 | 时间外验证 AUC | 最终检验 AUC | 比 agent 高（最终检验） | 能分辨的最小提升 | agent 来源 |",
          "|---|---|---|---|---|---|---|"]
    for name in AGENT_RUNS:
        a = agent_numbers(name)
        md.append(f"| {name} | agent 最终模型 | {a['agent_oot']:.4f} | {a['agent_holdout']:.4f} | — | — | {a['version']} |")
        md.append(f"| {name} | B1 FLAML | {a['b1_oot']:.4f} | {a['b1_holdout']:.4f} | {a['b1_holdout'] - a['agent_holdout']:+.4f} | — | {a['version']} |")
        for mode in MODES:
            r = run(name, mode)
            md.append(f"| {name} | AutoGluon {mode}（{r['best_model']}） | {r['oot_dev_auc']:.4f} | {r['holdout_auc']:.4f} | "
                      f"{r['holdout_auc'] - a['agent_holdout']:+.4f} | {r['mde']:.4f} | {a['version']} |")
    out = ROOT / "eval" / "trials" / "results" / "autogluon_vs_race.md"
    out.write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
