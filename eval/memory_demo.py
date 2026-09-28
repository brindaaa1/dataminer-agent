"""Memory 轻量演示（小样本，几分钟）：
  ① 历史任务 H（Lending Club 2012–13）→ 产出 Lesson 候选 + Prior
  ② 新任务 N（Lending Club 2013–14，指纹相似）：有/无 memory 各跑一次；lesson 在 N 上复现 → ACTIVE
  ③ 酒店任务 T（指纹很远）：冷启动，不复用任何经验
  ④ 收敛对比：N 的数据上，冷启动 vs 用 H 的 Prior warm-start，多个种子下的 best-so-far valid AUC（不做严格统计检验）
python -m eval.memory_demo [--sample 30000] [--seeds 5]"""
import argparse
import copy
import json
import shutil
from pathlib import Path

import numpy as np
import optuna

from agent.llm import PolicyMockLLM
from agent.orchestrator import Orchestrator
from agent.schemas import TaskSpec
from data.knowhow import load_config
from data.splits import build_splits
from evaluation.final_gate import FinalGate
from memory.retrieval import fingerprint_distance, nearest_task
from memory.store import MemoryStore
from modeling.inner_loop import run_study

ap = argparse.ArgumentParser()
ap.add_argument("--sample", type=int, default=60000)
ap.add_argument("--seeds", type=int, default=5)
a = ap.parse_args()

BASE = Path("artifacts_mem")
shutil.rmtree(BASE, ignore_errors=True)
cfg0 = load_config()
cfg0["fidelity"]["low"] = {"sample_frac": 1.0, "n_trials": 20, "early_stopping_rounds": 20}   # 数据已缩小，低保真不再抽样
cfg0["fidelity"]["full"]["n_trials"] = 15
cfg0["run"]["max_rounds"] = 4
cfg0["memory"]["db"] = str(BASE / "memory.db")
WIN = {"H": {"train_valid": ["2012-01", "2013-06"], "oot_dev": ["2013-07", "2013-09"], "holdout": ["2013-10", "2013-12"]},
       "N": {"train_valid": ["2013-01", "2014-06"], "oot_dev": ["2014-07", "2014-09"], "holdout": ["2014-10", "2014-12"]}}


def cfg_for(root, mem=None):
    c = copy.deepcopy(cfg0)
    c["paths"]["artifacts_root"] = str(BASE / root)
    if mem:
        c["memory"]["db"] = mem
    return c


for name in ("H", "N"):
    build_splits("lending_club", cfg_for(name), sample_n=a.sample, split_override=WIN[name])
build_splits("hotel_bookings", cfg_for("T"), sample_n=a.sample)


def run(dataset, root, task, autonomy="L0", **kw):
    c = cfg_for(root, kw.get("mem"))
    if kw.get("no_memory"):
        c["ablation"] = {"no_memory": True}
    o = Orchestrator(TaskSpec.from_knowhow(dataset, c, autonomy=autonomy), c, PolicyMockLLM(), FinalGate(dataset, c, task), task)
    o.run()
    ev = lambda t: [e["payload"] for e in o.events.query(t)]
    return o, ev


runs = {}
runs["H"] = run("lending_club", "H", "H")
print("H 完成", flush=True)
shutil.copy(BASE / "memory.db", BASE / "memory_H_only.db")         # 给"无 memory"对照组一份独立的空库（不读 H）
runs["N_on"] = run("lending_club", "N", "N_on")
print("N_on 完成", flush=True)
runs["N_off"] = run("lending_club", "N", "N_off", no_memory=True, mem=str(BASE / "memory_off.db"))
runs["T"] = run("hotel_bookings", "T", "T")
print("N_off / T 完成", flush=True)

mem = MemoryStore(str(BASE / "memory.db"))
fps = {t["fingerprint"]["task_id"]: t["fingerprint"] for t in mem.tasks()}
L = ["# Memory 轻量演示", "", f"小样本：每个任务约 {a.sample} 行；mock LLM（策略固定）；full 保真度 15 个 trial。", "",
     "## 1. 指纹与检索", "", "| 任务 | 数据 | n_samples | n_features | bad_rate | 检索结果（距离；阈值 %.1f） |" % cfg0["memory"]["max_distance"], "|---|---|---|---|---|---|"]
for k in ("H", "N_on", "T"):
    o, ev = runs[k]
    r = ev("memory_retrieval")[0]
    fp = fps[k]
    L.append(f"| {k} | {o.spec.dataset} | {fp['n_samples']} | {fp['n_features']} | {fp['bad_rate']:.3f} | "
             + (f"冷启动（无历史任务）" if r["distance"] is None else f"最近={r['nearest'] or '无'}；距离 {r['distance']:.2f}；{'冷启动' if r['cold_start'] else '复用 prior'}；ACTIVE 经验 {r['n_active_lessons']} 条") + " |")
L += ["", "## 2. Lesson 生命周期", "", "| key | 极性 | 状态 | 支撑任务 | 置信度 |", "|---|---|---|---|---|"]
for l in sorted(mem.lessons(), key=lambda l: (l["status"], l["key"])):
    L.append(f"| {l['key']} | {l['polarity']} | **{l['status']}** | {','.join(l['tasks'])} | {l['confidence']} |")
cons = runs["N_on"][1]("consolidated")[0]
L += ["", f"N_on 的 CONSOLIDATE：LLM 提出 {cons['proposed']} 条，规则准入 {cons['admitted']} 条，丢弃 {len(cons['dropped'])} 条；本次状态变化：{cons['lifecycle_changes']}", ""]

# ---- 收敛对比：同一份 N 数据，冷启动 vs 用 H 的 prior ----
cN = cfg_for("N")
near, _ = nearest_task(MemoryStore(str(BASE / "memory_H_only.db")), fps["N_on"], cN)
if not near or not near["prior"]:
    L += ["## 3. 收敛对比", "", "历史任务 H 没有产出 Prior（没有实验通过 guardrail，因此没有『最优实验』可提取——记忆不会保存未经验证的结果）。跳过收敛对比。", ""]
    Path("reports/memory_demo.md").write_text("\n".join(L))
    print("\n".join(L))
    raise SystemExit(0)
prior = near["prior"]
curves = {"cold": [], "warm": []}
for arm in curves:
    for s in range(a.seeds):
        eid = f"bench_{arm}_{s}"
        run_study(eid, "lending_club", "lgbm", cN, fidelity="low", seed=s, warm_start=prior["params"] if arm == "warm" else None)
        st = optuna.load_study(study_name=eid, storage=f"sqlite:///{BASE / 'N' / 'optuna.db'}")
        vals = [t.value for t in st.trials if t.value is not None]
        curves[arm].append(np.maximum.accumulate(vals))
KS = [1, 2, 3, 5, 10, 15]
L += ["## 3. 收敛对比（N 的数据，lgbm，20 个 trial，%d 个种子，best-so-far valid AUC 的均值 ± 标准差）" % a.seeds, "",
      f"warm-start 使用 H 的 prior：从历史任务取 {len(prior['params'])} 组参数，上限为 trial 总数的 10%（= 2 个初始点）。", "",
      "| 已跑 trial 数 | " + " | ".join(str(k) for k in KS) + " |", "|---|" + "---|" * len(KS)]
for arm, label in (("cold", "冷启动"), ("warm", "warm-start（用 H 的 prior）")):
    cells = []
    for k in KS:
        v = [c[min(k, len(c)) - 1] for c in curves[arm]]
        cells.append(f"{np.mean(v):.4f}±{np.std(v):.4f}")
    L.append(f"| {label} | " + " | ".join(cells) + " |")
tgt = np.mean([c[-1] for c in curves["cold"]]) - 0.002
tt = {arm: [next((i + 1 for i, x in enumerate(c) if x >= tgt), None) for c in curves[arm]] for arm in curves}
L += ["", f"达到目标（冷启动最终均值 − 0.002 = {tgt:.4f}）所需 trial 数：冷启动 {tt['cold']}，warm-start {tt['warm']}（None = 20 个 trial 内没达到）。", ""]
json.dump({"curves": {k: [c.tolist() for c in v] for k, v in curves.items()}, "trials_to_target": tt}, open("reports/memory_demo_data.json", "w"))
Path("reports/memory_demo.md").write_text("\n".join(L))
print("\n".join(L))
