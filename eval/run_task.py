"""命令行入口：python -m eval.run_task --dataset lending_club --task-id lc1 --llm mock|anthropic|kimi
holdout 的 FinalGate 在这里创建并注入（agent/ 下的代码不导入 final_gate）。"""
import argparse
import json

from agent.llm import make_llm
from agent.orchestrator import Orchestrator
from agent.schemas import TaskSpec
from data.knowhow import load_config
from evaluation.final_gate import FinalGate

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="lending_club")
ap.add_argument("--task-id", required=True)
ap.add_argument("--llm", default="mock", help="mock | anthropic | kimi | deepseek（见 config.yaml llm.providers）")
ap.add_argument("--autonomy", default="L1")
ap.add_argument("--full-trials", type=int, help="覆盖 full 保真度的 trial 数（开发时缩短用）")
ap.add_argument("--max-rounds", type=int)
ap.add_argument("--memory-db", help="跨任务记忆库路径（默认 config.memory.db）")
ap.add_argument("--no-memory", action="store_true", help="消融 A2：不读取 memory（仍会写入）")
a = ap.parse_args()

cfg = load_config()
if a.full_trials:
    cfg["fidelity"]["full"]["n_trials"] = a.full_trials
if a.max_rounds:
    cfg["run"]["max_rounds"] = a.max_rounds
if a.memory_db:
    cfg["memory"]["db"] = a.memory_db
if a.no_memory:
    cfg["ablation"] = {"no_memory": True}
llm = make_llm(a.llm, cfg)
spec = TaskSpec.from_knowhow(a.dataset, cfg, autonomy=a.autonomy)
o = Orchestrator(spec, cfg, llm, FinalGate(a.dataset, cfg, a.task_id), a.task_id)
print("最终状态:", o.run().value)
print(json.dumps({"best": o.p["best_exp_id"], "final_gate": o.p["final"], "stop": o.p["stop_reason"],
                  "await": o.p["await"], "oot_used": o.evaluator.budget.used, "budget_used": o.p["budget"]["used"]},
                 ensure_ascii=False, indent=1, default=str))
