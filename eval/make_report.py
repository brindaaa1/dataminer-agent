"""对已有任务重新生成模型卡（不重新训练）：python -m eval.make_report --task-id lc_mock2 [--dataset lending_club]"""
import argparse

from agent.llm import PolicyMockLLM
from agent.orchestrator import Orchestrator
from agent.schemas import TaskSpec
from data.knowhow import load_config
from report.model_card import build_report

ap = argparse.ArgumentParser()
ap.add_argument("--task-id", required=True)
ap.add_argument("--dataset", default="lending_club")
a = ap.parse_args()
cfg = load_config()
o = Orchestrator(TaskSpec.from_knowhow(a.dataset, cfg), cfg, PolicyMockLLM(), None, a.task_id)   # 不需要 final_gate：结果已在检查点里
r = build_report(cfg, o.spec, o.store, o.p, o.p["final"], o.evaluator.budget.used, o.llm)
print(r["model_card"], "\n".join(r["risks"]), sep="\n")
