"""建模子进程：python -m server.worker <task_id>。
流程与 app.py 的 live_run 相同（自治级别 L0，无人值守）；Orchestrator 每个状态结束写检查点，
同一个任务再启动一次就是续跑。上一次已经停在 FAILED 的任务续跑会立即结束（检查点就是终态）。"""
import json
import os
import sys

from agent.llm import make_llm, redact
from agent.orchestrator import Orchestrator
from agent.schemas import TaskSpec
from agent.states import TERMINAL, State
from evaluation.final_gate import FinalGate
from intake.writer import register
from runtime.events import EventLog
from runtime.tracing import make_tracer
from server import tasks


def run(tid: str):
    tasks.update(tid, pid=os.getpid())
    meta = tasks.load(tid)
    cfg = tasks.task_cfg(meta)
    written = json.loads((tasks.task_dir(tid) / "intake.json").read_text())["written"]
    register(cfg, written)
    ds = written["dataset_id"]
    tracer = make_tracer(tid, {"dataset": ds, "llm": meta["provider"], "source": "workbench"})
    if tracer.enabled:
        tasks.update(tid, langfuse_url=tracer.url())
    try:
        o = Orchestrator(TaskSpec.from_knowhow(ds, cfg, autonomy="L0"), cfg, make_llm(meta["provider"], cfg),
                         FinalGate(ds, cfg, tid), tid, tracer=tracer)
        while o.state not in TERMINAL and o.state != State.AWAIT_HUMAN:
            o.step()
        status, err = ("done", None) if o.state == State.DONE else ("failed", f"运行结束于 {o.state.value}")
    except Exception as e:
        status, err = "failed", redact(f"{type(e).__name__}: {str(e)[:500]}")
        EventLog(str(tasks.events_db(tid)), tid).append("worker_error", {"error": err})
    tasks.update(tid, status=status, error=err, pid=None)


if __name__ == "__main__":
    run(sys.argv[1])
