"""收集一个真实任务的输入（元数据、事件、实验记录、trial、接入状态、产出文件），交给 view.snapshot。"""
import json

import yaml

from memory.store import MemoryStore, Store
from runtime.events import EventLog
from runtime.trace import load_trials
from server import runner, tasks


def live(tid: str) -> dict:
    meta = runner.refresh(tasks.load(tid))
    d, db = tasks.task_dir(tid), str(tasks.events_db(tid))
    events = EventLog(db, tid).query()
    state = json.loads((d / "intake.json").read_text()) if (d / "intake.json").exists() else None
    records = [r.model_dump(mode="json") for r in Store(db).all(tid)]
    optuna_db = d / "artifacts" / "optuna.db"
    trials = load_trials(str(optuna_db), [r["exp_id"] for r in records]) if records and optuna_db.exists() else {}
    return {"meta": meta, "events": events, "records": records, "trials": trials, "intake": state,
            "data": {"csv_name": meta["csv_name"], "description": (d / "input" / "description.md").read_text()},
            "files": files(meta, events, state)}


def files(meta: dict, events: list[dict], state: dict | None) -> dict:
    """产出随流程出现：配置写出后有 YAML，跑完有模型卡、实验树和本次经验；完整事件一直都在。"""
    tid, out = meta["id"], {}
    rd = tasks.task_dir(tid) / "artifacts" / "reports"
    for n, pat in (("model_card.md", f"model_card_{tid}.md"), ("experiment_tree.mmd", f"experiment_tree_{tid}.mmd")):
        if (rd / pat).exists():
            out[n] = (rd / pat).read_text()
    if w := (state or {}).get("written"):
        out.update(w["yaml"])
        out["task_spec.yaml"] = yaml.safe_dump(w["task_spec"], allow_unicode=True, sort_keys=False)
    if meta["status"] in ("done", "accepted"):
        mem = MemoryStore(str(tasks.workspace() / "memory.db"))
        out["lessons.json"] = json.dumps([x for x in mem.lessons() if tid in x["tasks"]], ensure_ascii=False, indent=1)
    out["events.json"] = json.dumps(events, ensure_ascii=False, indent=1, default=str)
    return out
