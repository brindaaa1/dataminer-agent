"""把一次运行的事件日志和实验记录导出成 JSON，供 app.py 作为内置示例展示（不含原始数据行）。

    python -m eval.export_trace --db artifacts/state.db --task-id lc_kimi1 --note "..." \
        --meta '{"dataset": "Lending Club", "llm": "Kimi（kimi-k3）", "has_sample_data": false}'

meta 是页面上"运行档案"的内容（数据集、模型、设置等），只写能从运行命令或日志核实的信息。
"""
import argparse
import json
from pathlib import Path

from memory.store import Store
from runtime.env import ROOT
from runtime.events import EventLog


def export(db: str, task_id: str, note: str = "", meta: dict | None = None) -> dict:
    events = EventLog(db, task_id).query()
    if not events:
        raise SystemExit(f"{db} 里没有任务 {task_id} 的事件")
    return {"task_id": task_id, "note": note, "meta": meta or {}, "events": events,
            "records": [r.model_dump(mode="json") for r in Store(db).all(task_id)]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--task-id", required=True)
    ap.add_argument("--note", default="")
    ap.add_argument("--meta", default="{}", help="JSON：页面运行档案里展示的信息")
    ap.add_argument("--out-dir", default="examples/traces")
    a = ap.parse_args()
    out = Path(a.out_dir) / f"{a.task_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(export(a.db, a.task_id, a.note, json.loads(a.meta)), ensure_ascii=False, indent=1)
    out.write_text(text.replace(str(ROOT) + "/", ""))        # 事件里的绝对路径（含本机用户名）改成仓库相对路径
    print(out)
