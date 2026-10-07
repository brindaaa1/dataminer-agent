"""工作台的任务：元数据（task.json）、目录布局、按任务生成的 cfg。不含业务逻辑。

workspace/tasks/<task_id>/
  task.json      状态、provider、设置、备注
  input/         上传的 data.csv 与 description.md
  intake.json    IntakeSession.to_json()
  config/        接入写出的 know-how YAML
  artifacts/     state.db（事件、检查点、实验记录）、切分缓存、optuna.db、reports/
"""
import json
import os
import re
import shutil
import time
import uuid
from pathlib import Path

from data.knowhow import load_config
from data.sample import ROOT

DEFAULT_SETTINGS = {"max_rounds": 5, "full_trials": 5, "low_trials": None}


def workspace() -> Path:
    return Path(os.environ.get("DATAMINER_WORKSPACE") or ROOT / "workspace")


def task_dir(tid: str) -> Path:
    return workspace() / "tasks" / tid


def exists(tid: str) -> bool:
    return (task_dir(tid) / "task.json").exists()


def load(tid: str) -> dict:
    return json.loads((task_dir(tid) / "task.json").read_text())


def save(meta: dict) -> dict:
    p = task_dir(meta["id"]) / "task.json"
    tmp = p.with_name(f"task.json.{os.getpid()}.tmp")           # 先写临时文件再替换：worker 和 web 服务都会写
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    tmp.replace(p)
    return meta


def update(tid: str, **kw) -> dict:
    return save({**load(tid), **kw})


def create(csv: bytes, csv_name: str, description: str, provider: str, settings: dict | None = None,
           source: str | None = None) -> dict:
    tid = time.strftime("%m%d_%H%M%S_") + uuid.uuid4().hex[:4]
    d = task_dir(tid)
    (d / "input").mkdir(parents=True)
    (d / "input" / "data.csv").write_bytes(csv)
    (d / "input" / "description.md").write_text(description)
    return save({"id": tid, "title": f"{Path(csv_name).stem} · {time.strftime('%m-%d %H:%M')}", "csv_name": csv_name,
                 "provider": provider, "settings": {**DEFAULT_SETTINGS, **(settings or {})}, "status": "intake",
                 "busy": False, "error": None, "pid": None, "langfuse_url": None, "intake_langfuse_url": None,
                 "note": None, "source": source, "created_at": time.time()})


def list_tasks() -> list[dict]:
    root = workspace() / "tasks"
    metas = [json.loads(p.read_text()) for p in root.glob("*/task.json")] if root.exists() else []
    return sorted(metas, key=lambda m: m["created_at"], reverse=True)


def dataset_id(meta: dict) -> str:
    """know-how 目录名、切分缓存目录名都用它，只留 ASCII。"""
    return re.sub(r"[^A-Za-z0-9_]+", "_", Path(meta["csv_name"]).stem).strip("_").lower() or "data"


def events_db(tid: str) -> Path:
    art = task_dir(tid) / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    return art / "state.db"


def task_cfg(meta: dict) -> dict:
    """每个任务一套产物目录（state.db、切分、optuna、报告各自独立）；跨任务经验库共用。"""
    cfg = load_config(str(ROOT / "config.yaml"))
    art = events_db(meta["id"]).parent
    cfg["paths"].update(artifacts_root=str(art), reports_root=str(art / "reports"))
    cfg["memory"]["db"] = str(workspace() / "memory.db")
    cfg["fidelity"]["low"]["sample_frac"] = 0.5        # 同 data/sample.py：第一版用 hotel 样本测，10% 抽样在几千行上只剩噪声
    s = meta["settings"]
    cfg["run"]["max_rounds"], cfg["fidelity"]["full"]["n_trials"] = s["max_rounds"], s["full_trials"]
    if s.get("low_trials"):
        cfg["fidelity"]["low"]["n_trials"] = s["low_trials"]
    return cfg


def clone(tid: str) -> dict:
    """同一份数据和确认过的配置新建任务，直接进入"配置已确认"；路径里的旧任务目录换成新的。"""
    src, d0 = load(tid), task_dir(tid)
    new = create((d0 / "input" / "data.csv").read_bytes(), src["csv_name"], (d0 / "input" / "description.md").read_text(),
                 src["provider"], src["settings"], source=tid)
    d1 = task_dir(new["id"])
    shutil.copytree(d0 / "config", d1 / "config")
    (d1 / "intake.json").write_text((d0 / "intake.json").read_text().replace(str(d0), str(d1)))
    return update(new["id"], status="config_ready")
