"""examples/traces/ 里录制的真实运行，作为只读示例任务；渲染和真实任务走同一个 view.snapshot。
不填 API key 也能完整浏览一次流程。"""
import json
import os
from pathlib import Path

from data.sample import ROOT


def root() -> Path:
    return Path(os.environ.get("DATAMINER_EXAMPLES") or ROOT / "examples" / "traces")     # 与 app.py 共用这个环境变量


def names() -> list[str]:
    return sorted(p.stem for p in root().glob("*.json"))


def load(name: str) -> dict:
    rec = json.loads((root() / f"{name}.json").read_text())
    files = {}
    for n, pat in (("model_card.md", f"model_card_{name}.md"), ("experiment_tree.mmd", f"experiment_tree_{name}.mmd")):
        if (p := root() / "reports" / pat).exists():
            files[n] = p.read_text()
    files["events.json"] = json.dumps(rec["events"], ensure_ascii=False, indent=1)
    meta = {"id": f"ex-{name}", "title": f"示例 · {name}", "status": "done", "example": True,
            "provider": rec["meta"].get("llm", ""), "busy": False, "settings": {}, "note": None}
    return {"meta": meta, "events": rec["events"], "records": rec["records"], "trials": rec.get("trials") or {},
            "intake": None, "data": {"csv_name": rec["meta"].get("dataset", ""), "description": rec.get("note", "")},
            "files": files}
