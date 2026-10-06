"""一个评测版本的结构化结果 version.json（spec 2026-10-05 §7）：界面和回归对比都读它。"""
import json
import subprocess
import time
from pathlib import Path

KEEP = ("seed", "status", "error", "final_model", "holdout", "oot_dev", "overfit", "stop_reason", "n_rounds",
        "wall_sec", "cost_usd", "task_id", "process")


def git_state() -> dict:
    run = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
    return {"sha": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def build_version(label: str, tier: str, settings: dict, rows: list[dict], bases: dict, scenarios: list[dict], started: float) -> dict:
    allr = rows + scenarios
    return {"label": label, "tier": tier, "settings": settings, "git": git_state(), "started": started, "finished": time.time(),
            "wall_sec": round(sum(r.get("wall_sec") or 0 for r in allr), 1), "cost_usd": sum(r.get("cost_usd") or 0 for r in allr),
            "datasets": {ds: {"baselines": bases.get(ds, {}), "runs": [{k: r.get(k) for k in KEEP} for r in rows if r["dataset"] == ds]}
                         for ds in dict.fromkeys(r["dataset"] for r in rows)},
            "scenarios": scenarios}


def latest_version(root: Path, tier: str, exclude: str) -> dict | None:
    vs = [json.loads(p.read_text()) for p in Path(root).glob("*/version.json")]
    vs = [v for v in vs if v.get("tier") == tier and v.get("label") != exclude]
    return max(vs, key=lambda v: v.get("finished", 0)) if vs else None
