"""一个评测版本的结构化结果 version.json（spec 2026-10-05 §7）：界面和回归对比都读它。"""
import json
import re
import subprocess
import time
from pathlib import Path

KEEP = ("seed", "status", "error", "final_model", "holdout", "oot_dev", "overfit", "stop_reason", "n_rounds",
        "wall_sec", "cost_usd", "task_id", "process")


def git_state() -> dict:
    run = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
    return {"sha": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def _summary(rs: list[dict], b: dict) -> dict:
    """版本层面（spec §4）：各种子取平均、计比例；最差的那次（失败的优先，其次 holdout 最低）作为排查入口。"""
    ok = [r for r in rs if r["status"] == "ok" and r.get("holdout") is not None]
    b1 = (b.get("b1") or {}).get("holdout_auc")
    mean = lambda xs: sum(xs) / len(xs) if xs else None
    worst = min(rs, key=lambda r: (r["status"] == "ok" and r.get("holdout") is not None, r.get("holdout") or 0))
    fe = sum(bool(((r.get("process") or {}).get("fe_funnel") or {}).get("value")) for r in rs)
    return {"n": len(rs), "n_ok": len(ok), "holdout_mean": mean([r["holdout"] for r in ok]),
            "delta_b1_mean": None if b1 is None or not ok else mean([r["holdout"] - b1 for r in ok]),
            "rounds_mean": mean([((r.get("process") or {}).get("rounds") or {}).get("value") or 0 for r in rs]),
            "fe_in_final": f"{fe}/{len(rs)}", "wall_sec": round(sum(r.get("wall_sec") or 0 for r in rs), 1),
            "cost_usd": sum(r.get("cost_usd") or 0 for r in rs), "worst_task": worst.get("task_id")}


def build_version(label: str, tier: str, settings: dict, rows: list[dict], bases: dict, scenarios: list[dict], started: float,
                  status: str = "done", jobs: int = 1) -> dict:
    """status=running：运行中每跑完一个种子 / 情景就写一次，界面能看到正在跑的版本；回归对比不取它。"""
    allr = rows + scenarios
    return {"label": label, "tier": tier, "status": status, "jobs": jobs, "settings": settings, "git": git_state(), "started": started, "finished": time.time(),
            "wall_sec": round(sum(r.get("wall_sec") or 0 for r in allr), 1), "cost_usd": sum(r.get("cost_usd") or 0 for r in allr),
            "datasets": {ds: {"baselines": bases.get(ds, {}), "runs": [{k: r.get(k) for k in KEEP} for r in rows if r["dataset"] == ds],
                              "summary": _summary([r for r in rows if r["dataset"] == ds], bases.get(ds, {}))}
                         for ds in dict.fromkeys(r["dataset"] for r in rows)},
            "scenarios": scenarios}


def label_key(label: str) -> list:
    """版本号的自然顺序：v7a < v7b < v8 < v9a < v10。"""
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", label)]


def latest_version(root: Path, tier: str, before: str) -> dict | None:
    """同档、已跑完、版本号排在 before 之前的最后一个。按版本号而不是跑完的时间：重跑旧版本不会变成"上一版本"。"""
    vs = [json.loads(p.read_text()) for p in Path(root).glob("*/version.json")]
    vs = [v for v in vs if v.get("tier") == tier and v["status"] == "done" and label_key(v["label"]) < label_key(before)]
    return max(vs, key=lambda v: label_key(v["label"])) if vs else None
