"""评测运行作为只读任务（spec 2026-10-05 §8）：读 eval/suite/results/<版本>/version.json 和 data_raw/eval_runs 下的产物，
包装成与 examples.load 同形的 src，渲染走同一个 view.snapshot。产物目录的约定与 eval/suite/runner.art_dir 一致。"""
import json
import os
from pathlib import Path

from data.sample import ROOT
from memory.store import Store
from runtime.events import EventLog
from runtime.trace import load_trials

STATUS = {"ok": "done", "awaiting_human": "stopped"}          # 其余（failed 等）都按 failed


def results_root() -> Path:
    return Path(os.environ.get("DATAMINER_EVAL_RESULTS") or ROOT / "eval" / "suite" / "results")


def runs_root() -> Path:
    return Path(os.environ.get("DATAMINER_EVAL_RUNS") or ROOT / "data_raw" / "eval_runs")


def _versions() -> list[dict]:
    vs = [json.loads(p.read_text()) for p in results_root().glob("*/version.json")]
    return sorted(vs, key=lambda v: v.get("finished", 0), reverse=True)


def _regression(label: str) -> dict | None:
    p = results_root() / label / "regression.json"
    return json.loads(p.read_text()) if p.exists() else None


def index() -> dict[str, dict]:
    out = {}
    for v in _versions():
        lab, st = v["label"], v["settings"]
        base = {"label": lab, "tier": v["tier"]}
        for ds, d in v["datasets"].items():
            for r in d["runs"]:
                out[f"ev-{r['task_id']}"] = {**base, "kind": "run", "dataset": ds, "title": f"{ds} · 种子 {r['seed']}",
                                              "status": STATUS.get(r["status"], "failed"), "row": r, "max_rounds": st["max_rounds"],
                                              "art": runs_root() / lab / ds / f"s{r['seed']}"}
        for s in v.get("scenarios", []):
            out[f"ev-{s['task_id']}"] = {**base, "kind": "scenario", "dataset": s["base"], "title": f"情景 {s['scenario']} @ {s['base']}",
                                          "status": STATUS.get(s.get("status"), "failed"), "row": s, "max_rounds": st["scenario"]["max_rounds"],
                                          "art": runs_root() / lab / s["base"] / f"scn_{s['scenario']}"}
    return out


def exists(tid: str) -> bool:
    return tid in index()


def summaries() -> list[dict]:
    concl = {}
    out = []
    for tid, e in index().items():
        if e["label"] not in concl:
            concl[e["label"]] = (_regression(e["label"]) or {}).get("conclusion")
        out.append({"id": tid, "title": e["title"], "status": e["status"], "created_at": None, "example": False,
                    "eval": {"label": e["label"], "tier": e["tier"], "conclusion": concl[e["label"]]}})
    return out


def version_of(tid: str) -> tuple[dict, dict | None]:
    lab = index()[tid]["label"]
    return json.loads((results_root() / lab / "version.json").read_text()), _regression(lab)


def load(tid: str) -> dict:
    e, task_id = index()[tid], tid[3:]
    art, db = e["art"], e["art"] / "state.db"
    events = EventLog(str(db), task_id).query() if db.exists() else []
    records = [r.model_dump(mode="json") for r in Store(str(db)).all(task_id)] if db.exists() else []
    optuna = art / "optuna.db"
    trials = load_trials(str(optuna), [r["exp_id"] for r in records]) if records and optuna.exists() else {}
    files = {}
    for n, pat in (("model_card.md", f"model_card_{task_id}.md"), ("experiment_tree.mmd", f"experiment_tree_{task_id}.mmd")):
        if (p := art / "reports" / pat).exists():
            files[n] = p.read_text()
    files["events.json"] = json.dumps(events, ensure_ascii=False, indent=1, default=str)
    meta = {"id": tid, "title": f"{e['label']} · {e['title']}", "status": e["status"], "example": False, "eval": e["label"],
            "provider": "", "busy": False, "settings": {"max_rounds": e["max_rounds"]}, "note": None, "error": e["row"].get("error")}
    return {"meta": meta, "events": events, "records": records, "trials": trials, "intake": None,
            "data": {"csv_name": e["dataset"], "description": f"评测版本 {e['label']}（{e['tier']} 档）"}, "files": files}
