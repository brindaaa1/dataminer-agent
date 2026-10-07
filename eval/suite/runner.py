"""一次评测运行：手写 know-how（跳过 LLM 起草接入），L0，固定设置；产物放 data_raw/eval_runs/<label>/<数据集>/s<种子>/。
每个种子一个目录（切分、state.db、经验库都独立），可以单独补跑某个种子；基线放 <数据集>/baselines/。"""
import json
import os
import shutil
from pathlib import Path

from agent.llm import make_llm, redact
from agent.orchestrator import Orchestrator
from agent.schemas import TaskSpec
from agent.states import TERMINAL, State
from baselines import b0_default_lgbm, b1_automl
from data.knowhow import load_config
from data.splits import build_splits
from data.sample import ROOT
from eval.process import process_metrics
from eval.suite.datasets import DATASETS
from eval.suite.metrics import cost_usd, process_stats, usage_totals
from evaluation.final_gate import FinalGate
from memory.store import Store
from runtime.events import EventLog
from runtime.tracing import make_tracer

RUNS = ROOT / "data_raw" / "eval_runs"
SETTINGS = {"max_rounds": 6, "low_sample_frac": 0.3, "low_trials": 10, "full_trials": 20, "split_seed": 42, "autonomy": "L0"}


def art_dir(label: str, name: str, seed: int | str):
    return RUNS / label / name / (f"s{seed}" if isinstance(seed, int) else seed)


def prepare_run(label: str, name: str, seed: int | str):
    """同一版本号重跑：清掉这个种子的旧产物。否则 Orchestrator._train 会直接复用磁盘上同名实验的结果、从旧检查点续跑。"""
    shutil.rmtree(art_dir(label, name, seed), ignore_errors=True)


def merge_rows(old: list[dict], new: list[dict]) -> list[dict]:
    """补跑时按 (数据集, 种子) 替换，其余保留。"""
    done = {(r["dataset"], r["seed"]) for r in new}
    return [r for r in old if (r["dataset"], r["seed"]) not in done] + new


def build_cfg(name: str, label: str, seed: int, sub: str | None = None) -> dict:
    cfg = load_config(str(ROOT / "config.yaml"))
    d, art = DATASETS[name], art_dir(label, name, sub or seed)
    art.mkdir(parents=True, exist_ok=True)
    cfg["datasets"][d["dataset"]]["tables"] = d["tables"]
    cfg["paths"].update(artifacts_root=str(art), reports_root=str(art / "reports"))
    cfg["memory"]["db"] = str(art / f"memory_s{seed}.db")
    cfg["run"]["max_rounds"], cfg["run"]["seed"] = SETTINGS["max_rounds"], seed
    cfg["split"]["seed"] = SETTINGS["split_seed"]
    cfg["fidelity"]["low"].update(sample_frac=SETTINGS["low_sample_frac"], n_trials=SETTINGS["low_trials"])
    cfg["fidelity"]["full"]["n_trials"] = SETTINGS["full_trials"]
    # --jobs 并行时各进程分核，不然抢 CPU 把预算耗光（v11 第一次并行，hotel s0 因超预算失败）
    cfg["inner_loop"]["n_threads"] = max(1, cfg["inner_loop"]["n_threads"] // int(os.environ.get("DATAMINER_EVAL_JOBS", "1")))
    return cfg


def drive(o, answers: dict | None = None, auto_answers: dict | None = None, max_answers: int = 3):
    """跑到结束或停下问人。answers：这次续跑带的回答。auto_answers：情景用例的模拟回答——停在缺规格时自动答，
    最多 max_answers 次（推荐答案缺失时"确认"不被采纳，不能无限答下去）。"""
    if answers and o.state == State.AWAIT_HUMAN:
        o.resume_with({"answers": answers})
    n = 0
    while True:
        while o.state not in TERMINAL and o.state != State.AWAIT_HUMAN:
            o.step()
        w = o.p.get("await") or {}
        if o.state == State.AWAIT_HUMAN and auto_answers and n < max_answers and w.get("reason") == "spec_incomplete":
            o.resume_with({"answers": {k: v for k, v in auto_answers.items() if k in w["missing"]}})
            n += 1
            continue
        return o.state


def _run(cfg: dict, ds: str, tid: str, art: Path, llm, tracer, answers=None, auto_answers=None):
    """一次运行：驱动 → 收集事件和记录 → 行（结果层字段 + 过程指标）。run_agent 和 run_scenario 共用。"""
    row = {"task_id": tid, "status": "ok", "error": None, "final_model": None,
           "trace_id": getattr(getattr(tracer, "root", None), "trace_id", None)}
    o = None
    try:
        o = Orchestrator(TaskSpec.from_knowhow(ds, cfg, autonomy=SETTINGS["autonomy"]), cfg, llm, FinalGate(ds, cfg, tid), tid, tracer=tracer)
        drive(o, answers, auto_answers)
        if o.state == State.AWAIT_HUMAN:          # 要问人：不判失败，停下等回答（检查点已存，带 answers 续跑）
            qs = (o.p["await"] or {}).get("questions") or []
            row.update(status="awaiting_human", questions=qs,
                       error="等待人工回复：" + "、".join(q["key"] if isinstance(q, dict) else str(q) for q in qs) or str(o.p["await"]))
        elif o.state != State.DONE:
            row.update(status="failed", error=f"运行结束于 {o.state.value}")
    except Exception as e:
        row.update(status="failed", error=redact(f"{type(e).__name__}: {str(e)[:300]}"))
    events = EventLog(str(art / "state.db"), tid).query()
    records = [r.model_dump(mode="json") for r in Store(str(art / "state.db")).all(tid)]
    if row["status"] == "failed":            # Orchestrator 自己接住异常进入 FAILED 时，原始错误只在事件里：补上最后一条
        last = next((e["payload"].get("error") or "；".join(e["payload"]["errors"]) for e in reversed(events)
                     if (e["payload"] or {}).get("error") or (e["payload"] or {}).get("errors")), None)
        if last and last not in row["error"]:
            row["error"] += f"：{last}"
    row |= process_stats(events)
    if o and row["final_exp"]:
        row["final_model"] = o.store.get(row["final_exp"]).config.get("model")
    price = cfg["llm"].get("prices", {}).get(getattr(llm, "model", None))
    row["usage"] = usage_totals(events)
    row["cost_usd"] = cost_usd(row["usage"], price)
    row["process"] = process_metrics(events, records, cfg["run"]["max_rounds"], price)
    return row, events, records


def run_agent(name: str, label: str, seed: int, provider: str, llm=None, tracer=None, answers: dict | None = None) -> dict:
    """answers：上次停在等人回复（awaiting_human）时，带着回答从检查点续跑；key 是问题的 key，值是"确认"或人的原话。"""
    cfg, ds = build_cfg(name, label, seed), DATASETS[name]["dataset"]
    tid = f"{label}_{name}_s{seed}"
    llm = llm or make_llm(provider, cfg)
    tracer = tracer or make_tracer(tid, {"dataset": name, "llm": provider, "source": f"eval:{label}"})
    row, _, _ = _run(cfg, ds, tid, art_dir(label, name, seed), llm, tracer, answers)
    return {"dataset": name, "seed": seed, **row}


B1_BUDGET_SEC = 300.0     # B1 固定预算：与 agent 代码无关，按数据集缓存，换版本直接复用


def run_baselines(name: str, label: str) -> dict:
    """B0、B1 与 agent 代码无关：按数据集 + 切分种子 + B1 预算缓存（v9/v10：hotel 的 B1 每个版本重跑 7 分钟以上，
    预算还随 agent 用时变，"比 B1"跟着抖）。数据或基线代码改了，删掉 RUNS/_baselines 下对应文件即可重算。"""
    cache = RUNS / "_baselines" / f"{name}_split{SETTINGS['split_seed']}_b1{int(B1_BUDGET_SEC)}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    prepare_run(label, name, "baselines")
    cfg, ds = build_cfg(name, label, 0, "baselines"), DATASETS[name]["dataset"]
    build_splits(ds, cfg)                           # 基线目录独立：切分自己生成（与 agent 同一切分种子，结果一致）
    out = {"b0": b0_default_lgbm.run(ds, cfg, name=f"b0_{label}_{name}"),
           "b1": b1_automl.run(ds, cfg, B1_BUDGET_SEC, name=f"b1_{label}_{name}")}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out, ensure_ascii=False, default=float))
    return out


SCENARIO_SETTINGS = {"max_rounds": 5, "n_rows": 20_000, "seed": 0}     # 情景运行求快：抽 2 万行、1 个种子、5 轮


def run_scenario(name: str, base: str, label: str, provider: str, params: dict | None = None, llm=None, tracer=None) -> dict:
    """在底座数据集 base（DATASETS 里的评测名）上构造情景变体并跑一次，检查 agent 的反应。产物放在 <版本>/<底座>/scn_<情景>/。"""
    from eval.scenarios import SCENARIOS
    sc, params = SCENARIOS[name], {**SCENARIO_SETTINGS, **(params or {})}
    sub, ds = f"scn_{name}", DATASETS[base]["dataset"]
    prepare_run(label, base, sub)
    cfg, art = build_cfg(base, label, params["seed"], sub), art_dir(label, base, sub)
    tid = f"{label}_{sub}_{base}"
    out = {"scenario": name, "base": base, "task_id": tid}

    def broken(e):                           # build / prepare / check 不在 _run 的保护里：出错只记这一个情景不通过，不让整次评测崩掉（终审 Important 1）
        return {**out, "status": "failed", "error": redact(f"{type(e).__name__}: {str(e)[:300]}"), "passed": False,
                "reason": f"构造/检查出错：{type(e).__name__}: {str(e)[:200]}", "evidence": [], "usage": usage_totals([]),
                "cost_usd": None, "wall_sec": 0.0, "process": {}}
    try:
        v = sc.build(ds, cfg, art / "variant", params)
        cfg["datasets"][ds]["tables"], cfg["paths"]["knowhow_root"] = v.tables, v.knowhow_root
        cfg["run"]["max_rounds"] = params["max_rounds"]
        if v.prepare:
            v.prepare(cfg)
    except Exception as e:
        return broken(e)
    llm = llm or make_llm(provider, cfg)
    tracer = tracer or make_tracer(tid, {"dataset": base, "llm": provider, "source": f"eval:{label}:scenario"})
    row, events, records = _run(cfg, ds, tid, art, llm, tracer, auto_answers=v.answers)
    if row["status"] == "failed":            # 运行本身出错时 check 的结论没有意义（会误报"没有停下问人"之类），直接报运行失败
        return {**out, **row, "passed": False, "reason": f"运行失败：{row['error']}", "evidence": []}
    try:
        return {**out, **row, **sc.check(events, records, params)}
    except Exception as e:
        return {**broken(e), **{k: row[k] for k in ("usage", "cost_usd", "wall_sec", "process")}}

def pending_scenarios(tier_scenarios: list[dict], done: list[dict], resuming: bool) -> list[dict]:
    """续跑（--resume / --answer）时只跑还没有正常结果的情景，不把四个情景全部清掉重跑、再花一次钱（终审 Important 2）。"""
    if not resuming:
        return list(tier_scenarios)
    ok = {(r["scenario"], r["base"]) for r in done if r.get("status") == "ok"}
    return [s for s in tier_scenarios if (s["name"], s["base"]) not in ok]
