"""状态机主循环（§4.5）：while not terminal: state = step(state)，每步结束写检查点。
LLM 只出现在 PLAN / DECIDE；EXECUTE / EVALUATE / RECORD / STOP_CHECK / FINAL_GATE 全是代码。
final_gate 由外部注入（agent/ 下的代码不导入它，见 §4.0-1）。"""
import json
import time
from pathlib import Path

import optuna
from pydantic import ValidationError

from report.model_card import build_report

from agent.actions import allowed_models, apply_action, config_hash, model_profiles
from agent.decide import DecideFailed, Validator, decide
from agent.llm import TracedLLM, extract_json
from agent.schemas import Plan, TaskSpec
from agent.states import State as S, TRANSITIONS, TERMINAL, legal
from data.access import DataAccess
from data.knowhow import load_knowhow
from evaluation.evaluator import Evaluator, scan_features
from evaluation.metrics import primary
from evaluation.oot_budget import OOTBudgetExhausted
from evaluation.stop import stop_check
from memory.consolidate import Proposals, admit, allowed_keys, update_lifecycle, validate
from memory.retrieval import active_lessons, build_context, nearest_task
from memory.schema import Cost, ExperimentRecord
from memory.store import MemoryStore, Store
from modeling.inner_loop import date_spec
from runtime import budget as bud
from runtime.checkpoint import Checkpoint
from runtime.events import EventLog
from runtime.tracing import NoopTracer
from tools.query_data import investigate
from tools.run_experiment import run_experiment

PLAN_SYSTEM = (Path(__file__).parent / "prompts" / "plan.md").read_text()
CONS_SYSTEM = (Path(__file__).parent / "prompts" / "consolidate.md").read_text()
META = ("_row_id", "_label", "_split_time", "_obs_time")


class Orchestrator:
    def __init__(self, spec: TaskSpec, cfg: dict, llm, final_gate, task_id: str, crash_hook=None, tracer=None):
        """tracer：可选的 Langfuse 上报（runtime/tracing.py）；默认不上报。"""
        self.cfg, self.llm, self.final_gate, self.crash_hook = cfg, llm, final_gate, crash_hook
        self.tracer = tracer or NoopTracer()
        art = Path(cfg["paths"]["artifacts_root"])
        art.mkdir(parents=True, exist_ok=True)
        self.art, db = art, str(art / "state.db")
        self.task_id = task_id
        self.events, self.ckpt, self.store = EventLog(db, task_id, self.tracer.on_event), Checkpoint(db, task_id), Store(db)
        self.llm = TracedLLM(llm, self.events.append, lambda: self.state.value, self.tracer)
        self.evaluator_db = db
        self.mem = MemoryStore(cfg["memory"]["db"])
        saved = self.ckpt.load()
        if saved:
            self.state, self.p, _ = S(saved[0]), saved[1], saved[2]
        else:
            self.state = S.INTAKE
            self.p = {"task_id": task_id, "spec": spec.model_dump(), "all_features": [], "directions": [], "race": [],
                      "current_node": None, "best_exp_id": None, "round_no": 0, "history_verdicts": [],
                      "budget": bud.new_budget(spec.budget.model_dump()), "handled_leaks": [], "quarantined": [],
                      "pending": None, "pending_leaks": {}, "investigate_counts": {}, "investigations": {},
                      "seq": 0, "stop_reason": None, "final": None, "resume_state": None, "await": None, "last_codes": []}
        self.spec = TaskSpec(**self.p["spec"])
        cfg["metric"] = self.spec.metric.model_dump()     # 口径以检查点里的 spec 为准：任务开始前锁定，恢复运行也不变
        self._evaluator = None

    # ---------- 主循环 ----------
    def run(self, max_steps: int = 10_000):
        n = 0
        while self.state not in TERMINAL and self.state != S.AWAIT_HUMAN and n < max_steps:
            self.step()
            n += 1
        return self.state

    def step(self):
        t0 = time.time()
        self.tracer.begin_state(self.state.value)
        try:
            nxt = getattr(self, f"_{self.state.value.lower()}")()
            assert legal(self.state, nxt), f"非法转换 {self.state} → {nxt}"
            if self.crash_hook:
                self.crash_hook(self.state.value)          # 测试用：模拟在状态中途被打断（工作做完、检查点未写）
        except BaseException as e:
            self.tracer.finish(self.state.value, error=f"{type(e).__name__}: {str(e)[:300]}")
            raise
        eid = self.events.append("state_transition", {"from": self.state.value, "to": nxt.value, "sec": round(time.time() - t0, 2)})
        self.state = nxt
        self.ckpt.save(self.state.value, self.p, eid)
        if nxt in TERMINAL or nxt == S.AWAIT_HUMAN:
            self.tracer.finish(nxt.value, {"best_exp_id": self.p["best_exp_id"], "stop_reason": self.p["stop_reason"],
                                           "final": self.p["final"]})

    # ---------- 辅助 ----------
    @property
    def evaluator(self):
        if self._evaluator is None:
            self._evaluator = Evaluator(self.cfg, self.spec.dataset, self.task_id, self.evaluator_db)
        return self._evaluator

    def _remaining(self):
        return bud.remaining(self.p["budget"])

    def _result(self, exp_id):
        return json.loads((self.art / "experiments" / exp_id / "result.json").read_text())

    def _next_exp_id(self):
        self.p["seq"] += 1
        return f"{self.task_id}_e{self.p['seq']:02d}"

    def _llm_json(self, system, user):
        text, toks = self.llm.complete(system, user, 0.0)
        bud.spend(self.p["budget"], tokens=toks)
        return extract_json(text)

    def _warm_points(self, parent_id, config):
        """§4.10：特征集不变 → 上一轮 top10 trial；模型或特征集变了 → 只注入最优参数一个点；换了模型族则不注入。"""
        if not parent_id:                                     # 第 1 轮（模型赛跑）：只有跨任务 Prior；冷启动则为空
            pr_ = self.p.get("prior")
            return pr_["params"] if pr_ and pr_["model"] == config["model"] else []
        pr = self._result(parent_id)
        if pr["config"]["model"] != config["model"]:
            return []
        if pr["config"]["features"] == config["features"]:
            try:
                st = optuna.load_study(study_name=parent_id, storage=f"sqlite:///{self.art / 'optuna.db'}")
                done = [t for t in st.trials if t.value is not None]
                return [t.params for t in sorted(done, key=lambda t: -t.value)[:10]]
            except Exception:
                pass
        return [pr["best_params"]]

    def _train(self, exp_id, config, parent_id):
        """幂等：结果已在磁盘上就直接复用，不重复计算（resume）。"""
        if (self.art / "experiments" / exp_id / "result.json").exists():
            return {"status": "OK", **self._result(exp_id)}
        r = run_experiment(exp_id, self.spec.dataset, config["model"], self.cfg,
                           remaining_cpu_minutes=self._remaining()["cpu_minutes"], features=config["features"],
                           feature_sets=config.get("feature_sets"),
                           space=config.get("space"), fidelity=config["fidelity"], n_trials=config.get("n_trials"),
                           warm_start=self._warm_points(parent_id, config), monotone=config.get("monotone", False))
        return r

    def _write_record(self, exp_id, parent, hypothesis, action, diff, config, result, verdict, codes):
        m = (result or {}).get("metrics", {})
        self.store.add(ExperimentRecord(
            exp_id=exp_id, task_id=self.task_id, parent_exp_id=parent, hypothesis=hypothesis, action_type=action, diff=diff,
            config=config or {}, config_hash=config_hash(config) if config else "", fidelity=(config or {}).get("fidelity", "none") if config else "none",
            metrics=m, diagnosis_codes=codes, verdict=verdict,
            cost=Cost(cpu_minutes=(result or {}).get("cost", {}).get("wall_minutes", 0) * self.cfg["inner_loop"]["n_threads"],
                      wall_minutes=(result or {}).get("cost", {}).get("wall_minutes", 0)),
            artifact_refs={"result": f"experiments/{exp_id}/result.json"} if result else {}))

    # ---------- 状态处理 ----------
    def _intake(self):
        self.events.append("spec_locked", {"metric": self.spec.metric.model_dump()})   # 页面据此知道本次运行的主指标
        miss = self.spec.missing_required()
        if miss:
            self.p["await"] = {"reason": "spec_incomplete", "missing": miss,
                               "questions": [f"请提供 {m}（禁止 agent 自行假设）" for m in miss]}
            return S.CLARIFY
        return S.PROFILE

    def _clarify(self):
        self.events.append("clarify_questions", self.p["await"])
        return self._await_human(S.INTAKE)

    def _await_human(self, resume_state: S):
        self.p["resume_state"] = resume_state.value
        self.events.append("await_human", self.p["await"] or {})
        return S.AWAIT_HUMAN

    def resume_with(self, reply: dict):
        """人工回复：spec 更新 / 放行特征 / 隔离特征。写 USER_INSTRUCTION 事件，然后回到 resume_state 重新规划（§4.14）。"""
        assert self.state == S.AWAIT_HUMAN
        self.events.append("USER_INSTRUCTION", reply)
        if "spec" in reply:
            self.p["spec"] = {**self.p["spec"], **{k: v for k, v in reply["spec"].items() if k != "metric"}}   # 口径已锁定，人工回复不能改
            self.spec = TaskSpec(**self.p["spec"])
            banned = set(self.spec.constraints.banned_models)
            if banned:
                inv = self.store.invalidate(self.task_id, lambda r: r.config.get("model") in banned)
                self.events.append("invalidated", {"exp_ids": inv})
        self.p["handled_leaks"] += reply.get("approve", [])
        self.p["quarantined"] += reply.get("quarantine", [])
        for f in reply.get("approve", []) + reply.get("quarantine", []):
            self.p["pending_leaks"].pop(f, None)
        self.p["pending_leaks"] = {}
        self.p["await"] = None
        self.state = S(self.p["resume_state"])
        self.ckpt.save(self.state.value, self.p, self.events.append("resumed", {"to": self.state.value}))

    def _profile(self):
        art = self.art
        ds = self.spec.dataset
        if not (art / "splits" / ds / "train.parquet").exists():
            from data.splits import build_splits
            rep = build_splits(ds, self.cfg)
            if any(w["code"] == "DATA_FATAL" for w in rep["warnings"]):
                self.p["await"] = {"reason": "DATA_FATAL", "warnings": rep["warnings"]}
                self.events.append("data_fatal", self.p["await"])
                return S.FAILED
        tr = DataAccess(ds, str(art)).load("train")
        self.p["all_features"] = [c for c in tr.columns if c not in META]
        fp = self._fingerprint(tr)
        self.p["fingerprint"] = fp
        self.events.append("profile", {k: fp[k] for k in ("n_samples", "bad_rate", "n_features", "missing_rate", "time_span_months")})
        self.p["prior"], self.p["memory_ctx"] = None, None
        if not self.cfg.get("ablation", {}).get("no_memory"):           # A2：去掉 memory
            near, dist = nearest_task(self.mem, fp, self.cfg)
            lessons = active_lessons(self.mem, fp, self.cfg) if near else []   # 不相似 → 冷启动：既不复用 prior 也不复用 Lesson
            self.events.append("memory_retrieval", {"nearest": near["fingerprint"]["task_id"] if near else None,
                                                    "distance": dist, "cold_start": near is None, "n_active_lessons": len(lessons)})
            if near and near["prior"]:
                self.p["prior"] = {**near["prior"], "distance": dist}
            if near:
                self.p["memory_ctx"] = {"prior": {k: self.p["prior"][k] for k in ("from_task", "distance", "model")} if self.p["prior"] else None,
                                        "lessons": [{"key": l["key"], "statement": l["statement"], "polarity": l["polarity"],
                                                     "confidence": l["confidence"]} for l in lessons]}
        return S.PLAN

    def _fingerprint(self, tr) -> dict:
        from data.knowhow import field_table
        kh = load_knowhow(self.spec.dataset, self.cfg)
        fields, _, _ = field_table(kh, kh["dictionary"].get("label", {}).get("source_table"),
                                   self.cfg["task_specs"].get(self.spec.dataset, {}).get("derived_expr_overrides"))
        f = self.p["all_features"]
        tcol = "_split_time" if "_split_time" in tr else None
        span = int((tr[tcol].max().year - tr[tcol].min().year) * 12 + tr[tcol].max().month - tr[tcol].min().month) if tcol else 0
        return {"task_id": self.task_id, "n_samples": int(len(tr)), "n_features": len(f), "bad_rate": float(tr["_label"].mean()),
                "missing_rate": float(tr[f].isna().mean().mean()), "time_span_months": span, "label_def": self.spec.label_def or "",
                "feature_domains": sorted({fields.get(c, {}).get("availability", "application") for c in f})}

    def _plan(self):
        tr = DataAccess(self.spec.dataset, str(self.art)).load("train")
        dates = set(date_spec(self.spec.dataset, self.cfg)[0])                    # 文本形式的日期字段不算类别变量
        cats = [c for c in self.p["all_features"] if c not in dates and (tr[c].dtype == object or str(tr[c].dtype) == "category")]
        prof = {**{k: self.p["fingerprint"][k] for k in ("n_samples", "bad_rate", "missing_rate", "time_span_months")},
                "n_features": len(self.p["all_features"]), "n_categorical": len(cats),
                "max_cardinality": int(max((tr[c].nunique() for c in cats), default=0))}
        plan = Plan(directions=[])
        try:
            ctx = {"mode": "PLAN", "spec": self.spec.model_dump(mode="json"), "data_profile": prof, "models": model_profiles(self.spec)}
            plan = Plan.model_validate(self._llm_json(PLAN_SYSTEM, "```json\n" + json.dumps(ctx, ensure_ascii=False) + "\n```"))
            self.p["directions"] = [d.model_dump() for d in plan.directions]
        except (ValidationError, ValueError) as e:
            self.events.append("plan_invalid", {"error": str(e)[:200]})      # 规划失败不致命：没有方向也能靠 DECIDE 继续
        # 模型赛跑（§4.12）：LLM 提名候选，代码按约束过滤后在同特征集、同预算、低保真下各跑一个 study；没提名则全部赛跑
        allowed = allowed_models(self.spec)
        race_models = list(dict.fromkeys(c.model for c in plan.candidate_models if c.model in allowed)) or allowed
        self.events.append("plan_candidates", {"candidates": [c.model_dump() for c in plan.candidate_models], "raced": race_models})
        for i, m in enumerate(race_models):
            exp_id = f"{self.task_id}_race_{m}"
            cfg = {"model": m, "features": self.p["all_features"], "space": None, "fidelity": "low", "n_trials": None}
            res = self._train(exp_id, cfg, None)
            if res["status"] != "OK":
                continue
            base = self._result(self.p["race"][0]) if self.p["race"] else None
            out = self._evaluate_result(res, base, guard=False)
            self._write_record(exp_id, None, f"模型赛跑：{m}", "RACE", {"model": m}, cfg, res, out["verdict"], out["codes"])
            bud.spend(self.p["budget"], cpu_minutes=res["cost"]["wall_minutes"] * self.cfg["inner_loop"]["n_threads"],
                      wall_minutes=res["cost"]["wall_minutes"])
            if exp_id not in self.p["race"]:
                self.p["race"].append(exp_id)
            self.p["pending_leaks"].update(out["leaks"])
        pm = primary(self.cfg)
        best_race = max(self.p["race"], key=lambda e: self._result(e)["metrics"][f"oot_dev_{pm}"]) if self.p["race"] else None
        self.p["current_node"] = best_race
        try:
            from features.factory import load_templates
            self.p["templates"] = list(load_templates(self.spec.dataset, self.cfg))
        except FileNotFoundError:
            self.p["templates"] = []
        return S.DECIDE

    def _registered_features(self, res):
        """按来源分：模板产出（可信，走正常筛选）vs run_code 产出（一律可疑，见 tools/run_code.py）。"""
        fs = res["config"].get("feature_sets") or []
        if not fs:
            return set(), set()
        from features.registry import Registry
        reg = Registry(self.cfg)
        known, suspect = set(), set()
        for x in fs:
            m = reg.get(x)
            (suspect if m["source"] == "run_code" else known).update(m["kept"])
        return known, suspect

    def _evaluate_result(self, res, base_result, guard=True):
        kh = load_knowhow(self.spec.dataset, self.cfg)
        tr = DataAccess(self.spec.dataset, str(self.art)).load("train")
        feats = res["config"]["features"]
        scan = {} if self.cfg.get("ablation", {}).get("no_guardrail") else scan_features(tr, feats, kh, self.cfg, self.spec.dataset, *self._registered_features(res))
        try:
            out = self.evaluator.evaluate(res, base_result, scan, history_verdicts=self.p["history_verdicts"],
                                          remaining_budget_frac=bud.remaining_frac(self.p["budget"]),
                                          handled_leaks=self.p["handled_leaks"], guard=guard)
        except OOTBudgetExhausted:
            out = {"verdict": "INCONCLUSIVE", "diagnosis_codes": ["OOT_BUDGET_EXHAUSTED"], "diagnosis_details": {}}
        v = out["verdict"]
        out["verdict"] = getattr(v, "value", v)
        out["codes"] = out["diagnosis_codes"]
        out["leaks"] = out.get("diagnosis_details", {}).get("LEAK_SUSPECT", {})
        m, cmp = res.get("metrics", {}), out.get("compare")
        self.events.append("evaluated", {"verdict": out["verdict"], "codes": out["codes"],
                                         "guardrail_failures": out.get("guardrail_failures", []),
                                         "model": res["config"].get("model"), "fidelity": res["config"].get("fidelity"),
                                         "metrics": {k: v for k, v in m.items() if k.startswith(("valid_", "oot_dev_")) or k == "score_psi"},
                                         "baseline": base_result["exp_id"] if base_result else None,
                                         "compare": {k: cmp.get(k) for k in ("delta", "ci_low", "ci_high", "alpha", "significant", "k")} if cmp else None},
                           exp_id=res.get("exp_id"))
        return out

    def _decide(self):
        p = self.p
        if p["current_node"] is None:                                   # 预算不够跑任何实验：直接收尾，不让 LLM 空转
            p["stop_reason"] = "NO_RUNNABLE_EXPERIMENT（预算不足）"
            self.events.append("stop", {"reason": p["stop_reason"]})
            return S.FINAL_GATE
        if p["pending_leaks"] and self.spec.autonomy == "L0":           # L0：自动隔离并记录（§4.15），不经 LLM
            p["quarantined"] += list(p["pending_leaks"])
            self.events.append("auto_quarantine", p["pending_leaks"])
            p["pending_leaks"] = {}
        recs = {r.exp_id: r for r in self.store.all(self.task_id)}
        n_rows = len(DataAccess(self.spec.dataset, str(self.art)).load("train"))
        ctx = build_context(self.store, p, self.cfg, self.spec, self.evaluator.budget, self._remaining(), model_profiles(self.spec))
        v = Validator(self.cfg, self.spec, self.store, p, self._remaining(), n_rows)
        try:
            d, toks = decide(self.llm, ctx, v, self.cfg, on_event=lambda t, pl: self.events.append(t, pl))
        except DecideFailed as e:
            p["await"] = {"reason": "decide_failed", "errors": e.errors}
            if self.spec.autonomy == "L0":
                self.events.append("decide_failed", p["await"])
                return S.FAILED
            return self._await_human(S.DECIDE)
        bud.spend(p["budget"], tokens=toks)
        node = None
        if p["current_node"]:
            n = recs[p["current_node"]]
            node = {"exp_id": n.exp_id, "config": n.config}
        kind, new_cfg, parent = apply_action(d.action.value, d.params, node, recs, self.spec, p["quarantined"])
        exp_id = self._next_exp_id()
        p["pending"] = {"decision": d.model_dump(mode="json"), "kind": kind, "config": new_cfg, "parent": parent, "exp_id": exp_id}
        if kind == "stop":
            p["stop_reason"] = "LLM_STOP: " + str(d.params.get("reason", ""))
            self.events.append("stop", {"reason": p["stop_reason"]})
            p["pending"] = None
            return S.FINAL_GATE
        if kind == "escalate":
            p["await"] = {"reason": "escalate_human", **d.params}
            p["pending"] = None
            return self._await_human(S.DECIDE)
        return S.EXECUTE

    def _execute(self):
        pd = self.p["pending"]
        if pd["kind"] == "expand":
            node = self.store.get(self.p["current_node"])
            try:
                if pd["config"]["code"]:
                    from tools.run_code import RunCodeError, run_code
                    gen = run_code(pd["config"]["code"], self.spec.dataset, self.cfg)
                else:
                    from tools.generate_features import generate_features
                    gen = generate_features(pd["config"]["template_id"], self.spec.dataset, self.cfg)
            except Exception as e:
                pd["result"] = {"status": "EXPAND_FAILED", "error": f"{type(e).__name__}: {str(e)[:400]}"}
                return S.RECORD
            pd["expand"] = {"feature_set_id": gen["feature_set_id"], "kept": gen["kept"],
                            "leak_suspect": list(gen.get("leak_suspect", {}))}
            pd["kind"] = "train"
            pd["config"] = {**node.config, "features": sorted(set(node.config["features"]) | set(gen["kept"])),
                            "feature_sets": (node.config.get("feature_sets") or []) + [gen["feature_set_id"]],
                            "fidelity": "low", "n_trials": None}
        if pd["kind"] == "train":
            pd["result"] = self._train(pd["exp_id"], pd["config"], pd["parent"])
            return S.EVALUATE if pd["result"]["status"] == "OK" else S.RECORD
        if pd["kind"] == "nav":
            self.p["current_node"] = pd["parent"]
        if pd["kind"] == "investigate":
            node = self.store.get(self.p["current_node"])
            q = pd["decision"]["params"]["question"]
            pd["investigation"] = {q: investigate(self.spec.dataset, self.cfg, q, node.config["features"])}
        return S.RECORD

    def _evaluate(self):
        pd = self.p["pending"]
        res = pd["result"]
        best = self._result(self.p["best_exp_id"]) if self.p["best_exp_id"] else None
        pd["eval"] = self._evaluate_result(res, best)
        return S.RECORD

    def _record(self):
        p, pd = self.p, self.p["pending"]
        d, res, ev = pd["decision"], pd.get("result"), pd.get("eval")
        ok = res is not None and res.get("status") == "OK"
        verdict = ev["verdict"] if ok else "NONE"
        codes = ev["codes"] if ok else ([res["status"]] if res else [])
        diff = dict(d["params"])
        if pd.get("expand"):
            diff["expand"] = pd["expand"]
        if pd.get("investigation"):
            diff["result"] = pd["investigation"]
            node = str(p["current_node"])
            p["investigate_counts"][node] = p["investigate_counts"].get(node, 0) + 1
            p["investigations"][node] = {**p["investigations"].get(node, {}), **pd["investigation"]}
        self._write_record(pd["exp_id"], pd["parent"] if pd["kind"] != "nav" else p["current_node"], d["hypothesis"],
                           d["action"], diff, pd["config"] if pd["kind"] == "train" else None, res if ok else None, verdict, codes)
        if ok:
            bud.spend(p["budget"], cpu_minutes=res["cost"]["wall_minutes"] * self.cfg["inner_loop"]["n_threads"],
                      wall_minutes=res["cost"]["wall_minutes"])
            p["history_verdicts"].append(verdict)
            p["last_codes"] = codes
            if ev["leaks"]:
                p["pending_leaks"] = ev["leaks"]
            if verdict == "ACCEPT":
                p["best_exp_id"] = p["current_node"] = pd["exp_id"]
        p["round_no"] += 1
        self.events.append("recorded", {"verdict": verdict, "codes": codes, "action": d["action"]}, exp_id=pd["exp_id"])
        p["pending"] = None
        return S.STOP_CHECK

    def _stop_check(self):
        p = self.p
        best = self._result(p["best_exp_id"]) if p["best_exp_id"] else None
        reason = stop_check(cfg=self.cfg, oot_budget=self.evaluator.budget, round_no=p["round_no"],
                            max_rounds=self.cfg["run"]["max_rounds"], remaining=self._remaining(),
                            history_verdicts=p["history_verdicts"],
                            best_score=best["metrics"][f"oot_dev_{primary(self.cfg)}"] if best else None,
                            target_value=self.spec.target_value)
        if reason:
            p["stop_reason"] = reason
            self.events.append("stop", {"reason": reason})
            return S.FINAL_GATE
        return S.DECIDE

    def _final_gate(self):
        p = self.p
        if p["best_exp_id"]:
            p["final"] = self.final_gate.run_once(self._result(p["best_exp_id"]))
            self.events.append("final_gate", p["final"], exp_id=p["best_exp_id"])
        else:
            p["final"] = None
            self.events.append("final_gate_skipped", {"why": "没有通过 evaluator 的全保真度实验"})
        return S.REPORT

    def _report(self):
        """最小汇总（完整模型卡与实验树在第 7 项）。数据全部来自代码，不经 LLM。"""
        p = self.p
        recs = self.store.all(self.task_id)
        out = {"task_id": self.task_id, "dataset": self.spec.dataset, "stop_reason": p["stop_reason"], "final_gate": p["final"],
               "best_exp_id": p["best_exp_id"], "budget": p["budget"], "quarantined_features": p["quarantined"],
               "oot_dev_used": self.evaluator.budget.used,
               "experiments": [{"exp_id": r.exp_id, "parent": r.parent_exp_id, "action": r.action_type, "verdict": r.verdict,
                                "fidelity": r.fidelity, "oot_dev_auc": r.metrics.get("oot_dev_auc"), "codes": r.diagnosis_codes,
                                "invalidated": r.invalidated} for r in recs]}
        rd = Path(self.cfg["paths"]["reports_root"])
        rd.mkdir(exist_ok=True)
        (rd / f"summary_{self.task_id}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
        rep = build_report(self.cfg, self.spec, self.store, p, p["final"], self.evaluator.budget.used, self.llm)
        self.events.append("report_built", {"model_card": rep["model_card"], "n_risks": len(rep["risks"]), "narrative": bool(rep["narrative"])})
        return S.CONSOLIDATE

    def _consolidate(self):
        """LLM 提出候选 Lesson → 代码校验准入 → 规则推进生命周期 → 保存任务指纹与 Prior（§4.13）。"""
        p = self.p
        recs = self.store.all(self.task_id)
        allowed = allowed_keys(recs)
        ctx = {"mode": "CONSOLIDATE", "allowed_keys": allowed, "verdicts": {r.exp_id: r.verdict for r in recs},
               "experiments": [{"exp_id": r.exp_id, "action": r.action_type, "model": r.config.get("model"), "verdict": r.verdict,
                                "codes": r.diagnosis_codes} for r in recs if r.config]}
        try:
            props = Proposals.model_validate(self._llm_json(CONS_SYSTEM, "```json\n" + json.dumps(ctx, ensure_ascii=False) + "\n```"))
        except Exception as e:          # 收尾阶段：LLM 出错（含断网）不应让已完成的运行失败；本次只是不产出经验
            props = Proposals(lessons=[])
            self.events.append("consolidate_failed", {"error": f"{type(e).__name__}: {str(e)[:200]}"})
        ok, dropped = validate(props, recs, allowed)
        ids = admit(self.mem, ok, recs, self.task_id, p["fingerprint"], self.cfg)
        changes = update_lifecycle(self.mem, self.task_id, recs, p["best_exp_id"], p["final"], self.cfg)
        prior = None
        if p["best_exp_id"]:
            best = self._result(p["best_exp_id"])
            pts = self._warm_points(p["best_exp_id"], best["config"])[: self.cfg["memory"]["prior_top_k"]]
            prior = {"from_task": self.task_id, "model": best["config"]["model"], "params": pts}
        self.mem.put_task(p["fingerprint"], prior, p["final"])
        self.events.append("consolidated", {"proposed": len(props.lessons), "admitted": len(ids), "dropped": dropped,
                                            "lifecycle_changes": {k: list(v) for k, v in changes.items()}})
        return S.DONE
