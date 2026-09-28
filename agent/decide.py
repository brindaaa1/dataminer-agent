"""DECIDE 节点：上下文渲染 + LLM 调用 + pydantic 校验 + 业务规则校验；校验失败回灌错误重试（§4.7, §4.0-3）。"""
import json
from pathlib import Path

from pydantic import ValidationError

from agent.actions import ActionError, apply_action, config_hash
from agent.llm import extract_json
from agent.schemas import ActionType, Decision
from evaluation.guardrails import POLICY_PRIOR, decision_codes
from tools.run_experiment import estimate_cost_minutes

SYSTEM = (Path(__file__).parent / "prompts" / "decide.md").read_text()


class DecideFailed(RuntimeError):
    def __init__(self, errors):
        super().__init__("; ".join(errors))
        self.errors = errors


def render(ctx: dict, feedback: str | None = None) -> str:
    s = "以下是当前上下文（JSON）。请输出决策。\n\n```json\n" + json.dumps(ctx, ensure_ascii=False, indent=1, default=str) + "\n```\n"
    return s + (f"\n上一次输出未通过校验，请修正：\n{feedback}\n" if feedback else "")


class Validator:
    """校验规则（§4.7）：参数合法、成本在预算内、不在禁忌表、满足前置条件、有未处理泄漏时只能 ESCALATE。"""

    def __init__(self, cfg, spec, store, p, remaining, n_rows_train):
        self.cfg, self.spec, self.store, self.p, self.remaining, self.n_rows = cfg, spec, store, p, remaining, n_rows_train

    def check(self, d: Decision) -> list[str]:
        errs, a = [], d.action.value
        if self.p["pending_leaks"] and a != "ESCALATE_HUMAN":
            return ["存在未处理的 LEAK_SUSPECT，只允许 ESCALATE_HUMAN"]
        if a == "ESCALATE_HUMAN" and self.spec.autonomy == "L0":
            return ["L0 无人值守模式不能请求人工"]
        if d.est_cost.cpu_minutes > self.remaining["cpu_minutes"] or d.est_cost.tokens > self.remaining["tokens"]:
            errs.append(f"est_cost 超出剩余预算 {self.remaining}")
        recs = {r.exp_id: r for r in self.store.all(self.p["task_id"])}
        node = None
        if self.p["current_node"]:
            n = recs[self.p["current_node"]]
            node = {"exp_id": n.exp_id, "config": n.config}
        try:
            kind, new_cfg, _ = apply_action(a, d.params, node, recs, self.spec, self.p["quarantined"])
        except ActionError as e:
            return errs + [str(e)]
        if kind == "train":
            h = config_hash(new_cfg)
            if h in self.store.taboo_hashes(self.p["task_id"]):
                errs.append("该配置已经试过（禁忌表），请换一个改动")
            fid = self.cfg["fidelity"][new_cfg["fidelity"]]
            need = estimate_cost_minutes(self.n_rows * fid["sample_frac"], len(new_cfg["features"]),
                                         new_cfg.get("n_trials") or fid["n_trials"], self.cfg)
            if need > self.remaining["cpu_minutes"]:
                errs.append(f"代码估算成本 {need:.1f} 分钟超出剩余预算")
            mf = self.spec.constraints.max_features
            if mf and len(new_cfg["features"]) > mf:
                errs.append(f"特征数超过上限 {mf}")
        if a == "INVESTIGATE":
            if self.p["investigate_counts"].get(str(self.p["current_node"]), 0) >= self.cfg["decide"]["investigate_cap_per_branch"]:
                errs.append("该分支的 INVESTIGATE 次数已达上限，请换动作")
        prior = {x for c in decision_codes(self.p, list(recs.values())) for x in POLICY_PRIOR.get(c, [])}
        if prior and a not in prior and len(d.rationale.strip()) < self.cfg["decide"]["policy_deviation_min_rationale_chars"]:
            errs.append(f"偏离策略先验 {sorted(prior)} 时，rationale 必须写明理由")
        return errs


def decide(llm, ctx, validator: Validator, cfg, on_event=lambda *a, **k: None):
    """返回 (Decision, tokens)。重试用尽抛 DecideFailed。"""
    feedback, tokens, last = None, 0, []
    for attempt in range(cfg["decide"]["max_retries"] + 1):
        text, t = llm.complete(SYSTEM, render(ctx, feedback), cfg["llm"]["decide_temperature"])
        tokens += t
        try:
            d = Decision.model_validate(extract_json(text))
        except (ValidationError, ValueError) as e:
            last = [f"输出不是合法的 Decision JSON：{str(e)[:300]}"]
        else:
            last = validator.check(d)
            if not last:
                on_event("decision", {"attempt": attempt, "decision": d.model_dump(mode="json")})
                return d, tokens
        on_event("decision_rejected", {"attempt": attempt, "errors": last})
        feedback = "\n".join(f"- {e}" for e in last)
    raise DecideFailed(last)
