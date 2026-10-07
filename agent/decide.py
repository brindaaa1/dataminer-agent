"""DECIDE 节点：上下文渲染 + LLM 调用 + pydantic 校验 + 业务规则校验；校验失败回灌错误重试（§4.7, §4.0-3）。"""
import json
from pathlib import Path

from pydantic import ValidationError

from agent.actions import ActionError, apply_action, config_hash
from agent.llm import extract_json
from agent.schemas import ActionType, Decision
from evaluation.guardrails import POLICY_PRIOR, decision_codes
from modeling.inner_loop import effective_frac
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
        src = recs.get(d.params.get("exp_id")) if a == "PROMOTE_FIDELITY" else None
        if src and self.p.get("best_exp_id") and src.action_type != "RACE" and src.exp_id not in self.p.get("promote_hint", []):
            mde = (self.p.get("oot_power") or {}).get("mde")      # v7–v10：这类升全量 19 次 0 次采纳
            errs.append(f"{src.exp_id} 在低保真上比最优高出不到可分辨的最小提升（MDE {mde}），算打平：升到全量大概率仍是不确定，白花一次全量训练。"
                        "只有 promotable 里标为值得升的（差距 ≥ MDE）才能升；其他赛跑模型可以升；或换方向、STOP")
        if kind == "train":
            h = config_hash(new_cfg)
            if h in self.store.taboo_hashes(self.p["task_id"]):     # 点名撞的是哪个实验，LLM 才知道怎么改（否则会反复撞同一个）
                dup = next(r for r in self.store.all(self.p["task_id"]) if r.config_hash == h and not r.invalidated)
                head = f"该配置已经试过（禁忌表）：与 {dup.exp_id}（{dup.action_type}，{dup.config.get('model')}/{dup.fidelity}，结论 {dup.verdict}）完全相同。"
                if dup.fidelity == "full":       # 撞的是全量实验：别再让它升全量（v5 评测 home_credit s2 照提示又升，连撞 4 次）
                    errs.append(head + "它已经在全量数据上复验过，不要再升全量；换方向：换模板或特征、调参、换模型，或 STOP")
                else:
                    errs.append(head + "请改搜索空间或特征；想在全量数据上复验它，用 PROMOTE_FIDELITY")
            fid = self.cfg["fidelity"][new_cfg["fidelity"]]
            need = estimate_cost_minutes(self.n_rows * effective_frac(fid, self.n_rows), len(new_cfg["features"]),
                                         new_cfg.get("n_trials") or fid["n_trials"], self.cfg, new_cfg["model"],
                                         self.p.get("cost_obs", {}).get(new_cfg["model"]))
            if need > self.remaining["cpu_minutes"]:
                errs.append(f"代码估算成本 {need:.1f} 分钟超出剩余预算")
            mf = self.spec.constraints.max_features
            if mf and len(new_cfg["features"]) > mf:
                errs.append(f"特征数超过上限 {mf}")
        if kind == "expand":
            errs += self._expand_errors(d.params, recs)
        if a == "INVESTIGATE":
            if self.p["investigate_counts"].get(str(self.p["current_node"]), 0) >= self.cfg["decide"]["investigate_cap_per_branch"]:
                errs.append("该分支的 INVESTIGATE 次数已达上限，请换动作")
        prior = {x for c in decision_codes(self.p, list(recs.values()), self.cfg["run"]["max_rounds"] - self.p["round_no"]) for x in POLICY_PRIOR.get(c, [])}
        if prior and a not in prior and len(d.rationale.strip()) < self.cfg["decide"]["policy_deviation_min_rationale_chars"]:
            errs.append(f"偏离策略先验 {sorted(prior)} 时，rationale 必须写明理由")
        return errs


    def _expand_errors(self, params, recs) -> list[str]:
        """加特征的训练配置要生成后才知道，禁忌表拦不住：按『同一节点 + 同一模板 / 同一段代码』查重（v5 评测 home_credit s2）。
        模板只能从代码支持的列表里选（v5 s0：选了多级聚合模板，执行时 NotImplementedError）。"""
        tmpl, code = params.get("template_id"), params.get("code")
        if tmpl and tmpl not in self.p.get("templates", []):
            return [f"模板 {tmpl} 不可用；可用模板：{self.p.get('templates', [])}"]
        node = self.p["current_node"]
        dup = next((r for r in recs.values() if r.action_type == "EXPAND_FEATURES" and r.parent_exp_id == node and not r.invalidated
                    and ((tmpl and r.diff.get("template_id") == tmpl) or (code and r.diff.get("code") == code))), None)
        if dup:
            return [f"在当前节点 {node} 上已经用{'模板 ' + tmpl if tmpl else '同一段代码'}加过特征：{dup.exp_id}（结论 {dup.verdict}），"
                    f"再做一次结果相同。换模板或代码；想在全量数据上复验 {dup.exp_id}，用 PROMOTE_FIDELITY"]
        return []


def decide(llm, ctx, validator: Validator, cfg, on_event=lambda *a, **k: None):
    """返回 (Decision, tokens)。重试用尽抛 DecideFailed。"""
    feedback, tokens, last, seen = None, 0, [], []
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
        seen += [e for e in last if e not in seen]       # 带上本次决策所有被拒原因：只给最近一条，LLM 会在两个被拒选项间来回（v6 评测 home_credit）
        feedback = "\n".join(f"- {e}" for e in seen)
    raise DecideFailed(last)
