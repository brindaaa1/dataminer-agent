"""Evaluator：ACCEPT 的唯一来源（§4.0-2）。LLM 传入的任何"判定"都会被忽略。

流程：guardrail → 诊断码 → compare（消耗 OOT-dev 预算）→ 判定。
"""
from enum import Enum

import numpy as np

from data.access import DataAccess
from evaluation.compare import paired_bootstrap
from evaluation.guardrails import diagnose, single_feature_auc
from evaluation.metrics import METRICS, primary
from evaluation.oot_budget import OOTBudget


class Verdict(str, Enum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    INCONCLUSIVE = "INCONCLUSIVE"
    PROMISING = "PROMISING"


def scan_features(df, features, kh, cfg, dataset: str | None = None, extra_known=(), always_suspect=()) -> dict:
    """在 train 上扫描特征：单特征 AUC、可得时间未知/事后字段、命名泄漏模式。供 diagnose 使用。
    extra_known：已登记、可得时间有保证的特征（如特征工厂产出，按 PIT 规则生成）。"""
    import re
    from data.knowhow import field_table
    d = kh["dictionary"] or {}
    spec = cfg["task_specs"].get(dataset, {}) if dataset else {}
    fields, _, _ = field_table(kh, d.get("label", {}).get("source_table"), spec.get("derived_expr_overrides"))
    unregistered_ok = spec.get("unregistered_columns") == "application"      # 主表未登记字段的默认口径（与切分一致）
    pats = [re.compile(p, re.I) for p in ((kh["blacklist"] or {}).get("leakage_patterns", []))]
    bad_av = ("unknown", "post_origination", "post_outcome", "at_checkin", "updated_until_event")

    def unavailable(f):
        if f in always_suspect:                # run_code 产出：不管数据集策略如何，一律视为可得时间不可信
            return True
        if f in extra_known:
            return False
        if f not in fields:
            return not unregistered_ok
        return fields[f].get("availability") in bad_av
    return {
        "feature_aucs": {f: single_feature_auc(df[f], df["_label"]) for f in features if f in df},
        "unavailable_fields": [f for f in features if unavailable(f)],
        "leak_pattern_hits": [f for f in features if any(p.search(f) for p in pats)],
    }


class Evaluator:
    def __init__(self, cfg: dict, dataset: str, task_id: str, db_path: str | None = None):
        self.cfg = cfg
        self.reference_gap = None              # 由 Orchestrator 在 PROFILE 测出后设置（evaluation/reference.py）
        self.n_raw_features = None             # 原始特征数，由 Orchestrator 设置；TOO_MANY_FEATURES 按新增数算
        art = cfg["paths"]["artifacts_root"]
        self.art = art
        self.budget = OOTBudget(cfg, db_path or f"{art}/state.db", task_id)
        self.y_oot = DataAccess(dataset, art).load("oot_dev")["_label"].values

    def _pred(self, exp_id):
        return np.load(f"{self.art}/experiments/{exp_id}/oot_dev_pred.npy")

    def evaluate(self, cand: dict, best: dict | None, scan: dict | None = None, *,
                 history_verdicts=None, remaining_budget_frac=None, handled_leaks=(),
                 llm_claimed_verdict=None, guard: bool = True) -> dict:
        """cand/best: run_experiment 的返回 dict。llm_claimed_verdict 仅记录，不参与判定。
        guard：护栏只和『已接受的当前最优』比；模型赛跑里的同伴比较（base 只是先跑的那个）不检查护栏。"""
        c = self.cfg["evaluator"]
        scan = scan or {}
        diag = diagnose(cand["metrics"], self.cfg, n_features=cand["n_features"],
                        score_psi=cand["metrics"].get("score_psi"),
                        feature_aucs={f: a for f, a in scan.get("feature_aucs", {}).items() if f not in handled_leaks},
                        unavailable_fields=[f for f in scan.get("unavailable_fields", []) if f not in handled_leaks],
                        leak_pattern_hits=[f for f in scan.get("leak_pattern_hits", []) if f not in handled_leaks],
                        history_verdicts=history_verdicts, remaining_budget_frac=remaining_budget_frac,
                        best_iter=cand.get("best_iter"), n_rounds_max=self.cfg["inner_loop"]["n_rounds_max"],
                        model=cand["model"], reference_gap=self.reference_gap, n_raw_features=self.n_raw_features)
        codes = diag["codes"]
        fails = []
        if self.cfg.get("ablation", {}).get("no_guardrail"):     # A3：去掉 guardrail，只留 compare
            pass
        elif "OVERFIT_GAP" in codes:
            fails.append("OVERFIT_GAP")
        if not self.cfg.get("ablation", {}).get("no_guardrail"):
            if "LEAK_SUSPECT" in codes:
                fails.append("LEAK_SUSPECT")
            if "TOO_MANY_FEATURES" in diag["details"]:
                fails.append("TOO_MANY_FEATURES")
            if cand["metrics"].get("score_psi", 0) > c["psi_severe"]:
                fails.append("SCORE_PSI")
        out = {"diagnosis_codes": codes, "diagnosis_details": diag["details"], "guardrail_failures": fails,
               "llm_claimed_verdict_ignored": llm_claimed_verdict, "compare": None, "oot_used": self.budget.used}

        if best is None:                                          # 第一个实验：没有可比对象，不消耗 OOT 预算
            out["verdict"] = self._first(cand, fails)
            return out

        k, alpha = self.budget.consume(key=cand['exp_id'])
        cmp = paired_bootstrap(self.y_oot, self._pred(cand["exp_id"]), self._pred(best["exp_id"]), alpha,
                               c["bootstrap_n"], c["bootstrap_seed"], metric_fn=METRICS[primary(self.cfg)].fn)
        cmp["k"] = k
        out["compare"], out["oot_used"] = cmp, k
        if not cmp["significant"]:
            out["diagnosis_codes"] = codes + ["PLATEAU"]
        bm, cm = best["metrics"], cand["metrics"]
        guard = {g: {"best": bm[f"oot_dev_{g}"], "cand": cm[f"oot_dev_{g}"], "drop": bm[f"oot_dev_{g}"] - cm[f"oot_dev_{g}"]}
                 for g, tol in self.cfg["metric"]["guards"].items() if guard and bm[f"oot_dev_{g}"] - cm[f"oot_dev_{g}"] > tol}
        if guard:                                                 # 护栏只拦截，不参与排名
            fails.append("GUARD_FAIL")
            out["diagnosis_codes"] = out["diagnosis_codes"] + ["GUARD_FAIL"]
            out["diagnosis_details"]["GUARD_FAIL"] = guard
        mono = self.cfg["monotone"]
        cc, bc = cand.get("config", {}), best.get("config", {})
        noninferior = (mono["keep_if_psi_better"] and cc.get("monotone") and not bc.get("monotone") and cand["fidelity"] == "full"
                       and best["fidelity"] == "full" and cc.get("model") == bc.get("model") and cc.get("features") == bc.get("features")
                       and cmp["ci_high"] >= 0 and cand["metrics"].get("score_psi", 1) < best["metrics"].get("score_psi", 0) - mono["psi_margin"])
        if fails:
            out["verdict"] = Verdict.REJECT
        elif noninferior and not cmp["significant"]:                # 加约束：AUC 没有显著变差、分数 PSI 更优 → 保留（只有 evaluator 能给 ACCEPT）
            out["verdict"] = Verdict.ACCEPT
            out["monotone_noninferior"] = {"delta": cmp["delta"], "psi_new": cand["metrics"]["score_psi"], "psi_base": best["metrics"]["score_psi"]}
        elif cmp["significant"]:
            out["verdict"] = Verdict.ACCEPT if cand["fidelity"] == "full" else Verdict.PROMISING
        elif cmp["ci_high"] < 0:
            out["verdict"] = Verdict.REJECT                        # 显著变差
        else:
            out["verdict"] = Verdict.INCONCLUSIVE
        return out

    @staticmethod
    def _first(cand, fails):
        if fails:
            return Verdict.REJECT
        return Verdict.ACCEPT if cand["fidelity"] == "full" else Verdict.PROMISING
