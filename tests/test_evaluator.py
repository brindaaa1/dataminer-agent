"""第 3 项测试：诊断码、配对 bootstrap、OOT-dev 预算收紧与停止、LLM 判定不影响 verdict。"""
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.knowhow import load_config
from evaluation.compare import _auc, paired_bootstrap
from evaluation.evaluator import Evaluator, Verdict, scan_features
from evaluation.guardrails import diagnose, psi, single_feature_auc
from evaluation.oot_budget import OOTBudget, OOTBudgetExhausted
from evaluation.stop import history_mark, stop_check

CFG = load_config()
M = {"train_auc": .8, "valid_auc": .7, "oot_dev_auc": .69, "gap": .01, "score_psi": .02}


def test_psi_identical_vs_shifted():
    x = np.random.default_rng(0).normal(size=5000)
    assert psi(x, x) < 0.01 and psi(x, x + 1.0) > 0.25


def test_auc_matches_sklearn():
    from sklearn.metrics import roc_auc_score
    r = np.random.default_rng(1)
    y, p = r.integers(0, 2, 500), np.round(r.random(500), 1)      # 含并列
    assert _auc(y, p) == pytest.approx(roc_auc_score(y, p))


def test_diagnosis_codes():
    d = diagnose({**M, "valid_auc": .75, "oot_dev_auc": .70}, CFG, n_features=10)
    assert "OVERFIT_GAP" in d["codes"]
    d = diagnose(M, CFG, n_features=10, feature_aucs={"x": .9, "y": .6}, unavailable_fields=["z"],
                 leak_pattern_hits=["last_pymnt_d"])
    assert set(d["details"]["LEAK_SUSPECT"]) == {"x", "z", "last_pymnt_d"}
    assert "PSI_DRIFT" in diagnose({**M, "score_psi": .3}, CFG, n_features=1, score_psi=.3)["codes"]
    assert "NO_PROGRESS" in diagnose(M, CFG, n_features=1, history_verdicts=["ACCEPT", "REJECT", "INCONCLUSIVE", "REJECT"])["codes"]
    assert "BUDGET_LOW" in diagnose(M, CFG, n_features=1, remaining_budget_frac=0.1)["codes"]


def test_single_feature_auc_detects_leak_including_missingness():
    y = pd.Series(np.r_[np.ones(500), np.zeros(500)].astype(int))
    x = pd.Series(np.where(y == 1, 1.0, np.nan))                 # 只在正样本非缺失
    assert single_feature_auc(x, y) > 0.9


def test_paired_bootstrap_detects_real_gain_and_noise():
    r = np.random.default_rng(0)
    n = 6000
    y = r.integers(0, 2, n)
    base = y * 0.5 + r.normal(size=n)
    better = y * 1.0 + r.normal(size=n)
    assert paired_bootstrap(y, better, base, 0.05, 300)["significant"]
    noise = base + r.normal(scale=0.01, size=n)
    assert not paired_bootstrap(y, noise, base, 0.05, 300)["significant"]


def test_alpha_tightens_and_budget_stops(tmp_path):
    b = OOTBudget({"oot_budget": {"alpha_0": .05, "max_compares": 3, "policy": "log"}}, str(tmp_path / "s.db"), "t")
    alphas = [b.consume()[1] for _ in range(3)]
    assert alphas[0] == pytest.approx(.05) and alphas[0] > alphas[1] > alphas[2]
    assert b.exhausted
    with pytest.raises(OOTBudgetExhausted):
        b.consume()
    assert OOTBudget(b_cfg := {"oot_budget": {"alpha_0": .05, "max_compares": 3, "policy": "log"}}, str(tmp_path / "s.db"), "t").used == 3  # 重启不归零


def test_stop_check_oot_budget_and_others(tmp_path):
    b = OOTBudget({"oot_budget": {"alpha_0": .05, "max_compares": 1, "policy": "log"}}, str(tmp_path / "s.db"), "t")
    rem = {"tokens": 1, "cpu_minutes": 1, "wall_minutes": 1}
    kw = dict(cfg=CFG, oot_budget=b, round_no=1, max_rounds=10, remaining=rem, history_verdicts=["ACCEPT"])
    assert stop_check(**kw) is None
    b.consume()
    assert stop_check(**kw) == "OOT_BUDGET_EXHAUSTED"
    assert stop_check(**{**kw, "remaining": {**rem, "tokens": 0}}) == "BUDGET_EXHAUSTED"
    assert stop_check(**{**kw, "history_verdicts": ["REJECT"] * 3, "best_score": 0.8}) == "NO_PROGRESS"


def test_no_progress_does_not_stop_while_there_is_no_final_model(tmp_path):
    """还没有任何最终模型时，连续 REJECT 不提前停：停了就一定没有模型（v0 评测 hotel s0：3 轮全量复验都被 OVERFIT_GAP 拒后停在第 3 轮）。
    轮数上限仍然生效。"""
    b = OOTBudget({"oot_budget": {"alpha_0": .05, "max_compares": 9, "policy": "log"}}, str(tmp_path / "s.db"), "t")
    kw = dict(cfg=CFG, oot_budget=b, round_no=3, max_rounds=5, remaining={"tokens": 1, "cpu_minutes": 1, "wall_minutes": 1},
              history_verdicts=["REJECT"] * 3)
    assert stop_check(**kw) is None
    assert stop_check(**{**kw, "round_no": 5}) == "MAX_ROUNDS"


def test_low_fidelity_gain_without_significance_does_not_count_as_no_progress(tmp_path):
    """v4 评测 lending_club：加特征后低保真比对照高 0.004~0.006，但不显著 → INCONCLUSIVE，连续 3 轮就 NO_PROGRESS，
    agent 没机会把它升全量复验。低保真、不确定、但点估计更好的结果不计入停止计数；全量或变差的照旧计入。"""
    b = OOTBudget({"oot_budget": {"alpha_0": .05, "max_compares": 9, "policy": "log"}}, str(tmp_path / "s.db"), "t")
    kw = dict(cfg=CFG, oot_budget=b, round_no=4, max_rounds=10, remaining={"tokens": 1, "cpu_minutes": 1, "wall_minutes": 1}, best_score=0.8)
    up = history_mark("INCONCLUSIVE", "low", 0.004)
    assert stop_check(**kw, history_verdicts=["INCONCLUSIVE", "INCONCLUSIVE", up]) is None
    assert stop_check(**kw, history_verdicts=["INCONCLUSIVE", "INCONCLUSIVE", history_mark("INCONCLUSIVE", "low", -0.001)]) == "NO_PROGRESS"
    assert stop_check(**kw, history_verdicts=["INCONCLUSIVE", "INCONCLUSIVE", history_mark("INCONCLUSIVE", "full", 0.004)]) == "NO_PROGRESS"
    assert history_mark("REJECT", "low", 0.004) == "REJECT"


def test_tiny_low_fidelity_gain_counts_as_no_progress(tmp_path):
    """v8 评测 lending_club：低保真有行数下限后和全量几乎一样，候选与对照的差在 ±0.002 抛硬币，
    差 > 0 就不计入停止计数会让 agent 停不下来（两个种子都跑满 8 轮）。差要到 min_up（0.5×MDE）才算"更高"。"""
    b = OOTBudget({"oot_budget": {"alpha_0": .05, "max_compares": 9, "policy": "log"}}, str(tmp_path / "s.db"), "t")
    kw = dict(cfg=CFG, oot_budget=b, round_no=4, max_rounds=10, remaining={"tokens": 1, "cpu_minutes": 1, "wall_minutes": 1}, best_score=0.8)
    tiny = history_mark("INCONCLUSIVE", "low", 0.002, min_up=0.005)
    assert tiny == "INCONCLUSIVE"
    assert stop_check(**kw, history_verdicts=["INCONCLUSIVE", "INCONCLUSIVE", tiny]) == "NO_PROGRESS"
    assert history_mark("INCONCLUSIVE", "low", 0.006, min_up=0.005) == "INCONCLUSIVE_UP"


def test_too_many_features_counts_added_features_not_raw_width():
    """v5 评测 home_credit：主表原始特征就有 122 个，固定上限 60 让包括赛跑在内的全部实验被拒，没有最终模型。
    护栏防的是特征工程失控：有原始特征数时按『新增了多少』算，没有时退回总数上限。"""
    cfg = {**CFG, "evaluator": {**CFG["evaluator"], "max_features": 60, "max_added_features": 40}}
    assert "TOO_MANY_FEATURES" not in diagnose(M, cfg, n_features=122, n_raw_features=122)["details"]
    assert "TOO_MANY_FEATURES" not in diagnose(M, cfg, n_features=162, n_raw_features=122)["details"]
    assert diagnose(M, cfg, n_features=163, n_raw_features=122)["details"]["TOO_MANY_FEATURES"] == {"n_features": 163, "n_added": 41}
    assert "TOO_MANY_FEATURES" in diagnose(M, cfg, n_features=61)["details"]


@pytest.fixture
def ev(tmp_path):
    cfg = {**CFG, "paths": {**CFG["paths"], "artifacts_root": str(tmp_path / "art")}, "evaluator": {**CFG["evaluator"], "bootstrap_n": 200}}
    r = np.random.default_rng(0)
    n = 4000
    y = r.integers(0, 2, n)
    d = tmp_path / "art" / "splits" / "toy"
    d.mkdir(parents=True)
    pd.DataFrame({"_label": y}).to_parquet(d / "oot_dev.parquet")
    preds = {"base": y * .5 + r.normal(size=n), "good": y * 1.2 + r.normal(size=n), "bad": r.normal(size=n)}
    for k, p in preds.items():
        (tmp_path / "art" / "experiments" / k).mkdir(parents=True)
        np.save(tmp_path / "art" / "experiments" / k / "oot_dev_pred.npy", p)
    mk = lambda k, fid="full", **m: {"exp_id": k, "model": "lr_scorecard", "fidelity": fid, "n_features": 10, "metrics": {**M, **m}}
    return Evaluator(cfg, "toy", "t"), mk


def test_verdicts(ev):
    e, mk = ev
    base = mk("base")
    assert e.evaluate(base, None)["verdict"] == Verdict.ACCEPT                       # 首个实验
    assert e.evaluate(mk("good"), base)["verdict"] == Verdict.ACCEPT
    assert e.evaluate(mk("good", "low"), base)["verdict"] == Verdict.PROMISING      # 低保真最多 PROMISING
    assert e.evaluate(mk("bad"), base)["verdict"] == Verdict.REJECT                 # 显著变差
    assert e.evaluate(mk("good", valid_auc=.8, oot_dev_auc=.7, gap=.1), base)["verdict"] == Verdict.REJECT  # guardrail 拦截
    leak = {"feature_aucs": {"x": .95}}
    assert e.evaluate(mk("good"), base, leak)["verdict"] == Verdict.REJECT           # 未处理的 LEAK_SUSPECT
    assert e.evaluate(mk("good"), base, leak, handled_leaks=["x"])["verdict"] == Verdict.ACCEPT


def test_llm_claim_does_not_change_verdict(ev):
    e, mk = ev
    o = e.evaluate(mk("bad"), mk("base"), llm_claimed_verdict="ACCEPT")
    assert o["verdict"] == Verdict.REJECT and o["llm_claimed_verdict_ignored"] == "ACCEPT"


def test_only_evaluator_creates_accept():
    root = Path(__file__).resolve().parents[1]
    for d in ("agent", "tools", "modeling", "features", "memory", "runtime"):
        for f in (root / d).rglob("*.py"):
            for node in ast.walk(ast.parse(f.read_text())):
                assert not (isinstance(node, ast.Attribute) and node.attr == "ACCEPT"), f


def test_single_feature_auc_and_iv_accept_datetime_columns():
    from features.screen import iv
    y = pd.Series(np.r_[np.zeros(300), np.ones(300)].astype(int))
    d = pd.Series(pd.to_datetime("2017-01-01") + pd.to_timedelta(np.r_[np.arange(300), np.arange(300) + 400], unit="D"))
    assert single_feature_auc(d, y) > 0.9 and iv(d, y) > 0.5


def test_guard_rejects_when_guard_metric_drops(ev):
    e, mk = ev
    e.cfg["metric"] = {"primary": "auc", "guards": {"ks": 0.02}}
    out = e.evaluate(mk("good", oot_dev_ks=.30), mk("base", oot_dev_ks=.40))
    assert out["verdict"] == "REJECT" and "GUARD_FAIL" in out["diagnosis_codes"]
    assert abs(out["diagnosis_details"]["GUARD_FAIL"]["ks"]["drop"] - .10) < 1e-9


def test_guard_within_tolerance_keeps_verdict(ev):
    e, mk = ev
    e.cfg["metric"] = {"primary": "auc", "guards": {"ks": 0.02}}
    out = e.evaluate(mk("good", oot_dev_ks=.39), mk("base", oot_dev_ks=.40))
    assert out["verdict"] == "ACCEPT"


def _rec(i, action, fidelity, verdict, codes=(), model="lgbm", diff=None):
    from memory.schema import Cost, ExperimentRecord
    return ExperimentRecord(exp_id=i, task_id="t", parent_exp_id=None, hypothesis="h", action_type=action, diff=diff or {},
                            config={"model": model}, config_hash=i, fidelity=fidelity, metrics={"oot_dev_auc": .8, "gap": .04},
                            diagnosis_codes=list(codes), cost=Cost(), verdict=verdict)


def test_after_promote_rejected_for_overfit_prior_is_regularize_not_promote_next():
    """v0 评测 hotel s0：全量复验被 OVERFIT_GAP 拒后，NO_FINAL_MODEL 仍把"升下一个候选"放进先验，LLM 连升三个 gap 一样大的候选全被拒。
    剩余轮数够时，先验只留收紧复杂度的方向；只剩最后一轮还没有最终模型时，照常要求升一个。"""
    from evaluation.guardrails import POLICY_PRIOR, decision_codes
    recs = [_rec("race_lgbm", "RACE", "low", "PROMISING"), _rec("race_rf", "RACE", "low", "PROMISING", model="random_forest"),
            _rec("e01", "PROMOTE_FIDELITY", "full", "REJECT", ["OVERFIT_GAP"], diff={"exp_id": "race_lgbm"})]
    p = {"best_exp_id": None, "last_codes": ["OVERFIT_GAP"]}
    codes = decision_codes(p, recs, rounds_left=3)
    prior = {x for c in codes for x in POLICY_PRIOR[c]}
    assert "NO_FINAL_MODEL" not in codes and "PROMOTE_FIDELITY" not in prior and "TUNE" in prior
    assert "NO_FINAL_MODEL" in decision_codes(p, recs, rounds_left=1)
    other = recs[:2] + [_rec("e01", "TUNE", "low", "REJECT", ["OVERFIT_GAP"])]          # 低保真调参被拒：不受影响
    assert "NO_FINAL_MODEL" in decision_codes(p, other, rounds_left=3)


def test_overfit_gap_is_judged_against_the_reference_drift():
    """hotel 的 valid 与 OOT-dev 季节不同：默认 LightGBM 自己的 gap 就有 0.07（v1 评测），固定 0.03 会拒掉所有有用的模型。
    改成相对判定：超过 参照 gap + 容差 才算过拟合；没有参照时退回固定阈值。"""
    from evaluation.guardrails import diagnose
    cfg = {**CFG, "evaluator": {**CFG["evaluator"], "overfit_gap": {"mode": "relative", "tolerance": 0.05}}}
    m = {"valid_auc": 0.89, "oot_dev_auc": 0.82}
    assert "OVERFIT_GAP" not in diagnose(m, cfg, n_features=1, reference_gap=0.07)["codes"]
    d = diagnose({"valid_auc": 0.95, "oot_dev_auc": 0.82}, cfg, n_features=1, reference_gap=0.07)
    assert "OVERFIT_GAP" in d["codes"] and d["details"]["OVERFIT_GAP"]["reference_gap"] == 0.07
    assert d["details"]["OVERFIT_GAP"]["excess"] == pytest.approx(0.06)
    assert "OVERFIT_GAP" in diagnose(m, cfg, n_features=1)["codes"]                  # 没有参照：固定阈值 0.03


def test_cost_estimate_is_model_aware():
    """v1 评测 hotel_bookings：catboost 全量复验一次 18～21 分钟，估算按 lgbm 的系数低估了约 80 倍，预算检查形同虚设。"""
    from tools.run_experiment import estimate_cost_minutes
    cfg = load_config()
    lg = estimate_cost_minutes(21000, 27, 20, cfg, "lgbm")
    cb = estimate_cost_minutes(21000, 27, 20, cfg, "catboost")
    assert cb == pytest.approx(lg * cfg["inner_loop"]["cost_model_factor"]["catboost"]) and cb > 60    # 约 77 CPU 分钟
    assert 5 < lg < 20                                                                                 # 实测 lgbm 全量约 1.2 分钟 × 8 线程


def test_cost_estimate_is_calibrated_on_this_tasks_race():
    """v6 评测 home_credit（122 个特征）：静态公式在 lending_club/hotel（~35 个特征）上校准，这里高估 37–44 倍，
    catboost 升全量估 1186 分钟 > 预算，决策连拒 4 次失败。有本任务同一模型的实测（赛跑）时按它的行数/特征数/trial 数比例外推。"""
    from tools.run_experiment import estimate_cost_minutes
    cfg = load_config()
    obs = {"cpu_minutes": 4.0, "n_rows": 21600, "n_features": 122, "n_trials": 10}
    assert estimate_cost_minutes(72000, 122, 20, cfg, "catboost", obs=obs) == pytest.approx(4.0 * 72000 / 21600 * 2)
    assert estimate_cost_minutes(21600, 142, 10, cfg, "catboost", obs=obs) == pytest.approx(4.0 * 142 / 122)
    assert estimate_cost_minutes(72000, 122, 20, cfg, "catboost") > 1000              # 没有实测时仍用静态公式


def test_report_metrics_lift_recall_and_brier():
    """补充参考指标（只报告，不参与选模型和护栏）：头部 10% 提升度和召回、Brier。"""
    import numpy as np
    from evaluation.metrics import report_metrics
    y = np.array([1] * 10 + [0] * 90)
    perfect = np.linspace(1, 0, 100)                      # 正类全排在最前
    m = report_metrics(y, perfect)
    assert m["lift_top10"] == pytest.approx(10.0) and m["recall_top10"] == pytest.approx(1.0)
    assert report_metrics(y, np.full(100, 0.1))["brier"] == pytest.approx(0.09)
