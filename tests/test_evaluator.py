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
from evaluation.stop import stop_check

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
    assert stop_check(**{**kw, "history_verdicts": ["REJECT"] * 3}) == "NO_PROGRESS"


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


def test_no_guardrail_ablation_lets_leaky_candidate_through(ev):
    e, mk = ev
    e.cfg = {**e.cfg, "ablation": {"no_guardrail": True}}
    base = mk("base")
    leak = {"feature_aucs": {"x": .99}}
    out = e.evaluate(mk("good", valid_auc=.8, oot_dev_auc=.7, gap=.1), base, leak)
    assert out["guardrail_failures"] == [] and out["verdict"] == Verdict.ACCEPT      # A3：没有 guardrail，虚高的实验被接受


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
