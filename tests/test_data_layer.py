"""第 1 项测试：know-how 字段校验、列策略、四段切分、holdout 隔离（§4.0-1）。
使用合成的小数据集 + 合成 know-how，不依赖真实大文件。"""
import ast
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from data.access import DataAccess, HoldoutAccessError, issue_holdout_token
from data.knowhow import check_fields, column_policy, load_knowhow
from data.splits import ID, LABEL, build_splits
from evaluation.final_gate import FinalGate

DICT = """
dataset: toy
entity_key: id
observation_time_col: issue_d
label:
  source_col: loan_status
  positive: ["Charged Off"]
  negative: ["Fully Paid"]
  exclude: ["Current"]
suggested_split:
  train_valid: ["2012-01", "2012-12"]
  oot_dev: ["2013-01", "2013-06"]
  holdout: ["2013-07", "2013-12"]
fields:
  income:     {availability: application}
  fico:       {availability: bureau_at_orig}
  zip_code:   {availability: application, compliance: exclude}
  grade:      {availability: incumbent_model}
  total_pymnt: {availability: post_origination}
  funded_inv: {availability: unknown}
  hardship_x: {availability: post_origination}
"""
BLACKLIST = """
leakage: {action: drop_at_load, fields: [total_pymnt, loan_status], field_prefixes: [hardship_]}
incumbent_model: {action: exclude_by_default, fields: [grade]}
compliance: {exclude: [zip_code], review: []}
unknown_availability: [funded_inv]
meta: {action: drop_at_load, fields: [id]}
leakage_patterns: ["pymnt", "^last_"]
"""


@pytest.fixture
def toy(tmp_path):
    rng = np.random.default_rng(0)
    n = 4000
    months = pd.date_range("2012-01-01", "2013-12-01", freq="MS")
    m = rng.choice(months, n)
    df = pd.DataFrame({
        "id": range(n), "issue_d": [pd.Timestamp(x).strftime("%b-%Y") for x in m],
        "income": rng.normal(50, 10, n), "fico": rng.normal(680, 30, n), "zip_code": "100xx",
        "grade": "A", "total_pymnt": rng.normal(size=n), "funded_inv": rng.normal(size=n),
        "hardship_x": 0, "new_col": rng.normal(size=n), "last_thing": 1,
        "loan_status": rng.choice(["Fully Paid", "Charged Off", "Current"], n, p=[.7, .2, .1]),
        "term": " 36 months"})
    kh = tmp_path / "knowhow" / "toy"
    kh.mkdir(parents=True)
    (kh / "data_dictionary.yaml").write_text(DICT)
    (kh / "blacklist.yaml").write_text(BLACKLIST)
    csv = tmp_path / "toy.csv"
    df.to_csv(csv, index=False)
    cfg = {"paths": {"knowhow_root": str(tmp_path / "knowhow"), "artifacts_root": str(tmp_path / "art"),
                     "reports_root": str(tmp_path / "rep")},
           "datasets": {"toy": {"tables": {"main": str(csv)}}},
           "split": {"valid_frac": 0.2, "seed": 1},
           "thresholds": {"maturity_current_share_max": 0.05, "missing_shift_report": 0.2},
           "task_specs": {"toy": {"row_filter": "trim(term)='36 months'",
                                  "split_time_expr": "strptime(issue_d, '%b-%Y')",
                                  "observation_time_expr": "strptime(issue_d, '%b-%Y')"}}}
    return cfg, df


# ---------- know-how 字段校验 ----------
def test_field_check_reports_missing_field(toy):
    cfg, _ = toy
    assert check_fields("toy", cfg)["ok"]
    d = yaml.safe_load(open(Path(cfg["paths"]["knowhow_root"]) / "toy" / "data_dictionary.yaml"))
    d["fields"]["ghost_field"] = {"availability": "application"}
    (Path(cfg["paths"]["knowhow_root"]) / "toy" / "data_dictionary.yaml").write_text(yaml.safe_dump(d))
    rep = check_fields("toy", cfg)
    assert not rep["ok"] and "ghost_field" in rep["tables"]["main"]["missing_in_data_dictionary"]


# ---------- 列策略 ----------
def test_column_policy(toy):
    cfg, df = toy
    kh = load_knowhow("toy", cfg)
    p = column_policy("toy", list(df.columns), kh)
    assert set(p["keep"]) == {"income", "fico"}
    assert {"total_pymnt", "hardship_x", "loan_status"} <= set(p["drop"]["leakage"])   # 字段名 + 前缀
    assert p["drop"]["incumbent"] == ["grade"] and p["drop"]["compliance_exclude"] == ["zip_code"]
    assert {"funded_inv", "new_col", "term", "last_thing"} <= set(p["quarantine"])    # 未知可得时间 → 隔离
    assert "last_thing" in p["pattern_hits_on_quarantined"] and p["pattern_hits_on_kept"] == []
    assert "grade" in column_policy("toy", list(df.columns), kh, use_incumbent=True)["keep"]  # A7 开关


# ---------- 切分 ----------
def test_split_time_order_and_label(toy):
    cfg, raw = toy
    rep = build_splits("toy", cfg)
    da = DataAccess("toy", cfg["paths"]["artifacts_root"])
    tr, va, oo = da.load("train"), da.load("valid"), da.load("oot_dev")
    assert tr._obs_time.max() < oo._obs_time.min()
    assert set(tr.columns) == {ID, LABEL, "_split_time", "_obs_time", "income", "fico"}   # 泄漏/隔离字段不在特征表里
    assert set(tr[LABEL].unique()) == {0, 1}                                               # Current 已被排除
    assert abs(len(va) / (len(va) + len(tr)) - 0.2) < 0.01
    n_expected = raw[(raw.loan_status != "Current")].shape[0]
    assert sum(v["n"] for v in rep["counts"].values()) == n_expected
    ids = [set(x[ID]) for x in (tr, va, oo)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])


def test_maturity_warning(toy):
    cfg, _ = toy      # 合成数据 Current 占比约 10% > 5%
    rep = build_splits("toy", cfg)
    assert rep["warnings"] and rep["warnings"][0]["code"] == "DATA_FATAL"


# ---------- holdout 隔离（§4.0-1） ----------
def test_holdout_denied_without_token(toy):
    cfg, _ = toy
    build_splits("toy", cfg)
    da = DataAccess("toy", cfg["paths"]["artifacts_root"])
    with pytest.raises(HoldoutAccessError):
        da.load("holdout")
    with pytest.raises(HoldoutAccessError):
        da.load("holdout", token="guess")
    with pytest.raises(HoldoutAccessError):
        issue_holdout_token()          # 非 final_gate 模块无法换取 token


def test_final_gate_can_read_holdout(toy):
    cfg, _ = toy
    build_splits("toy", cfg)
    cfg2 = {**cfg, "final_gate": {"delta": 0.02}}
    (Path(cfg["paths"]["artifacts_root"])).mkdir(exist_ok=True)
    h = FinalGate("toy", cfg2, "t").load_holdout()
    assert len(h) > 0 and h._obs_time.min() >= pd.Timestamp("2013-07-01")


def test_agent_code_cannot_reach_final_gate():
    """静态检查：agent/、tools/ 下的代码不得导入 final_gate、issue_holdout_token，也不得引用 holdout 目录。"""
    root = Path(__file__).resolve().parents[1]
    bad = []
    for d in ("agent", "tools", "features", "modeling"):
        for f in (root / d).rglob("*.py"):
            src = f.read_text()
            for node in ast.walk(ast.parse(src)):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""] + [a.name for a in node.names]
                if any("final_gate" in n or "issue_holdout_token" in n for n in names):
                    bad.append(str(f))
            if "artifacts/holdout" in src or "holdout.parquet" in src:
                bad.append(str(f))
    assert not bad, bad


# ---------- A3：泄漏字段放回 ----------
def test_leak_back_puts_leakage_fields_back_but_not_label_source(toy):
    cfg, df = toy
    kh = load_knowhow("toy", cfg)
    p = column_policy("toy", list(df.columns), kh, leak_back=True)
    assert {"total_pymnt", "hardship_x"} <= set(p["keep"]) or {"total_pymnt", "hardship_x"} <= set(p["quarantine"]) | set(p["keep"])
    assert "total_pymnt" in p["leak_truth"] and "loan_status" not in p["leak_truth"]
    assert "loan_status" not in p["keep"]                  # label 来源永远不作为特征
    assert "total_pymnt" not in column_policy("toy", list(df.columns), kh)["keep"]   # 默认仍然剔除


# ---------- classic_derived：可得时间继承（§4.11） ----------
def test_derive_availability_takes_weakest_input():
    from data.knowhow import derive_availability
    known = {"a": {"availability": "application"}, "b": {"availability": "updated_until_event"}, "c": {"availability": "post_origination"}}
    assert derive_availability("a + 1", known)[0] == "application"
    assert derive_availability("a + b", known)[0] == "updated_until_event"
    assert derive_availability("coalesce(a, 0) + c", known)[0] == "post_origination"     # 函数名/常量被忽略
    eff, warn = derive_availability("a + c", known, declared="application")               # 声明比输入强 → 取较弱者并告警
    assert eff == "post_origination" and warn


def test_classic_derived_pipeline(toy):
    cfg, raw = toy
    root = Path(cfg["paths"]["knowhow_root"]) / "toy" / "data_dictionary.yaml"
    d = yaml.safe_load(open(root))
    d["classic_derived"] = [
        {"name": "d_ok", "expr": "income + fico", "availability": "application"},
        {"name": "d_chain", "expr": "d_ok * 2", "availability": "application"},                  # 引用另一个派生特征
        {"name": "d_unk", "expr": "income + funded_inv", "availability": "application"},         # 输入存疑 → 继承为存疑
        {"name": "d_post", "expr": "fico + total_pymnt", "availability": "application"},         # 输入为贷后 → 继承为泄漏
    ]
    root.write_text(yaml.safe_dump(d, allow_unicode=True))
    rep = build_splits("toy", cfg)
    tr = DataAccess("toy", cfg["paths"]["artifacts_root"]).load("train")
    assert {"d_ok", "d_chain"} <= set(tr.columns) and not ({"d_unk", "d_post"} & set(tr.columns))
    assert (tr["d_chain"] - 2 * (tr["income"] + tr["fico"])).abs().max() < 1e-6
    assert "d_unk" in rep["policy"]["quarantine"] and "d_post" in rep["policy"]["drop"]["leakage"]
    assert {w["name"] for w in rep["warnings"] if w["code"] == "DERIVED_AVAILABILITY"} == {"d_unk", "d_post"}


def test_no_dataset_specific_literals_in_core_code():
    """换场景只换 know-how：核心代码里不得出现任何数据集特有的字段名或业务字面量（T5 的检验）。"""
    root = Path(__file__).resolve().parents[1]
    banned = ["issue_d", "loan_status", "'Current'", '"Current"', "%b-%Y", "reservation_status", "SK_ID_CURR", "is_canceled",
              "fico", "lead_time", "DAYS_EMPLOYED", "Charged Off"]
    hits = []
    for d in ("agent", "data", "features", "modeling", "evaluation", "memory", "runtime", "report", "tools", "baselines"):
        for f in (root / d).rglob("*.py"):
            for i, line in enumerate(f.read_text().splitlines(), 1):
                code = line.split("#")[0]
                if any(b in code for b in banned):
                    hits.append(f"{f.relative_to(root)}:{i}: {line.strip()[:80]}")
    assert not hits, "\n".join(hits)


def test_scan_features_knows_derived_and_registered(toy):
    """回归：派生特征、已登记的特征集特征不能被误判为『可得时间未知』。"""
    from evaluation.evaluator import scan_features
    cfg, raw = toy
    root = Path(cfg["paths"]["knowhow_root"]) / "toy" / "data_dictionary.yaml"
    d = yaml.safe_load(open(root))
    d["classic_derived"] = [{"name": "d_ok", "expr": "income + fico", "availability": "application"},
                            {"name": "d_unk", "expr": "income + funded_inv", "availability": "application"}]
    root.write_text(yaml.safe_dump(d, allow_unicode=True))
    kh = load_knowhow("toy", cfg)
    import pandas as pd
    df = pd.DataFrame({c: [1.0, 2.0] for c in ("income", "fico", "d_ok", "d_unk", "fs_feat", "mystery")} | {"_label": [0, 1]})
    s = scan_features(df, ["income", "fico", "d_ok", "d_unk", "fs_feat", "mystery"], kh, cfg, "toy", extra_known={"fs_feat"})
    assert set(s["unavailable_fields"]) == {"d_unk", "mystery"}          # 派生继承出的存疑 / 完全未登记 → 可疑；特征集特征放行
