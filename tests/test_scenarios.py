"""情景用例（spec §6）：通用变换在玩具数据上构造；check 用构造的事件判定；runner 用 mock LLM 跑通。"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from eval.scenarios import SCENARIOS, common
from test_data_layer import toy  # noqa: F401  玩具 lending_club 风格数据：label 是 loan_status 的取值列表


def ev(i, t, p=None, x=None):
    return {"id": i, "ts": float(i), "type": t, "exp_id": x, "payload": p or {}}


def test_copy_knowhow_never_touches_the_original(toy, tmp_path):
    cfg, _ = toy
    kh = common.copy_knowhow("toy", cfg, tmp_path / "v")
    common.edit_dictionary(kh, "toy", lambda d: d.update(dataset="changed"))
    orig = yaml.safe_load((Path(cfg["paths"]["knowhow_root"]) / "toy" / "data_dictionary.yaml").read_text())
    assert orig["dataset"] == "toy" and yaml.safe_load((kh / "toy" / "data_dictionary.yaml").read_text())["dataset"] == "changed"


def test_copy_knowhow_keeps_domain_playbooks(toy, tmp_path):
    """情景运行的 know-how 副本要带上领域手册（knowhow/domains/）；数据集自己的 priors.yaml 随数据集目录一起复制。
    漏复制会让情景运行和正常运行不一样（计划 1 遗留：当时漏的是 domain_priors.yaml）。"""
    cfg, _ = toy
    root = Path(cfg["paths"]["knowhow_root"])
    (root / "domains" / "_default").mkdir(parents=True)
    (root / "domains" / "_default" / "playbook.yaml").write_text("name: _default\n")
    (root / "toy" / "priors.yaml").write_text("target_direction: {x: 1}\n")
    kh = common.copy_knowhow("toy", cfg, tmp_path / "v")
    assert (kh / "domains" / "_default" / "playbook.yaml").exists()
    assert yaml.safe_load((kh / "toy" / "priors.yaml").read_text()) == {"target_direction": {"x": 1}}


def test_no_signal_shuffles_labels_only_among_labelled_rows(toy, tmp_path):
    cfg, raw = toy
    v = SCENARIOS["no_signal"].build("toy", cfg, tmp_path / "v", {"seed": 0})
    df = pd.read_csv(v.tables["main"])
    assert (df.loan_status == "Current").sum() == (raw.loan_status == "Current").sum()        # 排除的行不动
    assert sorted(df.loan_status) == sorted(raw.loan_status)                                   # 只是打乱
    lab = {"source_col": "loan_status", "positive": ["Charged Off"]}
    y_new, y_old = common.label_binary(df, lab), common.label_binary(raw, lab)
    assert (y_new != y_old).mean() > 0.1


def test_label_helpers_accept_scalar_and_list_values():
    df = pd.DataFrame({"s": ["Charged Off", "Fully Paid", "Current"], "c": [1, 0, 1]})
    assert common.label_binary(df, {"source_col": "s", "positive": ["Charged Off"], "negative": ["Fully Paid"]}).tolist() == [1, 0, 0]
    assert common.label_mask(df, {"source_col": "s", "positive": ["Charged Off"], "negative": ["Fully Paid"]}).tolist() == [True, True, False]
    assert common.label_binary(df, {"col": "c", "positive": 1, "negative": 0}).tolist() == [1, 0, 1]      # hotel 写法
    assert common.label_binary(df, {"col": "c"}).tolist() == [1, 0, 1]                                    # home_credit 写法


def test_no_signal_check():
    fg = lambda h: [ev(1, "final_gate", {"exp_id": "e1", "metric": "auc", "holdout_auc": h})]
    c = SCENARIOS["no_signal"].check
    assert c(fg(0.53), [], {})["passed"] and not c(fg(0.62), [], {})["passed"]
    assert c([], [], {})["passed"]                                                             # 没有最终模型也算通过


def test_run_scenario_end_to_end_with_mock(eval_env):
    from agent.llm import PolicyMockLLM
    row = eval_env.run_scenario("no_signal", "hotel_sample", "t", "mock", params={"n_rows": 3000, "max_rounds": 1}, llm=PolicyMockLLM())
    assert row["scenario"] == "no_signal" and row["base"] == "hotel_sample" and row["status"] == "ok"
    assert isinstance(row["passed"], bool) and "rounds" in row["process"]
    assert row["task_id"] == "t_scn_no_signal_hotel_sample"


def test_leak_column_is_derived_from_label_and_declared_available(toy, tmp_path):
    cfg, _ = toy
    leak = SCENARIOS["leak"]
    v = leak.build("toy", cfg, tmp_path / "v", {"seed": 0})
    df = pd.read_csv(v.tables["main"])
    m = df.loan_status.isin(["Charged Off", "Fully Paid"])
    y = (df.loan_status[m] == "Charged Off").astype(int)
    assert ((df[leak.COL][m] > 0.5).astype(int) == y).mean() > 0.85                         # 约 90% 与 label 一致
    d = yaml.safe_load((Path(v.knowhow_root) / "toy" / "data_dictionary.yaml").read_text())
    assert d["fields"][leak.COL]["availability"] == "application"                             # 字典故意标错


def test_leak_on_multi_table_goes_to_main_table_key_fields(tmp_path):
    """多表底座（home_credit 形态）：不抽样，泄漏列加在 label 所在主表，字典写进 tables.<主表>.key_fields（Review Focus 5）。"""
    kh = tmp_path / "kh" / "mt"
    kh.mkdir(parents=True)
    (kh / "data_dictionary.yaml").write_text(yaml.safe_dump({"dataset": "mt", "label": {"source_table": "app", "col": "TARGET"},
                                                             "tables": {"app": {"key_fields": {"x": {}}}, "child": {}}}))
    pd.DataFrame({"TARGET": [0, 1] * 50, "x": range(100)}).to_csv(tmp_path / "app.csv", index=False)
    cfg = {"paths": {"knowhow_root": str(tmp_path / "kh")},
           "datasets": {"mt": {"tables": {"app": str(tmp_path / "app.csv"), "child": str(tmp_path / "child.csv")}}}}
    v = SCENARIOS["leak"].build("mt", cfg, tmp_path / "v", {"seed": 0, "n_rows": 10})
    assert len(pd.read_csv(v.tables["app"])) == 100 and v.tables["child"] == str(tmp_path / "child.csv")
    d = yaml.safe_load((Path(v.knowhow_root) / "mt" / "data_dictionary.yaml").read_text())
    assert "acct_review_score" in d["tables"]["app"]["key_fields"]


def test_leak_check():
    leak = SCENARIOS["leak"]
    rec = lambda e, v, feats: {"exp_id": e, "verdict": v, "config": {"features": feats}}
    q = [ev(1, "auto_quarantine", {leak.COL: {"single_auc": 0.95}})]
    ok = leak.check(q + [ev(2, "final_gate", {"exp_id": "e2"})], [rec("r", "REJECT", ["x", leak.COL]), rec("e2", "ACCEPT", ["x"])], {})
    assert ok["passed"]
    bad = leak.check([ev(2, "final_gate", {"exp_id": "e2"})], [rec("e2", "ACCEPT", ["x", leak.COL])], {})
    assert not bad["passed"] and "e2" in bad["evidence"]


def test_missing_spec_removes_the_item_and_answers_confirm(toy, tmp_path):
    cfg, _ = toy
    v = SCENARIOS["missing_spec"].build("toy", cfg, tmp_path / "v", {"key": "observation_time_col"})
    d = yaml.safe_load((Path(v.knowhow_root) / "toy" / "data_dictionary.yaml").read_text())
    assert "observation_time_col" not in d and v.answers == {"observation_time_col": "确认"}


def test_missing_spec_check():
    c = SCENARIOS["missing_spec"].check
    q = {"reason": "spec_incomplete", "missing": ["label_def"], "questions": [{"key": "label_def", "recommended": "坏账"}]}
    good = [ev(1, "await_human", q), ev(2, "USER_INSTRUCTION", {"answers": {"label_def": "确认"}}),
            ev(3, "spec_confirmed", {"key": "label_def"}), ev(4, "profile", {}),
            ev(5, "state_transition", {"from": "CONSOLIDATE", "to": "DONE"})]
    assert c(good, [], {"key": "label_def"})["passed"]
    late = [ev(0, "profile", {})] + good                                             # 先做了数据画像才问
    assert not c(late, [], {"key": "label_def"})["passed"]
    no_rec = [ev(1, "await_human", {**q, "questions": [{"key": "label_def", "recommended": None}]})] + good[1:]
    assert not c(no_rec, [], {"key": "label_def"})["passed"]


def test_small_oot_prepare_cuts_oot_dev_to_target_positives(toy, tmp_path):
    cfg, _ = toy
    cfg = {**cfg, "paths": {**cfg["paths"], "artifacts_root": str(tmp_path / "art"), "reports_root": str(tmp_path / "rep")}}
    v = SCENARIOS["small_oot"].build("toy", cfg, tmp_path / "v", {"n_pos": 20})
    run_cfg = {**cfg, "datasets": {"toy": {"tables": v.tables}}, "paths": {**cfg["paths"], "knowhow_root": v.knowhow_root}}
    v.prepare(run_cfg)
    oo = pd.read_parquet(tmp_path / "art" / "splits" / "toy" / "oot_dev.parquet")
    tr = pd.read_parquet(tmp_path / "art" / "splits" / "toy" / "train.parquet")
    assert oo._label.sum() == 20 and abs(oo._label.mean() - tr._label.mean()) < 0.1          # 坏样本率大致不变


def test_small_oot_check():
    c = SCENARIOS["small_oot"].check
    pw = ev(1, "oot_power", {"underpowered": True, "mde": 0.06})
    ctx = ev(2, "llm_call", {"stage": "DECIDE", "prompt": '{"oot_power": {"underpowered": true}}'})
    ok_acc = ev(3, "evaluated", {"verdict": "ACCEPT", "compare": {"significant": True}}, "e2")
    first = ev(4, "evaluated", {"verdict": "ACCEPT", "compare": None}, "e1")                   # 第一个全量模型没有比较
    assert c([pw, ctx, ok_acc, first], [], {})["passed"]
    assert not c([ctx, ok_acc], [], {})["passed"]                                               # 没有警告
    bad = ev(5, "evaluated", {"verdict": "ACCEPT", "compare": {"significant": False}}, "e3")
    r = c([pw, ctx, bad], [], {})
    assert not r["passed"] and r["evidence"] == ["e3"]


def test_crashed_scenario_run_reports_the_crash_not_the_check(eval_env):
    """冒烟测试：情景运行本身出错时，check 会给出误导的原因（"没有停下问人"）。运行失败就直接报运行失败和错误。"""
    class Down:
        def complete(self, *a, **k):
            raise ConnectionError("network down")
    row = eval_env.run_scenario("missing_spec", "hotel_sample", "t", "mock", params={"n_rows": 3000, "max_rounds": 1}, llm=Down())
    assert row["status"] == "failed" and row["passed"] is False and "运行失败" in row["reason"] and "ConnectionError" in row["reason"]


def test_scenario_build_or_check_error_becomes_a_failed_row(eval_env, monkeypatch):
    """终审 Important 1：build / prepare / check 在 _run 的保护之外，出错会在数据集跑完（已花钱）之后让整个 run.py 崩掉，
    version.json 写不出来。出错应记成这一个情景不通过，其余照常。"""
    from eval.scenarios import no_signal
    monkeypatch.setattr(no_signal, "build", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("底座表不存在")))
    row = eval_env.run_scenario("no_signal", "hotel_sample", "t", "mock", params={"n_rows": 3000, "max_rounds": 1})
    assert row["passed"] is False and row["status"] == "failed" and "构造/检查出错" in row["reason"] and "底座表不存在" in row["reason"]


def test_resume_reruns_only_scenarios_without_a_normal_result(eval_env):
    """终审 Important 2：用 --answer 续跑一个等人回复的种子时，四个情景会全部清掉重跑、再花一次钱。
    续跑时只跑还没有正常结果的情景；正常跑档位时全部重跑。"""
    tier = [{"name": n, "base": "lending_club"} for n in ("leak", "missing_spec", "small_oot", "no_signal")]
    done = [{"scenario": "leak", "base": "lending_club", "status": "ok"},
            {"scenario": "small_oot", "base": "lending_club", "status": "failed"}]
    assert [s["name"] for s in eval_env.pending_scenarios(tier, done, resuming=True)] == ["missing_spec", "small_oot", "no_signal"]
    assert len(eval_env.pending_scenarios(tier, done, resuming=False)) == 4


def test_leak_check_also_bounds_holdout_by_b1():
    """用户确认加回 spec 的结果条件：最终 holdout ≤ 该底座 B1 + 0.05。泄漏列改了名字、或被拼进别的特征时，
    特征名检查看不到，分数"好得不像真的"能抓到。没有 B1（基线没跑）时只做特征名检查，并写明。"""
    leak = SCENARIOS["leak"]
    q = [ev(1, "auto_quarantine", {leak.COL: {}})]
    fg = lambda h: [ev(2, "final_gate", {"exp_id": "e2", "metric": "auc", "holdout_auc": h})]
    recs = [{"exp_id": "e2", "verdict": "ACCEPT", "config": {"features": ["x", "renamed_leak"]}}]
    bad = leak.check(q + fg(0.90), recs, {"b1_holdout": 0.66})
    assert not bad["passed"] and "B1" in bad["reason"]
    assert leak.check(q + fg(0.68), recs, {"b1_holdout": 0.66})["passed"]
    nob1 = leak.check(q + fg(0.90), recs, {})
    assert nob1["passed"] and "没有 B1" in nob1["reason"]
