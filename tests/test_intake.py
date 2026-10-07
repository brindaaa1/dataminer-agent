import numpy as np
import pandas as pd

from intake.profile import profile


def _csv(tmp_path):
    r = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({"order_year": r.choice([2015, 2016, 2017], n), "order_month": r.choice(["January", "July"], n),
                       "lead_time": r.integers(0, 300, n), "channel": r.choice(["web", "agent"], n),
                       "is_canceled": r.integers(0, 2, n), "status_date": ["2016-01-02"] * n})
    p = tmp_path / "t.csv"
    df.to_csv(p, index=False)
    return str(p)


def test_profile_finds_dates_and_labels(tmp_path):
    pr = profile(_csv(tmp_path))
    assert pr["n_rows"] == 400
    assert {"order_year", "order_month", "status_date"} <= set(pr["date_candidates"])
    assert pr["label_candidates"] == ["is_canceled"]
    assert len(pr["columns"]["channel"]["examples"]) <= 5


DATE = "make_date(order_year, CASE order_month WHEN 'January' THEN 1 ELSE 7 END, 1)"


def _good():
    from intake.schemas import IntakeDraft
    return IntakeDraft.model_validate({
        "label": {"col": "is_canceled", "definition": "订单被取消"},
        "observation_time_expr": DATE, "split_time_expr": DATE,
        "windows": {"train_valid": ["2015-01", "2015-12"], "oot_dev": ["2016-01", "2016-12"], "holdout": ["2017-01", "2017-12"]},
        "fields": {"lead_time": {"type": "numeric", "availability": "at_prediction"}},
        "leakage": ["status_date"], "metric": {"primary": "auc", "guards": {"pr_auc": 0.02}}})


def test_validate_passes_good_draft(tmp_path):
    from intake.validate import validate
    assert validate(_good(), _csv(tmp_path)) == []


def test_validate_reports_each_problem(tmp_path):
    from intake.validate import validate
    csv = _csv(tmp_path)
    bad = _good().model_copy(update={"split_time_expr": "strptime(nope, '%Y')"})
    assert any("split_time_expr" in e for e in validate(bad, csv))
    bad = _good().model_copy(update={"leakage": ["ghost"]})
    assert any("ghost" in e for e in validate(bad, csv))
    bad = _good().model_copy(update={"windows": _good().windows.model_copy(update={"holdout": ["2019-01", "2019-12"]})})
    assert any("holdout" in e for e in validate(bad, csv))


def test_required_items_need_user_confirmation():
    from intake.validate import missing_confirmations, value
    qs = missing_confirmations(_good(), {k: {"value": value(_good(), k)} for k in ("label", "metric")})
    assert {q.key for q in qs} == {"observation_time_expr", "windows"}
    assert all(q.recommended for q in qs)


def test_perfect_draft_roundtrip_runs_full_pipeline(tmp_path):
    """手写的酒店配置读成草稿 → 写出 → 注册 → mock 跑完整流程；与手写配置对比应完全一致。"""
    from agent.llm import PolicyMockLLM
    from agent.orchestrator import Orchestrator
    from agent.schemas import TaskSpec
    from agent.states import State
    from data.knowhow import load_config
    from data.sample import SAMPLE_CSV, use_sample
    from evaluation.final_gate import FinalGate
    from intake.compare import compare
    from intake.writer import from_knowhow_dir, register, write
    cfg = use_sample(load_config(), tmp_path / "art")
    (tmp_path / "art").mkdir()
    cfg["run"]["max_rounds"], cfg["fidelity"]["full"]["n_trials"], cfg["fidelity"]["low"]["n_trials"] = 1, 3, 3
    d = from_knowhow_dir("hotel_bookings", "knowhow", cfg["task_specs"]["hotel_bookings"])
    w = write(d, "hotel_intake", str(SAMPLE_CSV), str(tmp_path / "intake"))
    register(cfg, w)
    spec = TaskSpec.from_knowhow("hotel_intake", cfg, autonomy="L0")
    o = Orchestrator(spec, cfg, PolicyMockLLM(), FinalGate("hotel_intake", cfg, "rt"), "rt")
    assert o.run() == State.DONE
    c = compare("hotel_intake", w["knowhow_root"], "knowhow", str(SAMPLE_CSV))
    assert c["leak_missed"] == [] and c["leak_extra"] == [] and c["quarantine_missed"] == [] and c["availability_agree"] == 1.0
    assert c["meta_missed"] == []                                   # arrival_date_year：只用于切分，不是泄漏，单独统计
    assert "arrival_date_year" not in o.store.all("rt")[0].config["features"]


def test_meta_fields_are_dropped_not_used_as_features(tmp_path):
    import yaml
    from intake.writer import write
    from intake.schemas import IntakeDraft
    d = IntakeDraft.model_validate({**_good().model_dump(), "fields": {"order_year": {"type": "numeric", "availability": "meta"}}})
    w = write(d, "m1", _csv(tmp_path), str(tmp_path / "o"))
    bl = yaml.safe_load(w["yaml"]["blacklist.yaml"])
    assert bl["meta"]["fields"] == ["order_year"]


def test_session_asks_then_writes(tmp_path):
    from agent.llm import ScriptedLLM
    from intake.session import IntakeSession
    csv = _csv(tmp_path)
    full = _good().model_dump(mode="json")
    first = {**full, "windows": None, "questions": [{"key": "lead_time", "question": "lead_time 何时已知？", "recommended": "下单时"}]}
    first_fixed = {**full, "questions": first["questions"]}
    bad = {**full, "split_time_expr": "strptime(nope, '%Y')"}
    llm = ScriptedLLM([first, first_fixed, bad, full])       # 每轮先给一个不合格的草稿，校验错误回灌后重写
    s = IntakeSession(llm, csv, "预测订单是否取消", "t_intake", str(tmp_path / "out"))
    qs = s.step()
    assert {q.key for q in qs} == {"lead_time", "label", "observation_time_expr", "windows", "metric"}
    s.answer({q.key: q.recommended for q in qs})
    assert s.step() == [] and s.written
    types = [e["type"] for e in s.events]
    assert types.count("validation_failed") == 2 and types[-1] == "written"
    assert "请修正" in llm.calls[1] and "请修正" in llm.calls[3]


def test_after_max_rounds_only_required_questions_remain(tmp_path):
    from agent.llm import ScriptedLLM
    from intake.session import IntakeSession
    ask = {**_good().model_dump(mode="json"), "questions": [{"key": "channel", "question": "q", "recommended": "r"}]}
    s = IntakeSession(ScriptedLLM([ask] * 5), _csv(tmp_path), "x", "t2", str(tmp_path / "o"), max_rounds=2)
    s.step()
    s.answer({"channel": "r"})
    ask["questions"] = [{"key": "lead_time", "question": "q2", "recommended": "r2"}]
    assert "lead_time" in {q.key for q in s.step()}           # 第 2 轮仍可追问
    s.answer({"lead_time": "r2"})
    ask["questions"] = [{"key": "adr", "question": "q3", "recommended": "r3"}]
    qs = s.step()                                   # 追问轮数用完：不再追问非必答项
    assert {q.key for q in qs} == {"label", "observation_time_expr", "windows", "metric"}


def test_session_state_roundtrips_through_json(tmp_path):
    import json
    from agent.llm import ScriptedLLM
    from intake.session import IntakeSession
    full = _good().model_dump(mode="json")
    s = IntakeSession(ScriptedLLM([full]), _csv(tmp_path), "x", "t3", str(tmp_path / "o"))
    s.step()
    s2 = IntakeSession.from_json(json.loads(json.dumps(s.to_json())), ScriptedLLM([full]))
    s2.answer({k: "确认" for k in ("label", "observation_time_expr", "windows", "metric")})
    assert s2.step() == [] and s2.written and len(s2.events) > len(s.events)


def test_schema_error_is_fed_back_for_rewrite(tmp_path):
    from agent.llm import ScriptedLLM
    from intake.session import IntakeSession
    full = _good().model_dump(mode="json")
    wrong = {**full, "fields": {"lead_time": {"type": "meta", "availability": "at_prediction"}}}
    llm = ScriptedLLM([wrong, full])
    s = IntakeSession(llm, _csv(tmp_path), "x", "t4", str(tmp_path / "o"))
    s.step()
    assert "请修正" in llm.calls[1] and "type" in llm.calls[1]


# ---------- review 修复 ----------
def _str_label_csv(tmp_path):
    r = np.random.default_rng(1)
    n = 600
    df = pd.DataFrame({"yr": r.choice([2015, 2016, 2017], n), "lead_time": r.integers(0, 300, n),
                       "order status": r.choice(["Canceled", "Kept"], n)})
    p = tmp_path / "s.csv"
    df.to_csv(p, index=False)
    return str(p), df


def _str_draft(pos="Canceled", neg="Kept"):
    from intake.schemas import IntakeDraft
    e = "make_date(yr, 6, 1)"
    return IntakeDraft.model_validate({
        "label": {"col": "order status", "positive": pos, "negative": neg, "definition": "取消"},
        "observation_time_expr": e, "split_time_expr": e,
        "windows": {"train_valid": ["2015-01", "2015-12"], "oot_dev": ["2016-01", "2016-12"], "holdout": ["2017-01", "2017-12"]},
        "fields": {"lead_time": {"type": "numeric", "availability": "at_prediction"}}, "metric": {"primary": "auc"}})


def test_label_values_drive_the_split(tmp_path):
    """正类取值必须真正用于打标签：文字标签能切分，确认"正类是 Kept"时标签随之翻转。"""
    from data.knowhow import load_config
    from data.splits import build_splits
    from data.access import DataAccess
    from intake.validate import validate
    from intake.writer import register, write
    csv, df = _str_label_csv(tmp_path)
    for pos, neg in (("Canceled", "Kept"), ("Kept", "Canceled")):
        d = _str_draft(pos, neg)
        assert validate(d, csv) == []
        cfg = load_config()
        cfg["paths"].update(artifacts_root=str(tmp_path / f"art_{pos}"), reports_root=str(tmp_path / f"rep_{pos}"))
        register(cfg, write(d, "s", csv, str(tmp_path / f"o_{pos}")))
        build_splits("s", cfg)
        tr = DataAccess("s", cfg["paths"]["artifacts_root"]).load("train")
        want = (df.loc[df.yr == 2015, "order status"] == pos).mean()
        assert abs(tr["_label"].mean() - want) < 0.1


def test_confirmed_values_are_what_gets_written(tmp_path):
    """用户改了必答项 → 下一版草稿里的新值要再确认一次；写出的必答项等于用户确认过的值。"""
    import yaml
    from agent.llm import ScriptedLLM
    from intake.session import IntakeSession
    full = _good().model_dump(mode="json")
    changed = {**full, "metric": {"primary": "auc", "guards": {"ks": 0.02}}}
    drifted = {**changed, "label": {**changed["label"], "definition": "换了个说法"}}
    s = IntakeSession(ScriptedLLM([full, changed, drifted]), _csv(tmp_path), "x", "c1", str(tmp_path / "o"))
    qs = s.step()
    s.answer({q.key: ("护栏改用 ks" if q.key == "metric" else q.recommended) for q in qs})
    qs = s.step()
    assert [q.key for q in qs] == ["metric"] and "ks" in qs[0].recommended      # 改过的值拿回来再确认
    s.answer({"metric": "确认"})
    assert s.step() == []
    dic = yaml.safe_load(s.written["yaml"]["data_dictionary.yaml"])
    assert dic["label"]["definition"] == "订单被取消"                               # 写出的是确认过的值，不是漂移后的文字
    assert s.written["task_spec"]["metric"]["guards"] == {"ks": 0.02}


def test_non_date_fields_typed_date_are_rejected(tmp_path):
    from intake.validate import validate
    d = _good().model_copy(update={"fields": {**_good().fields}})
    d.fields["order_year"] = type(d.fields["lead_time"])(type="date", availability="at_prediction")
    errs = validate(d, _csv(tmp_path))
    assert any("order_year" in e and "date" in e for e in errs)
    d.fields.pop("order_year")
    d.fields["status_date"] = type(d.fields["lead_time"])(type="date", availability="post_outcome")
    assert validate(d, _csv(tmp_path)) == []


def test_windows_must_be_ordered_and_disjoint(tmp_path):
    from intake.validate import validate
    w = _good().windows.model_copy(update={"oot_dev": ["2015-06", "2016-12"]})
    assert any("重叠" in e for e in validate(_good().model_copy(update={"windows": w}), _csv(tmp_path)))


def test_expressions_cannot_read_local_files(tmp_path):
    from intake.validate import validate
    secret = tmp_path / "secret.txt"
    secret.write_text("x")
    e = f"make_date(order_year, 1, 1) + to_days((SELECT length(content) FROM read_text('{secret}'))::INTEGER)"
    errs = validate(_good().model_copy(update={"observation_time_expr": e}), _csv(tmp_path))
    assert any("observation_time_expr" in x for x in errs)


def test_profile_reports_duckdb_types(tmp_path):
    pr = profile(_csv(tmp_path))
    assert pr["columns"]["status_date"]["sql_type"] == "DATE" and pr["columns"]["order_month"]["sql_type"] == "VARCHAR"


def test_accepting_keeps_the_value_visible_to_the_llm(tmp_path):
    """采纳时回答里要带上被确认的值：只存"确认"会冲掉之前的修改意见，LLM 下一轮就可能改回去。"""
    from agent.llm import ScriptedLLM
    from intake.session import IntakeSession
    full = _good().model_dump(mode="json")
    s = IntakeSession(ScriptedLLM([full, full]), _csv(tmp_path), "x", "k1", str(tmp_path / "o"))
    s.step()
    s.answer({"metric": "确认"})
    assert "pr_auc" in s.answers["metric"]


def test_domain_is_recommended_validated_and_written(tmp_path):
    """接入起草时 LLM 从领域手册里推荐一个领域（上下文里有可选领域），代码校验名字合法，写进 data_dictionary 的 domain。"""
    import yaml
    from agent.llm import ScriptedLLM
    from intake.session import IntakeSession
    from intake.validate import validate
    csv = _csv(tmp_path)
    assert any("domain" in e for e in validate(_good().model_copy(update={"domain": "nope"}), csv))
    full = {**_good().model_dump(mode="json"), "domain": "credit_risk"}
    llm = ScriptedLLM([full, full])
    s = IntakeSession(llm, csv, "预测借款人是否违约", "t_dom", str(tmp_path / "o"))
    s.answer({q.key: q.recommended for q in s.step()})
    assert s.step() == [] and "credit_risk" in llm.calls[0]
    assert yaml.safe_load(s.written["yaml"]["data_dictionary.yaml"])["domain"] == "credit_risk"


def test_generic_availability_values_are_accepted(tmp_path):
    from intake.schemas import IntakeDraft
    from intake.validate import validate
    d = _good().model_dump(mode="json")
    d["fields"]["lead_time"]["availability"] = "at_prediction"
    assert validate(IntakeDraft.model_validate(d), _csv(tmp_path)) == []
    d["fields"]["lead_time"]["availability"] = "before_outcome"
    IntakeDraft.model_validate(d)


def test_draft_carries_a_data_summary_written_by_the_llm():
    """过程页的数据预览不写代码做表格，由起草的 LLM 顺带写几句数据描述。"""
    from intake.schemas import IntakeDraft
    d = IntakeDraft.model_validate({**_good().model_dump(mode="json"), "data_summary": "每行是一笔预订。"})
    assert d.data_summary == "每行是一笔预订。" and IntakeDraft().data_summary == ""


def test_history_fields_are_drafted_and_written(tmp_path):
    """当前值是泄漏、但过去的取值在预测时已知的字段：起草时标 history（取前几条、过去几条的均值），写进数据字典。"""
    import yaml
    from intake.schemas import IntakeDraft
    from intake.writer import write
    d = _good().model_dump(mode="json")
    d["fields"]["status_date"] = {"type": "numeric", "availability": "post_outcome", "history": {"lags": [1, 2], "means": [48]}}
    w = write(IntakeDraft.model_validate(d), "t_hist", _csv(tmp_path), str(tmp_path / "o"))
    dic = yaml.safe_load(w["yaml"]["data_dictionary.yaml"])
    assert dic["fields"]["status_date"]["history"] == {"lags": [1, 2], "means": [48]}
    assert "history" not in dic["fields"]["lead_time"]
