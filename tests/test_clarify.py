"""前置检查缺必答项：推荐答案的代码校验、确认后写进 spec。"""
from agent import clarify
from agent.schemas import TaskSpec

COLS = ["SK_ID_CURR", "DAYS_BIRTH", "DAYS_CREDIT", "TARGET"]


def test_observation_time_accepts_a_column_or_a_relative_declaration():
    assert clarify.validate("observation_time_col", {"col": "DAYS_BIRTH"}, COLS) == []
    assert clarify.validate("observation_time_col", {"relative_to": "申请时刻", "offset_cols": ["DAYS_*"]}, COLS) == []
    assert clarify.validate("observation_time_col", {"col": "issue_d"}, COLS)                        # 列不存在
    assert clarify.validate("observation_time_col", {"relative_to": "申请时刻", "offset_cols": ["MONTHS_*"]}, COLS)   # 一列都匹配不上
    assert clarify.validate("observation_time_col", {"col": "DAYS_BIRTH", "relative_to": "申请时刻"}, COLS)          # 两种写法只能选一种
    assert clarify.validate("label_def", {"definition": ""}, COLS)
    assert clarify.validate("oot_windows", {"holdout": 0.15}, COLS)                                  # 至少要有 oot_dev


def test_confirmed_relative_time_satisfies_the_spec():
    """Home Credit 没有绝对时间：确认"相对申请时刻"之后，观察时间这一项就不再缺。"""
    spec = TaskSpec(dataset="d", label_def="x", oot_windows={"oot_dev": 0.1}, budget={"tokens": 1, "cpu_minutes": 1, "wall_minutes": 1})
    assert spec.missing_required() == ["observation_time_col"]
    d = clarify.apply(spec.model_dump(), "observation_time_col", {"relative_to": "申请时刻", "offset_cols": ["DAYS_*"]})
    s2 = TaskSpec(**d)
    assert s2.missing_required() == [] and s2.observation_time_col is None
    assert s2.observation_time_relative == {"relative_to": "申请时刻", "offset_cols": ["DAYS_*"]}
    assert TaskSpec(**clarify.apply(d, "observation_time_col", {"col": "DAYS_BIRTH"})).observation_time_col == "DAYS_BIRTH"


def test_malformed_json_is_fed_back_not_fatal(tmp_path):
    """v5 评测 home_credit s0：LLM 多写了一个右括号，JSON 解析异常直接让这个种子失败。格式错误应回灌重写（与接入阶段一致）。"""
    from agent.llm import extract_json
    replies = iter(['```json\n{"proposals": {"label_def": {"value": {"definition": "x"}}}}}\n```',
                   '```json\n{"proposals": {"label_def": {"question": "q", "recommended": "x", "reason": "r", "value": {"definition": "x"}}}}\n```'])
    seen = []
    llm_json = lambda system, user: (seen.append(user), extract_json(next(replies)))[1]
    cfg = {"paths": {"knowhow_root": str(tmp_path), "artifacts_root": str(tmp_path)}}
    out = clarify.draft(llm_json, "d", cfg, ["label_def"])
    assert out["label_def"]["value"] == {"definition": "x"} and "JSON" in seen[1]
