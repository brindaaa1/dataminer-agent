"""用户偏好（工作台本地）：同一结构的数据确认过的口径、每个领域上次用的指标；接入起草时作为推荐答案，仍由用户确认。"""
from memory.preferences import recall, remember

DRAFT = {"label": {"col": "y", "definition": "流失"}, "observation_time_expr": "t", "split_time_expr": "t",
         "windows": {"train_valid": ["2020-01", "2020-12"]}, "domain": "_default",
         "metric": {"primary": "pr_auc", "guards": {"auc": 0.02}}, "fields": {"a": {}}}


def test_same_structure_data_recalls_confirmed_answers(tmp_path):
    p = tmp_path / "preferences.json"
    assert recall(p, ["a", "y", "t"]) == {"same_data": None, "metric_by_domain": {}}
    remember(p, ["a", "y", "t"], DRAFT)
    r = recall(p, ["t", "y", "a"])                                    # 列顺序不同也算同一结构
    assert r["same_data"] == {k: DRAFT[k] for k in ("label", "observation_time_expr", "split_time_expr", "windows")}
    assert r["metric_by_domain"] == {"_default": DRAFT["metric"]}
    assert recall(p, ["a", "y", "t", "new_col"])["same_data"] is None


def test_session_puts_preferences_into_the_draft_prompt(tmp_path):
    from agent.llm import ScriptedLLM
    from intake.session import IntakeSession
    from test_intake import _csv, _good
    full = _good().model_dump(mode="json")
    llm = ScriptedLLM([full])
    s = IntakeSession(llm, _csv(tmp_path), "x", "t_pref", str(tmp_path / "o"), prefs={"metric_by_domain": {"_default": {"primary": "ks"}}})
    s.step()
    assert '"preferences"' in llm.calls[0] and '"ks"' in llm.calls[0]
