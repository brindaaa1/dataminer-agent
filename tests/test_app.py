"""app.py 的 AppTest：数据卡片、历史回放、现场运行（mock 跑通 + LLM 报错时的降级），都不需要网络和 key。"""
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from runtime.tracing import NoopTracer

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def texts(at) -> str:
    return "\n".join(str(x.value) for kind in ("markdown", "caption", "subheader", "error", "info", "warning", "success", "code")
                     for x in getattr(at, kind))


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("DATAMINER_SAMPLE_ART", str(tmp_path / "art"))      # 现场运行的产物写到临时目录
    monkeypatch.setattr("runtime.tracing.make_tracer", lambda *a, **k: NoopTracer())  # 测试不往 Langfuse 发数据
    return AppTest.from_file(APP, default_timeout=300).run()


def test_landing_shows_data_head_and_tabs(app):
    assert not app.exception
    assert app.metric[0].value == "7,997" and len(app.dataframe[0].value) == 8     # 样本前 8 行
    assert [t.label for t in app.tabs][:2] == ["① 历史回放：真实 LLM 的一次运行", "② 现场运行：现在跑一次"]
    t = texts(app)
    assert "历史回放" in t and "不是实时运行" in t and "--task-id" in t              # 说明回放性质和名字来历
    assert "回放结束" not in t                                                     # 点回放之前不展示结果


def test_default_example_is_deepseek_on_hotel_sample(app):
    assert app.selectbox(key="example").value.startswith("hotel_deepseek3")      # 有样本数据的示例排第一
    app.select_slider(key="speed").set_value("快").run()
    app.button(key="replay").click().run()
    assert not app.exception
    t = texts(app)
    assert "规划方向" in t and "模型赛跑" in t and "⚠️ 这次运行用的不是" not in t
    assert "holdout AUC" in t and "全量复验" in t                                    # 有最终模型
    app.toggle(key="full_replay").set_value(True).run()
    t = texts(app)
    assert "deepseek-chat" in t                                                      # 完整 trace：LLM 调用原文


def test_replay_plays_rounds_with_rationale(app):
    sb = app.selectbox(key="example")
    sb.select_index(next(i for i, o in enumerate(sb.options) if o.startswith("lc_kimi3"))).run()
    app.select_slider(key="speed").set_value("快").run()
    app.button(key="replay").click().run()
    assert not app.exception
    t = texts(app)
    assert "回放结束：5 轮决策" in t and "达到最大轮数" in t
    assert "<dt>假设</dt>" in t and "<dt>理由</dt>" in t and 'dm-pill dm-v-ACCEPT' in t
    assert "被代码校验拒绝" in t and "PLATEAU" in t                                  # kimi3 第 4 轮前被拒 3 次
    assert "⚠️ 这次运行用的不是上面的酒店样本" in t
    assert not [e for e in app.expander if e.label.startswith("第 ")]              # 完整 trace 默认收起
    app.toggle(key="full_replay").set_value(True).run()
    heads = [e.label for e in app.expander]
    assert sum(h.startswith("第 ") for h in heads) == 5                             # 完整 trace 的逐轮展开


def test_live_mock_run(app):
    app.selectbox(key="provider").set_value("mock（确定性策略，无需 key）").run()
    app.slider(key="max_rounds").set_value(2).run()
    app.button(key="run").click().run()
    assert not app.exception, app.exception
    t = texts(app)
    assert "最终状态 DONE" in t and "dm-pill" in t and "holdout AUC" in t
    assert "<b>lgbm</b>" in t                                                         # 准备段：模型赛跑
    app.toggle(key="full_live").set_value(True).run()
    assert "PolicyMockLLM" in texts(app)                                            # 完整 trace 里有 llm_call 原文


def test_live_llm_error_degrades_and_redacts_key(app, monkeypatch):
    class Broke:
        model = "fake"

        def complete(self, *a, **k):
            raise RuntimeError("Error code: 429 - account <ak-secret123> suspended due to insufficient balance")

    got = {}

    def fake_make_llm(provider, cfg, api_key=None):
        got["key"] = api_key
        return Broke()

    monkeypatch.setattr("agent.llm.make_llm", fake_make_llm)
    app.selectbox(key="provider").set_value("DeepSeek").run()
    app.text_input(key="api_key").input("sk-user-typed-key").run()
    app.button(key="run").click().run()
    assert not app.exception, app.exception
    assert got["key"] == "sk-user-typed-key"                                        # 填的 key 直接交给客户端
    t = texts(app)
    assert "账户余额不足或触发限流" in t and "已跑完的部分保留在下面" in t
    assert "ak-secret123" not in t and "sk-user-typed-key" not in t                 # 页面上不出现 key 片段
    app.toggle(key="full_live").set_value(True).run()
    assert "LLM 调用原文" in [tab.label for tab in app.tabs]                          # PLAN 失败前的部分仍然可看
    assert "ak-secret123" not in texts(app)


def test_ui_wording_avoids_fidelity_jargon():
    """界面自己的文字不用"保真度"（LLM 原文照录，不在此列）。"""
    import ast
    src = Path(APP).read_text()
    tree = ast.parse(src)
    strings = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    offenders = [x for x in strings if "保真" in x and "LLM 原文" not in x]
    assert not offenders, offenders


def test_live_failed_run_is_not_shown_as_success(app, monkeypatch):
    class Garbage:                                                   # PLAN 容错；DECIDE 连续输出非法 → L0 下 FAILED
        model = "garbage"

        def complete(self, system, user, temperature=0.0):
            return "不是 JSON", 0

    monkeypatch.setattr("agent.llm.make_llm", lambda provider, cfg, api_key=None: Garbage())
    app.selectbox(key="provider").set_value("DeepSeek").run()
    app.text_input(key="api_key").input("sk-test").run()
    app.button(key="run").click().run()
    assert not app.exception, app.exception
    assert any("最终状态 FAILED" in w.value for w in app.warning)
    assert not any("最终状态" in s.value for s in app.success)


def test_replay_shows_trial_curves_accepted_expanded(app):
    app.select_slider(key="speed").set_value("快").run()
    app.button(key="replay").click().run()
    assert not app.exception
    ex = [e for e in app.expander if e.label.startswith("调参过程")]
    assert ex and any("被剪枝" in e.label for e in ex)
    assert "Optuna 只看 valid" in texts(app)


def test_trace_without_trials_or_metric_renders(tmp_path, monkeypatch):
    ex = json.loads((Path(APP).parent / "examples/traces/hotel_deepseek3.json").read_text())
    ex.pop("trials")
    ex["events"] = [e for e in ex["events"] if e["type"] != "spec_locked"]
    (tmp_path / "old.json").write_text(json.dumps(ex, ensure_ascii=False))
    monkeypatch.setenv("DATAMINER_EXAMPLES", str(tmp_path))         # app.py 从这个目录读回放
    monkeypatch.setenv("DATAMINER_SAMPLE_ART", str(tmp_path / "art"))
    at = AppTest.from_file(APP, default_timeout=300).run()
    at.select_slider(key="speed").set_value("快").run()
    at.button(key="replay").click().run()
    assert not at.exception and "未保存调参过程" in texts(at) and "holdout AUC" in texts(at)


def test_primary_of_reads_locked_metric():
    from app_helpers import metric_label, primary_of
    assert primary_of([{"type": "spec_locked", "payload": {"metric": {"primary": "pr_auc"}}}]) == "pr_auc"
    assert primary_of([]) == "auc" and metric_label("pr_auc") == "PR-AUC"


def test_intake_tab_replays_and_runs_generated_config(app):
    assert [t.label for t in app.tabs][2] == "③ 接入新数据：从数据和说明生成配置"
    app.button(key="intake_replay").click().run()
    assert not app.exception
    t = texts(app)
    assert "必答项" in t and "与手写配置对比" in t
    assert {"data_dictionary.yaml", "blacklist.yaml"} <= {e.label for e in app.expander}
    app.button(key="intake_run").click().run()
    assert not app.exception
    assert "最终状态 DONE" in texts(app)


def test_trial_rows_only_count_pruned_as_pruned():
    from app_helpers import trial_rows
    tr = {"e1": [{"n": 0, "value": .7, "state": "COMPLETE"}, {"n": 1, "value": None, "state": "PRUNED"},
                 {"n": 2, "value": None, "state": "FAIL"}, {"n": 3, "value": .8, "state": "COMPLETE"}]}
    df = trial_rows(tr, ["e1"])
    assert list(df["状态"]) == ["完成", "剪枝", "完成"] and list(df["至今最优"]) == [.7, .7, .8]
