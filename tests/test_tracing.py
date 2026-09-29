"""Langfuse 上报：没 key 不启用；有 key 时用真实 SDK + 内存导出器离线校验 span 层级；上报出错不影响建模。"""
import sys
import uuid

import pytest

from agent.llm import PolicyMockLLM, ScriptedLLM
from agent.orchestrator import Orchestrator
from agent.states import State
from conftest import toy_spec
from evaluation.final_gate import FinalGate
import runtime.tracing as tracing
from runtime.tracing import NoopTracer, make_tracer


def build(cfg, llm, tracer, task="t1"):
    return Orchestrator(toy_spec(cfg), cfg, llm, FinalGate("toy", cfg, task), task, tracer=tracer)


def test_disabled_without_keys_or_sdk(monkeypatch):
    monkeypatch.setattr(tracing, "load_env", lambda: [])                  # 不读仓库里的 .env
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert type(make_tracer("t")) is NoopTracer
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    monkeypatch.setitem(sys.modules, "langfuse", None)                   # 有 key 但没装 SDK
    assert type(make_tracer("t")) is NoopTracer


@pytest.fixture
def otel():
    pytest.importorskip("langfuse")
    from langfuse import Langfuse
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    exp = InMemorySpanExporter()
    client = Langfuse(public_key=f"pk-lf-{uuid.uuid4().hex}", secret_key="sk-lf-test", host="http://127.0.0.1:9",  # 客户端按 key 单例
                      span_exporter=exp, tracer_provider=TracerProvider())
    return client, exp


def tree(exp):
    spans = exp.get_finished_spans()
    by_id = {s.context.span_id: s for s in spans}
    parent = lambda s: by_id.get(s.parent.span_id) if s.parent else None
    kind = lambda s: s.attributes.get("langfuse.observation.type")
    return spans, parent, kind


def test_span_tree_matches_rounds_and_llm_calls(toy_cfg, otel):
    client, exp = otel
    tr = make_tracer("t1", {"dataset": "toy", "llm": "mock"}, client=client)
    assert tr.enabled
    o = build(toy_cfg, PolicyMockLLM(), tr)
    assert o.run() == State.DONE
    spans, parent, kind = tree(exp)
    roots = [s for s in spans if parent(s) is None]
    assert len(roots) == 1 and roots[0].name == "task t1" and kind(roots[0]) == "agent"
    assert {s.attributes.get("session.id") for s in spans if kind(s) != "event"} == {"t1"}
    segs = [s.name for s in sorted(spans, key=lambda s: s.start_time) if parent(s) is roots[0]]
    to_decide = [e for e in o.events.query("state_transition") if e["payload"]["to"] == "DECIDE"]
    assert segs == ["准备"] + [f"第 {i} 轮" for i in range(1, len(to_decide) + 1)] + ["收尾"]
    gens = [s for s in spans if kind(s) == "generation"]
    calls = o.events.query("llm_call")
    assert len(gens) == len(calls) and all(s.attributes.get("langfuse.observation.model.name") == "PolicyMockLLM" for s in gens)
    assert {parent(g).name for g in gens} == {"PLAN", "DECIDE", "REPORT", "CONSOLIDATE"}   # 挂在所处状态下
    evald = [s for s in spans if s.name == "evaluated"]
    assert len(evald) == len(o.events.query("evaluated")) and all(kind(s) == "event" for s in evald)
    assert all(kind(parent(s)) == "span" for s in evald)


def test_llm_error_marks_generation_and_root(toy_cfg, otel):
    client, exp = otel

    class Boom:
        model = "boom"

        def complete(self, *a, **k):
            raise RuntimeError("429 insufficient balance <ak-secret123>")

    o = build(toy_cfg, Boom(), make_tracer("t2", client=client), task="t2")
    with pytest.raises(RuntimeError):
        o.run()
    spans, parent, kind = tree(exp)
    g = next(s for s in spans if kind(s) == "generation")
    assert g.attributes.get("langfuse.observation.level") == "ERROR"
    assert "ak-***" in g.attributes.get("langfuse.observation.status_message") and "secret123" not in str(g.attributes)
    root = next(s for s in spans if parent(s) is None)
    assert root.attributes.get("langfuse.observation.level") == "ERROR"


def test_tracer_failures_never_break_the_run(toy_cfg):
    class Obs:
        trace_id = "x"

        def start_observation(self, **kw):
            return Obs()

        def create_event(self, **kw):
            raise ConnectionError("langfuse down")

        def update(self, **kw):
            raise ConnectionError("langfuse down")

        def end(self, **kw):
            pass

    class Client(Obs):
        def flush(self):
            raise ConnectionError("langfuse down")

    tr = make_tracer("t3", client=Client())
    assert tr.enabled
    assert build(toy_cfg, PolicyMockLLM(), tr, task="t3").run() == State.DONE


def test_intake_session_reported_as_one_trace(otel):
    """接入会话：一条 trace，命名 dataminer · intake · <id>；画像、每轮起草（LLM 调用、校验、追问、回答）、写出、对比。"""
    import json
    from pathlib import Path
    from intake.tracing import report_intake
    client, exp = otel
    rec = json.loads((Path(__file__).resolve().parents[1] / "examples/traces/intake/intake_hotel_deepseek.json").read_text())
    report_intake(rec, client=client)
    spans, parent, kind = tree(exp)
    roots = [s for s in spans if parent(s) is None]
    assert len(roots) == 1 and roots[0].name == f"intake {rec['dataset_id']}" and kind(roots[0]) == "agent"
    assert {s.attributes.get("session.id") for s in spans if kind(s) != "event"} == {f"intake_{rec['dataset_id']}"}
    n_rounds = max(e["round"] for e in rec["events"])
    kids = [s.name for s in sorted(spans, key=lambda s: s.start_time) if parent(s) is roots[0]]
    assert kids == ["数据画像"] + [f"第 {i} 轮起草" for i in range(1, n_rounds + 1)] + ["写出配置", "与手写配置对比"]
    gens = [s for s in spans if kind(s) == "generation"]
    assert len(gens) == sum(e["type"] == "llm_call" for e in rec["events"])
    assert all(parent(g).name.endswith("轮起草") for g in gens)
    assert sum(s.name == "用户回答" for s in spans) == sum(e["type"] == "answers" for e in rec["events"])
