"""参照模型（默认 LightGBM）：开始时训练一次存盘，最终检验量 holdout 漂移时直接读，不重训。"""
from agent.llm import PolicyMockLLM
from agent.orchestrator import Orchestrator
from conftest import make_toy, toy_spec
from evaluation.final_gate import FinalGate


def test_reference_model_is_trained_once_per_task(tmp_path, monkeypatch):
    from evaluation import reference
    cfg = make_toy(tmp_path)
    cfg["evaluator"]["overfit_gap"] = {"mode": "relative", "tolerance": 0.05}
    n = []
    orig = reference._train
    monkeypatch.setattr(reference, "_train", lambda *a, **k: n.append(1) or orig(*a, **k))
    o = Orchestrator(toy_spec(cfg), cfg, PolicyMockLLM(), FinalGate("toy", cfg, "t1"), "t1")
    o.run()
    assert o.p["final"]["reference_drop"] is not None and len(n) == 1
