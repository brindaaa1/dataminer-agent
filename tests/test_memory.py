"""第 12 项测试：Lesson 准入与生命周期由规则决定、指纹检索与冷启动、跨任务 warm-start（≤10%）。"""
import copy

import pytest

from agent.llm import PolicyMockLLM
from agent.orchestrator import Orchestrator
from conftest import make_toy, toy_spec
from data.knowhow import load_config
from evaluation.final_gate import FinalGate
from memory.consolidate import Proposals, admit, allowed_keys, update_lifecycle, validate
from memory.retrieval import active_lessons, fingerprint_distance, nearest_task
from memory.schema import Cost, ExperimentRecord
from memory.store import MemoryStore

CFG = load_config()
FP = {"task_id": "A", "n_samples": 10000, "n_features": 30, "bad_rate": 0.13, "missing_rate": 0.1, "time_span_months": 24,
      "label_def": "x", "feature_domains": []}


def rec(i, parent, action, verdict, auc, model="lgbm", space=None):
    return ExperimentRecord(exp_id=i, task_id="A", parent_exp_id=parent, hypothesis="h", action_type=action,
                            diff={"space": space or {}}, config={"model": model}, config_hash=i, fidelity="full",
                            metrics={"oot_dev_auc": auc}, diagnosis_codes=[], cost=Cost(), verdict=verdict)


@pytest.fixture
def mem(tmp_path):
    return MemoryStore(str(tmp_path / "m.db"))


@pytest.fixture
def tree():
    return [rec("e1", None, "RACE", "PROMISING", .65), rec("e2", "e1", "TUNE", "ACCEPT", .67, space={"learning_rate": {}}),
            rec("e3", "e2", "TUNE", "REJECT", .66, space={"num_leaves": {}})]


def test_allowed_keys_derived_by_code(tree):
    a = allowed_keys(tree)
    assert a == {"race:lgbm": ["e1"], "tune:lgbm:learning_rate@full": ["e2"], "tune:lgbm:num_leaves@full": ["e3"]}


def test_invalid_proposals_are_dropped_and_llm_cannot_set_status(tree):
    a = allowed_keys(tree)
    props = Proposals.model_validate({"lessons": [
        {"key": "tune:lgbm:learning_rate@full", "statement": "s", "evidence": ["e2"], "polarity": "POSITIVE", "status": "ACTIVE"},   # 夹带 status：被忽略
        {"key": "made:up:key", "statement": "s", "evidence": ["e2"], "polarity": "POSITIVE"},                                   # 编造的 key
        {"key": "tune:lgbm:learning_rate@full", "statement": "s", "evidence": ["e99"], "polarity": "POSITIVE"},                      # 不存在的证据
        {"key": "tune:lgbm:num_leaves@full", "statement": "s", "evidence": ["e3"], "polarity": "POSITIVE"},                          # 极性与证据判定矛盾
        {"key": "tune:lgbm:learning_rate@full", "statement": "s", "evidence": ["e2"], "polarity": "NEGATIVE"},                       # 证据有效却说无效
    ]})
    ok, dropped = validate(props, tree, a)
    assert len(ok) == 1 and len(dropped) == 4
    assert not hasattr(ok[0], "status")


def test_lifecycle_candidate_validated_active_deprecated(mem, tree):
    fin_ok, fin_bad = {"drop": 0.005}, {"drop": 0.05}
    props = Proposals.model_validate({"lessons": [{"key": "tune:lgbm:learning_rate@full", "statement": "s", "evidence": ["e2"], "polarity": "POSITIVE"},
                                                  {"key": "tune:lgbm:num_leaves@full", "statement": "n", "evidence": ["e3"], "polarity": "NEGATIVE"}]})
    ok, _ = validate(props, tree, allowed_keys(tree))
    ids = admit(mem, ok, tree, "A", FP, CFG)
    st = lambda: {l["key"]: l["status"] for l in mem.lessons()}
    assert set(st().values()) == {"CANDIDATE"}
    # holdout 落差过大：不晋升
    update_lifecycle(mem, "A", tree, "e2", fin_bad, CFG)
    assert set(st().values()) == {"CANDIDATE"}
    # final_gate 通过且证据在最优实验路径上 → VALIDATED（负向经验不要求在路径上）
    update_lifecycle(mem, "A", tree, "e2", fin_ok, CFG)
    assert set(st().values()) == {"VALIDATED"}
    assert not active_lessons(mem, FP, CFG)                       # 只有 ACTIVE 才会被检索
    # 第二个任务复现同 key 同极性 → ACTIVE
    tree_b = [rec("b1", None, "RACE", "PROMISING", .64), rec("b2", "b1", "TUNE", "ACCEPT", .66, space={"learning_rate": {}})]
    okb, _ = validate(Proposals.model_validate({"lessons": [{"key": "tune:lgbm:learning_rate@full", "statement": "s", "evidence": ["b2"], "polarity": "POSITIVE"}]}), tree_b, allowed_keys(tree_b))
    admit(mem, okb, tree_b, "B", {**FP, "task_id": "B"}, CFG)
    update_lifecycle(mem, "B", tree_b, "b2", fin_ok, CFG)
    assert st()["tune:lgbm:learning_rate@full"] == "ACTIVE" and st()["tune:lgbm:num_leaves@full"] == "VALIDATED"
    assert [l["key"] for l in active_lessons(mem, FP, CFG)] == ["tune:lgbm:learning_rate@full"]
    # 第三个任务出现同 key 相反极性且已验证的证据（scope 相同）→ 原经验 DEPRECATED
    tree_c = [rec("c1", None, "RACE", "PROMISING", .64), rec("c2", "c1", "TUNE", "REJECT", .63, space={"learning_rate": {}}), rec("c3", "c1", "PROMOTE_FIDELITY", "ACCEPT", .65)]
    okc, _ = validate(Proposals.model_validate({"lessons": [{"key": "tune:lgbm:learning_rate@full", "statement": "x", "evidence": ["c2"], "polarity": "NEGATIVE"}]}), tree_c, allowed_keys(tree_c))
    admit(mem, okc, tree_c, "C", {**FP, "task_id": "C"}, CFG)
    update_lifecycle(mem, "C", tree_c, "c3", fin_ok, CFG)
    by = {(l["key"], l["polarity"]): l["status"] for l in mem.lessons()}
    assert by[("tune:lgbm:learning_rate@full", "POSITIVE")] == "DEPRECATED" and by[("tune:lgbm:learning_rate@full", "NEGATIVE")] == "VALIDATED"
    assert not active_lessons(mem, FP, CFG)


def test_admit_is_idempotent(mem, tree):
    ok, _ = validate(Proposals.model_validate({"lessons": [{"key": "race:lgbm", "statement": "s", "evidence": ["e1"], "polarity": "POSITIVE"}]}), tree, allowed_keys(tree))
    admit(mem, ok, tree, "A", FP, CFG)
    admit(mem, ok, tree, "A", FP, CFG)                             # resume 重放
    l = mem.lessons()
    assert len(l) == 1 and l[0]["tasks"] == ["A"] and l[0]["evidence"] == ["e1"]


def test_fingerprint_distance_and_cold_start(mem):
    far = {**FP, "task_id": "H", "n_samples": 120000, "n_features": 27, "bad_rate": 0.37, "time_span_months": 18}
    near = {**FP, "task_id": "N", "n_samples": 12000, "bad_rate": 0.14}
    assert fingerprint_distance(FP, near, CFG) < CFG["memory"]["max_distance"] < fingerprint_distance(FP, far, CFG)
    mem.put_task(far, {"from_task": "H", "model": "lgbm", "params": []}, None)
    t, d = nearest_task(mem, FP, CFG)
    assert t is None and d > CFG["memory"]["max_distance"]           # 只有不相似的任务 → 冷启动
    mem.put_task(near, {"from_task": "N", "model": "lgbm", "params": []}, None)
    t, d = nearest_task(mem, FP, CFG)
    assert t["fingerprint"]["task_id"] == "N"


# ---------- 集成：两个相似任务 + 一个不相似任务 ----------
def _run(cfg, task, llm=None, **kw):
    o = Orchestrator(toy_spec(cfg, **kw), cfg, llm or PolicyMockLLM(), FinalGate("toy", cfg, task), task)
    o.run()
    return o


def _shared(cfg, mem_db):
    cfg["memory"]["db"] = mem_db
    cfg["fidelity"]["low"]["n_trials"] = 20            # 10% 上限 = 2 个初始点
    return cfg


def test_second_similar_task_reuses_prior_and_lessons_become_active(tmp_path):
    db = str(tmp_path / "shared.db")
    a = _run(_shared(make_toy(tmp_path / "A"), db), "A")
    assert a.events.query("memory_retrieval")[0]["payload"]["cold_start"] is True          # 第一个任务：没有历史
    assert {t["fingerprint"]["task_id"] for t in MemoryStore(db).tasks()} == {"A"}
    b = _run(_shared(make_toy(tmp_path / "B", seed=1), db), "B")
    r = b.events.query("memory_retrieval")[0]["payload"]
    assert r["nearest"] == "A" and not r["cold_start"]
    race = b._result("B_race_lgbm")
    assert 0 < race["n_warm_start"] <= int(race["n_trials"] * CFG["inner_loop"]["warm_start_max_frac"])   # 跨任务 prior 注入，且 ≤10%
    ls = MemoryStore(db).lessons()
    assert ls and all(l["status"] in ("CANDIDATE", "VALIDATED", "ACTIVE") for l in ls)
    assert any(l["status"] == "ACTIVE" and set(l["tasks"]) == {"A", "B"} for l in ls)      # 两个任务复现 → ACTIVE


def test_dissimilar_task_cold_starts_and_ignores_lessons(tmp_path):
    db = str(tmp_path / "shared.db")
    _run(_shared(make_toy(tmp_path / "A"), db), "A")
    _run(_shared(make_toy(tmp_path / "B", seed=1), db), "B")                                # 让 memory 里有 ACTIVE Lesson
    mem = MemoryStore(db)
    for t in mem.tasks():                                                                     # 模拟"一个完全不同的新任务"：把历史指纹挪远
        t["fingerprint"].update(n_samples=2_000_000, bad_rate=0.5, n_features=300)
        mem.put_task(t["fingerprint"], t["prior"], t["final"])
    for l in mem.lessons():
        l["scope"].update(n_samples=2_000_000, bad_rate=0.5, n_features=300)
        mem.put_lesson(l)
    c = _run(_shared(make_toy(tmp_path / "C", seed=2), db), "C")
    r = c.events.query("memory_retrieval")[0]["payload"]
    assert r["cold_start"] and r["nearest"] is None and r["n_active_lessons"] == 0
    assert c._result("C_race_lgbm")["n_warm_start"] == 0 and c.p["memory_ctx"] is None      # 不相似 → 不复用


def test_no_memory_ablation_reads_nothing(tmp_path):
    db = str(tmp_path / "shared.db")
    _run(_shared(make_toy(tmp_path / "A"), db), "A")
    cfg = _shared(make_toy(tmp_path / "B", seed=1), db)
    cfg["ablation"] = {"no_memory": True}                                                    # A2
    b = _run(cfg, "B")
    assert not b.events.query("memory_retrieval") and b._result("B_race_lgbm")["n_warm_start"] == 0


def test_evidence_from_dissimilar_task_does_not_merge(mem, tree):
    far = {**FP, "task_id": "H", "n_samples": 120000, "bad_rate": 0.37, "n_features": 27}
    ok, _ = validate(Proposals.model_validate({"lessons": [{"key": "race:lgbm", "statement": "s", "evidence": ["e1"], "polarity": "POSITIVE"}]}), tree, allowed_keys(tree))
    admit(mem, ok, tree, "A", FP, CFG)
    admit(mem, ok, tree, "H", far, CFG)                                # 不相似任务提出同 key：单独成谱系，不计入 A 的经验
    ls = mem.lessons()
    assert len(ls) == 2 and sorted(l["tasks"] for l in ls) == [["A"], ["H"]]
    near = {**FP, "task_id": "B", "n_samples": 12000}
    admit(mem, ok, tree, "B", near, CFG)                               # 相似任务：并入 A 的谱系
    assert sorted(l["tasks"] for l in mem.lessons()) == [["A", "B"], ["H"]]
