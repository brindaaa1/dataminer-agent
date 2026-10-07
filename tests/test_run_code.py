"""第 1 档补项：run_code 逃生通道（§4.6、§4.0-4）。
核心主张：产出强制经过筛选漏斗；无论数据集策略如何都被标记为 LEAK_SUSPECT；子进程隔离（超时、代码出错不崩溃主进程）；
不触碰 holdout；EXPAND_FEATURES 动作能把它接入正常的 train→evaluate 流程。"""
import time

import numpy as np
import pandas as pd
import pytest

from agent.actions import ActionError, apply_action
from agent.llm import PolicyMockLLM, ScriptedLLM
from agent.orchestrator import Orchestrator
from agent.schemas import Budget, TaskSpec
from conftest import toy_spec
from evaluation.evaluator import scan_features
from evaluation.final_gate import FinalGate
from features.registry import Registry
from tools.run_code import RunCodeError, apply_code_to_df, run_code

GOOD_CODE = "def compute(df):\n    return (df['x1'] - df['x2']).rename('code_feat')\n"
BAD_CODE = "def compute(df):\n    raise ValueError('boom')\n"
SLOW_CODE = "def compute(df):\n    import time\n    time.sleep(10)\n    return df['x1']\n"
NO_COMPUTE = "y = 1\n"


def test_apply_code_produces_row_aligned_output(toy_cfg):
    from data.access import DataAccess
    df = DataAccess("toy", toy_cfg["paths"]["artifacts_root"]).load("train")
    out = apply_code_to_df(GOOD_CODE, df, toy_cfg)
    assert list(out.columns) == ["_row_id", "code_feat"]
    merged = df.merge(out, on="_row_id")
    assert np.allclose(merged["code_feat"], merged["x1"] - merged["x2"], equal_nan=True)


def test_bad_code_and_missing_compute_raise_cleanly(toy_cfg):
    with pytest.raises(RunCodeError, match="执行失败"):
        run_code(BAD_CODE, "toy", toy_cfg)
    with pytest.raises(RunCodeError, match="compute"):
        run_code(NO_COMPUTE, "toy", toy_cfg)


def test_timeout_kills_subprocess(toy_cfg):
    cfg = {**toy_cfg, "run_code": {**toy_cfg["run_code"], "timeout_sec": 1}}
    t0 = time.time()
    with pytest.raises(RunCodeError, match="超时"):
        run_code(SLOW_CODE, "toy", cfg)
    assert time.time() - t0 < 5


def test_too_many_features_rejected(toy_cfg):
    cfg = {**toy_cfg, "run_code": {**toy_cfg["run_code"], "max_features": 1}}
    code = "def compute(df):\n    return df[['x1','x2']].assign(z=df.x1+df.x2)\n"
    with pytest.raises(RunCodeError, match="最多"):
        run_code(code, "toy", cfg)


def test_output_goes_through_screening_funnel(toy_cfg):
    """信号强的特征应该被漏斗保留；漏斗产出的指标（IV 等）齐全。"""
    r = run_code(GOOD_CODE, "toy", toy_cfg)
    assert "code_feat" in r["kept"]
    assert "iv" in r["funnel"] or "candidates" in r["funnel"]
    reg = Registry(toy_cfg)
    meta = reg.get(r["feature_set_id"])
    assert meta["source"] == "run_code" and meta["code"] == GOOD_CODE


def test_run_code_output_always_flagged_suspect_regardless_of_policy(toy_cfg):
    """核心设计点：不管数据集的 unregistered_columns 策略如何，run_code 产出一律 LEAK_SUSPECT。"""
    from data.access import DataAccess
    from data.knowhow import load_knowhow
    r = run_code(GOOD_CODE, "toy", toy_cfg)
    df = DataAccess("toy", toy_cfg["paths"]["artifacts_root"]).load("train")
    df = Registry(toy_cfg).attach(df, [r["feature_set_id"]], "train")
    kh = load_knowhow("toy", toy_cfg)
    scan = scan_features(df, ["x1", "code_feat"], kh, toy_cfg, "toy", always_suspect={"code_feat"})
    assert "code_feat" in scan["unavailable_fields"] and "x1" not in scan["unavailable_fields"]


def test_run_code_never_touches_holdout():
    """静态检查（run_code.py 属于 tools/，已被现有的跨模块守卫覆盖，见 test_data_layer 的 no_dataset_specific_literals 一类）
    + 显式确认代码里唯一出现的切分字符串是 train/valid/oot_dev，没有 "holdout"。"""
    import ast
    tree = ast.parse(open("tools/run_code.py").read())
    strs = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert {"train", "valid", "oot_dev"} <= strs and "holdout" not in strs
    assert not any(isinstance(n, ast.Name) and n.id == "issue_holdout_token" for n in ast.walk(tree))


def test_run_code_holdout_only_reachable_through_final_gate(toy_cfg):
    """run_code 产出的特征也遵守『holdout 只能由 final_gate 读取』：模拟 final_gate 拿到 holdout 后才重算特征。"""
    r = run_code(GOOD_CODE, "toy", toy_cfg)
    fg = FinalGate("toy", toy_cfg, "t_reapply")
    holdout = fg.load_holdout()
    out = Registry(toy_cfg).attach(holdout.copy(), [r["feature_set_id"]], "holdout")
    assert np.allclose(out["code_feat"], holdout["x1"] - holdout["x2"], equal_nan=True)


def test_expand_features_action_requires_template_xor_code():
    spec = TaskSpec(dataset="toy", budget=Budget(tokens=1, cpu_minutes=1, wall_minutes=1))
    node = {"exp_id": "e1", "config": {"model": "lgbm", "features": ["x1"], "space": None, "fidelity": "low", "n_trials": None}}
    with pytest.raises(ActionError, match="需要"):
        apply_action("EXPAND_FEATURES", {}, node, {}, spec)
    with pytest.raises(ActionError, match="二选一"):
        apply_action("EXPAND_FEATURES", {"template_id": "t", "code": "c"}, node, {}, spec)
    kind, cfg, parent = apply_action("EXPAND_FEATURES", {"code": GOOD_CODE}, node, {}, spec)
    assert kind == "expand" and cfg == {"template_id": None, "code": GOOD_CODE} and parent == "e1"


def test_expand_features_integrates_into_decide_loop(toy_cfg):
    """端到端：EXPAND_FEATURES(code=...) → run_code（只看白名单输入）→ 新特征进入下一轮训练，不再一律 LEAK_SUSPECT 等人工批准。
    （第 5 轮之前这里断言的是"只能 ESCALATE_HUMAN"；没有白名单的旧特征集和单特征 AUC 过高的特征仍会被标为可疑。）"""
    ok = lambda a, p: dict(action=a, params=p, hypothesis="h", expected_gain="g", est_cost={}, rationale="r" * 30)

    def script(u):
        import json
        ctx = json.loads(u.split("```json")[-1].split("```")[0])
        if ctx.get("mode") == "PLAN":
            return {"directions": []}
        if ctx.get("mode") != "DECIDE":
            return {"lessons": []}
        return ok("EXPAND_FEATURES", {"code": GOOD_CODE}) if ctx["round_no"] == 1 else ok("STOP", {"reason": "x"})
    toy_cfg["race"]["rule_first_promotion"] = False                      # 脚本按 LLM 的第 1 轮写
    o = Orchestrator(toy_spec(toy_cfg), toy_cfg, ScriptedLLM(script), FinalGate("toy", toy_cfg, "t_expand"), "t_expand")
    from agent.states import State
    assert o.run() == State.DONE
    rec = next(r for r in o.store.all("t_expand") if r.action_type == "EXPAND_FEATURES")
    assert rec.diff["expand"]["feature_set_id"].startswith("fs_code_") and "code_feat" in rec.config["features"]
    assert "code_feat" not in o.p["pending_leaks"] and "code_feat" not in o.p["quarantined"]


def test_expand_features_failure_is_recorded_not_crashed(toy_cfg):
    ok = lambda a, p: dict(action=a, params=p, hypothesis="h", expected_gain="g", est_cost={}, rationale="r" * 30)

    def script(u):
        import json
        ctx = json.loads(u.split("```json")[-1].split("```")[0])
        if ctx.get("mode") == "PLAN":
            return {"directions": []}
        if ctx["round_no"] == 1:
            return ok("EXPAND_FEATURES", {"code": BAD_CODE})
        return ok("STOP", {"reason": "x"})
    toy_cfg["race"]["rule_first_promotion"] = False                      # 脚本按 LLM 的第 1 轮写
    o = Orchestrator(toy_spec(toy_cfg), toy_cfg, ScriptedLLM(script), FinalGate("toy", toy_cfg, "t_fail"), "t_fail")
    from agent.states import State
    assert o.run() == State.DONE
    rec = next(r for r in o.store.all("t_fail") if r.action_type == "EXPAND_FEATURES")
    assert rec.verdict == "NONE" and "EXPAND_FAILED" in rec.diagnosis_codes





# ---- 第 5 轮：按输入判定可信度（G3）、失败原因回灌（G1）、规则写进上下文（G2）----

def test_code_only_sees_the_inputs_it_is_given(toy_cfg):
    """看不到标签、切分时间和不在白名单里的字段：特征在时间上自然可信。"""
    ncol = "def compute(df):\n    return (df['x1'] + float(len(df.columns))).rename('ncol')\n"
    r = run_code(ncol, "toy", toy_cfg, inputs=["x1"])
    feat = pd.read_parquet(Registry(toy_cfg).cache_path(r["feature_set_id"], "train"))
    from data.access import DataAccess
    tr = DataAccess("toy", toy_cfg["paths"]["artifacts_root"]).load("train")
    assert np.allclose(feat["ncol"] - tr["x1"].values, 3, equal_nan=True)   # 列数 = _row_id、_obs_time、x1
    with pytest.raises(RunCodeError, match="_label"):
        run_code("def compute(df):\n    return df['_label'].rename('y')\n", "toy", toy_cfg, inputs=["x1"])


def test_holdout_recompute_uses_the_same_inputs(toy_cfg):
    """final_gate 在 holdout 上现算 run_code 特征时，同样看不到标签。"""
    code = "def compute(df):\n    return (df['x1'] + (1000.0 if '_label' in df.columns else 0.0)).rename('f')\n"
    r = run_code(code, "toy", toy_cfg, inputs=["x1", "x2"])
    h = FinalGate("toy", toy_cfg, "t_h").load_holdout()          # holdout 只能经 final_gate 读取
    out = Registry(toy_cfg).attach(h, [r["feature_set_id"]], "holdout")
    assert np.allclose(out["f"], out["x1"], equal_nan=True)


def test_reordered_output_is_rejected(toy_cfg):
    """compute 打乱行序会让特征与样本错位：必须报错（并把原因回给 LLM），不能静默错配。"""
    code = "def compute(df):\n    return df.sort_values('x1')['x1'].rename('s')\n"
    with pytest.raises(RunCodeError, match="一一对应"):
        run_code(code, "toy", toy_cfg, inputs=["x1"])


def test_run_code_features_usable_at_l0_and_failures_fed_back(toy_cfg):
    """L0 下：run_code 特征按输入判定可信（不再一律 LEAK_SUSPECT 被隔离）；失败原因写进事件并出现在下一轮上下文；
    上下文里有 run_code 的规则（上限、可用输入，不含标签与被隔离字段）。v2 lending_club：12 个特征超上限，LLM 不知原因。"""
    import json
    too_many = "def compute(df):\n    import pandas as pd\n    return pd.DataFrame({f'f{i}': df['x1'] * i for i in range(1, 13)})\n"
    exp = lambda code: dict(action="EXPAND_FEATURES", params={"code": code}, hypothesis="h", expected_gain="g", est_cost={}, rationale="r" * 30)
    stop = dict(action="STOP", params={"reason": "x"}, hypothesis="h", expected_gain="g", est_cost={}, rationale="r" * 30)
    seen = []

    def script(u):
        ctx = json.loads(u.split("```json")[-1].split("```")[0])
        if ctx.get("mode") == "PLAN":
            return {"directions": []}
        if ctx.get("mode") != "DECIDE":
            return {"lessons": []}
        seen.append(ctx)
        return [exp(too_many), exp(GOOD_CODE), stop][min(len(seen) - 1, 2)]
    o = Orchestrator(toy_spec(toy_cfg, autonomy="L0"), toy_cfg, ScriptedLLM(script), FinalGate("toy", toy_cfg, "t9"), "t9")
    o.run()
    rc = seen[0]["run_code"]
    assert rc["max_features"] == toy_cfg["run_code"]["max_features"] and "x1" in rc["inputs"] and "_label" not in rc["inputs"]
    assert "最多" in (seen[1].get("last_failure") or "")
    assert any("最多" in (e["payload"].get("error") or "") for e in o.events.query("recorded"))
    expanded = next(r for r in o.store.all("t9") if r.action_type == "EXPAND_FEATURES" and r.config)
    assert "code_feat" in expanded.config["features"] and "code_feat" not in o.p["quarantined"]
    assert "LEAK_SUSPECT" not in expanded.diagnosis_codes
