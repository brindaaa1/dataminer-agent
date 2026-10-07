"""回归对比（spec §5）：同档同设置才比；配对差超过噪声带且多数种子同向才算变化；硬性标记一律报出。"""
import pytest

from eval.regression import SettingsMismatch, calibrate, compare, render_md

NOISE = {"calibrated": True, "bands": {"holdout": 0.003, "delta_b1": 0.003, "wasted_rate": 0.1, "rejected_decisions": 1.0,
                                       "upgrade_hit_rate": 0.2, "fe_in_final": 0.5}}


def run(seed, holdout, wasted=0.1, status="ok", rej=0, kinds=None, df=False):
    return {"seed": seed, "status": status, "holdout": holdout, "task_id": f"t{seed}", "wall_sec": 100, "cost_usd": 0.01, "overfit": False,
            "process": {"wasted_rounds": {"value": wasted}, "rejected_decisions": {"value": rej, "by_kind": kinds or {}},
                        "upgrade_hit_rate": {"value": 0.5}, "fe_funnel": {"value": False}, "decide_failed": {"value": df}}}


def ver(label, runs, scen=None, tier="fast", settings=None):
    return {"label": label, "tier": tier, "settings": settings or {"max_rounds": 8},
            "datasets": {"d": {"baselines": {"b1": {"holdout_auc": 0.70}}, "runs": runs}}, "scenarios": scen or []}


def row(reg, metric):
    return next(t for t in reg["table"] if t["metric"] == metric)


def test_better_worse_and_same():
    old = ver("v6", [run(0, 0.70), run(1, 0.71)])
    assert row(compare(ver("v7", [run(0, 0.71), run(1, 0.72)]), old, NOISE), "holdout")["verdict"] == "better"
    assert row(compare(ver("v7", [run(0, 0.69), run(1, 0.70)]), old, NOISE), "holdout")["verdict"] == "worse"
    assert row(compare(ver("v7", [run(0, 0.702), run(1, 0.711)]), old, NOISE), "holdout")["verdict"] == "same"     # 在噪声带内
    assert row(compare(ver("v7", [run(0, 0.72), run(1, 0.705)]), old, NOISE), "holdout")["verdict"] == "same"      # 方向不一致
    r = compare(ver("v7", [run(0, 0.70, wasted=0.4), run(1, 0.71, wasted=0.4)]), old, NOISE)
    assert row(r, "wasted_rate")["verdict"] == "worse" and "有退步" in r["conclusion"]                               # 越低越好


def test_settings_mismatch_is_refused():
    with pytest.raises(SettingsMismatch) as e:
        compare(ver("v7", [], settings={"max_rounds": 5}), ver("v6", []), NOISE)
    assert "max_rounds" in e.value.diff
    with pytest.raises(SettingsMismatch):
        compare(ver("v7", [], tier="full"), ver("v6", []), NOISE)


def test_failed_run_is_a_hard_flag_not_a_crash():
    """某个种子失败、没有 holdout：不参与分数比较，但出硬性标记（Review Focus 3）。"""
    r = compare(ver("v7", [run(0, None, status="failed"), run(1, 0.71)]), ver("v6", [run(0, 0.70), run(1, 0.71)]), NOISE)
    assert row(r, "holdout")["n"] == 1
    assert any(f["kind"] == "failed_run" and f["seed"] == 0 for f in r["flags"])


def test_hard_flags_decide_failed_new_reject_kind_scenario_regression_and_cost():
    old = ver("v6", [run(0, 0.70, kinds={"duplicate": 1})], scen=[{"scenario": "leak", "base": "d", "passed": True}])
    new_runs = [run(0, 0.70, kinds={"duplicate": 1, "budget": 2}, df=True)]
    new_runs[0]["cost_usd"] = 0.02
    r = compare(ver("v7", new_runs, scen=[{"scenario": "leak", "base": "d", "passed": False, "task_id": "s"}]), old, NOISE)
    kinds = {f["kind"] for f in r["flags"]}
    assert {"decide_failed", "new_reject_kind", "scenario_regressed", "cost_up"} <= kinds


def test_uncalibrated_noise_is_marked_and_aa_calibration():
    old, new = ver("v7a", [run(0, 0.700), run(1, 0.710)]), ver("v7b", [run(0, 0.704), run(1, 0.708)])
    n = calibrate(new, old)
    assert n["calibrated"] and n["bands"]["holdout"] == pytest.approx(0.004) and n["from"] == ["v7b", "v7a"]
    r = compare(new, old, {"calibrated": False, "bands": NOISE["bands"]})
    assert r["noise_calibrated"] is False and "未校准" in render_md(r)


def test_no_previous_version_is_not_an_error(tmp_path):
    from eval.suite.version import latest_version
    (tmp_path / "v6").mkdir()
    (tmp_path / "v6" / "runs.jsonl").write_text("{}\n")          # 旧版本只有 runs.jsonl，没有 version.json
    assert latest_version(tmp_path, "fast", before="v7") is None


def ver2(label, hotel, lc, b0=None):
    """两个数据集：hotel 噪声大、lc 噪声小（v7a/v7b A/A 的形状）。"""
    v = ver(label, hotel)
    v["datasets"]["lc"] = {"baselines": {"b1": {"holdout_auc": 0.65}}, "runs": lc}
    return v


def test_noise_bands_are_per_dataset():
    """A/A 里 hotel 差 0.016、lc 差 0.004：lc 用全局最大值会对它的变化视而不见。"""
    a = ver2("a", [run(0, 0.780), run(1, 0.775)], [run(0, 0.658), run(1, 0.663)])
    b = ver2("b", [run(0, 0.796), run(1, 0.790)], [run(0, 0.662), run(1, 0.659)])
    n = calibrate(b, a)
    assert n["by_dataset"]["d"]["holdout"] == pytest.approx(0.016) and n["by_dataset"]["lc"]["holdout"] == pytest.approx(0.004)
    assert n["bands"]["holdout"] == pytest.approx(0.016)                                   # 全局仍是最大值，旧文件照常可用
    new = ver2("c", [run(0, 0.790), run(1, 0.785)], [run(0, 0.668), run(1, 0.673)])      # 两个数据集都 +0.010
    r = compare(new, a, n)
    assert [t["verdict"] for t in r["table"] if t["metric"] == "holdout"] == ["same", "better"]


def test_overfit_flag_is_relative_to_b0_drift_and_only_when_new():
    """hotel 的 B0 自己从 OOT-dev 到 holdout 就掉 0.043；固定 0.02 让每次 hotel 运行都报过拟合。"""
    def v(label, oot, holdout, agent_flag=True):
        x = ver(label, [{**run(0, holdout), "oot_dev": oot, "overfit": agent_flag}])
        x["datasets"]["d"]["baselines"]["b0"] = {"oot_dev_auc": 0.820, "holdout_auc": 0.777}
        return x
    kinds = lambda r: [f["kind"] for f in r["flags"]]
    assert "overfit" not in kinds(compare(v("n", 0.838, 0.795), v("o", 0.835, 0.792), NOISE))      # 掉 0.043，与 B0 相同
    assert "overfit" in kinds(compare(v("n", 0.850, 0.780), v("o", 0.835, 0.792), NOISE))          # 掉 0.070，比 B0 多 0.027
    assert "overfit" not in kinds(compare(v("n", 0.850, 0.780), v("o", 0.850, 0.781), NOISE))      # 旧版本同种子也过拟合：不是新问题


def test_cost_threshold_comes_from_aa():
    """A/A 里 hotel 选的模型不同，用时差 3 倍：固定 30% 对时间没有意义。"""
    def v(label, wall):
        x = ver(label, [run(0, 0.70)])
        x["datasets"]["d"]["runs"][0]["wall_sec"] = wall
        return x
    n = calibrate(v("b", 300), v("a", 100))
    assert n["cost_ratio"]["d"]["wall_sec"] == pytest.approx(3.0) and n["cost_ratio"]["d"]["cost_usd"] == pytest.approx(1.3)
    kinds = lambda r: [f["kind"] for f in r["flags"]]
    assert "cost_up" not in kinds(compare(v("n", 250), v("o", 100), n))
    assert "cost_up" in kinds(compare(v("n", 350), v("o", 100), n))
    assert "cost_up" in kinds(compare(v("n", 140), v("o", 100), NOISE))                  # 没校准过：仍按 30%


def test_aa_pair_does_not_flag_itself_after_calibration():
    """v7b vs v7a：用时比 3.2999… 舍成 3.3 后反而低于实际比值，校准用的那一对自己又报 cost_up。"""
    a, b = ver("a", [run(0, 0.70)]), ver("b", [run(0, 0.70)])
    b["datasets"]["d"]["runs"][0]["wall_sec"] = 333.4
    assert not compare(b, a, calibrate(b, a))["flags"]


def test_aa_calibration_refuses_different_settings():
    """--aa 两次的档位或设置不同（例如轮数上限改了），差异就不是噪声，不能拿来校准。"""
    with pytest.raises(SettingsMismatch):
        calibrate(ver("b", [run(0, 0.7)], settings={"max_rounds": 5}), ver("a", [run(0, 0.7)]))


def test_cost_change_is_reported_per_dataset():
    """汇总直接给出用时、花费的变化，不只在越线时出硬性标记。"""
    new = ver("v7", [run(0, 0.70), run(1, 0.71)])
    new["datasets"]["d"]["runs"][0]["wall_sec"] = 50
    r = compare(new, ver("v6", [run(0, 0.70), run(1, 0.71)]), NOISE)
    c = next(x for x in r["cost"] if x["dataset"] == "d" and x["key"] == "wall_sec")
    assert (c["new"], c["old"], c["ratio"]) == (150, 200, 0.75)
    assert "用时与花费" in render_md(r) and "×0.75" in render_md(r)


def test_previous_version_is_by_label_order_and_skips_running(tmp_path):
    """上一版本按版本号取（v9a 的上一版本是 v8，即使 v7b 是最近才跑完的）；正在跑的版本不参与对比。"""
    import json
    from eval.suite.version import latest_version
    for lab, fin, st in (("v7b", 9, "done"), ("v8", 5, "done"), ("v8x", 6, "running"), ("v9b", 7, "done")):
        (tmp_path / lab).mkdir()
        (tmp_path / lab / "version.json").write_text(json.dumps({"label": lab, "tier": "fast", "finished": fin, "status": st}))
    assert latest_version(tmp_path, "fast", before="v9a")["label"] == "v8"
    assert latest_version(tmp_path, "fast", before="v10")["label"] == "v9b"
    assert latest_version(tmp_path, "fast", before="v7a") is None


def test_cost_rows_only_for_datasets_with_pairs():
    """旧版本只跑了部分数据集（v10b 只有 lending_club）：没有配对的数据集不出"0 → 0"的空行。"""
    new = ver2("v11", [run(0, 0.79)], [run(0, 0.66)])
    old = ver("v10", [run(0, 0.79)])
    old["datasets"] = {"lc": {"baselines": {}, "runs": [run(0, 0.66)]}}
    r = compare(new, old, NOISE)
    assert {c["dataset"] for c in r["cost"]} == {"lc"}


def test_parallel_runs_skip_wall_time_comparison():
    """--jobs > 1 时各进程抢 CPU，用时被拉长：不比用时、不出用时上升的标记；花费（token）照常比。"""
    new, old = ver("v11", [run(0, 0.70)]), ver("v10", [run(0, 0.70)])
    new["jobs"] = 4
    new["datasets"]["d"]["runs"][0]["wall_sec"] = 1000
    r = compare(new, old, NOISE)
    assert {c["key"] for c in r["cost"]} == {"cost_usd"} and not any(f["kind"] == "cost_up" for f in r["flags"])
