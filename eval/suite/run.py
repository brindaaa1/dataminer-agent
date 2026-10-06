"""评测入口：

    python -m eval.suite.run --label v7 --tier fast            # 按档位跑（含情景），写 version.json
    python -m eval.suite.run --label v0 --datasets hotel_sample --seeds 0 1 2 --llm deepseek

结果写到 eval/suite/results/<label>/：runs.jsonl（每次运行一行）、baselines.json、summary.md；配了 Langfuse 就同时写数据集与实验。"""
import argparse
import json
import time
from pathlib import Path

from eval.regression import SettingsMismatch, compare, render_md
from eval.suite import langfuse_report, runner
from eval.suite.metrics import process_md, scenarios_md, summary_md
from eval.suite.tiers import load_tier
from eval.suite.version import build_version, latest_version

OUT = Path(__file__).parent / "results"

ap = argparse.ArgumentParser()
ap.add_argument("--label", required=True, help="评测版本号，如 v7；修完重跑换新版本号")
ap.add_argument("--tier", choices=["fast", "full"], help="按档位跑（eval/tiers.yaml）：数据集、种子、情景都从档位读")
ap.add_argument("--datasets", nargs="+", choices=list(runner.DATASETS), help="不用档位时指定；用档位时覆盖档位的数据集")
ap.add_argument("--seeds", nargs="+", type=int, help="不用档位时默认 0 1 2")
ap.add_argument("--llm", default="deepseek")
ap.add_argument("--keep-baselines", action="store_true", help="补跑个别种子时沿用已有的 B0/B1，不重跑")
ap.add_argument("--resume", action="store_true", help="从检查点续跑（不清旧产物），例如代码修复后接着跑中断的种子")
ap.add_argument("--answer", type=json.loads, help='回答等人回复的问题并从检查点续跑，如 \'{"observation_time_col": "确认"}\'')
ap.add_argument("--no-scenarios", action="store_true", help="用档位时不跑情景（只补跑数据集时用）")
a = ap.parse_args()
tier = load_tier(a.tier) if a.tier else {"datasets": [], "seeds": [0, 1, 2], "scenarios": []}
a.datasets, a.seeds = a.datasets or tier["datasets"], a.seeds or tier["seeds"]
if not a.datasets:
    ap.error("需要 --tier 或 --datasets")
started = time.time()

out = OUT / a.label
out.mkdir(parents=True, exist_ok=True)
# 同一版本号可以分几次跑：按 (数据集, 种子) 替换本次跑的，其余保留
rows = [json.loads(x) for x in (out / "runs.jsonl").read_text().splitlines()] if (out / "runs.jsonl").exists() else []
bases = json.loads((out / "baselines.json").read_text()) if (out / "baselines.json").exists() else {}
new = []
for name in a.datasets:
    rs = []
    for s in a.seeds:
        if not (a.resume or a.answer):
            runner.prepare_run(a.label, name, s)
        r = runner.run_agent(name, a.label, s, a.llm, answers=a.answer)
        rs.append(r)
        print(f"{name} s{s}: {r['status']} holdout={r['holdout']} 用量={r['usage']} 美元={r['cost_usd']}", flush=True)
        if r["status"] == "awaiting_human":      # 监控看这一行：回答后用 --answer 续跑
            print(f"AWAIT_HUMAN {name} s{s}: {json.dumps(r['questions'], ensure_ascii=False)}", flush=True)
    new += rs
    rows = runner.merge_rows(rows, rs)
    # 先落盘 agent 结果，再跑基线：基线出错不连累已经跑完的 agent（v2 lending_club 就是这样丢了三行）
    (out / "runs.jsonl").write_text("".join(json.dumps({"label": a.label, **r}, ensure_ascii=False, default=str) + "\n" for r in rows))
    if any(r["status"] == "awaiting_human" for r in rs):
        print(f"{name}：有种子在等人回复，基线等续跑时再跑", flush=True)
    elif not (a.keep_baselines and name in bases):
        bases[name] = runner.run_baselines(name, a.label, [r["wall_sec"] for r in rows if r["dataset"] == name and r["status"] == "ok"])
        (out / "baselines.json").write_text(json.dumps(bases, ensure_ascii=False, indent=1, default=float))
# 已有的情景结果：不跑情景（--no-scenarios）时原样沿用；续跑时只补跑没有正常结果的
scenarios = [json.loads(x) for x in (out / "scenarios.jsonl").read_text().splitlines()] if (out / "scenarios.jsonl").exists() else []
if a.tier and not a.no_scenarios:
    for s in runner.pending_scenarios(tier["scenarios"], scenarios, resuming=bool(a.resume or a.answer)):
        b1 = ((bases.get(s["base"]) or {}).get("b1") or {}).get("holdout_auc")      # 情景 a 的结果条件：holdout ≤ 底座 B1 + 0.05
        r = runner.run_scenario(s["name"], s["base"], a.label, a.llm, {**(s.get("params") or {}), "b1_holdout": b1})
        scenarios = [x for x in scenarios if (x["scenario"], x["base"]) != (s["name"], s["base"])] + [r]
        # 每跑完一个就落盘：后面的情景出问题，前面的结果不丢
        (out / "scenarios.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False, default=str) + "\n" for x in scenarios))
        print(f"情景 {s['name']}@{s['base']}: {'通过' if r.get('passed') else '不通过'} {r.get('reason')} 美元={r['cost_usd']}", flush=True)
settings = {**runner.SETTINGS, "llm": a.llm, "scenario": runner.SCENARIO_SETTINGS}
if a.tier:
    (out / "version.json").write_text(json.dumps(build_version(a.label, a.tier, settings, rows, bases, scenarios, started),
                                                 ensure_ascii=False, indent=1, default=str))
reg_md = ""
if a.tier:
    prev = latest_version(OUT, a.tier, exclude=a.label)
    if prev is None:
        reg_md = "## 回归对比\n\n没有可比的同档上一版本（旧版本没有 version.json）。\n"
    else:
        try:
            reg = compare(json.loads((out / "version.json").read_text()), prev)
            (out / "regression.json").write_text(json.dumps(reg, ensure_ascii=False, indent=1))
            reg_md = render_md(reg)
            print(reg["conclusion"], flush=True)
        except SettingsMismatch as e:
            reg_md = f"## 回归对比\n\n不能和 {prev['label']} 对比：{e.diff}\n"
(out / "baselines.json").write_text(json.dumps(bases, ensure_ascii=False, indent=1, default=float))
(out / "summary.md").write_text("\n".join([summary_md(a.label, {**settings, "tier": a.tier, "seeds": a.seeds}, rows, bases),
                                           reg_md, process_md(rows), scenarios_md(scenarios) if scenarios else ""]))
langfuse_report.report(a.label, new, bases)
print(f"汇总：{out / 'summary.md'}")
