"""回归对比（spec 2026-10-05 §5）：同档、同设置的两个版本，按（数据集，种子）配对。
分数和过程指标：平均配对差超过噪声带、且多数种子同向（≤2 对时全部同向，否则至少 2/3）才算变化。
噪声带来自 A/A 测试（同一代码跑两次），按数据集存 eval/noise.json；没有时用默认值并标"未校准"。
硬性标记不管噪声一律报出：运行失败 / 没有最终模型、决策失败收尾、新的被拒原因类别、情景从通过变不通过、
新出现的 OOT-dev 过拟合（比 B0 自己的漂移多掉 OVERFIT_EXCESS 以上）、时间或花费增加超过 A/A 观察到的倍数（至少 30%）。

    python -m eval.regression v7 --against v6
    python -m eval.regression v7a --aa v7b        # A/A：两次同一代码，更新噪声带"""
import argparse
import json
import math
from pathlib import Path

RESULTS = Path(__file__).parent / "suite" / "results"
NOISE_FILE = Path(__file__).parent / "noise.json"
DEFAULT_BANDS = {"holdout": 0.003, "delta_b1": 0.003, "wasted_rate": 0.1, "rejected_decisions": 1.0, "upgrade_hit_rate": 0.2, "fe_in_final": 0.5}
COST_UP = 1.3
OVERFIT_EXCESS = 0.02     # 同 config final_gate.delta，但相对 B0 的 OOT-dev → holdout 漂移


def _p(r, k):
    return ((r.get("process") or {}).get(k) or {}).get("value")


def _b1(b):
    return (b.get("b1") or {}).get("holdout_auc")


METRICS = {   # 名字 → (取值函数(行, 基线), 越大越好)
    "holdout": (lambda r, b: r.get("holdout"), True),
    "delta_b1": (lambda r, b: None if r.get("holdout") is None or _b1(b) is None else r["holdout"] - _b1(b), True),
    "wasted_rate": (lambda r, b: _p(r, "wasted_rounds"), False),
    "rejected_decisions": (lambda r, b: _p(r, "rejected_decisions"), False),
    "upgrade_hit_rate": (lambda r, b: _p(r, "upgrade_hit_rate"), True),
    "fe_in_final": (lambda r, b: None if _p(r, "fe_funnel") is None else float(_p(r, "fe_funnel")), True),
}


class SettingsMismatch(ValueError):
    def __init__(self, diff: dict):
        super().__init__(f"档位或设置不同，不能对比：{diff}")
        self.diff = diff


def load_noise() -> dict:
    return json.loads(NOISE_FILE.read_text()) if NOISE_FILE.exists() else {"calibrated": False, "bands": DEFAULT_BANDS}


def _drift(oot, holdout):
    return None if oot is None or holdout is None else oot - holdout


def _overfit(r: dict, b0: dict) -> bool:
    """OOT-dev → holdout 掉得比 B0 多 OVERFIT_EXCESS 以上。hotel 的 B0 自己就掉 0.043，固定阈值每次都报。"""
    d, ref = _drift(r.get("oot_dev"), r.get("holdout")), _drift(b0.get("oot_dev_auc"), b0.get("holdout_auc"))
    return d is not None and d - (ref or 0) > OVERFIT_EXCESS


def _required(n: int) -> int:
    return n if n <= 2 else math.ceil(2 * n / 3)


def _pairs(new: dict, old: dict):
    """[(数据集, 新行, 旧行, 新基线, 旧基线)]，按种子配对。"""
    out = []
    for ds, nd in new["datasets"].items():
        od = old["datasets"].get(ds)
        if not od:
            continue
        ob = {r["seed"]: r for r in od["runs"]}
        out += [(ds, r, ob[r["seed"]], nd["baselines"], od["baselines"]) for r in nd["runs"] if r["seed"] in ob]
    return out


def _diffs(pairs, f):
    out = []
    for _, a, b, ba, bb in pairs:
        x, y = f(a, ba), f(b, bb)
        if x is not None and y is not None:
            out.append((x - y, x, y, a))
    return out


def _check_comparable(new: dict, old: dict):
    diff = {k: (new["settings"].get(k), old["settings"].get(k)) for k in set(new["settings"]) | set(old["settings"])
            if new["settings"].get(k) != old["settings"].get(k)}
    if new["tier"] != old["tier"]:
        diff["tier"] = (new["tier"], old["tier"])
    if diff:
        raise SettingsMismatch(diff)


def compare(new: dict, old: dict, noise: dict | None = None) -> dict:
    _check_comparable(new, old)
    noise = noise or load_noise()
    table, flags, cost = [], [], []
    for ds in new["datasets"]:
        bands = noise.get("by_dataset", {}).get(ds, noise["bands"])
        ratio = noise.get("cost_ratio", {}).get(ds, {})
        pairs = [p for p in _pairs(new, old) if p[0] == ds]
        for m, (f, hib) in METRICS.items():
            ds_ = _diffs(pairs, f)
            if not ds_:
                continue
            mean = sum(d for d, *_ in ds_) / len(ds_)
            same_dir = sum(d != 0 and (d > 0) == (mean > 0) for d, *_ in ds_)
            moved = abs(mean) > bands.get(m, 0) and same_dir >= _required(len(ds_))
            verdict = "same" if not moved else ("better" if (mean > 0) == hib else "worse")
            worst = min(ds_, key=lambda t: t[0] if hib else -t[0])[3]
            table.append({"dataset": ds, "metric": m, "delta": round(mean, 4), "new": round(sum(x for _, x, _, _ in ds_) / len(ds_), 4),
                          "old": round(sum(y for _, _, y, _ in ds_) / len(ds_), 4), "n": len(ds_), "verdict": verdict,
                          "worst_task": worst.get("task_id")})
        keys = ("wall_sec", "cost_usd") if new.get("jobs", 1) == old.get("jobs", 1) else ("cost_usd",)   # 并行数不同时用时不可比
        for key in keys if pairs else ():       # 旧版本没跑这个数据集：没有可比的用时花费
            n_, o_ = sum(p[1].get(key) or 0 for p in pairs), sum(p[2].get(key) or 0 for p in pairs)
            cost.append({"dataset": ds, "key": key, "new": round(n_, 4), "old": round(o_, 4), "ratio": round(n_ / o_, 2) if o_ else None,
                         "threshold": ratio.get(key, COST_UP)})
            if o_ and n_ > ratio.get(key, COST_UP) * o_:
                flags.append({"kind": "cost_up", "dataset": ds, "detail": f"{key} {o_:.4g} → {n_:.4g}（阈值 ×{ratio.get(key, COST_UP)}）"})
        old_by = {r["seed"]: r for r in old["datasets"].get(ds, {}).get("runs", [])}
        b0n, b0o = new["datasets"][ds]["baselines"].get("b0", {}), old["datasets"].get(ds, {}).get("baselines", {}).get("b0", {})
        for r in new["datasets"][ds]["runs"]:
            base = {"dataset": ds, "seed": r["seed"], "task_id": r.get("task_id")}
            if r.get("status") != "ok" or r.get("holdout") is None:
                flags.append({**base, "kind": "failed_run", "detail": r.get("error") or "没有最终模型"})
            if _p(r, "decide_failed"):
                flags.append({**base, "kind": "decide_failed", "detail": "因决策失败收尾"})
            o = old_by.get(r["seed"])
            if _overfit(r, b0n) and not (o and _overfit(o, b0o)):
                flags.append({**base, "kind": "overfit", "detail": f"新出现的 OOT-dev 过拟合：比 B0 多掉 {_drift(r['oot_dev'], r['holdout']) - (_drift(b0n.get('oot_dev_auc'), b0n.get('holdout_auc')) or 0):.3f}"})
            if o:
                kinds = lambda x: set((((x.get("process") or {}).get("rejected_decisions") or {}).get("by_kind") or {}))
                for k in sorted(kinds(r) - kinds(o)):
                    flags.append({**base, "kind": "new_reject_kind", "detail": f"新出现的被拒原因：{k}"})
    old_s = {(s["scenario"], s["base"]): s for s in old.get("scenarios", [])}
    for s in new.get("scenarios", []):
        o = old_s.get((s["scenario"], s["base"]))
        if not s.get("passed") and (o is None or o.get("passed")):
            flags.append({"kind": "scenario_regressed" if o else "scenario_failed", "task_id": s.get("task_id"),
                          "detail": f"情景 {s['scenario']}@{s['base']}：{s.get('reason')}"})
    worse = [t for t in table if t["verdict"] == "worse"]
    n_bad = len(worse) + len(flags)
    return {"new": new["label"], "old": old["label"], "tier": new["tier"], "noise_calibrated": bool(noise.get("calibrated")),
            "table": table, "flags": flags, "cost": cost, "conclusion": "无退步" if not n_bad else f"有退步需看（{n_bad} 项）"}


def _bands(pairs) -> dict:
    # 只有两次运行，观察到的差可能是 0；噪声带不低于默认值，校准只会放宽、不会更严
    return {m: round(max([abs(d) for d, *_ in _diffs(pairs, f)] + [DEFAULT_BANDS[m]]), 4) for m, (f, _) in METRICS.items()}


def calibrate(a: dict, b: dict) -> dict:
    """A/A：同一代码跑两次。每个数据集、每个指标取配对差的最大绝对值（不低于默认值）作为噪声带；
    全局 bands 取所有数据集的最大值。时间、花费取两次总量之比（不低于 COST_UP）作为该数据集的上涨阈值。"""
    _check_comparable(a, b)                  # 设置不同时差异不是噪声
    pairs = _pairs(a, b)
    by_ds, ratio = {}, {}
    for ds in dict.fromkeys(p[0] for p in pairs):
        pp = [p for p in pairs if p[0] == ds]
        by_ds[ds] = _bands(pp)
        ratio[ds] = {}
        for key in ("wall_sec", "cost_usd"):
            x, y = sum(p[1].get(key) or 0 for p in pp), sum(p[2].get(key) or 0 for p in pp)
            ratio[ds][key] = math.ceil(100 * max(COST_UP, x / y if y else 0, y / x if x else 0)) / 100   # 向上取整，A/A 自己不越线
    return {"calibrated": True, "from": [a["label"], b["label"]], "bands": _bands(pairs), "by_dataset": by_ds, "cost_ratio": ratio}


ARROW = {"better": "↑ 变好", "worse": "↓ 变坏", "same": "无显著变化"}


def render_md(reg: dict) -> str:
    L = [f"## 回归对比：{reg['new']} vs {reg['old']}（{reg['tier']} 档）", "",
         f"**结论：{reg['conclusion']}**" + ("" if reg["noise_calibrated"] else "（噪声带未校准：用默认值，先跑一次 A/A）"), "",
         "| 数据集 | 指标 | 新 | 旧 | 差 | 配对数 | 判定 | 变化最大的运行 |", "|---|---|---|---|---|---|---|---|"]
    L += [f"| {t['dataset']} | {t['metric']} | {t['new']} | {t['old']} | {t['delta']:+} | {t['n']} | {ARROW[t['verdict']]} | {t['worst_task'] or ''} |"
          for t in reg["table"]]
    if reg.get("cost"):
        name = {"wall_sec": "用时（秒）", "cost_usd": "花费（美元）"}
        L += ["", "**用时与花费**（配对种子合计）", "", "| 数据集 | 项 | 新 | 旧 | 倍数 | 标记阈值 |", "|---|---|---|---|---|---|"]
        L += [f"| {c['dataset']} | {name[c['key']]} | {c['new']:.4g} | {c['old']:.4g} | " + (f"×{c['ratio']}" if c["ratio"] is not None else "—")
              + f" | ×{c['threshold']} |" for c in reg["cost"]]
    if reg["flags"]:
        L += ["", "**硬性标记**", ""] + [f"- {f['kind']}：{f['detail']}" + (f"（{f['task_id']}）" if f.get("task_id") else "") for f in reg["flags"]]
    return "\n".join(L + [""])


def _load(label: str) -> dict:
    return json.loads((RESULTS / label / "version.json").read_text())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("new")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--against")
    g.add_argument("--aa", help="A/A 测试的另一次：用两次的差异更新 eval/noise.json")
    a = ap.parse_args()
    if a.aa:
        n = calibrate(_load(a.new), _load(a.aa))
        NOISE_FILE.write_text(json.dumps(n, ensure_ascii=False, indent=1))
        print(json.dumps(n, ensure_ascii=False, indent=1))
    else:
        reg = compare(_load(a.new), _load(a.against))
        (RESULTS / a.new / "regression.json").write_text(json.dumps(reg, ensure_ascii=False, indent=1))
        print(render_md(reg))
