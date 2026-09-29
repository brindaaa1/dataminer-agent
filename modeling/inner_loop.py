"""内循环：一次 run_experiment = 一个完整的 Optuna study（目标函数只看 valid，OOT-dev 只在 study 结束后评估）。"""
import json
import time
from pathlib import Path

import numpy as np
import optuna

from data.access import DataAccess
from data.knowhow import load_knowhow
from evaluation.guardrails import psi
from evaluation.metrics import METRICS, all_metrics, primary
from features.registry import Registry
from modeling.monotone import constraints_for
from modeling.prep import prepare
from modeling.zoo import ZOO, sample_params

optuna.logging.set_verbosity(optuna.logging.WARNING)


def clip_to_space(params: dict, space: dict) -> dict:
    """落在新空间之外的初始点裁剪到边界（§4.10）。"""
    out = {}
    for k, s in space.items():
        if k not in params:
            continue
        v = params[k]
        out[k] = v if "choices" in s else min(max(v, s["low"]), s["high"])
    return out


def run_study(exp_id, dataset, model, cfg, features=None, space=None, fidelity="low",
              n_trials=None, warm_start=None, seed=0, resume=True, feature_sets=None, monotone=False):
    """warm_start: 初始点列表（参数 dict）；数量会被截断到 n_trials × warm_start_max_frac。"""
    t0 = time.time()
    fid = cfg["fidelity"][fidelity]
    n_trials = n_trials or fid["n_trials"]
    art = Path(cfg["paths"]["artifacts_root"])
    da = DataAccess(dataset, str(art))
    date_fields, dfmt = date_spec(dataset, cfg)

    tr, va, oo = da.load("train"), da.load("valid"), da.load("oot_dev")
    if feature_sets:
        reg = Registry(cfg)
        tr, va, oo = (reg.attach(f, feature_sets, s_) for f, s_ in ((tr, "train"), (va, "valid"), (oo, "oot_dev")))
    if fid["sample_frac"] < 1:
        tr = tr.sample(frac=fid["sample_frac"], random_state=seed)
        va = va.sample(frac=fid["sample_frac"], random_state=seed)
    Xtr, cats = prepare(tr, date_fields, None, dfmt)
    Xva, _ = prepare(va, date_fields, cats, dfmt)
    Xoo, _ = prepare(oo, date_fields, cats, dfmt)
    if features:
        Xtr, Xva, Xoo = Xtr[features], Xva[features], Xoo[features]
    ytr, yva, yoo = tr["_label"], va["_label"], oo["_label"]

    Spec = ZOO[model]
    pm = primary(cfg)
    space = space or Spec.default_space
    mono, mono_rep = None, None
    if monotone:
        if not Spec.supports["monotone_constraints"]:
            raise ValueError(f"{model} 不支持单调约束")
        mono, mono_rep = constraints_for(dataset, cfg, {"feature_sets": feature_sets or []}, Xtr, ytr)
    storage = f"sqlite:///{art / 'optuna.db'}"
    study = optuna.create_study(
        study_name=exp_id, storage=storage, direction="maximize", load_if_exists=resume,
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=cfg["inner_loop"]["pruner_startup_trials"]))
    n_warm = enqueue_warm_start(study, warm_start or [], space, n_trials, cfg)

    def objective(trial):
        params = sample_params(trial, space)
        cbs = []
        if model == "lgbm":
            every = cfg["inner_loop"]["report_every"]

            def pruning_cb(env):
                it = env.iteration + 1
                if it % every == 0:
                    trial.report(env.evaluation_result_list[0][2], it)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
            cbs = [pruning_cb]
        m = Spec(params, cfg, seed, mono).fit(Xtr, ytr, Xva, yva, fid["early_stopping_rounds"], cbs)
        return METRICS[pm].fn(yva.values, m.predict(Xva))          # 目标函数：valid 上的主指标

    remaining = n_trials - len([t for t in study.trials if t.state.is_finished()])
    if remaining > 0:
        study.optimize(objective, n_trials=remaining)
    best = study.best_trial
    m = Spec(best.params, cfg, seed, mono).fit(Xtr, ytr, Xva, yva, fid["early_stopping_rounds"])
    p_tr, p_va, p_oo = m.predict(Xtr), m.predict(Xva), m.predict(Xoo)
    metrics = {f"{seg}_{k}": v for seg, (y_, p_) in {"train": (ytr, p_tr), "valid": (yva, p_va), "oot_dev": (yoo, p_oo)}.items()
               for k, v in all_metrics(y_.values, p_).items()}
    metrics["gap"] = metrics[f"valid_{pm}"] - metrics[f"oot_dev_{pm}"]
    metrics["score_psi"] = psi(p_va, p_oo, cfg["evaluator"]["psi_bins"])
    out = art / "experiments" / exp_id
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "oot_dev_pred.npy", p_oo)
    np.save(out / "valid_pred.npy", p_va)       # 供 compare 做配对 bootstrap（第 3 项）
    res = {"exp_id": exp_id, "model": model, "fidelity": fidelity, "n_trials": len(study.trials),
           "n_pruned": sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
           "n_warm_start": n_warm, "best_params": best.params, "best_iter": getattr(m, "best_iter", None), "metrics": metrics,
           "cost": {"wall_minutes": (time.time() - t0) / 60}, "n_features": Xtr.shape[1],
           "config": {"model": model, "features": list(Xtr.columns), "space": space, "fidelity": fidelity,
                      "n_trials": n_trials, "seed": seed, "feature_sets": feature_sets or [],
                      **({"monotone": True} if monotone else {})},
           "monotone_report": mono_rep}
    (out / "result.json").write_text(json.dumps(res, indent=1))
    return res


def enqueue_warm_start(study, points, space, n_trials, cfg) -> int:
    cap = int(n_trials * cfg["inner_loop"]["warm_start_max_frac"])
    n = 0
    for p in points[:cap]:
        study.enqueue_trial(clip_to_space(p, space))
        n += 1
    return n


def load_dict(dataset, cfg):
    return load_knowhow(dataset, cfg)["dictionary"]


def date_spec(dataset, cfg):
    """(type: date 的字段, 日期格式)。字段来自 data_dictionary，格式来自 config.task_specs（know-how 里没有结构化格式）。"""
    d = load_dict(dataset, cfg)
    return [c for c, s in d.get("fields", {}).items() if s.get("type") == "date"], cfg["task_specs"].get(dataset, {}).get("date_format")


def fit_model(dataset, cfg, config: dict, best_params: dict):
    """用已选定的配置在 train 上重训。返回 (model, ctx)；ctx.prep(df, split) 把任意切分的 df 变成模型输入。"""
    fid = cfg["fidelity"][config["fidelity"]]
    da = DataAccess(dataset, cfg["paths"]["artifacts_root"])
    date_fields, dfmt = date_spec(dataset, cfg)
    tr, va = da.load("train"), da.load("valid")
    fsets = config.get("feature_sets") or []
    reg = Registry(cfg) if fsets else None
    if fsets:
        tr, va = reg.attach(tr, fsets, "train"), reg.attach(va, fsets, "valid")
    if fid["sample_frac"] < 1:
        tr = tr.sample(frac=fid["sample_frac"], random_state=config["seed"])
        va = va.sample(frac=fid["sample_frac"], random_state=config["seed"])
    feats = config["features"]
    Xtr, cats = prepare(tr, date_fields, None, dfmt)
    Xva, _ = prepare(va, date_fields, cats, dfmt)
    mono = constraints_for(dataset, cfg, config, Xtr[feats], tr["_label"])[0] if config.get("monotone") else None
    m = ZOO[config["model"]](best_params, cfg, config["seed"], mono).fit(Xtr[feats], tr["_label"], Xva[feats], va["_label"],
                                                                          fid["early_stopping_rounds"])

    def prep(df, split):
        if reg:
            df = reg.attach(df, fsets, split)
        return prepare(df, date_fields, cats, dfmt)[0][feats]
    return m, prep


def fit_and_predict(dataset, cfg, config: dict, best_params: dict, frames: dict) -> dict:
    """对 frames（如 holdout）打分。只有 final_gate 会用 holdout 调用它。"""
    m, prep = fit_model(dataset, cfg, config, best_params)
    return {k: m.predict(prep(f, "holdout" if k in ("holdout", "masked") else k)) for k, f in frames.items()}
