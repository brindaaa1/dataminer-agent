"""动作 → 新的实验配置（§4.7）。这里是纯函数：校验参数合法性 + 生成配置；不训练。"""
import hashlib
import json

from modeling.zoo import ZOO


class ActionError(ValueError):
    pass


def config_hash(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]


def allowed_models(spec) -> list[str]:
    ms = [m for m in ZOO if m not in spec.constraints.banned_models]
    return ["lr_scorecard"] if spec.constraints.interpretability == "high" and "lr_scorecard" in ms else \
        ([] if spec.constraints.interpretability == "high" else ms)


def model_profiles(spec) -> dict[str, str]:
    """给 LLM 看的模型档案：只列约束允许的模型。"""
    return {m: ZOO[m].profile for m in allowed_models(spec)}


def _check_space(model, space):
    default = ZOO[model].default_space
    for k, s in space.items():
        if k not in default:
            raise ActionError(f"参数 {k} 不在 {model} 的搜索空间中，可选：{list(default)}")
        if "choices" not in s and not (isinstance(s.get("low"), (int, float)) and isinstance(s.get("high"), (int, float)) and s["low"] < s["high"]):
            raise ActionError(f"参数 {k} 需要 low < high")


def apply_action(action: str, params: dict, node: dict | None, records: dict, spec, quarantined=()) -> tuple[str, dict | None, str | None]:
    """返回 (kind, 新配置, 父实验 id)。kind ∈ train / nav / investigate / escalate / stop。
    node: 当前节点 {"exp_id", "config"}；records: exp_id → ExperimentRecord。"""
    clean = lambda fs: [f for f in fs if f not in quarantined]
    if action == "STOP":
        return "stop", None, None
    if action == "ESCALATE_HUMAN":
        return "escalate", None, None
    if action == "INVESTIGATE":
        from tools.query_data import QUESTIONS
        if params.get("question") not in QUESTIONS:
            raise ActionError(f"question 必须是 {QUESTIONS}")
        return "investigate", None, None
    if action == "BACKTRACK":
        r = records.get(params.get("exp_id"))
        if not r or r.invalidated:
            raise ActionError("BACKTRACK 的 exp_id 不存在或已失效")
        return "nav", None, r.exp_id
    if action == "PROMOTE_FIDELITY":
        r = records.get(params.get("exp_id"))
        if not r or r.fidelity != "low":
            raise ActionError("PROMOTE_FIDELITY 的对象必须是已存在的低保真实验")
        if r.invalidated:
            raise ActionError("该实验已失效")
        return "train", {**r.config, "features": clean(r.config["features"]), "fidelity": "full", "n_trials": None}, r.exp_id
    if node is None:
        raise ActionError("当前没有可派生的实验节点")
    cfg, parent = dict(node["config"]), node["exp_id"]
    cfg["features"] = clean(cfg["features"])
    if action == "TUNE":
        space, mono = params.get("space"), params.get("monotone")
        if not space and mono is None:
            raise ActionError("TUNE 需要 params.space 或 params.monotone")
        if mono is not None:
            if mono and not ZOO[cfg["model"]].supports["monotone_constraints"]:
                raise ActionError(f"{cfg['model']} 不支持单调约束")
            if mono:
                cfg["monotone"] = True
            else:
                cfg.pop("monotone", None)
        if space:
            _check_space(cfg["model"], space)
            cfg["space"] = {**(cfg.get("space") or ZOO[cfg["model"]].default_space), **space}
        cfg.update(fidelity=params.get("fidelity", cfg["fidelity"]), n_trials=params.get("n_trials", cfg.get("n_trials")))
    elif action == "SWITCH_MODEL":
        m = params.get("model")
        if m not in ZOO:
            raise ActionError(f"模型 {m} 不在 zoo 中：{list(ZOO)}")
        if m not in allowed_models(spec):
            raise ActionError(f"模型 {m} 被 spec 约束禁用")
        if m == cfg["model"]:
            raise ActionError("SWITCH_MODEL 的目标与当前模型相同")
        cfg.update(model=m, space=None, fidelity=params.get("fidelity", "low"), n_trials=None)
    elif action == "PRUNE_FEATURES":
        drop = params.get("features") or []
        bad = [f for f in drop if f not in cfg["features"]]
        if not drop or bad:
            raise ActionError(f"PRUNE_FEATURES 需要 params.features（当前特征集内的字段）；不存在：{bad}")
        keep = [f for f in cfg["features"] if f not in drop]
        if not keep:
            raise ActionError("不能剔除全部特征")
        cfg.update(features=keep, fidelity=params.get("fidelity", cfg["fidelity"]))
    elif action == "EXPAND_FEATURES":
        tmpl, code = params.get("template_id"), params.get("code")
        if not tmpl and not code:
            raise ActionError("EXPAND_FEATURES 需要 params.template_id（模板库）或 params.code（run_code 逃生通道）")
        if tmpl and code:
            raise ActionError("template_id 和 code 只能二选一")
        return "expand", {"template_id": tmpl, "code": code}, parent
    else:
        raise ActionError(f"未知动作 {action}")
    return "train", cfg, parent
