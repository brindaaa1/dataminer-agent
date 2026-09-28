"""预算：Budget = {tokens, cpu_minutes, wall_minutes}，实时扣减（§4.16）。状态是普通 dict，随检查点持久化。"""

KEYS = ("tokens", "cpu_minutes", "wall_minutes")


def new_budget(limits: dict) -> dict:
    return {"limits": {k: limits[k] for k in KEYS}, "used": {k: 0.0 for k in KEYS}}


def remaining(b: dict) -> dict:
    return {k: b["limits"][k] - b["used"][k] for k in KEYS}


def remaining_frac(b: dict) -> float:
    return min(remaining(b)[k] / b["limits"][k] for k in KEYS)


def spend(b: dict, **kw):
    for k, v in kw.items():
        b["used"][k] += v
