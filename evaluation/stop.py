"""停止条件（§4.9），由代码在每轮 RECORD 之后检查，不依赖 LLM。"""


def stop_check(*, cfg, oot_budget, round_no: int, max_rounds: int, remaining: dict,
               history_verdicts: list, best_score: float | None = None, target_value: float | None = None,
               data_fatal: bool = False, llm_stop: bool = False) -> str | None:
    """返回停止原因（字符串）或 None。remaining: {tokens, cpu_minutes, wall_minutes}。"""
    if data_fatal:
        return "DATA_FATAL"
    if any(v <= 0 for v in remaining.values()):
        return "BUDGET_EXHAUSTED"
    if round_no >= max_rounds:
        return "MAX_ROUNDS"
    n = cfg["evaluator"]["no_progress_rounds"]
    if len(history_verdicts) >= n and all(v in ("REJECT", "INCONCLUSIVE") for v in history_verdicts[-n:]):
        return "NO_PROGRESS"
    if oot_budget.exhausted:
        return "OOT_BUDGET_EXHAUSTED"
    if target_value is not None and best_score is not None and best_score >= target_value:
        return "TARGET_REACHED"
    if llm_stop:
        return "LLM_STOP"                      # 软停止：仅当上面所有硬条件都未触发
    return None
