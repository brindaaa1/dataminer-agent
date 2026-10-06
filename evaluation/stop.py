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
    # 还没有最终模型（best_score 为空）时不提前停：停了就一定没有模型；轮数上限仍然兜底
    if best_score is not None and len(history_verdicts) >= n and all(v in ("REJECT", "INCONCLUSIVE") for v in history_verdicts[-n:]):
        return "NO_PROGRESS"
    if oot_budget.exhausted:
        return "OOT_BUDGET_EXHAUSTED"
    if target_value is not None and best_score is not None and best_score >= target_value:
        return "TARGET_REACHED"
    if llm_stop:
        return "LLM_STOP"                      # 软停止：仅当上面所有硬条件都未触发
    return None


def history_mark(verdict: str, fidelity: str | None, delta: float | None) -> str:
    """记入 history_verdicts 的标记。低保真、不确定、但点估计比对照好 → INCONCLUSIVE_UP，不计入 NO_PROGRESS：
    它可以升全量复验，连续 3 轮就停会让 agent 没机会去升（v4 评测 lending_club）。"""
    if verdict == "INCONCLUSIVE" and fidelity == "low" and delta is not None and delta > 0:
        return "INCONCLUSIVE_UP"
    return verdict
