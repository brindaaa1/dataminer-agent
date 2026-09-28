"""状态枚举与合法转换表（§4.5）。AWAIT_HUMAN / PAUSED / FAILED 可从任意状态进入。"""
from enum import Enum


class State(str, Enum):
    INTAKE = "INTAKE"
    CLARIFY = "CLARIFY"
    PROFILE = "PROFILE"
    PLAN = "PLAN"
    DECIDE = "DECIDE"
    EXECUTE = "EXECUTE"
    EVALUATE = "EVALUATE"
    RECORD = "RECORD"
    STOP_CHECK = "STOP_CHECK"
    FINAL_GATE = "FINAL_GATE"
    REPORT = "REPORT"
    CONSOLIDATE = "CONSOLIDATE"
    DONE = "DONE"
    AWAIT_HUMAN = "AWAIT_HUMAN"
    FAILED = "FAILED"


S = State
TRANSITIONS = {
    S.INTAKE: {S.CLARIFY, S.PROFILE},
    S.CLARIFY: {S.INTAKE},                    # 追问后转 AWAIT_HUMAN（横切），人回复后回到 INTAKE
    S.PROFILE: {S.PLAN},
    S.PLAN: {S.DECIDE},
    S.DECIDE: {S.EXECUTE, S.FINAL_GATE},
    S.EXECUTE: {S.EVALUATE, S.RECORD},        # 不训练的动作（BACKTRACK/INVESTIGATE）跳过 EVALUATE
    S.EVALUATE: {S.RECORD},
    S.RECORD: {S.STOP_CHECK},
    S.STOP_CHECK: {S.DECIDE, S.FINAL_GATE},
    S.FINAL_GATE: {S.REPORT},
    S.REPORT: {S.CONSOLIDATE},
    S.CONSOLIDATE: {S.DONE},
    S.DONE: set(), S.FAILED: set(), S.AWAIT_HUMAN: set(),   # AWAIT_HUMAN 恢复时回到 resume_state
}
CROSS_CUTTING = {S.AWAIT_HUMAN, S.FAILED}
TERMINAL = {S.DONE, S.FAILED}


def legal(src: State, dst: State) -> bool:
    return dst in TRANSITIONS[src] or dst in CROSS_CUTTING
