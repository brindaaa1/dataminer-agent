"""情景用例登记表（spec 2026-10-05 §6）：名字 → 模块（build + check）。"""
from eval.scenarios import leak, missing_spec, no_signal, small_oot

SCENARIOS = {"leak": leak, "missing_spec": missing_spec, "no_signal": no_signal, "small_oot": small_oot}
