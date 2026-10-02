import type { TaskEvent } from "@/api"
import { category, segmentOf } from "./trace"

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const ev = (id: number, type: string, payload: any = {}): TaskEvent => ({ id, ts: id, type, exp_id: null, payload })

test("分段与 build_trace 一致：转入 DECIDE 的那条 state_transition 仍属于上一段", () => {
  const es = [
    ev(1, "intake.profile"), ev(2, "state_transition", { from: "INTAKE", to: "PROFILE" }), ev(3, "llm_call"),
    ev(4, "state_transition", { from: "PLAN", to: "DECIDE" }), ev(5, "decision"),
    ev(6, "state_transition", { from: "STOP_CHECK", to: "DECIDE" }),
    ev(7, "state_transition", { from: "DECIDE", to: "FINAL_GATE" }), ev(8, "final_gate"),
  ]
  expect(segmentOf(es)).toEqual(["intake", "setup", "setup", "setup", "r1", "r1", "r2", "finale"])
})

test("类别", () => {
  expect(["intake.llm_call", "intake.draft", "llm_call", "evaluated", "decision_rejected", "report_built"].map(category))
    .toEqual(["llm", "intake", "llm", "eval", "decision", "other"])
})
