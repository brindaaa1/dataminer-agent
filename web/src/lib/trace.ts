import type { TaskEvent } from "@/api"

/** 每条事件属于哪一段。规则同 runtime/trace.build_trace：第一次转入 DECIDE 之前是准备阶段，
 *  每次转入 DECIDE 开始新一轮，转入 FINAL_GATE 进入收尾；转换事件本身算在转换之前那一段。 */
export function segmentOf(events: TaskEvent[]): string[] {
  let n = 0
  let finale = false
  return events.map((e) => {
    if (e.type.startsWith("intake.")) return "intake"
    const seg = finale ? "finale" : n === 0 ? "setup" : `r${n}`
    if (e.type === "state_transition") {
      if (e.payload?.to === "DECIDE") n += 1
      if (e.payload?.to === "FINAL_GATE") finale = true
    }
    return seg
  })
}

export function category(t: string): string {
  if (t === "intake.llm_call" || t === "llm_call") return "llm"
  if (t.startsWith("intake.")) return "intake"
  if (t === "state_transition") return "state"
  if (t === "evaluated") return "eval"
  if (t === "decision" || t === "decision_rejected") return "decision"
  return "other"
}

export function summary(e: TaskEvent): string {
  const p = e.payload ?? {}
  switch (e.type) {
    case "state_transition": return `${p.from} → ${p.to}`
    case "llm_call": return `${p.stage ?? ""} · ${p.model ?? ""}`
    case "intake.llm_call": return "接入起草"
    case "evaluated": return `${e.exp_id ?? ""} ${p.verdict}${p.codes?.length ? ` [${p.codes.join(", ")}]` : ""}`
    case "decision": return `决策：${p.decision?.action ?? ""}`
    default: return e.type
  }
}
