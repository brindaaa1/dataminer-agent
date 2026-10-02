import type { Status, Tone } from "@/api"

export const METRIC: Record<string, string> = { auc: "AUC", pr_auc: "PR-AUC", ks: "KS" }
export const QLABEL: Record<string, string> = { label: "标签", observation_time_expr: "预测时点", windows: "时间窗口", metric: "指标" }
export const PROVIDERS: [string, string][] = [
  ["mock", "mock（无需 key，用 hotel 样本）"], ["deepseek", "DeepSeek"], ["kimi", "Kimi"], ["anthropic", "Anthropic"],
]
export const VERDICT: Record<string, { label: string; cls: string; color: string }> = {
  ACCEPT: { label: "采纳", cls: "bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-300", color: "#16a34a" },
  PROMISING: { label: "有希望", cls: "bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300", color: "#d97706" },
  INCONCLUSIVE: { label: "不确定", cls: "bg-zinc-100 text-zinc-700 dark:bg-zinc-800 dark:text-zinc-300", color: "#71717a" },
  REJECT: { label: "拒绝", cls: "bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300", color: "#dc2626" },
  NONE: { label: "完成", cls: "bg-zinc-100 text-zinc-700 dark:bg-zinc-800 dark:text-zinc-300", color: "#71717a" },
}
export const TONE: Record<Tone, string> = {
  ok: "border-l-green-600", bad: "border-l-red-600", warn: "border-l-amber-500", neutral: "border-l-zinc-300", running: "border-l-blue-500",
}
export const DOT: Record<Status, string> = {
  intake: "bg-blue-500", intake_failed: "bg-red-500", config_ready: "bg-blue-500", running: "bg-blue-500 animate-pulse",
  stopped: "bg-zinc-400", failed: "bg-red-500", done: "bg-amber-500", accepted: "bg-green-600",
}
export const CLONEABLE: Status[] = ["config_ready", "running", "stopped", "failed", "done", "accepted"]
export const CODES: Record<string, string> = {
  OVERFIT_GAP: "验证集与 OOT-dev 的主指标差距超过阈值：过拟合或跨时间不稳定",
  GUARD_FAIL: "护栏指标比当前最优变差超过容差",
  LEAK_SUSPECT: "疑似泄漏",
  PSI_DRIFT: "模型分数在 OOT-dev 上的分布漂移超过告警线",
  SCORE_PSI: "分数 PSI 超过严重阈值，直接拒绝",
  PLATEAU: "与当前最优的配对比较不显著",
  NO_CONVERGE: "最佳迭代数过少，模型几乎没有学到东西",
  NO_FINAL_MODEL: "还没有全量训练出来的最优模型",
}
export const f3 = (x: number | null | undefined) => (x == null ? "—" : x.toFixed(3))
export const mmss = (s: number) => `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(Math.floor(s % 60)).padStart(2, "0")}`
