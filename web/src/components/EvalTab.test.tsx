import { fireEvent, render, screen } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import type { EvalView, Snapshot } from "@/api"
import { api } from "@/api"
import { baseSnap } from "@/test-fixtures"
import { EvalTab } from "./EvalTab"

const view: EvalView = {
  process: [
    { key: "wasted_rounds", label: "无效轮次", value: 0.25, detail: { n: 1 }, evidence: [{ ref: "e03", round: "r2" }, { ref: "e99", round: null }] },
    { key: "decide_failed", label: "决策失败收尾", value: false, detail: {}, evidence: [] },
    { key: "final_lineage", label: "最终模型来源", value: "赛跑 lgbm → 升全量", detail: {}, evidence: [] },
  ],
  version: { label: "v9", tier: "fast", running: false, against: "v8", conclusion: "有退步需看（1 项）", noise_calibrated: true,
    table: [{ dataset: "hotel", metric: "holdout", label: "holdout", new: 0.79, old: 0.8, delta: -0.01, n: 2, verdict: "worse", task: "ev-v9_hotel_s1" },
      { dataset: "hotel", metric: "delta_b1", label: "相对 B1 的差", new: 0.01, old: 0.0, delta: 0.01, n: 2, verdict: "same", task: null }],
    cost: [{ dataset: "hotel", label: "用时（秒）", new: 600, old: 1200, ratio: 0.5, threshold: 3.31 }],
    flags: [{ kind: "failed_run", label: "运行失败或没有最终模型", detail: "运行结束于 FAILED", task: "ev-v9_hotel_s1" }],
    summary: [{ dataset: "hotel", n: 2, n_ok: 2, holdout_mean: 0.7926, delta_b1_mean: 0.009, rounds_mean: 4, fe_in_final: "0/2",
      wall_sec: 1084, cost_usd: 0.011, worst_task: "ev-v9_hotel_s1" }],
    runs: [{ task: "ev-v9_hotel_s0", dataset: "hotel", seed: 0, status: "ok", holdout: 0.79, delta_b1: 0.01 }] },
  scenarios: [{ scenario: "leak", base: "hotel", passed: false, reason: "采纳了泄漏列", task: "ev-v9_scn_leak_hotel" }],
}
const snap: Snapshot = { ...baseSnap, id: "ev-v9_hotel_s0", status: "done", stage: "report", readonly: true, eval_label: "v9" }
const mount = (v: EvalView, s = snap, onTrace = vi.fn(), onSelect = vi.fn()) => {
  vi.spyOn(api, "evalView").mockResolvedValue(v)
  render(<QueryClientProvider client={new QueryClient()}><EvalTab snap={s} onTrace={onTrace} onSelect={onSelect} /></QueryClientProvider>)
  return { onTrace, onSelect }
}

test("本次运行：比率显示百分比，证据能跳到对应轮次，找不到轮次的不出按钮", async () => {
  const { onTrace } = mount(view)
  expect(await screen.findByText("25%")).toBeInTheDocument()
  expect(screen.getByText("否")).toBeInTheDocument()
  fireEvent.click(screen.getByRole("button", { name: "R2 · e03" }))
  expect(onTrace).toHaveBeenCalledWith("r2")
  expect(screen.queryByRole("button", { name: /e99/ })).toBeNull()
})

test("本版本：结论、变坏的行点了跳到那个种子；情景不通过可以跳过去", async () => {
  const { onSelect } = mount(view)
  fireEvent.click(await screen.findByRole("button", { name: "本版本" }))
  expect(screen.getByText(/有退步需看/)).toBeInTheDocument()
  fireEvent.click(screen.getByRole("cell", { name: "holdout" }))
  expect(onSelect).toHaveBeenCalledWith("ev-v9_hotel_s1")
  fireEvent.click(screen.getByRole("button", { name: "情景用例" }))
  fireEvent.click(screen.getByRole("button", { name: "看这次运行" }))
  expect(onSelect).toHaveBeenCalledWith("ev-v9_scn_leak_hotel")
})

test("用户任务只有本次运行", async () => {
  mount({ ...view, version: null, scenarios: null }, { ...snap, id: "t1", eval_label: null })
  await screen.findByText("25%")
  expect(screen.queryByRole("button", { name: "本版本" })).toBeNull()
})

test("本版本：指标显示中文名，并列出用时与花费的变化", async () => {
  mount(view)
  fireEvent.click(await screen.findByRole("button", { name: "本版本" }))
  expect(screen.getByRole("cell", { name: "相对 B1 的差" })).toBeInTheDocument()
  expect(screen.queryByRole("cell", { name: "delta_b1" })).toBeNull()
  expect(screen.getByRole("cell", { name: "用时（秒）" })).toBeInTheDocument()
  expect(screen.getByRole("cell", { name: "×0.5" })).toBeInTheDocument()
})

test("本版本：硬性标记显示中文说明，有每个数据集的合计，点最差的一次跳过去", async () => {
  const { onSelect } = mount(view)
  fireEvent.click(await screen.findByRole("button", { name: "本版本" }))
  expect(screen.getByText(/运行失败或没有最终模型：运行结束于 FAILED/)).toBeInTheDocument()
  expect(screen.queryByText(/failed_run/)).toBeNull()
  expect(screen.getByRole("cell", { name: "0.793" })).toBeInTheDocument()
  expect(screen.getByRole("cell", { name: "+0.009" })).toBeInTheDocument()
  fireEvent.click(screen.getByRole("button", { name: "最差的一次" }))
  expect(onSelect).toHaveBeenCalledWith("ev-v9_hotel_s1")
})
