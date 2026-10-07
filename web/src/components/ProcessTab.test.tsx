import { fireEvent, render, screen } from "@testing-library/react"
import type { Round, Snapshot } from "@/api"
import { baseSnap } from "@/test-fixtures"
import { ProcessTab } from "./ProcessTab"

const round = (key: string, title: string, by_rule: boolean): Round => ({
  key, title, action: "PROMOTE_FIDELITY", action_label: "全量复验", verdict: "ACCEPT", codes: [], by_rule,
  metrics: { train: null, valid: null, oot_dev: 0.65, oot_dev_ks: null, gap: 0.01 },
  hypothesis: "赛跑打平：lgbm、catboost 效果相当；按偏好顺序推荐 lgbm", alternatives: [], stop_reason: null, evaluations: [],
  llm_calls: 0, running: false, trials: {},
})

test("代码按规则做的决策标出「规则」，不显示成 LLM 的决策", () => {
  globalThis.ResizeObserver ??= class { observe() {} unobserve() {} disconnect() {} } as unknown as typeof ResizeObserver
  const snap: Snapshot = { ...baseSnap, status: "done", stage: "report",
    run: { metric: "auc", max_rounds: 6, rounds: [round("r1", "第 1 轮", true), round("r2", "第 2 轮", false)], chart: [], narration: [], final: null, tree: [] } }
  render(<ProcessTab snap={snap} onTrace={vi.fn()} />)
  expect(screen.getByText("规则")).toBeInTheDocument()                    // 列表里只有第 1 轮带标记
  fireEvent.click(screen.getByText(/第 1 轮 · 全量复验/))
  expect(screen.getByText("决策（规则）")).toBeInTheDocument()
  fireEvent.click(screen.getByText(/第 2 轮 · 全量复验/))
  expect(screen.getByText("决策（LLM）")).toBeInTheDocument()
})
