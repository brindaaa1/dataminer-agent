import { fireEvent, render, screen } from "@testing-library/react"
import type { Round, Snapshot } from "@/api"
import { baseSnap } from "@/test-fixtures"
import { act } from "@testing-library/react"
import { setDev } from "@/useDev"
import { ProcessTab } from "./ProcessTab"

const round = (key: string, title: string, by_rule: boolean): Round => ({
  key, title, action: "PROMOTE_FIDELITY", action_label: "全量复验", action_user: "用全量数据复验", verdict: "ACCEPT", codes: [], by_rule,
  metrics: { train: null, valid: null, oot_dev: 0.65, oot_dev_ks: null, gap: 0.01 },
  hypothesis: "赛跑打平：lgbm、catboost 效果相当；按偏好顺序推荐 lgbm", user_note: null, alternatives: [], stop_reason: null, evaluations: [],
  llm_calls: 0, running: false, trials: {},
})

test("开发者视图：代码按规则做的决策标出「规则」，不显示成 LLM 的决策", () => {
  act(() => setDev(true))
  globalThis.ResizeObserver ??= class { observe() {} unobserve() {} disconnect() {} } as unknown as typeof ResizeObserver
  const snap: Snapshot = { ...baseSnap, status: "done", stage: "report",
    run: { metric: "auc", max_rounds: 6, rounds: [round("r1", "第 1 轮", true), round("r2", "第 2 轮", false)], chart: [], narration: [], brief: [], final: null, tree: [] } }
  render(<ProcessTab snap={snap} onTrace={vi.fn()} />)
  expect(screen.getByText("规则")).toBeInTheDocument()                    // 列表里只有第 1 轮带标记
  fireEvent.click(screen.getByText(/第 1 轮 · 全量复验/))
  expect(screen.getByText("决策（规则）")).toBeInTheDocument()
  fireEvent.click(screen.getByText(/第 2 轮 · 全量复验/))
  expect(screen.getByText("决策（LLM）")).toBeInTheDocument()
})
