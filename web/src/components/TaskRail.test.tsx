import { fireEvent, render, screen } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import type { TaskSummary } from "@/api"
import { TaskRail } from "./TaskRail"

const t = (id: string, title: string, ev: TaskSummary["eval"] = null): TaskSummary =>
  ({ id, title, status: "done", created_at: null, example: false, eval: ev })
const v9 = { label: "v9", tier: "fast", conclusion: "无退步", running: false }
const tasks = [t("a", "hotel · 10-02"), t("ev-v9_hotel_s0", "hotel · 种子 0", v9), t("ev-v9_scn_leak_hotel", "情景 leak @ hotel", v9),
  t("ev-v8_hotel_s0", "hotel · 种子 0", { label: "v8", tier: "fast", conclusion: null, running: false }),
  t("ev-v10_hotel_s0", "hotel · 种子 0", { label: "v10", tier: "fast", conclusion: null, running: true })]

test("评测运行按版本分组，版本标题带档位和结论，点种子选中它", () => {
  const sel = vi.fn()
  render(<QueryClientProvider client={new QueryClient()}><TaskRail tasks={tasks} current={null} onSelect={sel} /></QueryClientProvider>)
  fireEvent.click(screen.getByLabelText("任务列表"))                       // 展开侧栏
  expect(screen.getByText("评测运行")).toBeInTheDocument()
  expect(screen.getByText("v9 · fast · 无退步")).toBeInTheDocument()
  expect(screen.getByText("v8 · fast")).toBeInTheDocument()
  expect(screen.getByText("v10 · fast · 运行中")).toBeInTheDocument()
  fireEvent.click(screen.getByText("情景 leak @ hotel"))
  expect(sel).toHaveBeenCalledWith("ev-v9_scn_leak_hotel")
  expect(screen.getAllByTitle("复制为新任务")).toHaveLength(1)          // 只有普通任务能复制
})
