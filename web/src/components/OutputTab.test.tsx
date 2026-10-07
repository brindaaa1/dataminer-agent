import { render, screen } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import type { ReactNode } from "react"
import type { Snapshot } from "@/api"
import { baseSnap } from "@/test-fixtures"
import { act } from "@testing-library/react"
import { setDev } from "@/useDev"
import { OutputTab } from "./OutputTab"

const wrap = (n: ReactNode) => <QueryClientProvider client={new QueryClient()}>{n}</QueryClientProvider>
const done: Snapshot = {
  ...baseSnap, status: "done", stage: "report",
  run: {
    metric: "auc", max_rounds: 5, rounds: [], chart: [], narration: [], brief: [], tree: [],
    final: { exp_id: "t1_e04", model: "lr_scorecard", oot_dev: 0.816, holdout: 0.731, holdout_ks: 0.335, drop: 0.086,
      overfit: true, n_experiments: 6, n_accepted: 1, lift_top10: 2.5, risks: ["疑似对 OOT-dev 过拟合"], risks_user: ["疑似对 OOT-dev 过拟合"] },
  },
}

test("开发者视图：结论卡过拟合时 holdout 标红，风险提示排在最前；确认前不能打包下载", () => {
  act(() => setDev(true))
  render(wrap(<OutputTab snap={done} />))
  expect(screen.getByText("0.731")).toHaveClass("text-red-600")
  expect(screen.getByText("疑似对 OOT-dev 过拟合")).toBeInTheDocument()
  expect(screen.getByTitle("确认报告后可下载")).toBeInTheDocument()
})

test("开发者视图：确认后出现打包下载链接", () => {
  act(() => setDev(true))
  render(wrap(<OutputTab snap={{ ...done, status: "accepted", readonly: true }} />))
  expect(screen.getByRole("link", { name: /打包下载/ })).toHaveAttribute("href", "/api/tasks/t1/bundle.zip")
})
