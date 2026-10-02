import { render, screen } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import type { ReactNode } from "react"
import type { Snapshot } from "@/api"
import { baseSnap } from "@/test-fixtures"
import { OutputTab } from "./OutputTab"

const wrap = (n: ReactNode) => <QueryClientProvider client={new QueryClient()}>{n}</QueryClientProvider>
const done: Snapshot = {
  ...baseSnap, status: "done", stage: "report",
  run: {
    metric: "auc", max_rounds: 5, rounds: [], chart: [], narration: [], tree: [],
    final: { exp_id: "t1_e04", model: "lr_scorecard", oot_dev: 0.816, holdout: 0.731, holdout_ks: 0.335, drop: 0.086,
      overfit: true, n_experiments: 6, n_accepted: 1, risks: ["疑似对 OOT-dev 过拟合"] },
  },
}

test("结论卡：过拟合时 holdout 标红，风险提示排在最前；确认前不能下载", () => {
  render(wrap(<OutputTab snap={done} />))
  expect(screen.getByText("0.731")).toHaveClass("text-red-600")
  expect(screen.getByText("疑似对 OOT-dev 过拟合")).toBeInTheDocument()
  expect(screen.getByText("确认报告后可下载")).toBeInTheDocument()
})

test("确认后出现下载链接", () => {
  render(wrap(<OutputTab snap={{ ...done, status: "accepted", readonly: true }} />))
  expect(screen.getByRole("link", { name: /打包下载/ })).toHaveAttribute("href", "/api/tasks/t1/bundle.zip")
})
