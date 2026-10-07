import { render, screen } from "@testing-library/react"
import type { Snapshot } from "@/api"
import { baseSnap } from "@/test-fixtures"
import { DataTab } from "./DataTab"

test("确认环节显示推荐的领域", () => {
  const snap: Snapshot = { ...baseSnap, intake: { round: 1, questions: [], changed: [], yaml: {},
    draft: { label: "loan_status 取 Charged Off → 1", observation_time_expr: "issue_d", windows: null, metric: "AUC（护栏 KS）",
      domain: "信贷风控（违约预测）", data_summary: "每行是一笔贷款。",
      fields: [{ name: "price", type: "numeric", availability: "post_outcome", reason: "当前电价", group: "leak", history: "历史值可用：前 1 条" }], confirmed: [] } } }
  render(<DataTab snap={snap} />)
  expect(screen.getByText("领域")).toBeInTheDocument()
  expect(screen.getByText("信贷风控（违约预测）")).toBeInTheDocument()
  expect(screen.getByText("历史值可用：前 1 条")).toBeInTheDocument()
})
