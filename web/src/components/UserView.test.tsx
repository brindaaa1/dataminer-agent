import { act, render, screen } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import type { ReactNode } from "react"
import type { Round, Snapshot } from "@/api"
import { baseSnap } from "@/test-fixtures"
import { setDev } from "@/useDev"
import { ChatRun } from "./ChatRun"
import { OutputTab } from "./OutputTab"
import { ProcessTab } from "./ProcessTab"
import { Workspace } from "./Workspace"

const wrap = (n: ReactNode) => <QueryClientProvider client={new QueryClient()}>{n}</QueryClientProvider>
const round: Round = {
  key: "r1", title: "第 1 轮", action: "PROMOTE_FIDELITY", action_label: "全量复验", action_user: "用全量数据复验", verdict: "REJECT",
  codes: ["OVERFIT_GAP"], by_rule: true, metrics: { train: 0.9, valid: 0.87, oot_dev: 0.833, oot_dev_ks: 0.5, gap: 0.041 },
  hypothesis: "升全量", user_note: null, alternatives: ["TUNE"], stop_reason: null, evaluations: [], llm_calls: 1, running: false, trials: {},
}
const snap: Snapshot = {
  ...baseSnap, status: "done", stage: "report", langfuse_url: "https://lf/x",
  artifacts: [{ name: "model_card.md", label: "模型卡" }, { name: "events.json", label: "完整事件" }],
  run: {
    metric: "auc", max_rounds: 5, rounds: [round], chart: [], tree: [],
    narration: [{ key: "r1", text: "第 1 轮 · 全量复验：拒绝 · 时间外 AUC 0.833，gap 0.041（OVERFIT_GAP）", tone: "bad" }],
    brief: [{ key: "stop", text: "停止：达到轮数上限", tone: "neutral" }],
    final: { exp_id: "t1_e04", model: "lr_scorecard", oot_dev: 0.816, holdout: 0.731, holdout_ks: 0.335, drop: 0.086, overfit: true,
      n_experiments: 6, n_accepted: 1, lift_top10: 2.5, risks: ["疑似对 OOT-dev 过拟合", "停止原因：LLM_STOP: final_model.exists"],
      risks_user: ["疑似对 OOT-dev 过拟合"] },
  },
}
beforeEach(() => act(() => setDev(false)))

test("用户视图：只有数据 / 过程 / 产出三个页签；开发者视图多出 Trace、评测", () => {
  render(wrap(<Workspace snap={snap} onSelect={vi.fn()} />))
  expect(screen.queryByRole("tab", { name: "Trace" })).toBeNull()
  act(() => setDev(true))
  expect(screen.getByRole("tab", { name: "Trace" })).toBeInTheDocument()
  expect(screen.getByRole("tab", { name: "评测" })).toBeInTheDocument()
})

test("用户视图：对话只播关键节点；开发者视图是完整播报", () => {
  render(wrap(<ChatRun snap={snap} />))
  expect(screen.getByText("停止：达到轮数上限")).toBeInTheDocument()
  expect(screen.queryByText(/OVERFIT_GAP/)).toBeNull()
  act(() => setDev(true))
  expect(screen.getByText(/OVERFIT_GAP/)).toBeInTheDocument()
})

test("用户视图：过程页没有诊断码、gap、LLM 原文按钮和规则标记，动作和结论用业务语言；Langfuse 链接两种视图都有", () => {
  globalThis.ResizeObserver ??= class { observe() {} unobserve() {} disconnect() {} } as unknown as typeof ResizeObserver
  render(<ProcessTab snap={snap} onTrace={vi.fn()} />)
  expect(screen.getAllByText(/用全量数据复验/).length).toBeGreaterThan(0)
  expect(screen.getAllByText("没通过").length).toBeGreaterThan(0)
  for (const t of [/gap/, /OVERFIT_GAP/, /看 LLM 原文/, /^规则$/, /考虑过/]) expect(screen.queryByText(t)).toBeNull()
  expect(screen.getByRole("link", { name: /Langfuse/ })).toHaveAttribute("href", "https://lf/x")
  act(() => setDev(true))
  expect(screen.getByText(/看 LLM 原文/)).toBeInTheDocument()
  expect(screen.getAllByText(/gap/).length).toBeGreaterThan(0)
})

test("用户视图：产出页只列要处理的风险、只有模型卡，卡片有头部 10% 提升度", () => {
  render(wrap(<OutputTab snap={snap} />))
  expect(screen.getByText("疑似对 OOT-dev 过拟合")).toBeInTheDocument()
  expect(screen.queryByText(/LLM_STOP/)).toBeNull()
  expect(screen.queryByText("完整事件")).toBeNull()
  expect(screen.getByText("头部 10% 提升度")).toBeInTheDocument()
  expect(screen.getByRole("button", { name: /下载报告/ })).toBeInTheDocument()            // 下载的是看到的这份报告
  expect(screen.getByRole("link", { name: /查看详细分析过程（Langfuse）/ })).toHaveAttribute("href", "https://lf/x")
  expect(screen.queryByText(/打包下载全部产出/)).toBeNull()                              // 中间产出的压缩包只在开发者视图
  expect(screen.queryByText(/OOT-dev AUC/)).toBeNull()
  act(() => setDev(true))
  expect(screen.getByText("完整事件")).toBeInTheDocument()
  expect(screen.getByText(/LLM_STOP/)).toBeInTheDocument()
  expect(screen.getByText(/打包下载全部产出/)).toBeInTheDocument()
})

test("用户版模型卡：章节按顺序编号，有建模过程（每步做什么、为什么、结果），指标表用中文段名", async () => {
  const { userCard } = await import("./OutputTab")
  const md = "# 模型卡：t\n\n> x\n\n## 执行摘要\n\n摘要\n\n## 1. 任务规格\n\nspec\n\n## 4. 各段指标\n\n| 段 | AUC |\n|---|---|\n| train | 0.9 |\n| OOT-dev | 0.8 |\n| holdout | 0.7 |\n\n- valid − OOT-dev gap：0.1\n\n## 6. 特征清单（按平均 |SHAP| 排序）\n\n| 特征 | 来源 |\n|---|---|\n| a | 原始字段 |\n\n## 8. 风险提示\n\n- 停止原因：x\n\n## 9. 模型设定（复现用）\n\n- 模型：lgbm\n\n| 参数 | 取值 | 含义 |\n|---|---|---|\n| num_leaves | 31 | 叶子数 |\n"
  const out = userCard(md, [{ ...round, verdict: "ACCEPT", user_note: "先用全部数据复验", metrics: { ...round.metrics, train: 0.85, oot_dev: 0.81 } }], "AUC")
  const heads = out.split("\n").filter((l) => l.startsWith("## "))
  expect(heads).toEqual(["## 执行摘要", "## 1. 建模过程", "## 2. 效果", "## 3. 主要特征", "## 4. 模型设定（复现用）"])
  expect(out).toContain("第 1 轮 · 用全量数据复验**（采纳）：先用全部数据复验。时间外验证 AUC 0.810（训练集 0.850）")
  for (const t of ["| 训练集 | 0.9 |", "| 最终检验 | 0.7 |", "| num_leaves | 31 | 叶子数 |"]) expect(out).toContain(t)
  for (const t of ["任务规格", "gap", "停止原因"]) expect(out).not.toContain(t)
})

test("没配置 Langfuse 的任务：入口照样显示，置灰并提示怎么开启", () => {
  render(wrap(<OutputTab snap={{ ...snap, langfuse_url: null }} />))
  expect(screen.getByText(/查看详细分析过程（Langfuse）/)).toHaveAttribute("title", "在 .env 配置 Langfuse 后，可以在这里查看每一步的完整过程")
  expect(screen.queryByRole("link", { name: /Langfuse/ })).toBeNull()
})
