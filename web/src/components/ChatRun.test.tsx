import { fireEvent, render, screen } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { baseSnap } from "@/test-fixtures"
import { ChatRun } from "./ChatRun"

test("续跑失败（比如 409）时要显示原因，不能像什么都没发生", async () => {
  globalThis.fetch = vi.fn(async () => new Response(JSON.stringify({ detail: "当前状态 running 不能执行这个操作" }), { status: 409 })) as typeof fetch
  render(<QueryClientProvider client={new QueryClient()}><ChatRun snap={{ ...baseSnap, status: "stopped", stage: "modeling", run: { metric: "auc", max_rounds: 3, rounds: [], chart: [], narration: [], final: null, tree: [] } }} /></QueryClientProvider>)
  fireEvent.click(screen.getByRole("button", { name: /从检查点续跑/ }))
  expect(await screen.findByText(/当前状态 running 不能执行这个操作/)).toBeInTheDocument()
})
