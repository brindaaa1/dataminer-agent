import { render } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { baseSnap } from "@/test-fixtures"
import { Chat } from "./Chat"

test("新版 Chrome 的 scrollIntoView 返回 Promise：自动滚动的 effect 不能把它当成清理函数", () => {
  Element.prototype.scrollIntoView = () => Promise.resolve() as unknown as void
  const qc = new QueryClient()
  const ui = (last: number) => <QueryClientProvider client={qc}><Chat snap={{ ...baseSnap, last_event_id: last }} onCreated={() => {}} /></QueryClientProvider>
  const { rerender, unmount } = render(ui(1))
  expect(() => { rerender(ui(2)); unmount() }).not.toThrow()
})

test("新建任务时，agent 先发一段使用说明：适合什么数据、怎么建模、最后给什么", () => {
  const { getByText } = render(<QueryClientProvider client={new QueryClient()}><Chat snap={null} onCreated={() => {}} /></QueryClientProvider>)
  for (const t of [/二分类/, /时间列/, /lgbm、catboost/, /一份报告/]) expect(getByText(t)).toBeInTheDocument()
})

test("示例和已有任务的对话也以使用说明开头，但不再提示上传", () => {
  Element.prototype.scrollIntoView = () => {}
  const { getByText, queryByText } = render(<QueryClientProvider client={new QueryClient()}><Chat snap={baseSnap} onCreated={() => {}} /></QueryClientProvider>)
  expect(getByText(/二分类/)).toBeInTheDocument()
  expect(queryByText(/在下面上传/)).toBeNull()
})
