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
