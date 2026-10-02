import { fireEvent, render, screen } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { Composer } from "./Composer"

test("分两次添加附件：先 CSV 再 .md，两个都保留，发送可用", () => {
  render(<QueryClientProvider client={new QueryClient()}><Composer onCreated={() => {}} /></QueryClientProvider>)
  const input = screen.getByTestId("file-input")
  fireEvent.change(input, { target: { files: [new File(["a,b"], "data.csv")] } })
  fireEvent.change(input, { target: { files: [new File(["说明"], "业务说明.md")] } })
  expect(screen.getByText(/data\.csv/)).toBeInTheDocument()
  expect(screen.getByText(/业务说明\.md/)).toBeInTheDocument()
  expect(screen.getByRole("button", { name: /发送/ })).toBeEnabled()
})
