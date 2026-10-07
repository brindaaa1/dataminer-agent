import { fireEvent, render, screen } from "@testing-library/react"
import type { Question } from "@/api"
import { buildReplies, QuestionCard } from "./QuestionCard"

const qs: Question[] = [
  { key: "label", question: "标签对吗？", recommended: '{"col":"is_canceled"}', display: "is_canceled = 1", required: true },
  { key: "lead_time", question: "何时已知？", recommended: "下单时", display: "下单时", required: false },
]

test("buildReplies: 采纳发推荐值，修改发文字，补充说明单独一项", () => {
  expect(buildReplies(qs, { label: { mode: "accept" }, lead_time: { mode: "edit", text: "入住时" } }, " 补充 "))
    .toEqual({ label: '{"col":"is_canceled"}', lead_time: "入住时", 补充说明: "补充" })
})

test("每题都答了才能提交", () => {
  const onSubmit = vi.fn()
  render(<QuestionCard questions={qs} extra="" pending={false} onSubmit={onSubmit} />)
  const submit = screen.getByRole("button", { name: /提交本轮/ })
  expect(submit).toBeDisabled()
  fireEvent.click(screen.getByRole("button", { name: "全部采纳" }))
  expect(submit).toBeEnabled()
  fireEvent.click(submit)
  expect(onSubmit).toHaveBeenCalledWith({ label: '{"col":"is_canceled"}', lead_time: "下单时" })
})
