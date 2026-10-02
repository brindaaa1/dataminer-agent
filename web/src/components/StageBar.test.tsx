import { render, screen } from "@testing-library/react"
import { StageBar } from "./StageBar"

test("marks the current stage", () => {
  render(<StageBar stage="modeling" />)
  expect(screen.getByText("建模")).toHaveAttribute("aria-current", "step")
  expect(screen.getByText("接入")).not.toHaveAttribute("aria-current")
})
