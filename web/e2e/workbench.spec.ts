import path from "path"
import { expect, test } from "@playwright/test"

const SAMPLE = path.resolve(process.cwd(), "../data_sample/hotel_bookings")

test("mock：上传 → 确认配置 → 建模 → 确认报告 → 下载", async ({ page }) => {
  await page.goto("/")
  await page.getByTestId("file-input").setInputFiles([path.join(SAMPLE, "hotel_bookings_sample.csv"), path.join(SAMPLE, "业务说明.md")])
  await page.getByLabel("最多轮数").selectOption("1")
  await page.getByRole("button", { name: "发送" }).click()
  await page.getByRole("button", { name: "全部采纳" }).click({ timeout: 60_000 })
  await page.getByRole("button", { name: /提交本轮/ }).click()
  await page.getByRole("button", { name: "确认配置并开始建模" }).click({ timeout: 60_000 })
  await page.getByRole("button", { name: "确认报告" }).click({ timeout: 300_000 })
  const dl = page.waitForEvent("download")
  await page.getByRole("link", { name: /打包下载/ }).click()
  expect((await dl).suggestedFilename()).toMatch(/\.zip$/)
})
