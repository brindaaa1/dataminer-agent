import { defineConfig } from "@playwright/test"

export default defineConfig({
  testDir: "./e2e",
  timeout: 400_000,
  use: { baseURL: "http://127.0.0.1:8765" },
  webServer: {
    // 先构建前端，再由 FastAPI 托管 dist；工作区放临时目录，不往 Langfuse 发
    command: "npm run build && cd .. && DATAMINER_WORKSPACE=$(mktemp -d) LANGFUSE_PUBLIC_KEY= python -m uvicorn server.app:app --port 8765",
    url: "http://127.0.0.1:8765/api/tasks",
    reuseExistingServer: false,
    timeout: 120_000,
  },
})
