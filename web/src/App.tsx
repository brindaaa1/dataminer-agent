import { useEffect, useState } from "react"
import { useQuery, useQueryClient } from "@tanstack/react-query"
import { api } from "@/api"
import { Chat } from "@/components/Chat"
import { TaskRail } from "@/components/TaskRail"
import { Workspace } from "@/components/Workspace"
import { useTaskStream } from "@/useTaskStream"

function remembered(): string | null {
  try { return localStorage.getItem("task") } catch { return null }
}

export default function App() {
  const [taskId, setTaskId] = useState<string | null>(remembered)
  // 评测运行由命令行在跑，没有事件流：运行中的版本定时重取
  const tasks = useQuery({ queryKey: ["tasks"], queryFn: api.tasks,
    refetchInterval: (q) => (q.state.data?.some((t) => t.eval?.running) ? 15000 : false) })
  const snap = useQuery({ queryKey: ["task", taskId], queryFn: () => api.task(taskId!), enabled: !!taskId, retry: false,
    refetchInterval: (q) => (q.state.data?.eval_label && q.state.data.status === "running" ? 5000 : false) })
  const evalRunning = !!snap.data?.eval_label && snap.data.status === "running"
  useTaskStream(taskId, !!snap.data && !snap.data.eval_label && (snap.data.status === "running" || snap.data.busy))
  const qc = useQueryClient()
  useEffect(() => { if (evalRunning) qc.invalidateQueries({ queryKey: ["events", taskId] }) }, [evalRunning, snap.data?.last_event_id, taskId, qc])
  const select = (id: string | null) => {
    setTaskId(id)
    try { if (id) localStorage.setItem("task", id); else localStorage.removeItem("task") } catch { /* 无痕模式等 */ }
  }
  useEffect(() => { if (snap.isError) select(null) }, [snap.isError])        // 记住的任务已经不在了
  return (
    <div className="flex h-screen bg-background text-foreground">
      <TaskRail tasks={tasks.data ?? []} current={taskId} onSelect={select} />
      <main className="flex min-w-0 flex-1 gap-3 p-3">
        <section className="flex w-[36%] min-w-[340px] flex-col overflow-hidden rounded-lg border">
          <Chat snap={snap.data ?? null} onCreated={select} />
        </section>
        <section className="flex min-w-0 flex-1 flex-col overflow-hidden rounded-lg border">
          {snap.data ? <Workspace snap={snap.data} onSelect={select} />
            : <p className="m-auto text-sm text-muted-foreground">新建任务，或在左侧选一个（示例任务可以直接看）。</p>}
        </section>
      </main>
    </div>
  )
}
