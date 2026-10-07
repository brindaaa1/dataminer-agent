import { useEffect } from "react"
import { useQueryClient } from "@tanstack/react-query"
import type { TaskEvent } from "@/api"

/** 任务活跃（建模中、接入起草中）时订阅 SSE：新事件按 id 去重后追加进事件缓存，并重取快照；收到 idle 就关掉。 */
export function useTaskStream(id: string | null, active: boolean) {
  const qc = useQueryClient()
  useEffect(() => {
    if (!id || !active) return
    const es = new EventSource(`/api/tasks/${id}/stream`)
    es.addEventListener("task", (m) => {
      const e = JSON.parse((m as MessageEvent).data) as TaskEvent
      qc.setQueryData<TaskEvent[]>(["events", id], (old) => (old && !old.some((x) => x.id === e.id) ? [...old, e] : old))
      qc.invalidateQueries({ queryKey: ["task", id] })
    })
    es.addEventListener("idle", () => {
      es.close()
      qc.invalidateQueries({ queryKey: ["task", id] })
      qc.invalidateQueries({ queryKey: ["tasks"] })
    })
    return () => es.close()
  }, [id, active, qc])
}
