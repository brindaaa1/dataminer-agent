import { useState } from "react"
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { api, type Snapshot } from "@/api"
import { Bubble } from "@/components/Bubble"
import { ErrorText } from "@/components/ErrorText"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"

export function ChatReport({ snap }: { snap: Snapshot }) {
  const [note, setNote] = useState("")
  const [withNote, setWithNote] = useState(false)
  const qc = useQueryClient()
  const accept = useMutation({
    mutationFn: () => api.accept(snap.id, note),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["task", snap.id] }); qc.invalidateQueries({ queryKey: ["tasks"] }) },
  })
  if (snap.example) return <Bubble from="system">示例任务：录制的真实运行，只读。</Bubble>
  if (snap.status === "accepted") return <Bubble from="system">报告已确认{snap.note ? `：${snap.note}` : ""}。产出可以在右侧打包下载。</Bubble>
  if (snap.status !== "done") return null
  return (
    <Bubble>报告已生成，在右侧「产出」。
      {withNote && <Textarea className="mt-2" rows={2} value={note} onChange={(e) => setNote(e.target.value)} placeholder="备注（会记进事件日志）" />}
      <div className="mt-2 flex gap-2">
        <Button size="sm" onClick={() => accept.mutate()} disabled={accept.isPending}>确认报告</Button>
        {!withNote && <Button size="sm" variant="outline" onClick={() => setWithNote(true)}>加备注后确认</Button>}
      </div>
      <ErrorText error={accept.error} />
    </Bubble>
  )
}
