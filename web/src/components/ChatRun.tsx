import { useMutation, useQueryClient } from "@tanstack/react-query"
import { Loader2, Play, Square } from "lucide-react"
import { api, type Snapshot } from "@/api"
import { Bubble } from "@/components/Bubble"
import { ErrorText } from "@/components/ErrorText"
import { Button } from "@/components/ui/button"

export function ChatRun({ snap }: { snap: Snapshot }) {
  const qc = useQueryClient()
  const refresh = () => { qc.invalidateQueries({ queryKey: ["task", snap.id] }); qc.invalidateQueries({ queryKey: ["tasks"] }) }
  const stop = useMutation({ mutationFn: () => api.stop(snap.id), onSuccess: refresh })
  const resume = useMutation({ mutationFn: () => api.resume(snap.id), onSuccess: refresh })
  const r = snap.run
  if (!r && snap.status !== "running") return null
  return (
    <>
      <Bubble from="system">开始建模{r?.max_rounds ? `（最多 ${r.max_rounds} 轮）` : ""}</Bubble>
      {r?.narration.map((n) => (
        <Bubble key={n.key} tone={n.tone}>
          {n.tone === "running" && <Loader2 className="mr-1 inline size-3 animate-spin" />}{n.text}
        </Bubble>
      ))}
      {snap.status === "running" && !snap.readonly && (
        <Button size="sm" variant="outline" onClick={() => stop.mutate()} disabled={stop.isPending}><Square className="size-3" />中止</Button>
      )}
      <ErrorText error={stop.error ?? resume.error} />
      {(snap.status === "stopped" || snap.status === "failed") && (
        <Bubble tone={snap.status === "failed" ? "bad" : "neutral"}>
          {snap.status === "stopped" ? "已中止。" : `运行失败：${snap.error ?? ""}`}
          {!snap.readonly && <div className="mt-2"><Button size="sm" onClick={() => resume.mutate()} disabled={resume.isPending}><Play className="size-3" />从检查点续跑</Button></div>}
        </Bubble>
      )}
    </>
  )
}
