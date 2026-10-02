import { useMutation, useQueryClient } from "@tanstack/react-query"
import { Loader2 } from "lucide-react"
import { api, type Snapshot } from "@/api"
import { Bubble } from "@/components/Bubble"
import { ErrorText } from "@/components/ErrorText"
import { QuestionCard } from "@/components/QuestionCard"
import { Button } from "@/components/ui/button"

export function ChatIntake({ snap, extra, onSent }: { snap: Snapshot; extra: string; onSent: () => void }) {
  const qc = useQueryClient()
  const refresh = () => qc.invalidateQueries({ queryKey: ["task", snap.id] })
  const answers = useMutation({ mutationFn: (r: Record<string, string>) => api.answers(snap.id, r), onSuccess: () => { onSent(); refresh() } })
  const retry = useMutation({ mutationFn: () => api.retry(snap.id), onSuccess: refresh })
  const run = useMutation({ mutationFn: () => api.run(snap.id), onSuccess: () => { refresh(); qc.invalidateQueries({ queryKey: ["tasks"] }) } })
  const it = snap.intake
  return (
    <>
      <Bubble from="user">
        <div className="font-medium">📎 {snap.data.csv_name}</div>
        {snap.data.description && (
          <details><summary className="cursor-pointer text-muted-foreground">业务说明</summary>
            <p className="whitespace-pre-wrap">{snap.data.description}</p></details>
        )}
      </Bubble>
      {snap.data.n_rows != null && <Bubble>读完了：{snap.data.n_rows.toLocaleString()} 行、{snap.data.n_cols} 列。草稿在右侧「数据」。</Bubble>}
      {snap.busy && <Bubble from="system"><Loader2 className="mr-1 inline size-3 animate-spin" />LLM 起草中…</Bubble>}
      {it && it.questions.length > 0 && (
        <QuestionCard key={it.round} questions={it.questions} extra={extra} pending={answers.isPending} onSubmit={(r) => answers.mutate(r)} />
      )}
      <ErrorText error={answers.error ?? retry.error ?? run.error} />
      {snap.status === "intake_failed" && (
        <Bubble tone="bad">接入没有完成：{(snap.error ?? "").slice(0, 120)}
          {(snap.error ?? "").length > 120 && (
            <details className="text-xs"><summary className="cursor-pointer text-muted-foreground">完整原因</summary>
              <p className="whitespace-pre-wrap break-all">{snap.error}</p></details>
          )}
          <div className="mt-2"><Button size="sm" onClick={() => retry.mutate()} disabled={retry.isPending}>重试本轮</Button></div>
        </Bubble>
      )}
      {it && Object.keys(it.yaml).length > 0 && (
        <Bubble>配置已写好，右侧「数据」里可以看完整配置。
          {snap.status === "config_ready" && (
            <div className="mt-2"><Button size="sm" onClick={() => run.mutate()} disabled={run.isPending}>确认配置并开始建模</Button></div>
          )}
        </Bubble>
      )}
    </>
  )
}
