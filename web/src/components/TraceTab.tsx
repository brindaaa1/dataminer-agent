import { useMemo, useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { api, type Snapshot, type TaskEvent } from "@/api"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { mmss } from "@/lib/format"
import { category, segmentOf, summary } from "@/lib/trace"

const CATS: [string, string][] = [["all", "全部"], ["state", "状态"], ["llm", "LLM"], ["eval", "评估"], ["decision", "决策"], ["intake", "接入"]]
const CAT_LABEL: Record<string, string> = { state: "状态", llm: "LLM", eval: "评估", decision: "决策", intake: "接入", other: "其他" }

function Row({ e, t0 }: { e: TaskEvent; t0: number }) {
  const c = category(e.type)
  const head = (
    <span className="flex items-center gap-2">
      <span className="w-12 shrink-0 tabular-nums text-muted-foreground">{mmss(e.ts - t0)}</span>
      <Badge variant="outline">{CAT_LABEL[c]}</Badge><span className="truncate">{summary(e)}</span>
    </span>
  )
  if (c !== "llm") return head
  const pre = "max-h-64 overflow-auto whitespace-pre-wrap rounded bg-muted p-2 text-xs"
  return (
    <details>
      <summary className="cursor-pointer list-none">{head}</summary>
      <div className="mt-1 space-y-1">
        <div className="text-xs text-muted-foreground">prompt</div><pre className={pre}>{e.payload.prompt}</pre>
        <div className="text-xs text-muted-foreground">回复</div><pre className={pre}>{e.payload.response}</pre>
      </div>
    </details>
  )
}

export function TraceTab({ snap, round, setRound }: { snap: Snapshot; round: string; setRound: (r: string) => void }) {
  const q = useQuery({ queryKey: ["events", snap.id], queryFn: () => api.events(snap.id) })
  const [cat, setCat] = useState("all")
  const events = useMemo(() => q.data ?? [], [q.data])
  const segs = useMemo(() => segmentOf(events), [events])
  const t0 = events[0]?.ts ?? 0
  const rows = events.map((e, i) => ({ e, seg: segs[i] }))
    .filter(({ e, seg }) => (round === "all" || seg === round) && (cat === "all" || category(e.type) === cat))
  const opts: [string, string][] = [["all", "全部轮次"], ["intake", "接入"], ...(snap.run?.rounds.map((r) => [r.key, r.title] as [string, string]) ?? []), ["finale", "收尾"]]
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <select aria-label="轮次" value={round} onChange={(e) => setRound(e.target.value)} className="h-8 rounded-md border bg-background px-2">
          {opts.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
        </select>
        {CATS.map(([v, l]) => <Button key={v} size="sm" variant={cat === v ? "default" : "outline"} onClick={() => setCat(v)}>{l}</Button>)}
        <span className="ml-auto flex gap-3">
          {snap.intake_langfuse_url && <a className="text-primary" href={snap.intake_langfuse_url} target="_blank" rel="noreferrer">接入 trace ↗</a>}
          {snap.langfuse_url && <a className="text-primary" href={snap.langfuse_url} target="_blank" rel="noreferrer">在 Langfuse 打开整条 trace ↗</a>}
        </span>
      </div>
      <ul className="divide-y text-sm">{rows.map(({ e }) => <li key={e.id} className="py-1"><Row e={e} t0={t0} /></li>)}</ul>
      {rows.length === 0 && <p className="text-sm text-muted-foreground">没有符合条件的事件。</p>}
    </div>
  )
}
