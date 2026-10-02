import { useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { Download } from "lucide-react"
import Markdown from "react-markdown"
import remarkGfm from "remark-gfm"
import { api, type Final, type Snapshot } from "@/api"
import { buttonVariants } from "@/components/ui/button"
import { f3, METRIC } from "@/lib/format"
import { cn } from "@/lib/utils"

function Conclusion({ f, metric }: { f: Final; metric: string }) {
  const L = METRIC[metric] ?? metric
  const kpis: [string, string, string][] = [
    ["最终模型", f.model ?? "—", f.exp_id], [`OOT-dev ${L}`, f3(f.oot_dev), ""],
    [`holdout ${L}`, f3(f.holdout), `KS ${f3(f.holdout_ks)}`], ["实验", String(f.n_experiments), `采纳 ${f.n_accepted}`],
  ]
  return (
    <div className="space-y-2">
      <div className="grid grid-cols-4 gap-2">
        {kpis.map(([k, v, sub], i) => (
          <div key={k} className="rounded-md border p-2">
            <div className="text-xs text-muted-foreground">{k}</div>
            <div className={cn("text-lg font-semibold tabular-nums", i === 2 && f.overfit && "text-red-600")}>{v}</div>
            <div className="truncate text-xs text-muted-foreground">{sub}</div>
          </div>
        ))}
      </div>
      {f.risks.map((r, i) => <div key={i} className="border-l-4 border-l-red-600 bg-red-50 px-3 py-1.5 text-sm dark:bg-red-950/30">{r}</div>)}
    </div>
  )
}

export function OutputTab({ snap }: { snap: Snapshot }) {
  const names = snap.artifacts.map((a) => a.name)
  const [sel, setSel] = useState<string | null>(null)
  const cur = sel && names.includes(sel) ? sel : names.includes("model_card.md") ? "model_card.md" : names[0]
  const body = useQuery({ queryKey: ["artifact", snap.id, cur, snap.last_event_id], queryFn: () => api.artifact(snap.id, cur!), enabled: !!cur })
  const canDownload = snap.example || snap.status === "accepted"
  return (
    <div className="grid grid-cols-[170px_minmax(0,1fr)] gap-4">
      <aside className="space-y-1 text-sm">
        <h4 className="text-xs text-muted-foreground">文件</h4>
        {snap.artifacts.map((a) => (
          <button key={a.name} onClick={() => setSel(a.name)}
            className={cn("block w-full truncate rounded px-2 py-1 text-left", a.name === cur && "bg-accent")}>{a.label}</button>
        ))}
        {canDownload
          ? <a href={`/api/tasks/${snap.id}/bundle.zip`} download className={cn(buttonVariants({ size: "sm", variant: "outline" }), "mt-3 w-full")}><Download className="size-4" />打包下载</a>
          : <><span className={cn(buttonVariants({ size: "sm", variant: "outline" }), "pointer-events-none mt-3 w-full opacity-50")}><Download className="size-4" />打包下载</span>
            <p className="text-xs text-muted-foreground">确认报告后可下载</p></>}
      </aside>
      <div className="min-w-0 space-y-3">
        {snap.run?.final && <Conclusion f={snap.run.final} metric={snap.run.metric} />}
        {cur && (cur.endsWith(".md")
          ? <article className="prose prose-sm max-w-none dark:prose-invert"><Markdown remarkPlugins={[remarkGfm]}>{body.data ?? ""}</Markdown></article>
          : <pre className="overflow-auto rounded bg-muted p-3 text-xs">{body.data}</pre>)}
        {!cur && <p className="text-sm text-muted-foreground">还没有产出。</p>}
      </div>
    </div>
  )
}
