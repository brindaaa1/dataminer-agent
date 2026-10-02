import type { FieldRow, Snapshot, Windows } from "@/api"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"

const GROUPS: Record<FieldRow["group"], [string, string]> = {
  ok: ["可用", "bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-300"],
  leak: ["泄漏", "bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300"],
  quarantine: ["隔离/待定", "bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300"],
  meta: ["仅用于切分", "bg-zinc-100 text-zinc-700 dark:bg-zinc-800 dark:text-zinc-300"],
}
const WIN: [keyof Windows, string, string][] = [
  ["train_valid", "训练/验证", "bg-blue-500"], ["oot_dev", "OOT-dev", "bg-amber-500"], ["holdout", "holdout", "bg-violet-500"],
]

export function months(a: string, b: string) {
  const [y1, m1] = a.split("-").map(Number)
  const [y2, m2] = b.split("-").map(Number)
  return (y2 - y1) * 12 + (m2 - m1) + 1
}

export function DataTab({ snap }: { snap: Snapshot }) {
  const d = snap.intake?.draft
  const changed = new Set(snap.intake?.changed ?? [])
  const header = (
    <p className="mb-3 text-sm">
      <b>{snap.data.csv_name}</b>{snap.data.n_rows != null && ` · ${snap.data.n_rows.toLocaleString()} 行 · ${snap.data.n_cols} 列`}
    </p>
  )
  if (!d) return <>{header}<p className="text-sm text-muted-foreground">{snap.example ? "示例任务没有录制接入过程。" : "草稿还没生成。"}</p></>
  const row = (k: string, label: string, value: string | null) => (
    <div key={k} className={cn("flex justify-between gap-4 border-b py-1.5 text-sm", changed.has(k) && "bg-amber-50 dark:bg-amber-950/30")}>
      <span className="shrink-0 text-muted-foreground">{label}</span>
      <span className="text-right">
        <span className="break-all">{value ?? "—"}</span>{" "}
        <Badge variant="outline">{d.confirmed.includes(k) ? "已确认" : "待确认"}</Badge>
        {changed.has(k) && <Badge className="ml-1">已修改</Badge>}
      </span>
    </div>
  )
  const byGroup = (g: FieldRow["group"]) => d.fields.filter((f) => f.group === g)
  return (
    <div className="space-y-4">
      {header}
      <section>
        <h3 className="mb-1 text-sm font-semibold">草稿配置 · 第 {snap.intake!.round} 版</h3>
        {row("label", "标签", d.label)}
        {row("observation_time_expr", "预测时点", d.observation_time_expr)}
        {row("metric", "指标", d.metric)}
        <div className={cn("mt-2 text-sm", changed.has("windows") && "bg-amber-50 dark:bg-amber-950/30")}>
          时间窗口 <Badge variant="outline">{d.confirmed.includes("windows") ? "已确认" : "待确认"}</Badge>
          {d.windows && (
            <div className="mt-1 flex h-7 overflow-hidden rounded text-[11px] text-white">
              {WIN.map(([k, name, color]) => (
                <div key={k} style={{ flex: months(...d.windows![k]) }} className={cn("min-w-0 truncate px-1 text-center leading-7", color)} title={`${name} ${d.windows![k][0]}~${d.windows![k][1]}`}>
                  {name} {d.windows![k][0]}~{d.windows![k][1]}
                </div>
              ))}
            </div>
          )}
        </div>
      </section>
      <section>
        <h3 className="mb-1 text-sm font-semibold">字段（{d.fields.length}）</h3>
        <div className="mb-2 flex flex-wrap gap-1">
          {(Object.keys(GROUPS) as FieldRow["group"][]).map((g) => <span key={g} className={cn("rounded px-2 py-0.5 text-xs", GROUPS[g][1])}>{GROUPS[g][0]} {byGroup(g).length}</span>)}
        </div>
        {(Object.keys(GROUPS) as FieldRow["group"][]).map((g) => byGroup(g).map((f) => (
          <div key={f.name} className={cn("flex gap-3 border-b py-1 text-sm", changed.has(`field:${f.name}`) && "bg-amber-50 dark:bg-amber-950/30")}>
            <span className="w-44 shrink-0 font-mono text-xs">{f.name}</span>
            <span className={cn("h-fit shrink-0 rounded px-1.5 text-xs", GROUPS[g][1])}>{GROUPS[g][0]}</span>
            <span className="text-muted-foreground">{f.type}{f.reason && ` · ${f.reason}`}</span>
          </div>
        )))}
      </section>
      {Object.keys(snap.intake!.yaml).length > 0 && (
        <section>
          <h3 className="mb-1 text-sm font-semibold">写出的配置</h3>
          {Object.entries(snap.intake!.yaml).map(([n, t]) => (
            <details key={n} className="text-sm"><summary className="cursor-pointer">{n}</summary>
              <pre className="overflow-auto rounded bg-muted p-2 text-xs">{t}</pre></details>
          ))}
        </section>
      )}
    </div>
  )
}
