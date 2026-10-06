import { useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { api, type ProcessRow, type Snapshot } from "@/api"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

const RATE = new Set(["wasted_rounds", "upgrade_hit_rate", "upgrade_success_rate", "recovery"])
const VERDICT = { better: ["↑ 变好", "text-green-700"], worse: ["↓ 变坏", "text-red-600"], same: ["无显著变化", "text-muted-foreground"] } as const
const roundName = (k: string) => (k === "setup" ? "赛跑" : k === "finale" ? "收尾" : k === "intake" ? "接入" : k.toUpperCase())

function fmt(r: ProcessRow): string {
  const v = r.value
  if (v === null || v === undefined) return "—"
  if (typeof v === "boolean") return v ? "是" : "否"
  if (typeof v === "number") return RATE.has(r.key) ? `${Math.round(v * 100)}%` : r.key === "cost" ? v.toFixed(4) : String(v)
  return v
}

function Process({ rows, onTrace }: { rows: ProcessRow[]; onTrace: (round: string) => void }) {
  return (
    <table className="w-full text-sm">
      <tbody className="divide-y">
        {rows.map((r) => (
          <tr key={r.key}>
            <td className="py-1.5 pr-3 text-muted-foreground">{r.label}</td>
            <td className="py-1.5 pr-3 tabular-nums">{fmt(r)}</td>
            <td className="flex flex-wrap gap-1 py-1.5">
              {r.evidence.filter((e) => e.round).map((e) => (
                <Button key={e.ref} size="sm" variant="outline" className="h-6 px-2 text-xs" onClick={() => onTrace(e.round!)}>
                  {`${roundName(e.round!)} · ${e.ref}`}
                </Button>
              ))}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

export function EvalTab({ snap, onTrace, onSelect }: { snap: Snapshot; onTrace: (round: string) => void; onSelect: (id: string) => void }) {
  const q = useQuery({ queryKey: ["eval", snap.id, snap.last_event_id], queryFn: () => api.evalView(snap.id) })
  const [view, setView] = useState<"run" | "version" | "scenarios">("run")
  const d = q.data
  if (!d) return <p className="text-sm text-muted-foreground">加载中…</p>
  const views: [typeof view, string][] = [["run", "本次运行"], ...(d.version ? [["version", "本版本"], ["scenarios", "情景用例"]] as [typeof view, string][] : [])]
  const v = d.version
  return (
    <div className="space-y-3">
      {views.length > 1 && (
        <div className="flex gap-2">{views.map(([k, l]) => <Button key={k} size="sm" variant={view === k ? "default" : "outline"} onClick={() => setView(k)}>{l}</Button>)}</div>
      )}
      {view === "run" && <Process rows={d.process} onTrace={onTrace} />}
      {view === "version" && v && (
        <div className="space-y-3 text-sm">
          <p><b>{v.conclusion}</b>{v.against && `（对比 ${v.against}，${v.tier} 档${v.noise_calibrated ? "" : "，噪声带未校准"}）`}</p>
          {v.table.length > 0 && (
            <table className="w-full"><thead className="text-left text-muted-foreground">
              <tr><th>数据集</th><th>指标</th><th>新</th><th>旧</th><th>差</th><th>判定</th></tr></thead>
              <tbody className="divide-y">{v.table.map((t) => (
                <tr key={t.dataset + t.metric} className={cn(t.task && "cursor-pointer hover:bg-accent")} onClick={() => t.task && onSelect(t.task)}>
                  <td>{t.dataset}</td><td>{t.metric}</td><td>{t.new}</td><td>{t.old}</td><td>{t.delta > 0 ? `+${t.delta}` : t.delta}</td>
                  <td className={VERDICT[t.verdict][1]}>{VERDICT[t.verdict][0]}</td>
                </tr>))}</tbody></table>
          )}
          {v.flags.length > 0 && (
            <ul className="list-disc pl-5">{v.flags.map((f, i) => (
              <li key={i}>{f.kind}：{f.detail}{f.task && <button className="ml-2 text-primary" onClick={() => onSelect(f.task!)}>看这次运行</button>}</li>))}</ul>
          )}
          <table className="w-full"><thead className="text-left text-muted-foreground"><tr><th>运行</th><th>状态</th><th>holdout</th><th>比 B1</th></tr></thead>
            <tbody className="divide-y">{v.runs.map((r) => (
              <tr key={r.task} className={cn("cursor-pointer hover:bg-accent", r.task === snap.id && "bg-accent")} onClick={() => onSelect(r.task)}>
                <td>{r.dataset} · 种子 {r.seed}</td><td>{r.status}</td><td>{r.holdout?.toFixed(3) ?? "—"}</td>
                <td>{r.delta_b1 === null ? "—" : `${r.delta_b1 >= 0 ? "+" : ""}${r.delta_b1.toFixed(3)}`}</td>
              </tr>))}</tbody></table>
        </div>
      )}
      {view === "scenarios" && d.scenarios && (
        <table className="w-full text-sm"><thead className="text-left text-muted-foreground"><tr><th className="pr-2">情景</th><th className="pr-2">底座</th><th className="whitespace-nowrap pr-2">结果</th><th>原因</th><th /></tr></thead>
          <tbody className="divide-y">{d.scenarios.map((s) => (
            <tr key={s.task}>
              <td className="pr-2">{s.scenario}</td><td className="pr-2">{s.base}</td>
              <td className={cn("whitespace-nowrap pr-2", s.passed ? "text-green-700" : "text-red-600")}>{s.passed ? "通过" : "不通过"}</td><td>{s.reason}</td>
              <td>{!s.passed && <Button size="sm" variant="outline" onClick={() => onSelect(s.task)}>看这次运行</Button>}</td>
            </tr>))}</tbody></table>
      )}
    </div>
  )
}
