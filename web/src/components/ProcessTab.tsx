import { Fragment, useState, type ReactElement } from "react"
import { Loader2 } from "lucide-react"
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"
import type { Round, Snapshot, Trial, TreeNode } from "@/api"
import { Badge } from "@/components/ui/badge"
import { Button, buttonVariants } from "@/components/ui/button"
import { CODES, f3, METRIC, VERDICT } from "@/lib/format"
import { cn } from "@/lib/utils"

function VerdictBadge({ v }: { v: string | null }) {
  if (!v) return null
  const x = VERDICT[v] ?? VERDICT.NONE
  return <span className={cn("rounded px-1.5 text-xs", x.cls)}>{x.label}</span>
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function VerdictDot(props: any) {
  const { cx, cy, payload } = props
  return <circle cx={cx} cy={cy} r={4} fill={VERDICT[payload.verdict]?.color ?? "#888"} />
}

function TrialLine({ trials }: { trials: Record<string, Trial[]> }) {
  const all = Object.values(trials).flat()
  if (!all.length) return null
  const done = all.filter((t) => t.state === "COMPLETE" && t.value != null)
  const best = Math.max(...done.map((t) => t.value!))
  return (
    <p className="text-xs text-muted-foreground">
      trial：完成 {done.length} · 剪枝 {all.filter((t) => t.state === "PRUNED").length}{done.length > 0 && ` · 验证集最好 ${best.toFixed(4)}`}
    </p>
  )
}

/** 代码按规则做的决策（赛跑打平后第一次升全量），不是 LLM 的。 */
function RuleBadge() {
  return <span className="rounded border px-1 text-xs text-muted-foreground" title="代码按规则做的决策，没有调 LLM">规则</span>
}

function Detail({ r, L, langfuse, onTrace }: { r: Round; L: string; langfuse: string | null; onTrace: (k: string) => void }) {
  const m: [string, number | null][] = [[`训练 ${L}`, r.metrics.train], [`验证 ${L}`, r.metrics.valid], [`时间外 ${L}`, r.metrics.oot_dev],
    ["时间外 KS", r.metrics.oot_dev_ks], ["gap", r.metrics.gap]]
  return (
    <div className="space-y-3 rounded-md border p-3 text-sm">
      <div className="flex items-center gap-2">
        <b>{r.title} · {r.action_label}</b>{r.by_rule && <RuleBadge />}<VerdictBadge v={r.verdict} />
        {r.running && <Loader2 className="size-3 animate-spin" />}
      </div>
      {(r.hypothesis || r.stop_reason) && (
        <section>
          <h4 className="text-xs text-muted-foreground">{r.by_rule ? "决策（规则）" : "决策（LLM）"}</h4>
          <p>{r.stop_reason ?? r.hypothesis}</p>
          {r.alternatives.length > 0 && (
            <ul className="mt-1 list-disc pl-5 text-muted-foreground">{r.alternatives.map((a, i) => <li key={i}>考虑过：{a}</li>)}</ul>
          )}
        </section>
      )}
      <section>
        <h4 className="text-xs text-muted-foreground">评估（代码）</h4>
        <dl className="grid grid-cols-[auto_1fr] gap-x-6">
          {m.map(([k, v]) => <Fragment key={k}><dt className="text-muted-foreground">{k}</dt><dd className="text-right tabular-nums">{f3(v)}</dd></Fragment>)}
        </dl>
        {r.codes.length > 0 && <div className="mt-1 flex flex-wrap gap-1">{r.codes.map((c) => <Badge key={c} variant="outline" title={CODES[c] ?? c}>{c}</Badge>)}</div>}
        {r.evaluations.length > 1 && (
          <table className="mt-2 w-full text-xs">
            <thead className="text-muted-foreground"><tr><th className="text-left">模型</th><th className="text-left">保真度</th><th className="text-left">结论</th><th className="text-right">时间外 {L}</th></tr></thead>
            <tbody>{r.evaluations.map((e) => (
              <tr key={e.exp_id}><td>{e.model}</td><td>{e.fidelity}</td><td><VerdictBadge v={e.verdict} /></td><td className="text-right tabular-nums">{f3(e.oot_dev)}</td></tr>
            ))}</tbody>
          </table>
        )}
        <TrialLine trials={r.trials} />
      </section>
      <div className="flex gap-2">
        <Button size="sm" variant="outline" onClick={() => onTrace(r.key)}>看 LLM 原文</Button>
        {langfuse && <a className={buttonVariants({ size: "sm", variant: "ghost" })} href={langfuse} target="_blank" rel="noreferrer">Langfuse ↗</a>}
      </div>
    </div>
  )
}

function Tree({ nodes }: { nodes: TreeNode[] }) {
  const ids = new Set(nodes.map((n) => n.exp_id))
  const kids = new Map<string, TreeNode[]>()
  nodes.forEach((n) => { if (n.parent && ids.has(n.parent)) kids.set(n.parent, [...(kids.get(n.parent) ?? []), n]) })
  const walk = (n: TreeNode, depth: number): ReactElement[] => [
    <div key={n.exp_id} style={{ paddingLeft: depth * 18 }} className="flex items-center gap-2 py-0.5 text-sm">
      <span className="font-mono text-xs text-muted-foreground">{n.exp_id.split("_").pop()}</span>{n.label}<VerdictBadge v={n.verdict} />
    </div>,
    ...(kids.get(n.exp_id) ?? []).flatMap((k) => walk(k, depth + 1)),
  ]
  return <div>{nodes.filter((n) => !n.parent || !ids.has(n.parent)).flatMap((n) => walk(n, 0))}</div>
}

export function ProcessTab({ snap, onTrace }: { snap: Snapshot; onTrace: (round: string) => void }) {
  const r = snap.run
  const [sel, setSel] = useState<string | null>(null)
  const [view, setView] = useState<"list" | "tree">("list")
  if (!r) return <p className="text-sm text-muted-foreground">建模还没开始。</p>
  const L = METRIC[r.metric] ?? r.metric
  const cur = r.rounds.find((x) => x.key === sel) ?? r.rounds[r.rounds.length - 1]
  return (
    <div className="space-y-3">
      {r.chart.length > 0 && (
        <div className="h-40">
          <ResponsiveContainer>
            <LineChart data={r.chart} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
              <XAxis dataKey="label" fontSize={11} />
              <YAxis domain={["auto", "auto"]} fontSize={11} width={48} tickFormatter={(v: number) => v.toFixed(3)} />
              <Tooltip formatter={(v) => Number(v).toFixed(4)} />
              <Line dataKey="value" name={`时间外 ${L}`} stroke="var(--primary)" dot={<VerdictDot />} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
      <div className="flex gap-1">
        <Button size="sm" variant={view === "list" ? "default" : "outline"} onClick={() => setView("list")}>轮次</Button>
        <Button size="sm" variant={view === "tree" ? "default" : "outline"} onClick={() => setView("tree")}>实验树</Button>
      </div>
      {view === "tree" ? <Tree nodes={r.tree} /> : (
        <div className="grid grid-cols-[minmax(0,2fr)_minmax(0,3fr)] gap-3">
          <ul className="space-y-1">
            {r.rounds.map((x) => (
              <li key={x.key}>
                <button onClick={() => setSel(x.key)}
                  className={cn("w-full rounded-md border p-2 text-left text-sm", cur?.key === x.key && "border-primary ring-1 ring-primary")}>
                  <div className="flex items-center gap-2">{x.title} · {x.action_label}{x.by_rule && <RuleBadge />}<VerdictBadge v={x.verdict} />{x.running && <Loader2 className="size-3 animate-spin" />}</div>
                  <div className="text-xs text-muted-foreground">
                    {x.metrics.oot_dev != null && `时间外 ${L} ${f3(x.metrics.oot_dev)}`}{x.metrics.gap != null && ` · gap ${f3(x.metrics.gap)}`}
                  </div>
                </button>
              </li>
            ))}
          </ul>
          {cur && <Detail r={cur} L={L} langfuse={snap.langfuse_url} onTrace={onTrace} />}
        </div>
      )}
    </div>
  )
}
