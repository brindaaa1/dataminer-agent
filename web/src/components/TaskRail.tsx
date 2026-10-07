import { useState } from "react"
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { Copy, Menu, Plus } from "lucide-react"
import { api, type TaskSummary } from "@/api"
import { ErrorText } from "@/components/ErrorText"
import { Button } from "@/components/ui/button"
import { CLONEABLE, DOT } from "@/lib/format"
import { cn } from "@/lib/utils"

function Item({ t, current, open, onSelect, onClone }: { t: TaskSummary; current: string | null; open: boolean;
  onSelect: (id: string) => void; onClone?: () => void }) {
  return (
    <li title={t.title} onClick={() => onSelect(t.id)}
      className={cn("group flex cursor-pointer items-center gap-2 px-4 py-1.5 text-sm", t.id === current && "bg-accent")}>
      <span className={cn("size-2 shrink-0 rounded-full", DOT[t.status])} />
      {open && <span className="flex-1 truncate">{t.title}</span>}
      {open && onClone && (
        <button className="invisible text-muted-foreground group-hover:visible" title="复制为新任务"
          onClick={(e) => { e.stopPropagation(); onClone() }}><Copy className="size-3.5" /></button>
      )}
    </li>
  )
}

export function TaskRail({ tasks, current, onSelect }: { tasks: TaskSummary[]; current: string | null; onSelect: (id: string | null) => void }) {
  const [open, setOpen] = useState(false)
  const qc = useQueryClient()
  const clone = useMutation({
    mutationFn: api.clone,
    onSuccess: (r) => { qc.invalidateQueries({ queryKey: ["tasks"] }); onSelect(r.id) },
  })
  const plain = tasks.filter((t) => !t.eval)
  const versions = new Map<string, TaskSummary[]>()
  for (const t of tasks) if (t.eval) versions.set(t.eval.label, [...(versions.get(t.eval.label) ?? []), t])
  return (
    <nav className={cn("flex shrink-0 flex-col border-r bg-muted/40 transition-[width]", open ? "w-64" : "w-12")}>
      <Button variant="ghost" size="icon" className="m-1" aria-label="任务列表" onClick={() => setOpen(!open)}><Menu className="size-4" /></Button>
      <Button variant="ghost" size={open ? "sm" : "icon"} className="m-1 justify-start" aria-label="新建任务" onClick={() => onSelect(null)}>
        <Plus className="size-4" />{open && "新建任务"}
      </Button>
      <ul className="flex-1 overflow-auto">
        {plain.map((t) => <Item key={t.id} t={t} current={current} open={open} onSelect={onSelect}
          onClone={!t.example && CLONEABLE.includes(t.status) ? () => clone.mutate(t.id) : undefined} />)}
        {open && versions.size > 0 && <li className="px-4 pt-3 pb-1 text-xs text-muted-foreground">评测运行</li>}
        {open && [...versions].map(([label, ts]) => {
          const e = ts[0].eval!
          return (
            <li key={label}>
              <details open={ts.some((t) => t.id === current)}>
                <summary className="cursor-pointer px-4 py-1 text-sm">{[label, e.tier, e.running ? "运行中" : e.conclusion].filter(Boolean).join(" · ")}</summary>
                <ul className="pl-2">{ts.map((t) => <Item key={t.id} t={t} current={current} open={open} onSelect={onSelect} />)}</ul>
              </details>
            </li>
          )
        })}
      </ul>
      {open && <div className="px-3 pb-2"><ErrorText error={clone.error} /></div>}
    </nav>
  )
}
