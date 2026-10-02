import type { Stage } from "@/api"
import { cn } from "@/lib/utils"

const STEPS: [Stage, string][] = [["intake", "接入"], ["confirm", "确认"], ["modeling", "建模"], ["report", "报告"]]

export function StageBar({ stage }: { stage: Stage }) {
  const cur = STEPS.findIndex(([s]) => s === stage)
  return (
    <ol className="flex gap-1 text-xs" aria-label="阶段">
      {STEPS.map(([s, label], i) => (
        <li key={s} aria-current={i === cur ? "step" : undefined}
          className={cn("flex-1 rounded px-2 py-1 text-center",
            i < cur && "bg-primary/20", i === cur && "bg-primary text-primary-foreground", i > cur && "bg-muted text-muted-foreground")}>
          {label}
        </li>
      ))}
    </ol>
  )
}
