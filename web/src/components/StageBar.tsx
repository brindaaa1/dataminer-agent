import type { Stage } from "@/api"
import { cn } from "@/lib/utils"

const STEPS: [Stage, string][] = [["intake", "接入"], ["confirm", "确认"], ["modeling", "建模"], ["report", "报告"]]

/** 进度条：只表示走到哪一步，不可点。 */
export function StageBar({ stage }: { stage: Stage }) {
  const cur = STEPS.findIndex(([s]) => s === stage)
  return (
    <div aria-label="进度">
      <div className="h-1.5 overflow-hidden rounded-full bg-muted">
        <div className="h-full rounded-full bg-primary transition-[width]" style={{ width: `${((cur + 1) / STEPS.length) * 100}%` }} />
      </div>
      <ol className="mt-1 grid grid-cols-4 text-[11px] text-muted-foreground">
        {STEPS.map(([s, label], i) => (
          <li key={s} aria-current={i === cur ? "step" : undefined} className={cn("text-center", i === cur && "font-semibold text-foreground")}>{label}</li>
        ))}
      </ol>
    </div>
  )
}
