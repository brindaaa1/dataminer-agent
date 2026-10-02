import type { ReactNode } from "react"
import type { Tone } from "@/api"
import { TONE } from "@/lib/format"
import { cn } from "@/lib/utils"

export function Bubble({ from = "agent", tone, children }: { from?: "agent" | "user" | "system"; tone?: Tone; children: ReactNode }) {
  return (
    <div className={cn("rounded-lg px-3 py-2 text-sm leading-relaxed",
      from === "agent" && "bg-blue-50 dark:bg-blue-950/40",
      from === "user" && "ml-8 bg-emerald-50 dark:bg-emerald-950/40",
      from === "system" && "bg-muted text-xs text-muted-foreground",
      tone && cn("border-l-4", TONE[tone]))}>
      {children}
    </div>
  )
}
