import type { ReactNode } from "react"
import type { Tone } from "@/api"
import { cn } from "@/lib/utils"

const DOT: Partial<Record<Tone, string>> = { ok: "bg-green-600", bad: "bg-red-600", warn: "bg-amber-500" }

/** 对话消息：用户消息靠右、浅底圆角；agent 消息是正文，状态只用一个小圆点；系统提示是小号灰字。 */
export function Bubble({ from = "agent", tone, children }: { from?: "agent" | "user" | "system"; tone?: Tone; children: ReactNode }) {
  if (from === "user") return <div className="ml-auto max-w-[85%] rounded-2xl bg-muted px-3.5 py-2 text-sm leading-relaxed">{children}</div>
  if (from === "system") return <div className="text-xs text-muted-foreground">{children}</div>
  return (
    <div className="flex gap-2 text-sm leading-relaxed">
      <span className={cn("mt-[0.55em] size-1.5 shrink-0 rounded-full", (tone && DOT[tone]) ?? "bg-zinc-300 dark:bg-zinc-600")} />
      <div className="min-w-0 flex-1">{children}</div>
    </div>
  )
}
