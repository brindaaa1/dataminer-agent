import { useEffect, useRef, useState } from "react"
import type { Snapshot, Stage } from "@/api"
import { Bubble } from "@/components/Bubble"
import { ChatIntake } from "@/components/ChatIntake"
import { ChatReport } from "@/components/ChatReport"
import { ChatRun } from "@/components/ChatRun"
import { Composer } from "@/components/Composer"
import { StageBar } from "@/components/StageBar"
import { Textarea } from "@/components/ui/textarea"

function Header({ stage, title }: { stage: Stage; title?: string }) {
  return (
    <div className="space-y-1 border-b p-3">
      <StageBar stage={stage} />
      {title && <div className="truncate text-xs text-muted-foreground">{title}</div>}
    </div>
  )
}

function Footer({ snap, extra, setExtra }: { snap: Snapshot; extra: string; setExtra: (s: string) => void }) {
  const asking = snap.status === "intake" && !snap.busy && (snap.intake?.questions.length ?? 0) > 0
  const hint = asking ? "也可以直接打字补充，随本轮一起提交…" : snap.status === "running" ? "运行中暂不支持插话"
    : snap.stage === "report" ? "对报告追问（第二版）" : "等待中…"
  return (
    <div className="border-t p-2">
      <Textarea rows={2} value={asking ? extra : ""} onChange={(e) => setExtra(e.target.value)} disabled={!asking} placeholder={hint} />
    </div>
  )
}

export function Chat({ snap, onCreated }: { snap: Snapshot | null; onCreated: (id: string) => void }) {
  const [extra, setExtra] = useState("")
  const end = useRef<HTMLDivElement>(null)
  const tick = snap ? `${snap.id}:${snap.last_event_id}:${snap.status}:${snap.busy}` : ""
  useEffect(() => { end.current?.scrollIntoView({ block: "end" }) }, [tick])      // 花括号：新版 Chrome 的 scrollIntoView 返回 Promise，不能当清理函数返回
  if (!snap) {
    return (
      <div className="flex h-full flex-col">
        <Header stage="intake" />
        <div className="flex-1 overflow-auto p-3"><Bubble>上传 CSV 和业务说明，我来起草建模配置。</Bubble></div>
        <Composer onCreated={onCreated} />
      </div>
    )
  }
  return (
    <div className="flex h-full flex-col">
      <Header stage={snap.stage} title={snap.title} />
      <div className="flex-1 space-y-2 overflow-auto p-3">
        <ChatIntake snap={snap} extra={extra} onSent={() => setExtra("")} />
        <ChatRun snap={snap} />
        <ChatReport snap={snap} />
        <div ref={end} />
      </div>
      <Footer snap={snap} extra={extra} setExtra={setExtra} />
    </div>
  )
}
