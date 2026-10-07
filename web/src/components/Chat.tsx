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

/** 每个对话开头 agent 先发的使用说明：适合什么数据、怎么建模、最后给什么。fresh：新建任务，提示怎么开始。 */
function Intro({ fresh = false }: { fresh?: boolean }) {
  return (
    <Bubble>
      <p>你好，我可以帮你把一张表做成一个能用的预测模型。</p>
      <p className="mt-2 font-medium">适合的数据</p>
      <ul className="list-disc pl-5">
        <li>一份 CSV，一行是一个要预测的对象（一笔订单、一个用户、一个时段……）</li>
        <li>有一列是二分类结果，比如是否取消、是否违约、是涨还是跌</li>
        <li>最好有一个时间列：我会拿最近的一段时间单独检验，看模型放到未来还稳不稳</li>
      </ul>
      <p className="mt-2 font-medium">我会怎么做</p>
      <ul className="list-disc pl-5">
        <li>先读数据、起草配置（标签怎么定、哪些字段在预测时还拿不到），关键的地方请你确认</li>
        <li>用 lgbm、catboost 等模型先小样本比一比，再一轮轮加特征、调参，每一步都告诉你为什么</li>
        <li>全程检查信息泄漏和过拟合，结果由代码按统计检验判定，不由我自己说了算</li>
      </ul>
      <p className="mt-2">最后会给你一份报告（效果、主要影响因素、风险提示）和可以下载的模型。</p>
      {fresh && <p className="mt-2 text-muted-foreground">在下面上传 CSV，写几句业务说明（要预测什么、结果怎么定义），就可以开始了。</p>}
    </Bubble>
  )
}

function Footer({ snap, extra, setExtra }: { snap: Snapshot; extra: string; setExtra: (s: string) => void }) {
  const asking = snap.status === "intake" && !snap.busy && (snap.intake?.questions.length ?? 0) > 0
  const hint = asking ? "也可以直接打字补充，随本轮一起提交…" : snap.status === "running" ? "运行中暂不支持插话" : "等待中…"
  if (snap.stage === "report") return null          // 对报告追问还没做：建模完成后不出输入框
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
        <div className="flex-1 space-y-3 overflow-auto p-3"><Intro fresh /></div>
        <Composer onCreated={onCreated} />
      </div>
    )
  }
  return (
    <div className="flex h-full flex-col">
      <Header stage={snap.stage} title={snap.title} />
      <div className="flex-1 space-y-2 overflow-auto p-3">
        <Intro />
        <ChatIntake snap={snap} extra={extra} onSent={() => setExtra("")} />
        <ChatRun snap={snap} />
        <ChatReport snap={snap} />
        <div ref={end} />
      </div>
      <Footer snap={snap} extra={extra} setExtra={setExtra} />
    </div>
  )
}
