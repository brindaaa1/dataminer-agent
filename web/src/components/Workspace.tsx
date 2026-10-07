import { useEffect, useState } from "react"
import type { Snapshot, Stage } from "@/api"
import { DataTab } from "@/components/DataTab"
import { EvalTab } from "@/components/EvalTab"
import { OutputTab } from "@/components/OutputTab"
import { ProcessTab } from "@/components/ProcessTab"
import { TraceTab } from "@/components/TraceTab"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { setDev, useDev } from "@/useDev"

type Tab = "process" | "trace" | "output" | "eval"
const DEFAULT: Record<Stage, Tab> = { intake: "process", confirm: "process", modeling: "process", report: "output" }
const first = (s: Snapshot): Tab => (s.eval_label ? "eval" : DEFAULT[s.stage])         // 评测运行默认打开「评测」

export function Workspace({ snap, onSelect }: { snap: Snapshot; onSelect: (id: string) => void }) {
  const dev = useDev()
  const [picked, setTab] = useState<Tab>(first(snap))
  const tab = !dev && (picked === "trace" || picked === "eval") ? DEFAULT[snap.stage] : picked      // 用户视图没有这两个页签
  const [round, setRound] = useState("all")
  useEffect(() => { setTab(first(snap)); setRound("all") }, [snap.id, snap.stage])
  const trace = (r: string) => { setRound(r); setTab("trace") }
  const pane = "min-h-0 flex-1 overflow-auto px-4 pb-4"
  return (
    <Tabs value={tab} onValueChange={(v) => setTab(v as Tab)} className="flex h-full flex-col">
      <div className="m-2 flex items-center justify-between">
        <TabsList>
          <TabsTrigger value="process">过程</TabsTrigger>
          {dev && <TabsTrigger value="trace">Trace</TabsTrigger>}
          <TabsTrigger value="output">产出</TabsTrigger>
          {dev && <TabsTrigger value="eval">评测</TabsTrigger>}
        </TabsList>
        <label className="flex items-center gap-1 text-xs text-muted-foreground">
          <input type="checkbox" checked={dev} onChange={(e) => setDev(e.target.checked)} />开发者视图
        </label>
      </div>
      <TabsContent value="process" className={pane}><div className="space-y-6"><DataTab snap={snap} /><ProcessTab snap={snap} onTrace={trace} /></div></TabsContent>
      <TabsContent value="trace" className={pane}><TraceTab snap={snap} round={round} setRound={setRound} /></TabsContent>
      <TabsContent value="output" className={pane}><OutputTab snap={snap} /></TabsContent>
      <TabsContent value="eval" className={pane}><EvalTab snap={snap} onTrace={trace} onSelect={onSelect} /></TabsContent>
    </Tabs>
  )
}
