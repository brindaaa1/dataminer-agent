import { useEffect, useState } from "react"
import type { Snapshot, Stage } from "@/api"
import { DataTab } from "@/components/DataTab"
import { EvalTab } from "@/components/EvalTab"
import { OutputTab } from "@/components/OutputTab"
import { ProcessTab } from "@/components/ProcessTab"
import { TraceTab } from "@/components/TraceTab"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

type Tab = "data" | "process" | "trace" | "output" | "eval"
const DEFAULT: Record<Stage, Tab> = { intake: "data", confirm: "data", modeling: "process", report: "output" }
const first = (s: Snapshot): Tab => (s.eval_label ? "eval" : DEFAULT[s.stage])         // 评测运行默认打开「评测」

export function Workspace({ snap, onSelect }: { snap: Snapshot; onSelect: (id: string) => void }) {
  const [tab, setTab] = useState<Tab>(first(snap))
  const [round, setRound] = useState("all")
  useEffect(() => { setTab(first(snap)); setRound("all") }, [snap.id, snap.stage])
  const trace = (r: string) => { setRound(r); setTab("trace") }
  const pane = "min-h-0 flex-1 overflow-auto px-4 pb-4"
  return (
    <Tabs value={tab} onValueChange={(v) => setTab(v as Tab)} className="flex h-full flex-col">
      <TabsList className="m-2">
        <TabsTrigger value="data">数据</TabsTrigger>
        <TabsTrigger value="process">过程</TabsTrigger>
        <TabsTrigger value="trace">Trace</TabsTrigger>
        <TabsTrigger value="output">产出</TabsTrigger>
        <TabsTrigger value="eval">评测</TabsTrigger>
      </TabsList>
      <TabsContent value="data" className={pane}><DataTab snap={snap} /></TabsContent>
      <TabsContent value="process" className={pane}><ProcessTab snap={snap} onTrace={trace} /></TabsContent>
      <TabsContent value="trace" className={pane}><TraceTab snap={snap} round={round} setRound={setRound} /></TabsContent>
      <TabsContent value="output" className={pane}><OutputTab snap={snap} /></TabsContent>
      <TabsContent value="eval" className={pane}><EvalTab snap={snap} onTrace={trace} onSelect={onSelect} /></TabsContent>
    </Tabs>
  )
}
