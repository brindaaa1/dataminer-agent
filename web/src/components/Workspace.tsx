import { useEffect, useState } from "react"
import type { Snapshot, Stage } from "@/api"
import { DataTab } from "@/components/DataTab"
import { OutputTab } from "@/components/OutputTab"
import { ProcessTab } from "@/components/ProcessTab"
import { TraceTab } from "@/components/TraceTab"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

type Tab = "data" | "process" | "trace" | "output"
const DEFAULT: Record<Stage, Tab> = { intake: "data", confirm: "data", modeling: "process", report: "output" }

export function Workspace({ snap }: { snap: Snapshot }) {
  const [tab, setTab] = useState<Tab>(DEFAULT[snap.stage])
  const [round, setRound] = useState("all")
  useEffect(() => { setTab(DEFAULT[snap.stage]); setRound("all") }, [snap.id, snap.stage])
  const pane = "min-h-0 flex-1 overflow-auto px-4 pb-4"
  return (
    <Tabs value={tab} onValueChange={(v) => setTab(v as Tab)} className="flex h-full flex-col">
      <TabsList className="m-2">
        <TabsTrigger value="data">数据</TabsTrigger>
        <TabsTrigger value="process">过程</TabsTrigger>
        <TabsTrigger value="trace">Trace</TabsTrigger>
        <TabsTrigger value="output">产出</TabsTrigger>
      </TabsList>
      <TabsContent value="data" className={pane}><DataTab snap={snap} /></TabsContent>
      <TabsContent value="process" className={pane}><ProcessTab snap={snap} onTrace={(r) => { setRound(r); setTab("trace") }} /></TabsContent>
      <TabsContent value="trace" className={pane}><TraceTab snap={snap} round={round} setRound={setRound} /></TabsContent>
      <TabsContent value="output" className={pane}><OutputTab snap={snap} /></TabsContent>
    </Tabs>
  )
}
