import { useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { Download } from "lucide-react"
import Markdown from "react-markdown"
import remarkGfm from "remark-gfm"
import { api, type Final, type Round, type Snapshot } from "@/api"
import { buttonVariants } from "@/components/ui/button"
import { f3, METRIC, VERDICT } from "@/lib/format"
import { cn } from "@/lib/utils"
import { useDev } from "@/useDev"

const SEG: [RegExp, string][] = [[/^\| train \|/, "| 训练集 |"], [/^\| valid \|/, "| 验证集 |"], [/^\| OOT-dev \|/, "| 时间外验证 |"], [/^\| holdout \|/, "| 最终检验 |"]]
const pick = (secs: string[], title: string) => secs.find((s) => s.replace(/^\d+\. /, "").startsWith(title)) ?? ""
const body = (sec: string) => sec.slice(sec.indexOf("\n") + 1)

/** 每一步：做了什么、为什么（LLM 给业务人员的说明）、结果。 */
function steps(rounds: Round[], L: string): string {
  return rounds.filter((r) => !r.running && r.action !== "STOP").map((r) => {
    const v = r.verdict === "REJECT" ? "没通过" : (VERDICT[r.verdict ?? ""]?.label ?? "")
    const perf = r.metrics.oot_dev != null ? `时间外验证 ${L} ${f3(r.metrics.oot_dev)}${r.metrics.train != null ? `（训练集 ${f3(r.metrics.train)}）` : ""}` : ""
    return `- **${r.title} · ${r.action_user}**${v ? `（${v}）` : ""}：${r.user_note ? r.user_note.replace(/[。.]$/, "") + "。" : ""}${perf}`
  }).join("\n")
}

/** 用户视图的模型卡：执行摘要、建模过程、效果、主要特征（前 10 个），按顺序编号；风险只在上方的提示框里出现一次。完整版在开发者视图。 */
export function userCard(md: string, rounds: Round[], L: string): string {
  if (!md) return ""
  const [head, ...secs] = md.split(/^## /m)
  const metrics = body(pick(secs, "各段指标")).split("\n").filter((l) => !l.startsWith("- valid − OOT-dev gap"))
    .map((l) => SEG.reduce((x, [re, to]) => x.replace(re, to), l)).join("\n")
  const feat = body(pick(secs, "特征清单")).split("\n")
  const rows = feat.filter((l) => l.startsWith("| ") && !l.startsWith("| 特征"))
  const features = feat.filter((l) => !l.startsWith("| ") || l.startsWith("| 特征") || rows.slice(0, 10).includes(l)).join("\n")
  return [head.split("\n").filter((l) => !l.startsWith(">")).join("\n").trim(), "",
    "## 执行摘要", body(pick(secs, "执行摘要")).trim(), "",
    "## 1. 建模过程", steps(rounds, L), "",
    "## 2. 效果", metrics.trim(), "",
    "## 3. 主要特征", features.trim(), "",
    "## 4. 模型设定（复现用）", body(pick(secs, "模型设定")).trim(), ""].join("\n")
}

function save(text: string, name: string) {
  const a = document.createElement("a")
  a.href = URL.createObjectURL(new Blob([text], { type: "text/markdown" }))
  a.download = name
  a.click()
  URL.revokeObjectURL(a.href)
}

function Conclusion({ f, metric }: { f: Final; metric: string }) {
  const dev = useDev()
  const L = METRIC[metric] ?? metric
  const kpis: [string, string, string, boolean][] = dev ? [
    ["最终模型", f.model ?? "—", f.exp_id, false], [`OOT-dev ${L}`, f3(f.oot_dev), "", false],
    [`holdout ${L}`, f3(f.holdout), `KS ${f3(f.holdout_ks)}`, f.overfit], ["实验", String(f.n_experiments), `采纳 ${f.n_accepted}`, false],
  ] : [
    ["最终模型", f.model ?? "—", "", false], [`最终检验 ${L}`, f3(f.holdout), "", f.overfit],
    ["头部 10% 提升度", f.lift_top10 == null ? "—" : `${f.lift_top10.toFixed(2)} 倍`, "分数最高的 10% 里正类浓度是整体的几倍", false],
  ]
  return (
    <div className="space-y-2">
      <div className={cn("grid gap-2", dev ? "grid-cols-4" : "grid-cols-3")}>
        {kpis.map(([k, v, sub, bad]) => (
          <div key={k} className="rounded-md border p-2">
            <div className="text-xs text-muted-foreground">{k}</div>
            <div className={cn("text-lg font-semibold tabular-nums", bad && "text-red-600")}>{v}</div>
            <div className="text-xs text-muted-foreground">{sub}</div>
          </div>
        ))}
      </div>
      {(dev ? f.risks : f.risks_user).map((r, i) => <div key={i} className="border-l-4 border-l-red-600 bg-red-50 px-3 py-1.5 text-sm dark:bg-red-950/30">{r}</div>)}
    </div>
  )
}

export function OutputTab({ snap }: { snap: Snapshot }) {
  const dev = useDev()
  const files = dev ? snap.artifacts : snap.artifacts.filter((a) => a.name === "model_card.md")
  const names = files.map((a) => a.name)
  const [sel, setSel] = useState<string | null>(null)
  const cur = sel && names.includes(sel) ? sel : names.includes("model_card.md") ? "model_card.md" : names[0]
  const text = useQuery({ queryKey: ["artifact", snap.id, cur, snap.last_event_id], queryFn: () => api.artifact(snap.id, cur!), enabled: !!cur })
  const canDownload = snap.example || snap.status === "accepted"
  const shown = dev || cur !== "model_card.md" ? text.data ?? "" : userCard(text.data ?? "", snap.run?.rounds ?? [], METRIC[snap.run?.metric ?? ""] ?? "")
  const isCard = cur === "model_card.md"
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        {files.length > 1 && files.map((a) => (
          <button key={a.name} onClick={() => setSel(a.name)}
            className={cn("rounded-full border px-3 py-1 text-xs", a.name === cur ? "border-primary bg-primary text-primary-foreground" : "hover:bg-accent")}>{a.label}</button>
        ))}
        <div className="ml-auto flex flex-wrap items-center gap-2">
          {snap.langfuse_url
            ? <a href={snap.langfuse_url} target="_blank" rel="noreferrer" className={buttonVariants({ size: "sm", variant: "outline" })}>查看详细分析过程（Langfuse）↗</a>
            : <span className={cn(buttonVariants({ size: "sm", variant: "outline" }), "cursor-not-allowed opacity-50")}
                title="在 .env 配置 Langfuse 后，可以在这里查看每一步的完整过程">查看详细分析过程（Langfuse）</span>}
          {isCard && (
            <button className={buttonVariants({ size: "sm" })} title="下载页面上的这份报告（Markdown）"
              onClick={() => save(shown, `report_${snap.id}.md`)}><Download className="size-4" />下载报告</button>
          )}
          {dev && (canDownload
            ? <a href={`/api/tasks/${snap.id}/bundle.zip`} download className={buttonVariants({ size: "sm", variant: "outline" })} title="配置、事件、实验树、模型卡等全部产出"><Download className="size-4" />打包下载全部产出</a>
            : <span className={cn(buttonVariants({ size: "sm", variant: "outline" }), "pointer-events-none opacity-50")} title="确认报告后可下载"><Download className="size-4" />打包下载全部产出</span>)}
        </div>
      </div>
      {snap.run?.final && <Conclusion f={snap.run.final} metric={snap.run.metric} />}
      {cur && (cur.endsWith(".md")
        ? <article className="prose prose-sm max-w-none dark:prose-invert"><Markdown remarkPlugins={[remarkGfm]}>{shown}</Markdown></article>
        : <pre className="overflow-auto rounded bg-muted p-3 text-xs">{text.data}</pre>)}
      {!cur && <p className="text-sm text-muted-foreground">还没有产出。</p>}
    </div>
  )
}
