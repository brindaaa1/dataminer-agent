import { useState } from "react"
import { Check } from "lucide-react"
import type { Question } from "@/api"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"
import { QLABEL } from "@/lib/format"
import { cn } from "@/lib/utils"

export type Answer = { mode: "accept" } | { mode: "edit"; text: string }

export function buildReplies(questions: Question[], ans: Record<string, Answer>, extra: string): Record<string, string> {
  const r: Record<string, string> = Object.fromEntries(questions.map((q) => {
    const a = ans[q.key]
    return [q.key, a.mode === "accept" ? q.recommended : a.text]
  }))
  if (extra.trim()) r["补充说明"] = extra.trim()
  return r
}

export function QuestionCard({ questions, extra, pending, onSubmit }: {
  questions: Question[]; extra: string; pending: boolean; onSubmit: (replies: Record<string, string>) => void
}) {
  const [ans, setAns] = useState<Record<string, Answer>>({})
  const set = (k: string, a: Answer) => setAns({ ...ans, [k]: a })
  const done = questions.filter((q) => { const a = ans[q.key]; return a && (a.mode === "accept" || a.text.trim()) }).length
  return (
    <Card className="gap-2 p-3">
      <p className="text-sm">这一轮有 {questions.length} 处需要你看一下：</p>
      {questions.map((q) => {
        const a = ans[q.key]
        return (
          <div key={q.key} className={cn("rounded-md border bg-background p-2 text-sm", q.required && "border-l-4 border-l-amber-500")}>
            <div className="flex items-center gap-2">
              <b>{QLABEL[q.key] ?? q.key}</b>
              {q.required && <Badge variant="outline" className="text-amber-700">必答</Badge>}
              {a?.mode === "accept" && <Check className="size-4 text-green-600" />}
            </div>
            <p className="text-muted-foreground">{q.question}</p>
            <p className="mt-1">{q.display}</p>
            {q.display !== q.recommended && (
              <details className="text-xs"><summary className="cursor-pointer text-muted-foreground">看原始值</summary>
                <code className="break-all">{q.recommended}</code></details>
            )}
            {a?.mode === "edit"
              ? <Textarea className="mt-1" rows={2} value={a.text} onChange={(e) => set(q.key, { mode: "edit", text: e.target.value })} />
              : (
                <div className="mt-1 flex gap-1">
                  <Button size="sm" variant={a ? "default" : "outline"} onClick={() => set(q.key, { mode: "accept" })}>采纳</Button>
                  <Button size="sm" variant="ghost" onClick={() => set(q.key, { mode: "edit", text: q.recommended })}>修改…</Button>
                </div>
              )}
          </div>
        )
      })}
      <div className="flex justify-end gap-2">
        <Button size="sm" variant="ghost" onClick={() => setAns(Object.fromEntries(questions.map((q) => [q.key, { mode: "accept" } as Answer])))}>全部采纳</Button>
        <Button size="sm" disabled={done < questions.length || pending} onClick={() => onSubmit(buildReplies(questions, ans, extra))}>
          提交本轮（{done}/{questions.length} 已答）
        </Button>
      </div>
    </Card>
  )
}
