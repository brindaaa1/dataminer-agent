import { useRef, useState } from "react"
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { Paperclip, Send } from "lucide-react"
import { api } from "@/api"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { PROVIDERS } from "@/lib/format"

const select = "h-8 rounded-md border bg-background px-2 text-sm"

export function Composer({ onCreated }: { onCreated: (id: string) => void }) {
  const [files, setFiles] = useState<File[]>([])
  const [text, setText] = useState("")
  const [provider, setProvider] = useState("mock")
  const [rounds, setRounds] = useState(5)
  const input = useRef<HTMLInputElement>(null)
  const qc = useQueryClient()
  const csv = files.find((f) => f.name.toLowerCase().endsWith(".csv"))
  const doc = files.find((f) => /\.(md|txt)$/i.test(f.name))
  const create = useMutation({
    mutationFn: () => {
      const fd = new FormData()
      fd.append("csv", csv!)
      if (doc) fd.append("description_file", doc)
      fd.append("description", text)
      fd.append("provider", provider)
      fd.append("max_rounds", String(rounds))
      return api.create(fd)
    },
    onSuccess: (s) => { qc.invalidateQueries({ queryKey: ["tasks"] }); qc.setQueryData(["task", s.id], s); onCreated(s.id) },
  })
  return (
    <div className="space-y-2 border-t p-2">
      {files.length > 0 && <div className="flex flex-wrap gap-1">{files.map((f) => <Badge key={f.name} variant="secondary">📎 {f.name}</Badge>)}</div>}
      <Textarea rows={3} value={text} onChange={(e) => setText(e.target.value)} placeholder="业务说明：要预测什么、标签怎么定义……（也可以附 .md 文件）" />
      <div className="flex items-center gap-2">
        <input ref={input} type="file" multiple accept=".csv,.md,.txt" className="hidden" data-testid="file-input"
          onChange={(e) => { const add = Array.from(e.target.files ?? []); setFiles((old) => [...old.filter((f) => !add.some((x) => x.name === f.name)), ...add]); e.target.value = "" }} />
        <Button variant="ghost" size="icon" aria-label="添加附件" onClick={() => input.current?.click()}><Paperclip className="size-4" /></Button>
        <select aria-label="LLM" className={select} value={provider} onChange={(e) => setProvider(e.target.value)}>
          {PROVIDERS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
        </select>
        <select aria-label="最多轮数" className={select} value={rounds} onChange={(e) => setRounds(Number(e.target.value))}>
          {[1, 2, 3, 5, 8].map((n) => <option key={n} value={n}>最多 {n} 轮</option>)}
        </select>
        <Button className="ml-auto" size="sm" disabled={!csv || create.isPending} onClick={() => create.mutate()}><Send className="size-4" />发送</Button>
      </div>
      {create.error && <p className="text-xs text-red-600">{create.error.message}</p>}
    </div>
  )
}
