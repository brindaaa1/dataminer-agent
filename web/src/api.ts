export type Status = "intake" | "intake_failed" | "config_ready" | "running" | "stopped" | "failed" | "done" | "accepted"
export type Stage = "intake" | "confirm" | "modeling" | "report"
export type Tone = "ok" | "bad" | "warn" | "neutral" | "running"

export interface TaskSummary { id: string; title: string; status: Status; created_at: number | null; example: boolean }
export interface Question { key: string; question: string; recommended: string; display: string; required: boolean }
export interface FieldRow { name: string; type: string; availability: string; reason: string; group: "ok" | "leak" | "quarantine" | "meta" }
export type Windows = Record<"train_valid" | "oot_dev" | "holdout", [string, string]>
export interface Draft { label: string | null; observation_time_expr: string | null; windows: Windows | null; metric: string | null; fields: FieldRow[]; confirmed: string[] }
export interface Intake { round: number; questions: Question[]; draft: Draft | null; changed: string[]; yaml: Record<string, string> }
export interface Trial { n: number; value: number | null; state: string }
export interface Evaluation { exp_id: string; model: string | null; fidelity: string | null; verdict: string | null; codes: string[]; oot_dev: number | null }
export interface Round {
  key: string; title: string; action: string | null; action_label: string; verdict: string | null; codes: string[]
  metrics: { train: number | null; valid: number | null; oot_dev: number | null; oot_dev_ks: number | null; gap: number | null }
  hypothesis: string | null; alternatives: string[]; stop_reason: string | null; evaluations: Evaluation[]
  llm_calls: number; running: boolean; trials: Record<string, Trial[]>
}
export interface Narration { key: string; text: string; tone: Tone }
export interface Final {
  exp_id: string; model: string | null; oot_dev: number | null; holdout: number | null; holdout_ks: number | null
  drop: number | null; overfit: boolean; n_experiments: number; n_accepted: number; risks: string[]
}
export interface TreeNode { exp_id: string; parent: string | null; verdict: string; label: string }
export interface Run {
  metric: string; max_rounds: number | null; rounds: Round[]; chart: { label: string; value: number; verdict: string | null }[]
  narration: Narration[]; final: Final | null; tree: TreeNode[]
}
export interface Snapshot {
  id: string; title: string; status: Status; stage: Stage; example: boolean; readonly: boolean; busy: boolean
  error: string | null; provider: string; langfuse_url: string | null; intake_langfuse_url: string | null; note: string | null
  data: { csv_name: string; description: string; n_rows: number | null; n_cols: number | null }
  intake: Intake | null; run: Run | null; artifacts: { name: string; label: string }[]; last_event_id: number
}
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export interface TaskEvent { id: number; ts: number; type: string; exp_id: string | null; payload: any }

async function check(r: Response) {
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail ?? r.statusText)
  return r
}
const getJSON = async <T,>(url: string): Promise<T> => (await check(await fetch(url))).json()
const post = async <T = { ok: boolean },>(url: string, body?: unknown): Promise<T> => {
  const form = body instanceof FormData
  const r = await fetch(url, {
    method: "POST",
    headers: form || body === undefined ? undefined : { "Content-Type": "application/json" },
    body: form ? body : body === undefined ? undefined : JSON.stringify(body),
  })
  return (await check(r)).json()
}

const t = (id: string) => `/api/tasks/${id}`
export const api = {
  tasks: () => getJSON<TaskSummary[]>("/api/tasks"),
  task: (id: string) => getJSON<Snapshot>(t(id)),
  events: (id: string) => getJSON<TaskEvent[]>(`${t(id)}/events`),
  create: (form: FormData) => post<Snapshot>("/api/tasks", form),
  answers: (id: string, replies: Record<string, string>) => post(`${t(id)}/answers`, { replies }),
  retry: (id: string) => post(`${t(id)}/retry`),
  run: (id: string) => post(`${t(id)}/run`),
  stop: (id: string) => post(`${t(id)}/stop`),
  resume: (id: string) => post(`${t(id)}/resume`),
  accept: (id: string, note: string) => post(`${t(id)}/accept`, { note }),
  clone: (id: string) => post<{ id: string }>(`${t(id)}/clone`),
  artifact: async (id: string, name: string) => (await check(await fetch(`${t(id)}/artifacts/${name}`))).text(),
}
