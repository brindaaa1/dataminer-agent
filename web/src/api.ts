export type Status = "intake" | "intake_failed" | "config_ready" | "running" | "stopped" | "failed" | "done" | "accepted"
export type Stage = "intake" | "confirm" | "modeling" | "report"
export type Tone = "ok" | "bad" | "warn" | "neutral" | "running"

export interface EvalInfo { label: string; tier: string; conclusion: string | null; running: boolean }
export interface TaskSummary { id: string; title: string; status: Status; created_at: number | null; example: boolean; eval: EvalInfo | null }
export interface Question { key: string; question: string; recommended: string; display: string; required: boolean }
export interface FieldRow { name: string; type: string; availability: string; reason: string; group: "ok" | "leak" | "quarantine" | "meta"; history: string | null }
export type Windows = Record<"train_valid" | "oot_dev" | "holdout", [string, string]>
export interface Draft { label: string | null; observation_time_expr: string | null; windows: Windows | null; metric: string | null
  domain: string | null; data_summary: string | null; fields: FieldRow[]; confirmed: string[] }
export interface Intake { round: number; questions: Question[]; draft: Draft | null; changed: string[]; yaml: Record<string, string> }
export interface Trial { n: number; value: number | null; state: string }
export interface Evaluation { exp_id: string; model: string | null; fidelity: string | null; verdict: string | null; codes: string[]; oot_dev: number | null }
export interface Round {
  key: string; title: string; action: string | null; action_label: string; verdict: string | null; codes: string[]
  metrics: { train: number | null; valid: number | null; oot_dev: number | null; oot_dev_ks: number | null; gap: number | null }
  hypothesis: string | null; user_note: string | null; alternatives: string[]; stop_reason: string | null; evaluations: Evaluation[]
  llm_calls: number; running: boolean; trials: Record<string, Trial[]>; by_rule: boolean; action_user: string
}
export interface Narration { key: string; text: string; tone: Tone }
export interface Final {
  exp_id: string; model: string | null; oot_dev: number | null; holdout: number | null; holdout_ks: number | null
  drop: number | null; overfit: boolean; n_experiments: number; n_accepted: number; lift_top10: number | null
  risks: string[]; risks_user: string[]
}
export interface TreeNode { exp_id: string; parent: string | null; verdict: string; label: string }
export interface Run {
  metric: string; max_rounds: number | null; rounds: Round[]; chart: { label: string; value: number; verdict: string | null }[]
  narration: Narration[]; brief: Narration[]; final: Final | null; tree: TreeNode[]
}
export interface Snapshot {
  id: string; title: string; status: Status; stage: Stage; example: boolean; eval_label: string | null; readonly: boolean; busy: boolean
  error: string | null; provider: string; langfuse_url: string | null; intake_langfuse_url: string | null; note: string | null
  data: { csv_name: string; description: string; n_rows: number | null; n_cols: number | null }
  intake: Intake | null; run: Run | null; artifacts: { name: string; label: string }[]; last_event_id: number
}
export interface Evidence { ref: string; round: string | null }
export interface ProcessRow { key: string; label: string; value: number | boolean | string | null; detail: Record<string, unknown>; evidence: Evidence[] }
export interface RegRow { dataset: string; metric: string; label: string; new: number; old: number; delta: number; n: number; verdict: "better" | "worse" | "same"; task: string | null }
export interface Flag { kind: string; label: string; detail: string; task: string | null }
export interface SummaryRow { dataset: string; n: number; n_ok: number; holdout_mean: number | null; delta_b1_mean: number | null
  rounds_mean: number | null; fe_in_final: string; wall_sec: number; cost_usd: number; worst_task: string | null }
export interface RunRow { task: string; dataset: string; seed: number; status: string; holdout: number | null; delta_b1: number | null }
export interface CostRow { dataset: string; label: string; new: number; old: number; ratio: number | null; threshold: number }
export interface VersionView { label: string; tier: string; running: boolean; against: string | null; conclusion: string; noise_calibrated: boolean
  table: RegRow[]; cost: CostRow[]; flags: Flag[]; runs: RunRow[]; summary: SummaryRow[] }
export interface ScenarioRow { scenario: string; base: string; passed: boolean; reason: string; task: string }
export interface EvalView { process: ProcessRow[]; version: VersionView | null; scenarios: ScenarioRow[] | null }
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
  evalView: (id: string) => getJSON<EvalView>(`${t(id)}/eval`),
  artifact: async (id: string, name: string) => (await check(await fetch(`${t(id)}/artifacts/${name}`))).text(),
}
