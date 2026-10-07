import { useSyncExternalStore } from "react"

/** 开发者视图开关：默认关（用户视图），记在浏览器里；任何组件切换，其他组件立即跟着变。 */
const KEY = "dev"
const subs = new Set<() => void>()
let on = (() => { try { return localStorage.getItem(KEY) === "1" } catch { return false } })()

export function setDev(v: boolean) {
  on = v
  try { localStorage.setItem(KEY, v ? "1" : "0") } catch { /* 无痕模式等 */ }
  subs.forEach((f) => f())
}

export function useDev(): boolean {
  return useSyncExternalStore((cb) => { subs.add(cb); return () => { subs.delete(cb) } }, () => on, () => false)
}
