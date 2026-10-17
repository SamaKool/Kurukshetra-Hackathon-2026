import type { GraphData } from '../api'

export const humanMinutes = (m: number) => {
  const h = Math.floor(Math.abs(m) / 60)
  const min = Math.abs(m) % 60
  return h && min ? `${h}h ${min}m` : h ? `${h}h` : `${min} min`
}

const byId = (a: string, b: string) => a.localeCompare(b, undefined, { numeric: true })

/** One task per column, left to right: every prerequisite before the tasks that wait for it, otherwise in the order
 * the tasks were created (T1, T2, T3...). Tasks caught in a cycle (the server flags it) go last, still in id order. */
export function sequence(g: GraphData): Record<string, number> {
  const waits: Record<string, number> = {}
  const next: Record<string, string[]> = {}
  g.nodes.forEach((n) => { waits[n.id] = 0; next[n.id] = [] })
  g.edges.forEach((e) => {
    if (waits[e.target] == null || next[e.source] == null) return
    waits[e.target]++
    next[e.source].push(e.target)
  })
  const ready = g.nodes.map((n) => n.id).filter((id) => waits[id] === 0)
  const order: string[] = []
  while (ready.length) {
    ready.sort(byId)
    const id = ready.shift()!
    order.push(id)
    next[id].forEach((t) => { if (--waits[t] === 0) ready.push(t) })
  }
  g.nodes.map((n) => n.id).filter((id) => !order.includes(id)).sort(byId).forEach((id) => order.push(id))
  return Object.fromEntries(order.map((id, i) => [id, i]))
}
