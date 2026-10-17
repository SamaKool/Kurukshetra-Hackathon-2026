import { useCallback, useEffect, useState } from 'react'
import type { AgentActionRow, Forecast, GraphData, RiskEvent, Status } from '../api'
import { api } from '../api'
import { Callout, Empty, fmtTime, PageHeader, plain, StatusTag, Tag } from '../ui'
import { humanMinutes, sequence } from './forecastUtil'
import { Timeline } from '../components/Timeline'

const SEVERITY_COLOR = { HIGH: 'red', MEDIUM: 'orange', LOW: 'gray' } as const
const CONFIDENCE_COLOR = { HIGH: 'green', MEDIUM: 'yellow', LOW: 'red' } as const
const ACTION_LABEL: Record<string, string> = {
  REQUEST_ETA: 'Asked for an ETA', REMIND_OWNER: 'Reminded', REQUEST_STATUS: 'Asked for status',
  REQUEST_CLARIFICATION: 'Asked a question', ESCALATE_BLOCKER: 'Escalated to the lead',
  NOTIFY_DEPENDENT_OWNER: 'Warned a dependent', REQUEST_HANDOFF: 'Asked for a handoff',
}
const ACTION_STATUS_COLOR = { pending: 'gray', sent: 'blue', committed: 'yellow', resolved: 'green', expired: 'gray', failed: 'red', unreachable: 'orange' } as const

const STATUS_FILL: Record<Status, string> = { Pending: 'var(--c-gray)', 'In Progress': 'var(--blue-bg)', Blocked: 'var(--red-bg)', Complete: 'var(--green-bg)' }

function DependencyGraph({ graph }: { graph: GraphData }) {
  const col = sequence(graph)
  const cols = Math.max(1, graph.nodes.length)
  const W = 170, H = 54, GX = 54
  // a link that skips over tasks arcs above the row instead of running through their boxes
  const span = (e: { source: string; target: string }) => Math.abs((col[e.target] ?? 0) - (col[e.source] ?? 0))
  const lift = (n: number) => Math.min(90, 20 + n * 14)
  const top = 8 + Math.max(0, ...graph.edges.filter((e) => span(e) > 1).map((e) => lift(span(e))))
  const pos = (id: string) => ({ x: 8 + (col[id] ?? 0) * (W + GX), y: top })
  const width = 16 + cols * W + (cols - 1) * GX
  const height = top + H + 8
  return (
    <div className="table-wrap" role="img" aria-label="Task dependency graph">
      <svg viewBox={`0 0 ${width} ${height}`} width={width} height={height} style={{ display: 'block' }}>
        <defs>
          <marker id="arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">
            <path d="M0,0 L8,4 L0,8 z" fill="currentColor" opacity="0.5" />
          </marker>
        </defs>
        {graph.edges.map((e, i) => {
          const a = pos(e.source), b = pos(e.target)
          const d = span(e) === 1 && a.x < b.x
            ? `M${a.x + W},${a.y + H / 2} L${b.x},${b.y + H / 2}`
            : `M${a.x + W / 2},${a.y} C${a.x + W / 2},${a.y - lift(span(e))} ${b.x + W / 2},${b.y - lift(span(e))} ${b.x + W / 2},${b.y}`
          return <path key={i} d={d} data-edge={`${e.source}-${e.target}`}
            fill="none" stroke="currentColor" opacity="0.4" strokeWidth="1.5" markerEnd="url(#arrow)" />
        })}
        {graph.nodes.map((n) => {
          const p = pos(n.id)
          const stroke = n.blocked ? 'var(--dot-red)' : n.late_min > 0 ? 'var(--orange-fg)' : n.critical ? 'var(--dot-blue)' : 'var(--line)'
          return (
            <g key={n.id} transform={`translate(${p.x},${p.y})`}>
              <rect width={W} height={H} rx="6" fill={STATUS_FILL[n.status]} stroke={stroke} strokeWidth={n.blocked || n.late_min > 0 || n.critical ? 2 : 1} />
              <text x="10" y="21" fontSize="12.5" fontWeight="600" fill="currentColor">{n.title.length > 22 ? n.title.slice(0, 21) + '…' : n.title}</text>
              <text x="10" y="40" fontSize="11.5" fill="currentColor" opacity="0.65">
                {n.owner} · {n.status}{n.late_min > 0 ? ` · late ${humanMinutes(n.late_min)}` : ''}
              </text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}

function Risk({ r }: { r: RiskEvent }) {
  return (
    <li className="risk">
      <div className="change-line">
        <Tag color={SEVERITY_COLOR[r.severity]}>{r.severity}</Tag>
        <b>{r.risk_type.replace(/_/g, ' ').toLowerCase()}</b>
        {r.attribution !== 'direct' && r.risk_type === 'DEADLINE_RISK' &&
          <Tag color="gray" title="The cause is not certain">{r.attribution === 'possible' ? 'cause: possible' : 'cause: unclear'}</Tag>}
      </div>
      <div>{r.summary}</div>
      {r.evidence.length > 0 && (
        <details className="toggle"><summary>Why the agent thinks so</summary>
          <ul>{r.evidence.map((e, i) => <li key={i} className="small muted">{e}</li>)}</ul>
        </details>
      )}
    </li>
  )
}

export function ForecastPage({ tick }: { tick: number }) {
  const [f, setF] = useState<Forecast | null>(null)
  const [graph, setGraph] = useState<GraphData | null>(null)
  const [acts, setActs] = useState<AgentActionRow[]>([])
  const [busy, setBusy] = useState(false)
  const [preview, setPreview] = useState<{ recipient: string; text: string; type: string }[] | null>(null)

  const load = useCallback(() => {
    Promise.all([api.forecast(), api.graph(), api.agentActions()]).then(([a, b, c]) => { setF(a); setGraph(b); setActs(c) }).catch(() => undefined)
  }, [])
  useEffect(load, [load, tick])

  const replan = async (send: boolean) => {
    setBusy(true)
    try {
      const r = await api.replan(send)
      setPreview(send ? null : r.actions)
      load()
    } finally { setBusy(false) }
  }

  const none = !f || f.status === 'NONE'
  return (
    <>
      <PageHeader icon="📈" title="Forecast & agent"
        description="The dependency graph feeds a CP-SAT scheduler. When it finds a problem, the agent decides who to contact and why.">
        <div className="review-actions">
          <button className="btn-primary small" disabled={busy} onClick={() => replan(true)}>Replan now</button>
          <button className="btn-secondary small" disabled={busy} onClick={() => replan(false)}>Preview, send nothing</button>
        </div>
      </PageHeader>

      {preview && (
        <Callout icon="👀" color="yellow">
          {preview.length === 0 ? 'Nothing to send: no problem needs a message right now.' : (
            <>Would send {preview.length}: {preview.map((p, i) => <div key={i}><b>{p.type}</b> to <b>{p.recipient}</b>: “{p.text}”</div>)}</>
          )}
        </Callout>
      )}

      {none ? <Empty>No forecast yet. It appears after the first status change, or press “Replan now”.</Empty> : f && (
        <>
          <div className="block">
            <div className="change-line">
              <Tag color={f.status === 'OPTIMAL' ? 'green' : f.status === 'INVALID_GRAPH' ? 'red' : 'yellow'}>{f.status.replace('_', ' ')}</Tag>
              {f.solver && <Tag color="gray">{f.solver === 'cp-sat' ? `CP-SAT · ${f.solve_ms} ms` : 'fallback scheduler'}</Tag>}
              {f.confidence && <Tag color={CONFIDENCE_COLOR[f.confidence]} title="How much to trust the numbers">{f.confidence.toLowerCase()} confidence</Tag>}
              {f.project_end && <span>Projected finish <b>{fmtTime(f.project_end)}</b>
                {f.previous_project_end && f.previous_project_end !== f.project_end && <span className="muted small"> (was {fmtTime(f.previous_project_end)})</span>}</span>}
              {(f.total_late_min ?? 0) > 0
                ? <Tag color="red">{humanMinutes(f.total_late_min!)} past deadlines</Tag> : <Tag color="green">on track</Tag>}
            </div>
            {f.note && <p className="small muted">{f.note}</p>}
            {f.analysis.issues && f.analysis.issues.length > 0 && <Callout icon="⚠️" color="red">{f.analysis.issues.join('; ')}</Callout>}
            {f.assumptions && f.assumptions.length > 0 && (
              <details className="toggle"><summary>{f.assumptions.length} assumption{f.assumptions.length > 1 ? 's' : ''} behind this forecast</summary>
                <ul>{f.assumptions.map((a, i) => <li key={i} className="small muted">{a}</li>)}</ul>
              </details>
            )}
          </div>

          <div className="grid-2">
            {f.risks.length > 0 && (
              <div className="block"><div className="block-title">Risks</div>
                <ul className="history">{f.risks.map((r, i) => <Risk key={i} r={r} />)}</ul>
              </div>
            )}

            {graph && graph.nodes.length > 0 && (
              <div className="block"><div className="block-title">Dependency graph</div>
                <p className="small muted">Arrows point from a prerequisite to the task that waits for it. Red outline: blocked. Orange: forecast to miss its deadline. Blue: on the critical path.</p>
                <DependencyGraph graph={graph} />
              </div>
            )}
          </div>

          <div className="block"><div className="block-title">Timeline</div>
            <Timeline tasks={f.tasks} />
          </div>

          <div className="block"><div className="block-title">Schedule</div>
            <div className="table-wrap">
              <table className="db-table">
                <thead><tr><th>Task</th><th>Owner</th><th>Status</th><th>Finishes</th><th>Deadline</th><th>Slack</th><th>Priority</th></tr></thead>
                <tbody>
                  {f.tasks.map((t) => (
                    <tr key={t.id} style={{ cursor: 'default' }}>
                      <td className="cell-title">{t.title} {t.critical && <Tag color="blue" title="On the critical path">critical</Tag>}</td>
                      <td>{t.owner}</td>
                      <td>{t.status && <StatusTag status={t.status} />}</td>
                      <td>{fmtTime(t.end)}</td>
                      <td>{t.deadline ? fmtTime(t.deadline) : <span className="muted">none</span>}</td>
                      <td>{t.slack_min == null ? <span className="muted">n/a</span>
                        : t.late_min > 0 ? <Tag color="red">late {humanMinutes(t.late_min)}</Tag> : <span>{humanMinutes(t.slack_min)}</span>}</td>
                      <td title="0-100: deadline urgency, dependency impact, blocker severity, resource risk">{t.priority ?? ''}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      <div className="block"><div className="block-title">What the agent did</div>
        {acts.length === 0 && <Empty>The agent has not needed to contact anyone yet.</Empty>}
        <ol className="feed">
          {acts.map((a) => (
            <li key={a.id} className="feed-row">
              <span className="muted small ts">{fmtTime(a.created_at)}</span>
              <span>
                <Tag color="purple">{ACTION_LABEL[a.type] ?? a.type}</Tag> <b>{a.recipient}</b> about <b>{a.task}</b>{' '}
                <Tag color={ACTION_STATUS_COLOR[a.status]} title={a.status === 'unreachable' ? 'Not connected to Slack or Telegram yet, so nothing was sent' : undefined}>{a.status}</Tag>
                {a.commitment_at && <span className="muted small"> promised {fmtTime(a.commitment_at)}</span>}
                <div className="muted">{plain(a.text)}</div>
                {a.response_text && <div className="small">↩ “{a.response_text}”</div>}
                {a.evidence.length > 0 && (
                  <details className="toggle"><summary>Why this message</summary>
                    <ul>{a.evidence.map((e, i) => <li key={i} className="small muted">{e}</li>)}</ul>
                  </details>
                )}
              </span>
            </li>
          ))}
        </ol>
      </div>
    </>
  )
}
