import '@testing-library/jest-dom/vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { Forecast, GraphData, Status } from '../api'
import { ForecastPage } from './Forecast'
import { humanMinutes, sequence } from './forecastUtil'

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

const graph: GraphData = {
  nodes: [
    { id: 'T3', title: 'Get sandbox test credentials', owner: 'Priya', status: 'In Progress', end: null, late_min: 0, critical: true, blocked: false },
    { id: 'T2', title: 'Connect the payment API', owner: 'Kabir', status: 'Blocked', end: null, late_min: 120, critical: true, blocked: true },
    { id: 'T4', title: 'Build the checkout page', owner: 'Meera', status: 'In Progress', end: null, late_min: 0, critical: false, blocked: false },
  ],
  edges: [{ source: 'T3', target: 'T2' }, { source: 'T2', target: 'T4' }],
}

const task = (id: string, title: string, owner: string, status: Status, late = 0) => ({
  id, title, owner, status, start: '2026-10-17T09:00:00+00:00', end: '2026-10-17T11:00:00+00:00', duration_min: 120,
  deadline: '2026-10-17T10:00:00+00:00', late_min: late, slack_min: late ? -late : 60, critical: late > 0, notes: [], priority: 61,
})

const forecast: Forecast = {
  status: 'OPTIMAL', solver: 'cp-sat', confidence: 'LOW', version: 'abc', project_end: '2026-10-17T14:00:00+00:00',
  previous_project_end: '2026-10-17T11:00:00+00:00', total_late_min: 120, solve_ms: 12,
  assumptions: ["'Connect the payment API' blocked by 'test credentials': no resolution time known; assumed 240 min"],
  tasks: [task('T2', 'Connect the payment API', 'Kabir', 'Blocked', 120), task('T4', 'Build the checkout page', 'Meera', 'In Progress')],
  risks: [{ risk_type: 'CONFIRMED_BLOCKER', severity: 'HIGH', task_id: 'T2', summary: "'Connect the payment API' is blocked: test credentials.",
    evidence: ['Kabir reported it blocked', 'dependency path: Connect the payment API -> Build the checkout page'], attribution: 'direct', delay_minutes: 0, confidence: 'LOW' }],
  analysis: {},
}

const actions = [{
  id: 1, created_at: '2026-10-17T09:05:00+00:00', type: 'REQUEST_ETA', risk_type: 'CONFIRMED_BLOCKER', task: 'Connect the payment API', task_id: 'T2',
  recipient: 'Priya', text: '@Priya Kabir is blocked on Connect the payment API', status: 'committed', channel: 'telegram',
  commitment_at: '2026-10-17T09:25:00+00:00', response_text: 'in 20 minutes', evidence: ['Kabir reported it blocked'], severity: 'HIGH',
}]

function mockApi(f: unknown = forecast, acts: unknown = actions) {
  const calls: string[] = []
  vi.stubGlobal('fetch', vi.fn((url: string, init?: RequestInit) => {
    calls.push(`${init?.method ?? 'GET'} ${url}`)
    const body = url.startsWith('/api/intel/forecast') ? f : url.startsWith('/api/intel/graph') ? graph
      : url.startsWith('/api/intel/actions') ? acts : url.startsWith('/api/intel/replan') ? { actions: [{ type: 'REQUEST_ETA', recipient: 'Priya', text: 'would say hi' }] } : {}
    return Promise.resolve(new Response(JSON.stringify(body), { status: 200 }))
  }))
  return calls
}

describe('helpers', () => {
  it('humanMinutes', () => {
    expect(humanMinutes(45)).toBe('45 min')
    expect(humanMinutes(60)).toBe('1h')
    expect(humanMinutes(125)).toBe('2h 5m')
    expect(humanMinutes(-30)).toBe('30 min')
  })
  it('sequence puts every prerequisite before the task that waits for it', () => {
    expect(sequence(graph)).toEqual({ T3: 0, T2: 1, T4: 2 })
  })
  it('sequence lays independent tasks out one after the other in creation order, not stacked in one column', () => {
    const node = graph.nodes[0]
    const free: GraphData = { nodes: ['T10', 'T2', 'T1', 'T3'].map((id) => ({ ...node, id })), edges: [] }
    expect(sequence(free)).toEqual({ T1: 0, T2: 1, T3: 2, T10: 3 })
  })
  it('sequence survives a dependency cycle and still gives every task its own column', () => {
    const cyc: GraphData = { nodes: graph.nodes.slice(0, 2), edges: [{ source: 'T3', target: 'T2' }, { source: 'T2', target: 'T3' }] }
    expect(Object.values(sequence(cyc)).sort()).toEqual([0, 1])
  })
})

describe('ForecastPage', () => {
  it('invites the first replan when there is no forecast', async () => {
    mockApi({ status: 'NONE', tasks: [], risks: [], analysis: {} }, [])
    render(<ForecastPage tick={0} />)
    expect(await screen.findByText(/No forecast yet/)).toBeInTheDocument()
    expect(screen.getByText(/has not needed to contact anyone/)).toBeInTheDocument()
  })

  it('shows solver, confidence, lateness and the schedule', async () => {
    mockApi()
    render(<ForecastPage tick={0} />)
    expect(await screen.findByText('OPTIMAL')).toBeInTheDocument()
    expect(screen.getByText(/CP-SAT · 12 ms/)).toBeInTheDocument()
    expect(screen.getByText('low confidence')).toBeInTheDocument()
    expect(screen.getByText('2h past deadlines')).toBeInTheDocument()
    expect(screen.getByText('late 2h')).toBeInTheDocument()
    expect(screen.getAllByText('Connect the payment API').length).toBeGreaterThan(0)
  })

  it('is honest about assumptions and why a risk was raised', async () => {
    mockApi()
    render(<ForecastPage tick={0} />)
    expect(await screen.findByText(/1 assumption behind this forecast/)).toBeInTheDocument()
    expect(screen.getByText('HIGH')).toBeInTheDocument()
    expect(screen.getByText(/dependency path: Connect the payment API -> Build the checkout page/)).toBeInTheDocument()
  })

  it('draws every task and dependency in the graph', async () => {
    mockApi()
    const { container } = render(<ForecastPage tick={0} />)
    await screen.findByText('OPTIMAL')
    expect(container.querySelectorAll('svg g').length).toBe(3)
    expect(container.querySelectorAll('svg path[marker-end]').length).toBe(2)
  })

  it('lists what the agent did with the reply it got', async () => {
    mockApi()
    render(<ForecastPage tick={0} />)
    expect(await screen.findByText('Asked for an ETA')).toBeInTheDocument()
    expect(screen.getByText('committed')).toBeInTheDocument()
    expect(screen.getByText(/in 20 minutes/)).toBeInTheDocument()
    expect(screen.getByText('Why this message')).toBeInTheDocument()
  })

  it('"Preview" asks for a dry run and shows what would be sent', async () => {
    const calls = mockApi()
    render(<ForecastPage tick={0} />)
    await screen.findByText('OPTIMAL')
    fireEvent.click(screen.getByRole('button', { name: 'Preview, send nothing' }))
    await waitFor(() => expect(calls).toContain('POST /api/intel/replan?send=false'))
    expect(await screen.findByText(/would say hi/)).toBeInTheDocument()
  })

  it('"Replan now" really sends', async () => {
    const calls = mockApi()
    render(<ForecastPage tick={0} />)
    await screen.findByText('OPTIMAL')
    fireEvent.click(screen.getByRole('button', { name: 'Replan now' }))
    await waitFor(() => expect(calls).toContain('POST /api/intel/replan?send=true'))
  })

  it('refetches when the live tick changes', async () => {
    const calls = mockApi()
    const { rerender } = render(<ForecastPage tick={0} />)
    await screen.findByText('OPTIMAL')
    const before = calls.filter((c) => c.includes('/intel/forecast')).length
    rerender(<ForecastPage tick={1} />)
    await waitFor(() => expect(calls.filter((c) => c.includes('/intel/forecast')).length).toBe(before + 1))
  })

  it('flags an invalid graph prominently', async () => {
    mockApi({ ...forecast, status: 'INVALID_GRAPH', tasks: [], risks: [], analysis: { issues: ['dependency cycle: T2 -> T3 -> T2'] } }, [])
    render(<ForecastPage tick={0} />)
    expect(await screen.findByText(/dependency cycle: T2 -> T3 -> T2/)).toBeInTheDocument()
  })
})
