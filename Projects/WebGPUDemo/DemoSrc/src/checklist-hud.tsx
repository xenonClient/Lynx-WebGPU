import { useState } from '@lynx-js/react'
import './demo.css'
import './elements.d.ts'

/**
 * The checklist HUD — the log card the scenes share.
 *
 * **It has to be collapsible.** Past ten check items, with error lines attached, the card covers nearly the
 * whole screen and hides the very render result you wanted to see (`threelab` did that). Pressing the header
 * collapses it, and **the one-line summary stays** even collapsed — the pass count must always be visible.
 */

export interface Check {
  label: string
  /** `skip` is something the device cannot do — not a failure (compressed formats and the like). */
  state: 'wait' | 'ok' | 'fail' | 'skip'
  detail?: string
}

const ICON = { wait: '○', ok: '✓', fail: '✗', skip: '–' }

interface Props {
  title: string
  subtitle?: string
  checks: Check[]
  /** The summary line (it survives collapsing). Generated from the pass count if absent. */
  summary?: string
  /** Error and dump lines. They hide along with the collapse. */
  errors?: string[]
  /** Whether to start collapsed (expanded by default). */
  initiallyCollapsed?: boolean
}

export function ChecklistHud(props: Props) {
  const [collapsed, setCollapsed] = useState(!!props.initiallyCollapsed)

  const passed = props.checks.filter((check) => check.state === 'ok').length
  const failed = props.checks.filter((check) => check.state === 'fail').length
  const skipped = props.checks.filter((check) => check.state === 'skip').length
  const total = props.checks.length - skipped
  const summary = props.summary ?? (
    `passed ${passed}/${total}`
    + (skipped ? ` · unsupported ${skipped}` : '')
    + (failed ? ` · failed ${failed}` : '')
  )
  const errors = props.errors || []

  return (
    <view className={`three-hud${collapsed ? ' hud-collapsed' : ''}`}>
      {/* The whole header is the button — a small arrow alone would be hard to hit with a finger. */}
      <view className="hud-header" bindtap={() => setCollapsed(!collapsed)}>
        <view className="hud-header-text">
          <text className="title">{props.title}</text>
          {props.subtitle ? <text className="subtitle">{props.subtitle}</text> : null}
        </view>
        <text className="hud-toggle">{collapsed ? '▸' : '▾'}</text>
      </view>

      {collapsed ? null : props.checks.map((check, index) => (
        <text className={`check-row check-${check.state}`} key={`check-${index}`}>
          {ICON[check.state]} {check.label}{check.detail ? ` — ${check.detail}` : ''}
        </text>
      ))}

      <text className="check-stats">
        {summary}{collapsed && failed ? ' · expand to see' : ''}
      </text>

      {collapsed ? null : errors.map((line, index) => (
        <text className="check-row check-fail" key={`err-${index}`}>{line}</text>
      ))}
    </view>
  )
}
