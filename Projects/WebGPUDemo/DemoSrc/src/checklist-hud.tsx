import { useState } from '@lynx-js/react'
import './demo.css'
import './elements.d.ts'

/**
 * 체크리스트 HUD — 씬들이 공유하는 로그 카드.
 *
 * **접을 수 있어야 한다.** 검증 항목이 열 개를 넘고 오류 줄까지 붙으면 카드가 화면을 거의
 * 다 덮어, 정작 확인하려던 렌더 결과가 안 보인다 (`threelab`이 그랬다). 머리글을 누르면
 * 접히고, 접힌 상태에서도 **요약 한 줄은 남긴다** — 통과 수는 항상 보여야 한다.
 */

export interface Check {
  label: string
  /** `skip`은 기기가 못 하는 것 — 실패가 아니다 (압축 계열 등). */
  state: 'wait' | 'ok' | 'fail' | 'skip'
  detail?: string
}

const ICON = { wait: '○', ok: '✓', fail: '✗', skip: '–' }

interface Props {
  title: string
  subtitle?: string
  checks: Check[]
  /** 요약 줄 (접어도 남는다). 없으면 통과 수로 자동 생성한다. */
  summary?: string
  /** 오류·덤프 줄. 접으면 같이 숨는다. */
  errors?: string[]
  /** 처음부터 접어 둘 것인가 (기본은 펼침). */
  initiallyCollapsed?: boolean
}

export function ChecklistHud(props: Props) {
  const [collapsed, setCollapsed] = useState(!!props.initiallyCollapsed)

  const passed = props.checks.filter((check) => check.state === 'ok').length
  const failed = props.checks.filter((check) => check.state === 'fail').length
  const skipped = props.checks.filter((check) => check.state === 'skip').length
  const total = props.checks.length - skipped
  const summary = props.summary ?? (
    `통과 ${passed}/${total}`
    + (skipped ? ` · 미지원 ${skipped}` : '')
    + (failed ? ` · 실패 ${failed}` : '')
  )
  const errors = props.errors || []

  return (
    <view className={`three-hud${collapsed ? ' hud-collapsed' : ''}`}>
      {/* 머리글 전체가 버튼이다 — 작은 화살표만 누르게 하면 손가락으로 맞히기 어렵다. */}
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
        {summary}{collapsed && failed ? ' · 펼쳐서 보기' : ''}
      </text>

      {collapsed ? null : errors.map((line, index) => (
        <text className="check-row check-fail" key={`err-${index}`}>{line}</text>
      ))}
    </view>
  )
}
