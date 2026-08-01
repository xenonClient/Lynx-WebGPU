import { root, useEffect, useState } from '@lynx-js/react'
import gpu from '../webgpu.js'
import { GPUBufferUsage } from '../webgpu.js'
import { encodeBase64 } from './base64.js'
import '../demo.css'
import '../elements.d.ts'

/**
 * 바이너리 브리지 비용 측정 — base64 문자열 vs `ArrayBuffer`.
 *
 * 두 경로가 **네이티브에서 같은 코드를 탄다**는 점이 이 비교를 성립시킨다.
 * `WGPUValueReader.requiredData`가 세 표현을 모두 받으므로, 커맨드의 `data`에 무엇을
 *싣느냐만 바꾸면 나머지(스테이징 + blit)는 완전히 같다.
 *
 * 두 가지를 따로 잰다:
 *
 * - **인코딩** — JS가 페이로드를 만드는 비용. base64는 바이트마다 도는 루프이고,
 *   ArrayBuffer는 버퍼 전체를 덮는 뷰라면 그대로 넘기므로 사실상 공짜다.
 * - **제출** — 만들어 둔 페이로드로 `execute`가 돌아올 때까지. 브리지 통과 +
 *   네이티브 디코딩 + 스테이징이 여기 들어간다. base64는 33% 더 큰 것을 넘긴다.
 *
 * 캔버스를 쓰지 않는다 — 버퍼 업로드만 재는 씬이라 표면이 필요 없다.
 */

const SIZES = [
  { label: '유니폼', bytes: 1024 },
  { label: '동적 텍스처 128²', bytes: 64 * 1024 },
  { label: '텍스처 256²', bytes: 256 * 1024 },
  { label: '정점 스트림', bytes: 1024 * 1024 },
]

/** 측정 예산. 작은 페이로드는 반복이 많이 필요하고, 큰 것은 GPU 큐가 밀리므로 묶어 둔다. */
const BUDGET_MS = 150
const MAX_ITERATIONS = 400

function makePattern(length: number): Uint8Array {
  const bytes = new Uint8Array(length)
  for (let index = 0; index < length; index += 1) bytes[index] = (index * 31 + 7) & 0xff
  return bytes
}

function formatBytes(n: number): string {
  return n < 1024 ? `${n} B` : n < 1024 * 1024 ? `${n / 1024} KB` : `${n / 1024 / 1024} MB`
}

/**
 * 1회 평균 시간(ms).
 *
 * `Date.now()`는 ms 해상도라 작은 작업은 한 번으로 못 잰다 — 예산을 채울 때까지 돌려
 * 총 시간을 횟수로 나눈다.
 */
function measure(run: () => void, maxIterations = MAX_ITERATIONS): number {
  run() // 워밍업 (JIT·할당자)
  const start = Date.now()
  let iterations = 0
  let now = start
  while (now - start < BUDGET_MS && iterations < maxIterations) {
    run()
    iterations += 1
    now = Date.now()
  }
  return (now - start) / Math.max(iterations, 1)
}

/** UI가 진행 상황을 그릴 틈을 준다 — 안 그러면 측정이 끝날 때까지 화면이 멈춘다. */
function yieldToUI(): Promise<void> {
  return new Promise((resolve) => setTimeout(() => resolve(), 0))
}

interface Row {
  label: string
  size: number
  b64Encode: number
  b64Submit: number
  abEncode: number
  abSubmit: number
}

function totalOf(row: Row, base64: boolean): number {
  return base64 ? row.b64Encode + row.b64Submit : row.abEncode + row.abSubmit
}

function Bench() {
  const [rows, setRows] = useState<Row[]>([])
  const [progress, setProgress] = useState('준비 중…')
  const [error, setError] = useState('')

  useEffect(() => {
    let disposed = false
    let device: any = null

    async function boot() {
      const adapter = await gpu.requestAdapter()
      if (!adapter) throw new Error('WebGPU 어댑터 없음')
      device = await adapter.requestDevice()
      device.onError((_e: any, text: string) => setError(text))

      const collected: Row[] = []

      for (const testCase of SIZES) {
        if (disposed) return
        setProgress(`측정 중 — ${testCase.label} (${formatBytes(testCase.bytes)})`)
        await yieldToUI()

        const bytes = makePattern(testCase.bytes)
        const buffer = device.createBuffer({
          size: testCase.bytes,
          usage: GPUBufferUsage.COPY_DST,
          label: `bench.${testCase.bytes}`,
        })
        // 큰 페이로드는 반복이 곧 GPU 큐 부담이다 — 총 전송량을 8MB 정도로 묶는다.
        const cap = Math.max(3, Math.floor((8 * 1024 * 1024) / testCase.bytes))

        // --- 인코딩: JS가 페이로드를 만드는 비용 ---
        const b64Encode = measure(() => {
          encodeBase64(bytes)
        })
        await yieldToUI()

        // 버퍼 전체를 덮는 뷰라 셰임은 백킹 버퍼를 그대로 넘긴다 (복사 없음).
        const abEncode = measure(() => {
          toArrayBufferLikeShim(bytes)
        })
        await yieldToUI()

        // --- 제출: 만들어 둔 페이로드로 브리지를 건너는 비용 ---
        // 두 경로가 완전히 대칭이 되도록 커맨드를 직접 기록한다 (셰임 경유 차이 제거).
        const b64Payload = encodeBase64(bytes)
        const abPayload = toArrayBufferLikeShim(bytes)

        const submitWith = (payload: unknown) => () => {
          device._recorder.push({
            op: 'writeBuffer',
            buffer: buffer.id,
            bufferOffset: 0,
            data: payload,
          })
          device.queue.submit([])
        }

        const b64Submit = measure(submitWith(b64Payload), cap)
        await yieldToUI()
        const abSubmit = measure(submitWith(abPayload), cap)
        await yieldToUI()

        buffer.destroy()
        device.queue.submit([])

        collected.push({
          label: testCase.label,
          size: testCase.bytes,
          b64Encode,
          b64Submit,
          abEncode,
          abSubmit,
        })
        if (disposed) return
        setRows(collected.slice())
      }

      setProgress('완료')
    }

    boot().catch((e) => setError(String(e && e.message ? e.message : e)))

    return () => {
      disposed = true
      if (device) device.destroy()
    }
  }, [])

  return (
    <view className="page bench-page">
      <text className="title">바이너리 브리지 비용</text>
      <text className="subtitle">base64 문자열 vs ArrayBuffer · {progress}</text>

      <view className="bench-row bench-head">
        <text className="bench-cell bench-label">페이로드</text>
        <text className="bench-cell">base64</text>
        <text className="bench-cell">ArrayBuffer</text>
        <text className="bench-cell">배수</text>
      </view>

      {rows.map((row) => {
        const b64 = totalOf(row, true)
        const ab = totalOf(row, false)
        const ratio = ab > 0 ? b64 / ab : 0
        return (
          <view key={row.label} className="bench-group">
            <view className="bench-row">
              <text className="bench-cell bench-label">
                {row.label} · {formatBytes(row.size)}
              </text>
              <text className="bench-cell bench-slow">{b64.toFixed(3)} ms</text>
              <text className="bench-cell bench-fast">{ab.toFixed(3)} ms</text>
              <text className="bench-cell bench-ratio">
                {ratio >= 1 ? `${ratio.toFixed(1)}×` : `${(1 / ratio).toFixed(1)}× 느림`}
              </text>
            </view>
            <view className="bench-row bench-detail">
              <text className="bench-cell bench-label">└ 인코딩 / 제출</text>
              <text className="bench-cell">
                {row.b64Encode.toFixed(3)} / {row.b64Submit.toFixed(3)}
              </text>
              <text className="bench-cell">
                {row.abEncode.toFixed(3)} / {row.abSubmit.toFixed(3)}
              </text>
              <text className="bench-cell" />
            </view>
          </view>
        )
      })}

      {error ? <text className="status">{error}</text> : null}
      <text className="bench-foot">
        인코딩 = JS가 페이로드를 만드는 비용 · 제출 = execute가 돌아올 때까지
        (브리지 + 네이티브 디코딩 + 스테이징)
      </text>
    </view>
  )
}

/**
 * 셰임의 `toArrayBuffer`와 같은 판단을 한다 — 버퍼 전체를 덮으면 그대로, 아니면 잘라 낸다.
 *
 * 셰임의 것은 모듈 밖으로 내보내지 않으므로 같은 규칙을 여기 다시 적는다.
 */
function toArrayBufferLikeShim(view: Uint8Array): ArrayBuffer {
  const backing = view.buffer as ArrayBuffer
  return view.byteOffset === 0 && view.byteLength === backing.byteLength
    ? backing
    : backing.slice(view.byteOffset, view.byteOffset + view.byteLength)
}

root.render(<Bench />)
