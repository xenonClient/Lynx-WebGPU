import { root, useEffect, useState } from '@lynx-js/react'
import gpu from '../webgpu.js'
import { GPUBufferUsage } from '../webgpu.js'
import { encodeBase64 } from './base64.js'
import '../demo.css'
import '../elements.d.ts'

/**
 * Measuring the binary bridge cost — a base64 string vs an `ArrayBuffer`.
 *
 * What makes this comparison hold is that **the two paths run the same code natively**.
 * `WGPUValueReader.requiredData` accepts all three representations, so changing only what is put on the
 *command's `data` leaves everything else (staging plus blit) exactly the same.
 *
 * Two things are measured separately:
 *
 * - **Encoding** — the cost of JS building the payload. base64 is a per-byte loop, while an ArrayBuffer
 *   whose view covers the whole buffer is passed through as is and is effectively free.
 * - **Submission** — from a prepared payload until `execute` returns. The bridge crossing, native decoding
 *   and staging all live here. base64 hands across 33% more.
 *
 * It uses no canvas — the scene measures buffer uploads only, so no surface is needed.
 */

const SIZES = [
  { label: 'uniform', bytes: 1024 },
  { label: 'dynamic texture 128²', bytes: 64 * 1024 },
  { label: 'texture 256²', bytes: 256 * 1024 },
  { label: 'vertex stream', bytes: 1024 * 1024 },
]

/** The measurement budget. Small payloads need many repeats, and large ones back the GPU queue up, so they are capped. */
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
 * The mean time of one run (ms).
 *
 * `Date.now()` has ms resolution, so a small piece of work cannot be measured once — it runs until the
 * budget is filled and the total is divided by the count.
 */
function measure(run: () => void, maxIterations = MAX_ITERATIONS): number {
  run() // warm-up (JIT, allocator)
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

/** Gives the UI room to draw progress — without it the screen freezes until the measurement ends. */
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
  const [progress, setProgress] = useState('getting ready…')
  const [error, setError] = useState('')

  useEffect(() => {
    let disposed = false
    let device: any = null

    async function boot() {
      const adapter = await gpu.requestAdapter()
      if (!adapter) throw new Error('no WebGPU adapter')
      device = await adapter.requestDevice()
      device.onError((_e: any, text: string) => setError(text))

      const collected: Row[] = []

      for (const testCase of SIZES) {
        if (disposed) return
        setProgress(`measuring — ${testCase.label} (${formatBytes(testCase.bytes)})`)
        await yieldToUI()

        const bytes = makePattern(testCase.bytes)
        const buffer = device.createBuffer({
          size: testCase.bytes,
          usage: GPUBufferUsage.COPY_DST,
          label: `bench.${testCase.bytes}`,
        })
        // For a large payload, repetition is GPU queue pressure — the total transfer is capped around 8MB.
        const cap = Math.max(3, Math.floor((8 * 1024 * 1024) / testCase.bytes))

        // --- Encoding: the cost of JS building the payload ---
        const b64Encode = measure(() => {
          encodeBase64(bytes)
        })
        await yieldToUI()

        // The view covers the whole buffer, so the shim hands the backing buffer over as is (no copy).
        const abEncode = measure(() => {
          toArrayBufferLikeShim(bytes)
        })
        await yieldToUI()

        // --- Submission: the cost of crossing the bridge with a prepared payload ---
        // The commands are recorded directly so the two paths are perfectly symmetric (removing the difference of going through the shim).
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

      setProgress('done')
    }

    boot().catch((e) => setError(String(e && e.message ? e.message : e)))

    return () => {
      disposed = true
      if (device) device.destroy()
    }
  }, [])

  return (
    <view className="page bench-page">
      <text className="title">Binary bridge cost</text>
      <text className="subtitle">a base64 string vs an ArrayBuffer · {progress}</text>

      <view className="bench-row bench-head">
        <text className="bench-cell bench-label">payload</text>
        <text className="bench-cell">base64</text>
        <text className="bench-cell">ArrayBuffer</text>
        <text className="bench-cell">factor</text>
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
                {ratio >= 1 ? `${ratio.toFixed(1)}×` : `${(1 / ratio).toFixed(1)}× slower`}
              </text>
            </view>
            <view className="bench-row bench-detail">
              <text className="bench-cell bench-label">└ encoding / submission</text>
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
        encoding = the cost of JS building the payload · submission = until execute returns
        (bridge + native decoding + staging)
      </text>
    </view>
  )
}

/**
 * Makes the same judgement as the shim's `toArrayBuffer` — covering the whole buffer passes through, otherwise it slices.
 *
 * The shim's is not exported out of the module, so the same rule is written again here.
 */
function toArrayBufferLikeShim(view: Uint8Array): ArrayBuffer {
  const backing = view.buffer as ArrayBuffer
  return view.byteOffset === 0 && view.byteLength === backing.byteLength
    ? backing
    : backing.slice(view.byteOffset, view.byteOffset + view.byteLength)
}

root.render(<Bench />)
