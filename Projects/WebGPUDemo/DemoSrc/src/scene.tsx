import { useEffect, useRef, useState } from '@lynx-js/react'
import gpu, { startFrameLoop } from './webgpu.js'
import './demo.css'
import './elements.d.ts'

export interface SceneContext {
  device: any
  context: any
  format: string
  /** 씬이 HUD에 한 줄 띄우고 싶을 때 (리드백 결과 등). 프레임마다 부르지 말 것. */
  report: (text: string) => void
  /**
   * 캔버스를 누르고 있는 지점 (0~1 정규화). 놓으면 null.
   *
   * 프레임 루프가 매 프레임 읽는 값이라 state가 아니라 ref다 — state로 두면 손가락이
   * 움직일 때마다 리렌더가 붙는다 (`interactive` 씬과 같은 이유).
   */
  pointer: { current: { x: number; y: number } | null }
}

/** 매 프레임 호출되는 렌더 함수. */
export type SceneRenderer = (frame: { delta: number; width: number; height: number }) => void

/**
 * 데모 씬들이 공유하는 껍데기 — 어댑터/디바이스/캔버스 준비, 프레임 루프, FPS 표시, 정리.
 *
 * 각 씬은 `setup`에서 리소스를 만들고 **매 프레임 호출될 함수 하나**를 돌려주면 된다.
 */
export function DemoScene(props: {
  title: string
  subtitle: string
  /** 애셋을 받아 와야 하는 씬은 Promise를 돌려줘도 된다 — 준비될 때까지 프레임 루프를 늦춘다. */
  setup: (scene: SceneContext) => SceneRenderer | Promise<SceneRenderer>
  /** 캔버스 위에 얹을 조작 UI. 씬이 직접 그려 넘긴다. */
  controls?: any
}) {
  const [fps, setFps] = useState(0)
  const [status, setStatus] = useState('')
  const [note, setNote] = useState('')

  const pointer = useRef<{ x: number; y: number } | null>(null)
  /** 터치 좌표는 CSS px 기준이므로 픽셀 크기가 아니라 CSS 크기로 나눠야 한다. */
  const canvasCss = useRef({ width: 1, height: 1 })

  function normalize(event: any) {
    const touch = event?.touches?.[0] ?? event?.changedTouches?.[0]
    if (!touch) return null
    const { width, height } = canvasCss.current
    return {
      x: Math.min(Math.max(touch.x / Math.max(width, 1), 0), 1),
      y: Math.min(Math.max(touch.y / Math.max(height, 1), 0), 1),
    }
  }

  useEffect(() => {
    let stop: (() => void) | null = null
    let disposed = false
    let device: any = null

    async function boot() {
      const adapter = await gpu.requestAdapter()
      if (!adapter) throw new Error('WebGPU 어댑터 없음')
      device = await adapter.requestDevice()
      device.onError((_error: any, text: string) => setStatus(text))

      const context = gpu.getCanvasContext('main')
      const format = gpu.getPreferredCanvasFormat()
      context.configure({ device, format })

      const render = await props.setup({ device, context, format, report: setNote, pointer })
      if (disposed) return

      // 캔버스 크기는 프레임마다 물어보면 브리지 왕복이 늘어난다 — 한 번 읽고 캐시한다.
      let size = context.getSize()
      let sizeCheck = 0
      let frames = 0
      let elapsed = 0

      stop = startFrameLoop(({ delta }: { delta: number }) => {
        if (disposed) return

        // 회전/리사이즈 대응: 30프레임마다 한 번만 확인한다.
        if (++sizeCheck >= 30) {
          sizeCheck = 0
          size = context.getSize()
        }
        if (size.width === 0 || size.height === 0) return

        render({ delta, width: size.width, height: size.height })

        frames += 1
        elapsed += delta
        if (elapsed >= 1000) {
          setFps(Math.round((frames * 1000) / elapsed))
          frames = 0
          elapsed = 0
        }
      })
    }

    boot().catch((error) => setStatus(String(error && error.message ? error.message : error)))

    return () => {
      disposed = true
      if (stop) stop()
      if (device) device.destroy()
    }
  }, [])

  return (
    <view className="page">
      <webgpu-canvas
        canvas-id="main"
        className="canvas"
        bindcanvasresize={(event: any) => {
          const { width, height, pixelRatio } = event.detail
          canvasCss.current = { width: width / pixelRatio, height: height / pixelRatio }
        }}
        bindtouchstart={(event: any) => {
          pointer.current = normalize(event)
        }}
        bindtouchmove={(event: any) => {
          const point = normalize(event)
          if (point) pointer.current = point
        }}
        bindtouchend={() => {
          pointer.current = null
        }}
        bindtouchcancel={() => {
          pointer.current = null
        }}
      />
      <view className="hud">
        <text className="title">{props.title}</text>
        <text className="subtitle">
          {props.subtitle} · {fps} fps
        </text>
        {note ? <text className="note">{note}</text> : null}
        {status ? <text className="status">{status}</text> : null}
      </view>
      <text className="badge">WebGPU on Lynx</text>
      {props.controls ?? null}
    </view>
  )
}
