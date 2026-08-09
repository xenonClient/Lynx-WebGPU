import { useEffect, useRef, useState } from '@lynx-js/react'
import gpu, { startFrameLoop } from './webgpu.js'
import './demo.css'
import './elements.d.ts'

export interface SceneContext {
  device: any
  context: any
  format: string
  /** For when a scene wants to put a line on the HUD (a readback result, say). Do not call it per frame. */
  report: (text: string) => void
  /**
   * The point being pressed on the canvas (normalized 0~1). null when released.
   *
   * The frame loop reads it every frame, so it is a ref rather than state — as state it would attach a
   * re-render on every finger movement (the same reason as in the `interactive` scene).
   */
  pointer: { current: { x: number; y: number } | null }
}

/** The render function called every frame. */
export type SceneRenderer = (frame: { delta: number; width: number; height: number }) => void

/**
 * The shell the demo scenes share — adapter/device/canvas setup, the frame loop, the FPS display, cleanup.
 *
 * Each scene builds its resources in `setup` and returns **one function to be called every frame**.
 */
export function DemoScene(props: {
  title: string
  subtitle: string
  /** A scene that has to fetch assets may return a Promise — the frame loop waits until it is ready. */
  setup: (scene: SceneContext) => SceneRenderer | Promise<SceneRenderer>
  /** Control UI laid over the canvas. The scene draws and passes it in itself. */
  controls?: any
}) {
  const [fps, setFps] = useState(0)
  const [status, setStatus] = useState('')
  const [note, setNote] = useState('')

  const pointer = useRef<{ x: number; y: number } | null>(null)
  /** Touch coordinates are in CSS px, so they must be divided by the CSS size rather than the pixel size. */
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
      if (!adapter) throw new Error('no WebGPU adapter')
      device = await adapter.requestDevice()
      device.onError((_error: any, text: string) => setStatus(text))

      const context = gpu.getCanvasContext('main')
      const format = gpu.getPreferredCanvasFormat()
      context.configure({ device, format })

      const render = await props.setup({ device, context, format, report: setNote, pointer })
      if (disposed) return

      // Asking for the canvas size every frame adds bridge crossings — it is read once and cached.
      let size = context.getSize()
      let sizeCheck = 0
      let frames = 0
      let elapsed = 0

      stop = startFrameLoop(({ delta }: { delta: number }) => {
        if (disposed) return

        // Handling rotation and resize: checked only once every 30 frames.
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
