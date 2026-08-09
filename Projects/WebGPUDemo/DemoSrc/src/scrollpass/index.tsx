import { root, useEffect, useRef, useState } from '@lynx-js/react'
import gpu, { GPUBufferUsage, startFrameLoop } from '../webgpu.js'
import '../demo.css'
import '../elements.d.ts'

/**
 * Scroll passthrough verification — a screen for checking whether scrolling works purely through
 * `passthrough-touches` hit test handling when a canvas overlaps a `<scroll-view>` **above it** as a sibling.
 *
 * Drag the canvas band vertically:
 *   - **passthrough ON**  → the list scrolls, and the canvas receives a `touchcancel` the moment scrolling
 *                           wins after `touchstart` (the same contention rules as any other element).
 *   - **passthrough OFF** → the same as the web default. The canvas blocks the UIKit hit test so scrolling
 *                           dies, and the canvas keeps receiving `touchmove`.
 *
 * The HUD's scroll offset and touch log show that difference in numbers — the simulator has no way to
 * inject touches, so the verdict comes from touching a device or the simulator by hand.
 */

/** A flowing diagonal band — seeing time pass means the canvas is alive and running independently of scrolling. */
const SHADER = /* wgsl */ `
struct Uniforms {
  time: f32,
  aspect: f32,
  _pad0: f32,
  _pad1: f32,
};
@group(0) @binding(0) var<uniform> u: Uniforms;

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOutput {
  var corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
    vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
  );
  let corner = corners[index];
  var out: VertexOutput;
  out.position = vec4f(corner, 0.0, 1.0);
  out.uv = corner * 0.5 + vec2f(0.5, 0.5);
  return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
  let stripe = 0.5 + 0.5 * sin((in.uv.x * u.aspect + in.uv.y) * 14.0 - u.time * 2.4);
  let pulse = 0.5 + 0.5 * sin(u.time * 0.8);
  let base = mix(vec3f(0.08, 0.13, 0.26), vec3f(0.16, 0.35, 0.68), stripe);
  let accent = vec3f(0.42, 0.86, 0.62) * pulse * fill_edge(in.uv.y);
  return vec4f(base + accent * 0.25, 1.0);
}

/// Brightens the band's top and bottom edges to reveal the canvas boundary.
fn fill_edge(y: f32) -> f32 {
  let edge = min(y, 1.0 - y);
  return 1.0 - smoothstep(0.0, 0.08, edge);
}
`

const ROWS = Array.from({ length: 40 }, (_, index) => index)

function ScrollPassScene() {
  const [passthrough, setPassthrough] = useState(true)
  const [status, setStatus] = useState('')
  const [scrollTop, setScrollTop] = useState(0)
  const [touchLog, setTouchLog] = useState('none yet')
  const touchCount = useRef(0)

  function logTouch(kind: string) {
    touchCount.current += 1
    setTouchLog(`${kind} · ${touchCount.current} total`)
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

      const module = device.createShaderModule({ code: SHADER, label: 'scrollpass' })
      const pipeline = device.createRenderPipeline({
        layout: 'auto',
        vertex: { module, entryPoint: 'vs_main' },
        fragment: { module, entryPoint: 'fs_main', targets: [{ format }] },
      })
      const uniformBuffer = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      })
      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
      })

      const uniforms = new Float32Array(4)
      let time = 0
      let size = context.getSize()
      let sizeCheck = 0

      stop = startFrameLoop(({ delta }: { delta: number }) => {
        if (disposed) return
        if (++sizeCheck >= 30) {
          sizeCheck = 0
          size = context.getSize()
        }
        if (size.width === 0 || size.height === 0) return

        time += delta / 1000
        uniforms[0] = time
        uniforms[1] = size.width / Math.max(size.height, 1)
        device.queue.writeBuffer(uniformBuffer, 0, uniforms)

        const encoder = device.createCommandEncoder()
        const pass = encoder.beginRenderPass({
          colorAttachments: [{
            view: context.getCurrentTexture().createView(),
            loadOp: 'clear',
            storeOp: 'store',
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
          }],
        })
        pass.setPipeline(pipeline)
        pass.setBindGroup(0, bindGroup)
        pass.draw(6)
        pass.end()
        device.queue.submit([encoder.finish()])
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
      <scroll-view
        scroll-y
        className="scroll-list"
        bindscroll={(event: any) => {
          const top = event?.detail?.scrollTop
          if (typeof top === 'number') setScrollTop(Math.round(top))
        }}
      >
        <view className="scroll-content">
          {ROWS.map((index) => (
            <view className="scroll-row" key={index}>
              <text className="scroll-row-text">
                {`row ${index + 1} — scrolling must continue behind the canvas band too`}
              </text>
            </view>
          ))}
        </view>
      </scroll-view>

      <webgpu-canvas
        canvas-id="main"
        className="overlay-canvas"
        passthrough-touches={passthrough}
        bindtouchstart={() => logTouch('touchstart')}
        bindtouchmove={() => logTouch('touchmove')}
        bindtouchend={() => logTouch('touchend')}
        bindtouchcancel={() => logTouch('touchcancel ← scrolling took it')}
      />

      <view className="hud">
        <text className="title">Scroll passthrough</text>
        <text className="subtitle">Drag the canvas band vertically — with passthrough ON the list moves</text>
        <text className="note">{`scroll offset: ${scrollTop}px`}</text>
        <text className="note">{`canvas Lynx events: ${touchLog}`}</text>
        {status ? <text className="status">{status}</text> : null}
      </view>
      <text className="badge">WebGPU on Lynx</text>

      <view className="controls">
        <view
          className={passthrough ? 'control-button control-button-on' : 'control-button'}
          bindtap={() => setPassthrough((value: boolean) => !value)}
        >
          <text>{`passthrough-touches: ${passthrough ? 'ON' : 'OFF'}`}</text>
        </view>
      </view>
    </view>
  )
}

root.render(<ScrollPassScene />)
