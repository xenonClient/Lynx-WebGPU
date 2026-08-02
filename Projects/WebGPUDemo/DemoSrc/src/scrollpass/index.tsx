import { root, useEffect, useRef, useState } from '@lynx-js/react'
import gpu, { GPUBufferUsage, startFrameLoop } from '../webgpu.js'
import '../demo.css'
import '../elements.d.ts'

/**
 * 스크롤 통과 검증 — `<scroll-view>` **위에** 캔버스가 형제로 겹칠 때,
 * `passthrough-touches` 히트 테스트 처리만으로 스크롤이 되는지 확인하는 화면.
 *
 * 캔버스 밴드를 세로로 드래그해 본다:
 *   - **통과 ON**  → 리스트가 스크롤되고, 캔버스는 `touchstart` 후 스크롤이 이기는 순간
 *                    `touchcancel`을 받는다 (다른 엘리먼트와 같은 경쟁 규칙).
 *   - **통과 OFF** → 웹 기본과 같다. 캔버스가 UIKit 히트 테스트를 막아 스크롤이 죽고,
 *                    캔버스는 `touchmove`를 계속 받는다.
 *
 * HUD의 스크롤 오프셋·터치 로그가 그 차이를 숫자로 보여 준다 — 시뮬레이터에는 터치 주입
 * 수단이 없으므로 판정은 실기기/시뮬레이터를 손으로 만져서 한다.
 */

/** 흐르는 대각선 띠 — 시간이 보이면 캔버스가 살아 있고, 스크롤과 무관하게 돈다는 뜻. */
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

/// 밴드 위·아래 가장자리를 밝혀 캔버스 경계를 드러낸다.
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
  const [touchLog, setTouchLog] = useState('아직 없음')
  const touchCount = useRef(0)

  function logTouch(kind: string) {
    touchCount.current += 1
    setTouchLog(`${kind} · 누적 ${touchCount.current}건`)
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
                {`행 ${index + 1} — 캔버스 밴드 뒤에서도 스크롤이 이어져야 한다`}
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
        bindtouchcancel={() => logTouch('touchcancel ← 스크롤이 가져감')}
      />

      <view className="hud">
        <text className="title">스크롤 통과</text>
        <text className="subtitle">캔버스 밴드를 세로로 드래그 — 통과 ON이면 리스트가 움직인다</text>
        <text className="note">{`스크롤 오프셋: ${scrollTop}px`}</text>
        <text className="note">{`캔버스 Lynx 이벤트: ${touchLog}`}</text>
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
