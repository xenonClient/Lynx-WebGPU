import { root, useEffect, useRef, useState } from '@lynx-js/react'
import gpu, { GPUBufferUsage, startFrameLoop } from '../webgpu.js'
import '../demo.css'
import '../elements.d.ts'

/**
 * 셰이더를 입힌 인터랙티브 표면 + 그 위아래에 놓인 평범한 Lynx 컴포넌트.
 *
 * 확인하려는 것 두 가지:
 * 1. 터치가 셰이더 유니폼까지 흘러가 프레임에 반영되는가
 * 2. **캔버스 위에 겹친 Lynx 컴포넌트가 터치를 먼저 가져가는가** (웹의 z-order/pointer-events와 같은가)
 *
 * 터치는 `<webgpu-canvas>`가 따로 만든 이벤트가 아니라 **Lynx 표준 터치 이벤트**를 쓴다.
 * 그래야 히트 테스트·버블링·`catch` 접두사·스크롤 제스처가 전부 Lynx 규칙(=웹 규칙)을 따른다.
 */

const RIPPLES = 6

const SHADER = /* wgsl */ `
struct Ripple {
  origin: vec2f,
  start: f32,
  _pad: f32,
};

struct Uniforms {
  resolution: vec2f,
  time: f32,
  pressed: f32,
  pointer: vec2f,
  _pad: vec2f,
  ripples: array<Ripple, ${RIPPLES}>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
  var corners = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  return vec4f(corners[index], 0.0, 1.0);
}

/// 둥근 사각형까지의 부호 있는 거리 (음수면 안쪽).
fn rounded_box(point: vec2f, half: vec2f, radius: f32) -> f32 {
  let q = abs(point) - half + vec2f(radius, radius);
  return length(max(q, vec2f(0.0, 0.0))) + min(max(q.x, q.y), 0.0) - radius;
}

@fragment
fn fs_main(@builtin(position) fragment: vec4f) -> @location(0) vec4f {
  let pixel = fragment.xy;
  let uv = pixel / u.resolution;
  let short = min(u.resolution.x, u.resolution.y);

  // 카드 모양으로 잘라 낸다 — 화면 전체가 아니라 "컴포넌트"로 보이게.
  let center = u.resolution * 0.5;
  let half = center - vec2f(u.resolution.x * 0.06, u.resolution.y * 0.05);
  let distanceToCard = rounded_box(pixel - center, half, short * 0.09);
  let inside = smoothstep(1.5, -1.5, distanceToCard);

  // 천천히 흐르는 바탕 그라디언트.
  let flow = uv.y + 0.18 * sin(uv.x * 3.4 + u.time * 0.5);
  var surface = mix(vec3f(0.09, 0.16, 0.38), vec3f(0.40, 0.15, 0.50), clamp(flow, 0.0, 1.0));
  surface = surface + 0.06 * vec3f(
    sin(u.time * 0.7 + uv.x * 3.0),
    sin(u.time * 0.9 + uv.y * 3.0),
    sin(u.time * 1.1)
  );

  // 누르고 있는 동안 손가락을 따라다니는 하이라이트.
  let pointerPixel = u.pointer * u.resolution;
  let glow = exp(-length(pixel - pointerPixel) / (short * 0.18));
  surface = surface + glow * u.pressed * vec3f(0.5, 0.62, 0.95);

  // 누른 자리에서 퍼져 나가는 물결.
  for (var i = 0u; i < ${RIPPLES}u; i = i + 1u) {
    let ripple = u.ripples[i];
    let age = u.time - ripple.start;
    if (ripple.start <= 0.0 || age < 0.0 || age > 1.8) {
      continue;
    }
    let radius = age * short * 0.9;
    let width = short * 0.025;
    let ring = exp(-abs(length(pixel - ripple.origin * u.resolution) - radius) / width);
    let fade = 1.0 - age / 1.8;
    surface = surface + ring * fade * fade * vec3f(0.85, 0.93, 1.0) * 0.9;
  }

  // 카드 테두리 하이라이트.
  let rim = exp(-abs(distanceToCard) / 2.0) * 0.6;
  var color = mix(vec3f(0.043, 0.055, 0.08), surface, inside);
  color = color + vec3f(0.4, 0.5, 0.75) * rim * inside;

  return vec4f(color, 1.0);
}
`

interface Pointer {
  x: number
  y: number
  pressed: boolean
}

function InteractiveScene() {
  const [fps, setFps] = useState(0)
  const [status, setStatus] = useState('')
  // 어떤 엘리먼트가 마지막 입력을 가져갔는지 — 레이어링 확인용.
  const [routed, setRouted] = useState('아직 없음')

  // 터치 상태는 렌더 루프가 매 프레임 읽으므로 ref에 둔다 (setState는 리렌더를 부른다).
  const pointer = useRef<Pointer>({ x: 0.5, y: 0.5, pressed: false })
  const ripples = useRef<Array<{ x: number; y: number; start: number }>>([])
  const canvasCss = useRef({ width: 1, height: 1 })
  const elapsed = useRef(0)

  /** Lynx 터치 이벤트의 엘리먼트 기준 좌표(CSS px)를 0~1로 정규화한다. */
  function normalize(event: any) {
    const touch = event?.touches?.[0] ?? event?.changedTouches?.[0]
    if (!touch) return null
    const { width, height } = canvasCss.current
    return {
      x: Math.min(Math.max(touch.x / Math.max(width, 1), 0), 1),
      y: Math.min(Math.max(touch.y / Math.max(height, 1), 0), 1),
    }
  }

  function handleTouchStart(event: any) {
    const point = normalize(event)
    if (!point) return
    pointer.current = { ...point, pressed: true }
    ripples.current.push({ ...point, start: elapsed.current })
    if (ripples.current.length > RIPPLES) ripples.current.shift()
    setRouted(`캔버스 — (${point.x.toFixed(2)}, ${point.y.toFixed(2)})`)
  }

  function handleTouchMove(event: any) {
    const point = normalize(event)
    if (!point) return
    pointer.current = { ...point, pressed: true }
  }

  function handleTouchEnd() {
    pointer.current = { ...pointer.current, pressed: false }
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

      const module = device.createShaderModule({ code: SHADER, label: 'interactive' })
      // resolution(8) + time(4) + pressed(4) + pointer(8) + pad(8) + ripples(6 × 16) = 128B
      const uniformBuffer = device.createBuffer({
        size: 128,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      })
      const pipeline = device.createRenderPipeline({
        layout: 'auto',
        vertex: { module, entryPoint: 'vs_main' },
        fragment: { module, entryPoint: 'fs_main', targets: [{ format }] },
      })
      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
      })

      const uniforms = new Float32Array(32)
      let size = context.getSize()
      let sizeCheck = 0
      let frames = 0
      let accumulated = 0

      stop = startFrameLoop(({ delta }: { delta: number }) => {
        if (disposed) return
        if (++sizeCheck >= 30) {
          sizeCheck = 0
          size = context.getSize()
        }
        if (size.width === 0 || size.height === 0) return

        elapsed.current += delta / 1000
        uniforms[0] = size.width
        uniforms[1] = size.height
        uniforms[2] = elapsed.current
        uniforms[3] = pointer.current.pressed ? 1 : 0
        uniforms[4] = pointer.current.x
        uniforms[5] = pointer.current.y

        for (let index = 0; index < RIPPLES; index++) {
          const base = 8 + index * 4
          const ripple = ripples.current[index]
          uniforms[base + 0] = ripple ? ripple.x : 0
          uniforms[base + 1] = ripple ? ripple.y : 0
          uniforms[base + 2] = ripple ? ripple.start : -1
          uniforms[base + 3] = 0
        }
        device.queue.writeBuffer(uniformBuffer, 0, uniforms)

        const encoder = device.createCommandEncoder()
        const pass = encoder.beginRenderPass({
          colorAttachments: [{
            view: context.getCurrentTexture().createView(),
            loadOp: 'clear',
            storeOp: 'store',
            clearValue: { r: 0.043, g: 0.055, b: 0.08, a: 1 },
          }],
        })
        pass.setPipeline(pipeline)
        pass.setBindGroup(0, bindGroup)
        pass.draw(3)
        pass.end()
        device.queue.submit([encoder.finish()])

        frames += 1
        accumulated += delta
        if (accumulated >= 1000) {
          setFps(Math.round((frames * 1000) / accumulated))
          frames = 0
          accumulated = 0
        }
      })
    }

    boot().catch((error) => setStatus(String(error?.message ?? error)))
    return () => {
      disposed = true
      if (stop) stop()
      if (device) device.destroy()
    }
  }, [])

  return (
    <view className="page">
      {/* 셰이더 표면 — 페이지 전체를 덮는다 */}
      <webgpu-canvas
        canvas-id="main"
        className="canvas"
        bindcanvasresize={(event) => {
          const { width, height, pixelRatio } = event.detail
          canvasCss.current = { width: width / pixelRatio, height: height / pixelRatio }
        }}
        bindtouchstart={handleTouchStart}
        bindtouchmove={handleTouchMove}
        bindtouchend={handleTouchEnd}
        bindtouchcancel={handleTouchEnd}
      />

      {/* 캔버스 "위"에 겹친 평범한 Lynx 컴포넌트 — 여길 누르면 캔버스로 가면 안 된다 */}
      <view
        className="layer-card"
        bindtap={() => setRouted('겹친 Lynx 카드 (캔버스로 안 감)')}
      >
        <text className="layer-title">겹친 Lynx 카드</text>
        <text className="layer-hint">여길 누르면 물결이 생기지 않아야 한다</text>
      </view>

      {/* 화면 위쪽 컴포넌트 */}
      <view className="hud">
        <text className="title">인터랙티브 표면</text>
        <text className="subtitle">Lynx 표준 터치 → 셰이더 유니폼 · {fps} fps</text>
        <text className="note">마지막 입력: {routed}</text>
        {status ? <text className="status">{status}</text> : null}
      </view>

      {/* 화면 아래쪽 컴포넌트 — 캔버스를 덮고 있으므로 여기도 터치를 먼저 가져간다 */}
      <view className="bottom-bar" bindtap={() => setRouted('아래쪽 Lynx 바 (캔버스로 안 감)')}>
        <text className="bottom-text">아래쪽 Lynx 바 — 탭해 보세요</text>
      </view>
    </view>
  )
}

root.render(<InteractiveScene />)
