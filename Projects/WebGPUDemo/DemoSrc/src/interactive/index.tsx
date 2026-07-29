import { root, useEffect, useInitData, useRef, useState } from '@lynx-js/react'
import gpu, { GPUBufferUsage } from '../webgpu.js'
import { startFrameLoop } from '../webgpu.js'
import * as mat from '../matrix.js'
import '../demo.css'
import '../elements.d.ts'

/**
 * 홀로그래픽 트레이딩 카드 — 잡고 움직이면 포일 무늬가 각도를 따라 흐른다.
 *
 * 카드는 화면 중앙에 고정되고 **기울기만** 손가락을 따라간다 (실물 카드를 손에 들고
 * 빛에 비춰 보는 동작). 포일은 눈속임이 아니라 실제 3D 자세에서 계산한다 —
 * 프래그먼트마다 시선 벡터와 법선을 구해 그 각도로 무지개 띠·정반사·반짝임을 만든다.
 * 그래서 기울일 때 무늬가 "따라 도는" 게 아니라 **표면 위를 흐른다**.
 *
 * 위아래로 겹친 Lynx 컴포넌트는 그대로 두었다 — 입력 라우팅이 웹과 같은지
 * 눈으로 확인하는 장치다 (`docs/LYNX-INTEGRATION.md` §5).
 */

/** 실물 포켓몬 카드 비율 (63mm × 88mm). */
const CARD_ASPECT = 63 / 88

const CARD_SHADER = /* wgsl */ `
struct Uniforms {
  mvp: mat4x4<f32>,
  model: mat4x4<f32>,
  cameraPosition: vec3f,
  time: f32,
  halfSize: vec2f,
  held: f32,
  _pad: f32,
};
@group(0) @binding(0) var<uniform> u: Uniforms;

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
  @location(1) worldPosition: vec3f,
  @location(2) normal: vec3f,
};

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOutput {
  var corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
    vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
  );
  let corner = corners[index];
  let local = vec4f(corner.x * u.halfSize.x, corner.y * u.halfSize.y, 0.0, 1.0);

  var out: VertexOutput;
  out.position = u.mvp * local;
  out.uv = corner * 0.5 + vec2f(0.5, 0.5);
  out.worldPosition = (u.model * local).xyz;
  // 카드는 평면이라 법선이 상수다 — 모델 행렬로 돌리기만 하면 된다.
  out.normal = (u.model * vec4f(0.0, 0.0, 1.0, 0.0)).xyz;
  return out;
}

// ── 보조 ─────────────────────────────────────────────────────────────

/// 둥근 사각형까지의 부호 있는 거리.
fn rounded_box(point: vec2f, half: vec2f, radius: f32) -> f32 {
  let q = abs(point) - half + vec2f(radius, radius);
  return length(max(q, vec2f(0.0, 0.0))) + min(max(q.x, q.y), 0.0) - radius;
}

/// SDF를 0~1 채움으로 바꾼다 (경계에서 부드럽게).
fn fill(distance: f32, softness: f32) -> f32 {
  return 1.0 - smoothstep(-softness, softness, distance);
}

/// 코사인 팔레트 무지개 — HSV 변환보다 싸고 이어짐이 매끄럽다.
fn spectrum(t: f32) -> vec3f {
  return 0.5 + 0.5 * cos(6.2831853 * (vec3f(t) + vec3f(0.0, 0.33, 0.67)));
}

fn hash21(p: vec2f) -> f32 {
  var q = fract(p * vec2f(123.34, 456.21));
  q = q + vec2f(dot(q, q + 45.32));
  return fract(q.x * q.y);
}

/// 아트 창 안에 그리는 절차적 그림 (에너지 코어).
fn artwork(p: vec2f, time: f32) -> vec3f {
  let radius = length(p);
  let angle = atan2(p.y, p.x);
  let swirl = sin(angle * 5.0 + radius * 13.0 - time * 0.9);
  let core = exp(-radius * 3.4);

  var color = mix(vec3f(0.04, 0.08, 0.20), vec3f(0.16, 0.42, 0.86), exp(-radius * 1.4));
  color = color + vec3f(1.0, 0.82, 0.42) * core * 0.9;
  color = color + vec3f(0.30, 0.62, 1.00) * max(swirl, 0.0) * exp(-radius * 2.2) * 0.55;
  return color;
}

// ── 프래그먼트 ────────────────────────────────────────────────────────

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
  let aspect = u.halfSize.x / u.halfSize.y;
  // p.y ∈ [-1, 1], p.x ∈ [-aspect, aspect] — 카드 비율을 그대로 쓴다.
  let p = (in.uv - vec2f(0.5, 0.5)) * vec2f(2.0 * aspect, 2.0);

  let normal = normalize(in.normal);
  let viewDirection = normalize(u.cameraPosition - in.worldPosition);
  let lightDirection = normalize(vec3f(-0.30, 0.50, 0.81));
  // 정면일수록 1, 기울일수록 작아진다. 포일 무늬를 흐르게 하는 핵심 값.
  let facing = clamp(dot(normal, viewDirection), 0.0, 1.0);
  let halfway = normalize(viewDirection + lightDirection);

  // ── 1) 카드 인쇄면
  var color = vec3f(0.86, 0.72, 0.30);                       // 금색 테두리
  let borderNoise = hash21(floor(in.uv * 180.0)) * 0.04;
  color = color - vec3f(borderNoise);

  // 안쪽 판
  let innerPlate = fill(rounded_box(p, vec2f(aspect - 0.10, 0.90), 0.05), 0.006);
  color = mix(color, vec3f(0.10, 0.13, 0.22), innerPlate);

  // 아트 창
  let artBox = rounded_box(p - vec2f(0.0, 0.22), vec2f(aspect - 0.17, 0.42), 0.03);
  let artMask = fill(artBox, 0.005);
  color = mix(color, artwork((p - vec2f(0.0, 0.22)) * vec2f(1.6, 1.9), u.time), artMask);

  // 제목 바 / 설명 바 / 하단 텍스트 줄 — 카드처럼 보이게 하는 최소 요소
  let titleBar = fill(rounded_box(p - vec2f(0.0, 0.78), vec2f(aspect - 0.17, 0.09), 0.03), 0.005);
  color = mix(color, vec3f(0.20, 0.26, 0.42), titleBar);

  let statBox = fill(rounded_box(p - vec2f(0.0, -0.30), vec2f(aspect - 0.17, 0.14), 0.03), 0.005);
  color = mix(color, vec3f(0.14, 0.18, 0.30), statBox);

  for (var line = 0u; line < 4u; line = line + 1u) {
    let y = -0.55 - f32(line) * 0.10;
    let width = aspect - 0.22 - f32(line % 2u) * 0.12;
    let bar = fill(rounded_box(p - vec2f(0.0, y), vec2f(width, 0.018), 0.018), 0.004);
    color = mix(color, vec3f(0.42, 0.48, 0.62), bar * 0.75);
  }

  // ── 2) 홀로 포일
  // 포일은 인쇄면 전체가 아니라 아트 창 + 제목 바에만 입힌다 (실물 홀로 카드와 같다).
  let foilMask = clamp(artMask + titleBar * 0.7, 0.0, 1.0);

  // 시선각(facing)을 무늬 위상에 섞으면, 기울일 때 띠가 표면 위를 흐른다.
  let bandA = in.uv.x * 1.7 + in.uv.y * 0.9 + facing * 2.8 + u.time * 0.02;
  let bandB = in.uv.x * -3.3 + in.uv.y * 2.6 + facing * 5.2;
  let foil = spectrum(fract(bandA)) * 0.62 + spectrum(fract(bandB)) * 0.38;
  // 기울일수록 강해진다 — 정면에서는 은은하고, 눕히면 확 산다.
  let sheen = pow(1.0 - facing, 1.3);
  color = color + foil * foilMask * (0.38 + sheen * 1.6);

  // ── 3) 반짝이 — 셀 안의 작은 점 하나. 각도가 바뀔 때만 깜빡인다.
  let grid = in.uv * vec2f(44.0, 62.0);
  let cell = floor(grid);
  let seed = hash21(cell);
  // 셀마다 점 위치를 흩어 격자로 보이지 않게 한다.
  let jitter = (vec2f(hash21(cell + vec2f(3.7, 1.3)), hash21(cell + vec2f(9.1, 5.5))) - vec2f(0.5)) * 0.7;
  let speck = exp(-length(fract(grid) - vec2f(0.5) - jitter) * 22.0);
  let twinkle = pow(max(sin(seed * 44.0 + facing * 26.0 + u.time * 0.7), 0.0), 30.0);
  color = color + speck * twinkle * foilMask * vec3f(1.0, 0.97, 0.88) * 2.2;

  // ── 4) 라미네이트 정반사 — 광원이 표면에 비친 하이라이트
  let specular = pow(max(dot(normal, halfway), 0.0), 60.0);
  color = color + specular * vec3f(1.0, 0.98, 0.94) * (0.7 + u.held * 0.6);

  // ── 5) 가장자리 프레넬
  color = color + pow(1.0 - facing, 3.2) * vec3f(0.45, 0.60, 1.0) * 0.30;

  // 카드 바깥은 잘라 낸다 (둥근 모서리 + 안티에일리어싱).
  let alpha = fill(rounded_box(p, vec2f(aspect, 1.0), 0.10), 0.008);
  return vec4f(color, alpha);
}
`

const BACKGROUND_SHADER = /* wgsl */ `
struct Background {
  resolution: vec2f,
  shadowCenter: vec2f,
  shadowScale: vec2f,
  held: f32,
  time: f32,
};
@group(0) @binding(0) var<uniform> b: Background;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
  var corners = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  return vec4f(corners[index], 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) fragment: vec4f) -> @location(0) vec4f {
  let uv = fragment.xy / b.resolution;
  // 바탕 — 가운데가 살짝 밝은 무대 조명
  let toCenter = (uv - vec2f(0.5, 0.45)) * vec2f(b.resolution.x / b.resolution.y, 1.0);
  var color = mix(vec3f(0.10, 0.12, 0.19), vec3f(0.030, 0.035, 0.055), clamp(length(toCenter) * 1.1, 0.0, 1.0));

  // 카드 아래 그림자 — 들어 올리면 흐려지고 넓어진다
  let offset = (uv - b.shadowCenter) / b.shadowScale;
  let shadow = exp(-dot(offset, offset) * (2.6 - b.held * 0.9));
  color = color * (1.0 - shadow * (0.55 - b.held * 0.15));

  return vec4f(color, 1.0);
}
`

interface Tilt {
  x: number
  y: number
  velocityX: number
  velocityY: number
}

function HoloCardScene() {
  const [fps, setFps] = useState(0)
  const [status, setStatus] = useState('')
  const [routed, setRouted] = useState('카드를 잡고 기울여 보세요')

  // 렌더 루프가 매 프레임 읽는 값 — setState를 쓰면 리렌더가 붙으므로 ref에 둔다.
  const pointer = useRef({ x: 0.5, y: 0.5, held: false })
  // 하네스용 고정 기울기 (`-cardTilt` 런치 인자). 0이면 평소대로 터치를 따른다.
  const initData = useInitData() as { forceTilt?: number } | undefined
  const forcedTilt = useRef(0)
  forcedTilt.current = typeof initData?.forceTilt === 'number' ? initData.forceTilt : 0
  const canvasCss = useRef({ width: 1, height: 1 })

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

  function grab(event: any) {
    const point = normalize(event)
    if (!point) return
    pointer.current = { ...point, held: true }
    setRouted(`카드 — (${point.x.toFixed(2)}, ${point.y.toFixed(2)})`)
  }

  function move(event: any) {
    const point = normalize(event)
    if (!point) return
    pointer.current = { ...point, held: true }
  }

  function release() {
    pointer.current = { ...pointer.current, held: false }
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

      // ── 배경 (무대 + 그림자)
      const backgroundModule = device.createShaderModule({ code: BACKGROUND_SHADER, label: 'stage' })
      const backgroundBuffer = device.createBuffer({
        size: 32,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      })
      const backgroundPipeline = device.createRenderPipeline({
        layout: 'auto',
        vertex: { module: backgroundModule, entryPoint: 'vs_main' },
        fragment: { module: backgroundModule, entryPoint: 'fs_main', targets: [{ format }] },
      })
      const backgroundBind = device.createBindGroup({
        layout: backgroundPipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: backgroundBuffer } }],
      })

      // ── 카드
      const cardModule = device.createShaderModule({ code: CARD_SHADER, label: 'holo-card' })
      const cardBuffer = device.createBuffer({
        size: 160,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      })
      const cardPipeline = device.createRenderPipeline({
        layout: 'auto',
        vertex: { module: cardModule, entryPoint: 'vs_main' },
        fragment: {
          module: cardModule,
          entryPoint: 'fs_main',
          targets: [{
            format,
            // 둥근 모서리를 알파로 깎으므로 일반(비 premultiplied) 알파 합성.
            blend: {
              color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
              alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            },
          }],
        },
        primitive: { topology: 'triangle-list' },
      })
      const cardBind = device.createBindGroup({
        layout: cardPipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: cardBuffer } }],
      })

      const cardUniforms = new Float32Array(40)
      const backgroundUniforms = new Float32Array(8)
      const cameraZ = 3.4

      const tilt: Tilt = { x: 0, y: 0, velocityX: 0, velocityY: 0 }
      let time = 0
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

        const step = Math.min(delta, 33) / 1000
        time += step

        // ── 기울기 스프링
        // 잡고 있으면 손가락 쪽으로 기울고, 놓으면 평평하게 돌아온다.
        // 놓았을 때도 아주 약하게 흔들려 포일이 죽어 보이지 않게 한다.
        const forced = forcedTilt.current
        const targetY = forced !== 0
          ? forced
          : pointer.current.held
            ? (pointer.current.x - 0.5) * 0.85
            : Math.sin(time * 0.55) * 0.075
        const targetX = forced !== 0
          ? -forced * 0.66
          : pointer.current.held
            ? (pointer.current.y - 0.5) * 0.85
            : Math.sin(time * 0.4 + 1.0) * 0.05

        const stiffness = 150
        const damping = 17
        tilt.velocityX += ((targetX - tilt.x) * stiffness - tilt.velocityX * damping) * step
        tilt.velocityY += ((targetY - tilt.y) * stiffness - tilt.velocityY * damping) * step
        tilt.x += tilt.velocityX * step
        tilt.y += tilt.velocityY * step

        // ── 행렬
        const aspect = size.width / size.height
        // 카메라에서 본 화면의 실제 크기로 카드를 맞춘다 — 가로/세로 어느 쪽으로도 넘치지 않게.
        const fovY = Math.PI / 4.4
        const viewHalfHeight = Math.tan(fovY / 2) * cameraZ
        const viewHalfWidth = viewHalfHeight * aspect
        const halfHeight = Math.min(viewHalfHeight * 0.72, (viewHalfWidth * 0.80) / CARD_ASPECT)
        const halfWidth = halfHeight * CARD_ASPECT
        const lift = pointer.current.held || forced !== 0 ? 0.22 : 0

        const projection = mat.perspective(fovY, aspect, 0.1, 100)
        const view = mat.translation(0, 0, -cameraZ)
        const model = mat.multiplyAll(
          mat.translation(0, 0, lift),
          mat.rotationY(tilt.y),
          mat.rotationX(tilt.x)
        )
        const mvp = mat.multiplyAll(projection, view, model)

        cardUniforms.set(mvp, 0)
        cardUniforms.set(model, 16)
        cardUniforms[32] = 0
        cardUniforms[33] = 0
        cardUniforms[34] = cameraZ         // cameraPosition
        cardUniforms[35] = time
        cardUniforms[36] = halfWidth       // halfSize
        cardUniforms[37] = halfHeight
        cardUniforms[38] = pointer.current.held ? 1 : 0
        cardUniforms[39] = 0
        device.queue.writeBuffer(cardBuffer, 0, cardUniforms)

        // 그림자: 카드 중심을 화면에 투영해 그 아래에 깐다.
        const center = mat.project(mvp, 0, -0.12, 0)
        backgroundUniforms[0] = size.width
        backgroundUniforms[1] = size.height
        backgroundUniforms[2] = center.x * 0.5 + 0.5
        backgroundUniforms[3] = -center.y * 0.5 + 0.5 + 0.06
        backgroundUniforms[4] = (halfWidth / viewHalfWidth) * 0.62
        backgroundUniforms[5] = (halfHeight / viewHalfHeight) * 0.42
        backgroundUniforms[6] = pointer.current.held ? 1 : 0
        backgroundUniforms[7] = time
        device.queue.writeBuffer(backgroundBuffer, 0, backgroundUniforms)

        const encoder = device.createCommandEncoder()
        const pass = encoder.beginRenderPass({
          colorAttachments: [{
            view: context.getCurrentTexture().createView(),
            loadOp: 'clear',
            storeOp: 'store',
            clearValue: { r: 0.03, g: 0.035, b: 0.055, a: 1 },
          }],
        })
        pass.setPipeline(backgroundPipeline)
        pass.setBindGroup(0, backgroundBind)
        pass.draw(3)

        pass.setPipeline(cardPipeline)
        pass.setBindGroup(0, cardBind)
        pass.draw(6)
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
      <webgpu-canvas
        canvas-id="main"
        className="canvas"
        bindcanvasresize={(event) => {
          const { width, height, pixelRatio } = event.detail
          canvasCss.current = { width: width / pixelRatio, height: height / pixelRatio }
        }}
        bindtouchstart={grab}
        bindtouchmove={move}
        bindtouchend={release}
        bindtouchcancel={release}
      />

      {/* 캔버스 위에 겹친 평범한 Lynx 카드 — 여길 누르면 카드가 기울면 안 된다 */}
      <view className="layer-card" bindtap={() => setRouted('겹친 Lynx 카드 (캔버스로 안 감)')}>
        <text className="layer-title">겹친 Lynx 카드</text>
        <text className="layer-hint">여길 누르면 카드가 안 움직여야 한다</text>
      </view>

      <view className="hud">
        <text className="title">홀로그래픽 카드</text>
        <text className="subtitle">Lynx 터치 → 3D 기울기 → 포일 · {fps} fps</text>
        <text className="note">마지막 입력: {routed}</text>
        {status ? <text className="status">{status}</text> : null}
      </view>

      <view className="bottom-bar" bindtap={() => setRouted('아래쪽 Lynx 바 (캔버스로 안 감)')}>
        <text className="bottom-text">아래쪽 Lynx 바 — 탭해 보세요</text>
      </view>
    </view>
  )
}

root.render(<HoloCardScene />)
