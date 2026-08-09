import { root, useEffect, useInitData, useRef, useState } from '@lynx-js/react'
import gpu, { GPUBufferUsage } from '../webgpu.js'
import { startFrameLoop } from '../webgpu.js'
import * as mat from '../matrix.js'
import '../demo.css'
import '../elements.d.ts'

/**
 * A holographic trading card — grab and move it and the foil pattern flows with the angle.
 *
 * The card stays fixed at the center of the screen and **only its tilt** follows the finger (the motion of
 * holding a real card up to the light). The foil is not a trick but computed from the actual 3D pose —
 * the view vector and normal are found per fragment and the angle between them makes the rainbow bands,
 * specular and glitter. So when you tilt it the pattern does not "turn with" the card but **flows across the surface**.
 *
 * The Lynx components overlapping above and below are left as they are — a device for verifying by eye
 * that input routing matches the web (`docs/LYNX-INTEGRATION.md` §5).
 */

/** The real Pokémon card ratio (63mm × 88mm). */
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
  // The card is flat, so the normal is constant — it only needs rotating by the model matrix.
  out.normal = (u.model * vec4f(0.0, 0.0, 1.0, 0.0)).xyz;
  return out;
}

// ── Helpers ──────────────────────────────────────────────────────────

/// The signed distance to a rounded rectangle.
fn rounded_box(point: vec2f, half: vec2f, radius: f32) -> f32 {
  let q = abs(point) - half + vec2f(radius, radius);
  return length(max(q, vec2f(0.0, 0.0))) + min(max(q.x, q.y), 0.0) - radius;
}

/// Turns an SDF into a 0~1 fill (soft at the boundary).
fn fill(distance: f32, softness: f32) -> f32 {
  return 1.0 - smoothstep(-softness, softness, distance);
}

/// A cosine-palette rainbow — cheaper than an HSV conversion and smoother in its continuity.
fn spectrum(t: f32) -> vec3f {
  return 0.5 + 0.5 * cos(6.2831853 * (vec3f(t) + vec3f(0.0, 0.33, 0.67)));
}

fn hash21(p: vec2f) -> f32 {
  var q = fract(p * vec2f(123.34, 456.21));
  q = q + vec2f(dot(q, q + 45.32));
  return fract(q.x * q.y);
}

/// The procedural picture drawn inside the art window (an energy core).
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

// ── Fragment ─────────────────────────────────────────────────────────

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
  let aspect = u.halfSize.x / u.halfSize.y;
  // p.y ∈ [-1, 1], p.x ∈ [-aspect, aspect] — the card ratio is used as is.
  let p = (in.uv - vec2f(0.5, 0.5)) * vec2f(2.0 * aspect, 2.0);

  let normal = normalize(in.normal);
  let viewDirection = normalize(u.cameraPosition - in.worldPosition);
  let lightDirection = normalize(vec3f(-0.30, 0.50, 0.81));
  // 1 head on, smaller as it tilts. The key value that makes the foil pattern flow.
  let facing = clamp(dot(normal, viewDirection), 0.0, 1.0);
  let halfway = normalize(viewDirection + lightDirection);

  // ── 1) The printed face of the card
  var color = vec3f(0.86, 0.72, 0.30);                       // the gold border
  let borderNoise = hash21(floor(in.uv * 180.0)) * 0.04;
  color = color - vec3f(borderNoise);

  // The inner panel
  let innerPlate = fill(rounded_box(p, vec2f(aspect - 0.10, 0.90), 0.05), 0.006);
  color = mix(color, vec3f(0.10, 0.13, 0.22), innerPlate);

  // The art window
  let artBox = rounded_box(p - vec2f(0.0, 0.22), vec2f(aspect - 0.17, 0.42), 0.03);
  let artMask = fill(artBox, 0.005);
  color = mix(color, artwork((p - vec2f(0.0, 0.22)) * vec2f(1.6, 1.9), u.time), artMask);

  // The title bar / description bar / bottom text lines — the minimum that makes it read as a card
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

  // ── 2) The holo foil
  // The foil covers only the art window and title bar rather than the whole printed face (as on a real holo card).
  let foilMask = clamp(artMask + titleBar * 0.7, 0.0, 1.0);

  // Mixing the view angle (facing) into the pattern phase makes the bands flow across the surface as it tilts.
  let bandA = in.uv.x * 1.7 + in.uv.y * 0.9 + facing * 2.8 + u.time * 0.02;
  let bandB = in.uv.x * -3.3 + in.uv.y * 2.6 + facing * 5.2;
  let foil = spectrum(fract(bandA)) * 0.62 + spectrum(fract(bandB)) * 0.38;
  // It strengthens as it tilts — subtle head on, vivid laid flat.
  let sheen = pow(1.0 - facing, 1.3);
  color = color + foil * foilMask * (0.38 + sheen * 1.6);

  // ── 3) Glitter — one small dot per cell. It only flickers as the angle changes.
  let grid = in.uv * vec2f(44.0, 62.0);
  let cell = floor(grid);
  let seed = hash21(cell);
  // The dot positions are scattered per cell so it does not read as a grid.
  let jitter = (vec2f(hash21(cell + vec2f(3.7, 1.3)), hash21(cell + vec2f(9.1, 5.5))) - vec2f(0.5)) * 0.7;
  let speck = exp(-length(fract(grid) - vec2f(0.5) - jitter) * 22.0);
  let twinkle = pow(max(sin(seed * 44.0 + facing * 26.0 + u.time * 0.7), 0.0), 30.0);
  color = color + speck * twinkle * foilMask * vec3f(1.0, 0.97, 0.88) * 2.2;

  // ── 4) The laminate specular — the highlight of the light source on the surface
  let specular = pow(max(dot(normal, halfway), 0.0), 60.0);
  color = color + specular * vec3f(1.0, 0.98, 0.94) * (0.7 + u.held * 0.6);

  // ── 5) Edge Fresnel
  color = color + pow(1.0 - facing, 3.2) * vec3f(0.45, 0.60, 1.0) * 0.30;

  // Everything outside the card is cut away (rounded corners plus antialiasing).
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
  // The backdrop — stage lighting, slightly brighter in the middle
  let toCenter = (uv - vec2f(0.5, 0.45)) * vec2f(b.resolution.x / b.resolution.y, 1.0);
  var color = mix(vec3f(0.10, 0.12, 0.19), vec3f(0.030, 0.035, 0.055), clamp(length(toCenter) * 1.1, 0.0, 1.0));

  // The shadow beneath the card — it blurs and widens as the card is lifted
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
  const [routed, setRouted] = useState('grab the card and tilt it')

  // A value the render loop reads every frame — setState would attach a re-render, so it lives in a ref.
  const pointer = useRef({ x: 0.5, y: 0.5, held: false })
  // A fixed tilt for the harness (the `-cardTilt` launch argument). 0 follows the touch as usual.
  const initData = useInitData() as { forceTilt?: number } | undefined
  const forcedTilt = useRef(0)
  forcedTilt.current = typeof initData?.forceTilt === 'number' ? initData.forceTilt : 0
  const canvasCss = useRef({ width: 1, height: 1 })

  /** Normalizes a Lynx touch event's element-relative coordinates (CSS px) to 0~1. */
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
    setRouted(`card — (${point.x.toFixed(2)}, ${point.y.toFixed(2)})`)
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
      if (!adapter) throw new Error('no WebGPU adapter')
      device = await adapter.requestDevice()
      device.onError((_error: any, text: string) => setStatus(text))

      const context = gpu.getCanvasContext('main')
      const format = gpu.getPreferredCanvasFormat()
      context.configure({ device, format })

      // ── The background (stage plus shadow)
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

      // ── The card
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
            // The rounded corners are carved out with alpha, so ordinary (non-premultiplied) alpha compositing.
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

        // ── The tilt spring
        // Held, it tilts toward the finger; released, it returns flat.
        // It also sways very faintly when released, so the foil does not look dead.
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

        // ── Matrices
        const aspect = size.width / size.height
        // The card is fitted to the screen's actual size as seen from the camera — so it overflows neither horizontally nor vertically.
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

        // The shadow: the card's center is projected onto the screen and laid beneath it.
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

      {/* An ordinary Lynx card overlapping the canvas — pressing here must not tilt the card */}
      <view className="layer-card" bindtap={() => setRouted('the overlapping Lynx card (it never reached the canvas)')}>
        <text className="layer-title">Overlapping Lynx card</text>
        <text className="layer-hint">pressing here must not move the card</text>
      </view>

      <view className="hud">
        <text className="title">Holographic card</text>
        <text className="subtitle">Lynx touch → 3D tilt → foil · {fps} fps</text>
        <text className="note">last input: {routed}</text>
        {status ? <text className="status">{status}</text> : null}
      </view>

      <view className="bottom-bar" bindtap={() => setRouted('the bottom Lynx bar (it never reached the canvas)')}>
        <text className="bottom-text">The bottom Lynx bar — try tapping it</text>
      </view>
    </view>
  )
}

root.render(<HoloCardScene />)
