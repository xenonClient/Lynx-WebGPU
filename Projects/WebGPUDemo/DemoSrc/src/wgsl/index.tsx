import { root } from '@lynx-js/react'
import { DemoScene, type SceneContext } from '../scene.jsx'
import { GPUBufferUsage, GPUMapMode, GPUTextureUsage } from '../webgpu.js'

/** The number of palette colors. The shader does not know this value — it counts with `arrayLength()`. */
const PALETTE = 7
const FRAME_SIZE = 8

/**
 * Animates the palette.
 *
 * The count is not taken as a uniform but measured with `arrayLength(&palette)`. A wrong buffer size table,
 * which the runtime plugs into a reserved index, goes wrong right here (the CPU checks it via `info[0]`).
 */
const COMPUTE_SHADER = /* wgsl */ `
struct Params {
  time: f32,
};
@group(0) @binding(0) var<storage, read_write> palette: array<vec4f>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> info: array<u32>;

@compute @workgroup_size(8)
fn animate(@builtin(global_invocation_id) id: vec3u) {
  let count = arrayLength(&palette);
  if (id.x == 0u) {
    info[0] = count;
    info[1] = arrayLength(&info);
  }
  if (id.x >= count) {
    return;
  }

  let t = f32(id.x) / f32(count);
  let phase = params.time * 0.3 + t;
  palette[id.x] = vec4f(
    0.5 + 0.5 * sin(6.2831853 * (phase + 0.00)),
    0.5 + 0.5 * sin(6.2831853 * (phase + 0.33)),
    0.5 + 0.5 * sin(6.2831853 * (phase + 0.67)),
    1.0,
  );
}
`

/**
 * Splits the screen into three bands to show three features at once.
 *
 * Top: an untyped constant expression (AbstractInt) and an array sized by a function-local `const`.
 * Middle: the same texture, sampled on the left with `textureSampleBaseClampToEdge` (an external texture) and on the right normally.
 *         The coordinates are deliberately pushed outside [0,1] on a repeat sampler, so **only the right side wraps the opposite edge in**.
 * Bottom: the number of cells `arrayLength()` decided — a wrong length makes the cell count visibly differ.
 */
const RENDER_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> palette: array<vec4f>;
@group(0) @binding(1) var frame: texture_external;
@group(0) @binding(2) var frameSampler: sampler;
@group(0) @binding(3) var frame2d: texture_2d<f32>;

// A constant expression with no type written is an abstract integer (AbstractInt) in WGSL — it settles into the type of where it is used.
const HIGHLIGHT = vec3(1);
const TILES = vec2(8, 2);

struct Out {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> Out {
  var corners = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var out: Out;
  out.position = vec4f(corners[index], 0.0, 1.0);
  out.uv = corners[index] * vec2f(0.5, -0.5) + vec2f(0.5, 0.5);
  return out;
}

// A const declared inside a function is a compile-time constant too — it can be used as an array size.
fn ramp(t: f32) -> vec3f {
  const STEPS = 5u;
  var stops: array<vec3f, STEPS>;
  // Dark at the top where the HUD sits, brighter toward the bottom — so the white text stays readable.
  stops[0] = vec3f(0.06, 0.07, 0.13);
  stops[1] = vec3f(0.09, 0.16, 0.31);
  stops[2] = vec3f(0.12, 0.31, 0.36);
  stops[3] = vec3f(0.85, 0.55, 0.25);
  stops[4] = vec3f(0.95, 0.34, 0.40);
  let index = min(u32(t * f32(STEPS)), STEPS - 1u);
  return stops[index];
}

// uv * TILES — an abstract integer vector settles to f32 in an f32 context.
fn checker(uv: vec2f) -> f32 {
  let cell = floor(uv * TILES);
  return select(0.82, 1.0, (u32(cell.x) + u32(cell.y)) % 2u == 0u);
}

@fragment
fn fs_main(in: Out) -> @location(0) vec4f {
  let uv = in.uv;

  // Middle: the same texture and coordinates, sampled with edge clamping on the left only.
  // The samples are taken first, **outside the branch (in uniform control flow)** — a textureSample inside
  // a varying branch is a uniformity violation and the spec validator (Dawn/browsers) rejects the pipeline.
  let local = vec2f(fract(uv.x * 2.0), (uv.y - 0.40) / 0.36);
  // Deliberately pushed outside [0,1] — being a repeat sampler, without clamping the opposite edge wraps in.
  let coord = local * 1.30 - vec2f(0.15, 0.15);
  let clamped = textureSampleBaseClampToEdge(frame, frameSampler, coord);
  let wrapped = textureSample(frame2d, frameSampler, coord);

  // Top: the untyped constant expression. The dark side is put at the top because the HUD sits there.
  if (uv.y < 0.40) {
    let t = uv.y / 0.40;
    let base = ramp(t) * checker(vec2f(uv.x, t));
    // HIGHLIGHT is vec3(1) — the same constant settles to f32 in an f32 vector slot.
    return vec4f(mix(base, HIGHLIGHT, 0.06), 1.0);
  }

  // The middle band — it picks between the two samples taken above.
  if (uv.y < 0.76 && uv.y >= 0.40) {
    // A thin line at the boundary of the two panels, to make the left/right comparison obvious.
    if (abs(uv.x - 0.5) < 0.004) {
      return vec4f(0.04, 0.05, 0.08, 1.0);
    }
    return select(wrapped, clamped, uv.x < 0.5);
  }

  // Bottom: the number of cells arrayLength() decided — a wrong length changes the cell count.
  let count = arrayLength(&palette);
  let index = min(u32(uv.x * f32(count)), count - 1u);
  let edge = fract(uv.x * f32(count));
  let seam = smoothstep(0.0, 0.04, edge) * smoothstep(1.0, 0.96, edge);
  return vec4f(palette[index].rgb * mix(0.5, 1.0, seam), 1.0);
}
`

/** An 8x8 "video frame". The edge columns/rows are painted in contrasting colors so a seam shows immediately. */
function makeFrame(step: number): Uint8Array {
  const pixels = new Uint8Array(FRAME_SIZE * FRAME_SIZE * 4)
  const highlight = step % FRAME_SIZE
  for (let y = 0; y < FRAME_SIZE; y += 1) {
    for (let x = 0; x < FRAME_SIZE; x += 1) {
      const offset = (y * FRAME_SIZE + x) * 4
      let r = 40 + Math.round((x / (FRAME_SIZE - 1)) * 60)
      let g = 90 + Math.round((y / (FRAME_SIZE - 1)) * 90)
      let b = 150 - Math.round((x / (FRAME_SIZE - 1)) * 40)
      if (x === highlight) {
        r = 250
        g = 240
        b = 180
      }
      if (x === 0) {
        r = 255
        g = 40
        b = 200
      } // left edge: magenta
      if (x === FRAME_SIZE - 1) {
        r = 30
        g = 220
        b = 120
      } // right edge: green
      if (y === 0) {
        g = Math.min(255, g + 60)
      }
      if (y === FRAME_SIZE - 1) {
        r = Math.round(r * 0.35)
        g = Math.round(g * 0.35)
        b = Math.round(b * 0.35)
      }
      pixels[offset] = r
      pixels[offset + 1] = g
      pixels[offset + 2] = b
      pixels[offset + 3] = 255
    }
  }
  return pixels
}

function setup({ device, context, format, report }: SceneContext) {
  const computeModule = device.createShaderModule({ code: COMPUTE_SHADER, label: 'wgsl.animate' })
  const renderModule = device.createShaderModule({ code: RENDER_SHADER, label: 'wgsl.render' })

  const palette = device.createBuffer({
    size: PALETTE * 16,
    usage: GPUBufferUsage.STORAGE,
    label: 'palette',
  })
  const params = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })
  const info = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    label: 'info',
  })
  // MAP_READ can only combine with COPY_DST (spec) — the readback goes into a dedicated staging buffer.
  const infoStaging = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    label: 'info.staging',
  })

  // An ordinary texture view is bound into the external texture slot (the same shape as a single-plane video frame).
  const frame = device.createTexture({
    size: { width: FRAME_SIZE, height: FRAME_SIZE },
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    label: 'frame',
  })
  const frameView = frame.createView()
  // A repeat sampler — without clamping, the opposite edge wraps in.
  const frameSampler = device.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
    addressModeU: 'repeat',
    addressModeV: 'repeat',
  })

  const computePipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module: computeModule, entryPoint: 'animate' },
  })
  const computeBindGroup = device.createBindGroup({
    layout: computePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: palette } },
      { binding: 1, resource: { buffer: params } },
      { binding: 2, resource: { buffer: info } },
    ],
  })

  const renderPipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module: renderModule, entryPoint: 'vs_main' },
    fragment: { module: renderModule, entryPoint: 'fs_main', targets: [{ format }] },
  })
  const renderBindGroup = device.createBindGroup({
    layout: renderPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: palette } },
      { binding: 1, resource: frameView },
      { binding: 2, resource: frameSampler },
      { binding: 3, resource: frameView },
    ],
  })

  const paramsData = new Float32Array(4)
  let time = 0
  let frames = 0
  let reading = false
  let reported = false

  return ({ delta }: { delta: number }) => {
    time += delta / 1000
    paramsData[0] = time
    device.queue.writeBuffer(params, 0, paramsData)

    // The frame is refreshed so it flows like "video" (8x8, so 256 bytes).
    if (frames % 6 === 0) {
      device.queue.writeTexture(
        { texture: frame },
        makeFrame(Math.floor(frames / 6)),
        { bytesPerRow: FRAME_SIZE * 4 },
        { width: FRAME_SIZE, height: FRAME_SIZE }
      )
    }

    const encoder = device.createCommandEncoder()

    const compute = encoder.beginComputePass()
    compute.setPipeline(computePipeline)
    compute.setBindGroup(0, computeBindGroup)
    compute.dispatchWorkgroups(1)
    compute.end()

    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: { r: 0.04, g: 0.05, b: 0.08, a: 1 },
        },
      ],
    })
    pass.setPipeline(renderPipeline)
    pass.setBindGroup(0, renderBindGroup)
    pass.draw(3)
    pass.end()

    // The CPU checks the length the shader measured once — the size table is verified in numbers, not by eye.
    frames += 1
    const wantsReadback = !reported && frames > 10 && !reading
    if (wantsReadback) encoder.copyBufferToBuffer(info, 0, infoStaging, 0, 16)

    device.queue.submit([encoder.finish()])

    if (wantsReadback) {
      reading = true
      infoStaging
        .mapAsync(GPUMapMode.READ)
        .then((buffer: ArrayBuffer) => {
          const values = new Uint32Array(buffer)
          const ok = values[0] === PALETTE && values[1] === 4
          reported = ok
          report(
            `arrayLength: palette ${values[0]} (expected ${PALETTE}) · info ${values[1]} (expected 4)` +
              (ok ? ' ✓' : ' ✗')
          )
        })
        .catch((error: unknown) => report(`readback failed: ${String(error)}`))
        .finally(() => {
          infoStaging.unmap()
          reading = false
        })
    }
  }
}

root.render(
  <DemoScene
    title="WGSL compatibility"
    subtitle="arrayLength · an external texture (edge-clamped on the left only) · untyped constant expressions"
    setup={setup}
  />
)
