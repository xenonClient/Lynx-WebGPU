import { root, useInitData, useRef, useState } from '@lynx-js/react'
import { DemoScene, type SceneContext } from '../scene.jsx'
import { GPUBufferUsage, GPUMapMode } from '../webgpu.js'

/**
 * GPU-driven rendering — **the CPU does not know how many are drawn.**
 *
 * A compute pass decides this frame's count and writes it into argument buffers, and the dispatch and draw
 * that follow read those buffers. JS passes **only handles** to `dispatchWorkgroupsIndirect` / `drawIndexedIndirect`.
 *
 * Comparing with direct mode makes the difference clear — direct mode does not know the count, so it always
 * draws the maximum. For the CPU to know the count would take a readback round trip, and that is exactly the cost indirect draw removes.
 */
const MAX_PARTICLES = 3072
const WORKGROUP_SIZE = 64
const MIN_PARTICLES = 96

/** One workgroup decides this frame's count and writes it into the two argument buffers. */
const PLAN_SHADER = /* wgsl */ `
struct Control {
  time: f32,
  maxCount: f32,
  minCount: f32,
  _pad: f32,
};
@group(0) @binding(0) var<uniform> control: Control;
@group(0) @binding(1) var<storage, read_write> drawArgs: array<u32>;
@group(0) @binding(2) var<storage, read_write> dispatchArgs: array<u32>;

@compute @workgroup_size(1)
fn plan() {
  let wave = sin(control.time * 0.55) * 0.5 + 0.5;
  let live = control.minCount + wave * (control.maxCount - control.minCount);
  let count = u32(live);

  // The 5 drawIndexedIndirect argument slots — indexCount, instanceCount, firstIndex, baseVertex, firstInstance
  drawArgs[0] = 6u;
  drawArgs[1] = count;
  drawArgs[2] = 0u;
  drawArgs[3] = 0u;
  drawArgs[4] = 0u;

  // The 3 dispatchWorkgroupsIndirect argument slots — x, y, z
  dispatchArgs[0] = (count + ${WORKGROUP_SIZE}u - 1u) / ${WORKGROUP_SIZE}u;
  dispatchArgs[1] = 1u;
  dispatchArgs[2] = 1u;
}
`

/** The position is a pure function of (index, time) — the picture stays coherent whichever subset runs. */
const UPDATE_SHADER = /* wgsl */ `
struct Particle {
  position: vec2f,
  size: f32,
  hue: f32,
};

struct Params {
  time: f32,
  aspect: f32,
  _a: f32,
  _b: f32,
};

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn update(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= arrayLength(&particles)) {
    return;
  }
  let index = f32(id.x);
  let ring = floor(index / 96.0);
  let slot = index - ring * 96.0;
  let radius = 0.16 + ring * 0.055;
  let speed = 0.85 / (0.6 + ring * 0.4);
  let angle = slot * 0.06545 + params.time * speed + ring * 0.9;

  var particle: Particle;
  particle.position = vec2f(cos(angle) * radius / params.aspect, sin(angle) * radius);
  particle.size = 0.011 + 0.005 * sin(index * 0.73);
  particle.hue = fract(index * 0.017);
  particles[id.x] = particle;
}
`

/** One instance is one quad — 4 vertices run through 6 indices. */
const RENDER_SHADER = /* wgsl */ `
struct Particle {
  position: vec2f,
  size: f32,
  hue: f32,
};

struct Params {
  time: f32,
  aspect: f32,
  _a: f32,
  _b: f32,
};

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: Params;

struct Out {
  @builtin(position) position: vec4f,
  @location(0) offset: vec2f,
  @location(1) hue: f32,
};

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32,
           @builtin(instance_index) instanceIndex: u32) -> Out {
  var corners = array<vec2f, 4>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0), vec2f(1.0, 1.0),
  );
  let corner = corners[vertexIndex];
  let particle = particles[instanceIndex];

  var out: Out;
  out.position = vec4f(
    particle.position + vec2f(corner.x * particle.size / params.aspect, corner.y * particle.size),
    0.0, 1.0
  );
  out.offset = corner;
  out.hue = particle.hue;
  return out;
}

@fragment
fn fs_main(in: Out) -> @location(0) vec4f {
  let falloff = clamp(1.0 - length(in.offset), 0.0, 1.0);
  let intensity = falloff * falloff;
  let color = mix(vec3f(0.28, 0.68, 1.0), vec3f(1.0, 0.6, 0.35), in.hue);
  return vec4f(color * intensity, intensity);
}
`

function setup({ device, context, format, report }: SceneContext, indirectRef: { current: boolean }) {
  const planModule = device.createShaderModule({ code: PLAN_SHADER, label: 'gpudriven.plan' })
  const updateModule = device.createShaderModule({ code: UPDATE_SHADER, label: 'gpudriven.update' })
  const renderModule = device.createShaderModule({ code: RENDER_SHADER, label: 'gpudriven.render' })

  const particles = device.createBuffer({
    size: MAX_PARTICLES * 16,
    usage: GPUBufferUsage.STORAGE,
    label: 'particles',
  })

  // A buffer the compute pass writes (STORAGE) and the command processor reads (INDIRECT).
  const drawArgs = device.createBuffer({
    size: 20,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_SRC,
    label: 'drawArgs',
  })
  // The staging buffer the HUD reads the values back through — MAP_READ can only combine with COPY_DST (spec).
  const drawArgsStaging = device.createBuffer({
    size: 20,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    label: 'drawArgs.staging',
  })
  const dispatchArgs = device.createBuffer({
    size: 12,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT,
    label: 'dispatchArgs',
  })

  const control = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })
  const params = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })

  // One quad — 4 vertices run through 6 indices.
  const indices = device.createBuffer({
    size: 12,
    usage: GPUBufferUsage.INDEX,
    mappedAtCreation: true,
    label: 'quad.indices',
  })
  new Uint16Array(indices.getMappedRange()).set([0, 1, 2, 2, 1, 3])
  indices.unmap()

  const planPipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module: planModule, entryPoint: 'plan' },
  })
  const planGroup = device.createBindGroup({
    layout: planPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: control } },
      { binding: 1, resource: { buffer: drawArgs } },
      { binding: 2, resource: { buffer: dispatchArgs } },
    ],
  })

  const updatePipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module: updateModule, entryPoint: 'update' },
  })
  const updateGroup = device.createBindGroup({
    layout: updatePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: particles } },
      { binding: 1, resource: { buffer: params } },
    ],
  })

  const renderPipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module: renderModule, entryPoint: 'vs_main' },
    fragment: {
      module: renderModule,
      entryPoint: 'fs_main',
      targets: [{
        format,
        blend: {
          color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
          alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
        },
      }],
    },
  })
  const renderGroup = device.createBindGroup({
    layout: renderPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: particles } },
      { binding: 1, resource: { buffer: params } },
    ],
  })

  const controlData = new Float32Array([0, MAX_PARTICLES, MIN_PARTICLES, 0])
  const paramsData = new Float32Array(4)
  const maxWorkgroups = Math.ceil(MAX_PARTICLES / WORKGROUP_SIZE)

  let time = 0
  let frame = 0
  let reading = false

  return ({ delta, width, height }: { delta: number; width: number; height: number }) => {
    time += delta / 1000
    const indirect = indirectRef.current

    controlData[0] = time
    device.queue.writeBuffer(control, 0, controlData)
    paramsData[0] = time
    paramsData[1] = width / height
    device.queue.writeBuffer(params, 0, paramsData)

    const encoder = device.createCommandEncoder()

    const compute = encoder.beginComputePass()
    if (indirect) {
      // 1) Decide the count — the CPU never sees this result.
      compute.setPipeline(planPipeline)
      compute.setBindGroup(0, planGroup)
      compute.dispatchWorkgroups(1)
    }
    // 2) Update only that many. The workgroup count comes from a GPU buffer.
    compute.setPipeline(updatePipeline)
    compute.setBindGroup(0, updateGroup)
    if (indirect) {
      compute.dispatchWorkgroupsIndirect(dispatchArgs)
    } else {
      compute.dispatchWorkgroups(maxWorkgroups)
    }
    compute.end()

    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: 0.02, g: 0.03, b: 0.055, a: 1 },
      }],
    })
    pass.setPipeline(renderPipeline)
    pass.setBindGroup(0, renderGroup)
    pass.setIndexBuffer(indices, 'uint16')
    // 3) The number drawn comes from the same buffer.
    if (indirect) {
      pass.drawIndexedIndirect(drawArgs)
    } else {
      pass.drawIndexed(6, MAX_PARTICLES)
    }
    pass.end()

    // The HUD — the GPU-decided value is read back only once every 20 frames (a readback waits for GPU completion).
    const wantsReadback = ++frame % 20 === 0 && !reading && indirect
    if (wantsReadback) encoder.copyBufferToBuffer(drawArgs, 0, drawArgsStaging, 0, 20)

    device.queue.submit([encoder.finish()])

    if (frame % 20 === 0 && !indirect) {
      report(`direct draw — with no idea of the count it always draws the maximum ${MAX_PARTICLES}`)
      return
    }
    if (wantsReadback) {
      reading = true
      drawArgsStaging
        .mapAsync(GPUMapMode.READ)
        .then((buffer: ArrayBuffer) => {
          const values = new Uint32Array(buffer)
          report(
            `${values[1]} instances decided by the GPU / max ${MAX_PARTICLES} · ` +
              `${Math.ceil(values[1] / WORKGROUP_SIZE)} workgroups — the CPU never passed them`
          )
        })
        .catch((error: unknown) => report(`argument readback failed: ${String(error)}`))
        .finally(() => {
          drawArgsStaging.unmap()
          reading = false
        })
    }
  }
}

function GpuDrivenScene() {
  // With `-altMode 1` it starts in direct draw (for automated capture — the same convention as the `bundle` scene).
  // Lynx may move a boolean across as a number, so it is read as truthy rather than ===.
  const alt = !!(useInitData() as { altMode?: unknown } | undefined)?.altMode
  const [indirect, setIndirect] = useState(!alt)

  const indirectRef = useRef(!alt)
  indirectRef.current = indirect

  return (
    <DemoScene
      title="GPU-driven rendering"
      subtitle="A compute pass decides the count and indirect dispatch/draw read that buffer"
      setup={(scene) => setup(scene, indirectRef)}
      controls={
        <view className="controls">
          <text className="control-value">{indirect ? 'the GPU decides the count' : 'fixed at the maximum'}</text>
          <text
            className={indirect ? 'control-button control-button-on' : 'control-button'}
            bindtap={() => setIndirect(true)}
          >
            Indirect
          </text>
          <text
            className={indirect ? 'control-button' : 'control-button control-button-on'}
            bindtap={() => setIndirect(false)}
          >
            Direct
          </text>
        </view>
      }
    />
  )
}

root.render(<GpuDrivenScene />)
