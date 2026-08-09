import { root } from '@lynx-js/react'
import { DemoScene, type SceneContext } from '../scene.jsx'
import { GPUBufferUsage, GPUMapMode } from '../webgpu.js'

const BARS = 64
const WORKGROUP_SIZE = 32

/** Computes the bar heights on the GPU and writes them into a storage buffer. */
const COMPUTE_SHADER = /* wgsl */ `
struct Params {
  time: f32,
  count: u32,
};
@group(0) @binding(0) var<storage, read_write> heights: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= params.count) {
    return;
  }
  let t = f32(id.x) / f32(params.count);
  let wave = sin(t * 12.0 + params.time * 2.0) * 0.5 + 0.5;
  let envelope = sin(t * 3.14159265);
  heights[id.x] = wave * envelope;
}
`

/** The vertex shader reads the same storage buffer and draws the bars. */
const RENDER_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> heights: array<f32>;
@group(0) @binding(1) var<uniform> count: vec4f;   // x = the bar count

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) shade: f32,
};

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32,
           @builtin(instance_index) instanceIndex: u32) -> VertexOutput {
  var corners = array<vec2f, 6>(
    vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0),
    vec2f(0.0, 1.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0),
  );
  let corner = corners[vertexIndex];
  let total = count.x;
  let width = 2.0 / total;
  let height = heights[instanceIndex] * 1.4;

  var out: VertexOutput;
  out.position = vec4f(
    -1.0 + (f32(instanceIndex) + corner.x * 0.85) * width,
    -0.7 + corner.y * height,
    0.0, 1.0
  );
  out.shade = heights[instanceIndex];
  return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
  return vec4f(mix(vec3f(0.2, 0.45, 0.9), vec3f(1.0, 0.75, 0.3), in.shade), 1.0);
}
`

function setup({ device, context, format, report }: SceneContext) {
  const computeModule = device.createShaderModule({ code: COMPUTE_SHADER, label: 'bars.update' })
  const renderModule = device.createShaderModule({ code: RENDER_SHADER, label: 'bars.render' })

  const heights = device.createBuffer({
    size: BARS * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    label: 'heights',
  })
  // The readback goes into a **dedicated staging buffer**. MAP_READ can only combine with COPY_DST (spec),
  // and a buffer being mapped is rejected by queue operations, so mapping the compute buffer directly would block the next frame.
  const staging = device.createBuffer({
    size: BARS * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    label: 'heights.staging',
  })
  const computeParams = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })
  const renderParams = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })
  device.queue.writeBuffer(renderParams, 0, new Float32Array([BARS, 0, 0, 0]))

  const computePipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module: computeModule, entryPoint: 'main' },
  })
  const computeBindGroup = device.createBindGroup({
    layout: computePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: heights } },
      { binding: 1, resource: { buffer: computeParams } },
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
      { binding: 0, resource: { buffer: heights } },
      { binding: 1, resource: { buffer: renderParams } },
    ],
  })

  const paramsData = new ArrayBuffer(16)
  const paramsFloats = new Float32Array(paramsData)
  const paramsUints = new Uint32Array(paramsData)
  paramsUints[1] = BARS

  const workgroups = Math.ceil(BARS / WORKGROUP_SIZE)
  let time = 0
  let frame = 0
  let reading = false

  return ({ delta }: { delta: number }) => {
    time += delta / 1000
    paramsFloats[0] = time
    device.queue.writeBuffer(computeParams, 0, paramsData)

    const encoder = device.createCommandEncoder()

    const compute = encoder.beginComputePass()
    compute.setPipeline(computePipeline)
    compute.setBindGroup(0, computeBindGroup)
    compute.dispatchWorkgroups(workgroups)
    compute.end()

    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: 0.043, g: 0.055, b: 0.08, a: 1 },
      }],
    })
    pass.setPipeline(renderPipeline)
    pass.setBindGroup(0, renderBindGroup)
    pass.draw(6, BARS)
    pass.end()

    // Only read to the CPU once every 30 frames — it waits for GPU completion, so it must not be per frame.
    const wantsReadback = ++frame % 30 === 0 && !reading
    if (wantsReadback) encoder.copyBufferToBuffer(heights, 0, staging, 0, BARS * 4)

    device.queue.submit([encoder.finish()])

    if (wantsReadback) {
      reading = true
      staging
        .mapAsync(GPUMapMode.READ)
        .then((buffer: ArrayBuffer) => {
          const values = new Float32Array(buffer)
          let max = 0
          let sum = 0
          for (const value of values) {
            if (value > max) max = value
            sum += value
          }
          report(
            `${values.length} values read by the CPU · max ${max.toFixed(3)} · mean ${(sum / values.length).toFixed(3)}`
          )
        })
        .catch((error: unknown) => report(`readback failed: ${String(error)}`))
        .finally(() => {
          // The mapping must be released for the next frame's copyBufferToBuffer to use this buffer again.
          staging.unmap()
          reading = false
        })
    }
  }
}

root.render(
  <DemoScene title="Compute · readback" subtitle="The CPU checks GPU-computed values with mapAsync" setup={setup} />
)
