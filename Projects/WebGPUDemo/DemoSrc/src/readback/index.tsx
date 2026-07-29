import { root } from '@lynx-js/react'
import { DemoScene, type SceneContext } from '../scene.jsx'
import { GPUBufferUsage, GPUMapMode } from '../webgpu.js'

const BARS = 64
const WORKGROUP_SIZE = 32

/** 막대 높이를 GPU에서 계산해 스토리지 버퍼에 쓴다. */
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

/** 같은 스토리지 버퍼를 정점 셰이더가 읽어 막대를 그린다. */
const RENDER_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> heights: array<f32>;
@group(0) @binding(1) var<uniform> count: vec4f;   // x = 막대 수

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
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.MAP_READ,
    label: 'heights',
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

    device.queue.submit([encoder.finish()])

    // 30프레임에 한 번만 CPU로 읽는다 — GPU 완료를 기다리는 경로라 프레임마다 하면 안 된다.
    if (++frame % 30 === 0 && !reading) {
      reading = true
      heights
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
            `CPU가 읽은 값 ${values.length}개 · 최대 ${max.toFixed(3)} · 평균 ${(sum / values.length).toFixed(3)}`
          )
        })
        .catch((error: unknown) => report(`리드백 실패: ${String(error)}`))
        .finally(() => {
          reading = false
        })
    }
  }
}

root.render(
  <DemoScene title="컴퓨트 · 리드백" subtitle="GPU가 계산한 값을 mapAsync로 CPU가 확인" setup={setup} />
)
