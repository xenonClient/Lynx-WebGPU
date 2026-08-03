import { root, useInitData, useRef, useState } from '@lynx-js/react'
import { DemoScene, type SceneContext } from '../scene.jsx'
import { GPUBufferUsage, GPUMapMode } from '../webgpu.js'

/**
 * GPU-driven 렌더링 — **CPU는 몇 개를 그리는지 모른다.**
 *
 * 컴퓨트가 이번 프레임의 개수를 정해 인자 버퍼에 쓰고, 이어지는 디스패치와 드로우가 그 버퍼를
 * 읽는다. JS는 `dispatchWorkgroupsIndirect` / `drawIndexedIndirect`에 **핸들만** 넘긴다.
 *
 * 직접 모드와 비교하면 차이가 분명하다 — 직접 모드는 개수를 모르니 늘 최대치를 그린다.
 * 개수를 CPU가 알려면 리드백 왕복이 필요한데, 그게 바로 간접 드로우가 없애는 비용이다.
 */
const MAX_PARTICLES = 3072
const WORKGROUP_SIZE = 64
const MIN_PARTICLES = 96

/** 워크그룹 하나가 이번 프레임의 개수를 정해 두 인자 버퍼에 쓴다. */
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

  // drawIndexedIndirect 인자 5칸 — indexCount, instanceCount, firstIndex, baseVertex, firstInstance
  drawArgs[0] = 6u;
  drawArgs[1] = count;
  drawArgs[2] = 0u;
  drawArgs[3] = 0u;
  drawArgs[4] = 0u;

  // dispatchWorkgroupsIndirect 인자 3칸 — x, y, z
  dispatchArgs[0] = (count + ${WORKGROUP_SIZE}u - 1u) / ${WORKGROUP_SIZE}u;
  dispatchArgs[1] = 1u;
  dispatchArgs[2] = 1u;
}
`

/** 위치는 (인덱스, 시간)의 순수 함수다 — 어떤 부분집합만 돌아도 그림이 일관된다. */
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

/** 인스턴스 하나가 사각형 하나 — 정점 4개를 인덱스 6개로 돈다. */
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

  // 컴퓨트가 쓰고(STORAGE) 커맨드 프로세서가 읽는(INDIRECT) 버퍼.
  // HUD가 값을 되짚어 보려고 MAP_READ도 함께 준다.
  const drawArgs = device.createBuffer({
    size: 20,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT
      | GPUBufferUsage.COPY_SRC | GPUBufferUsage.MAP_READ,
    label: 'drawArgs',
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

  // 사각형 하나 — 정점 4개를 6개 인덱스로 돈다.
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
      // 1) 개수를 정한다 — 이 결과를 CPU는 보지 않는다.
      compute.setPipeline(planPipeline)
      compute.setBindGroup(0, planGroup)
      compute.dispatchWorkgroups(1)
    }
    // 2) 정해진 만큼만 갱신한다. 워크그룹 수가 GPU 버퍼에서 온다.
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
    // 3) 그리는 개수도 같은 버퍼에서 온다.
    if (indirect) {
      pass.drawIndexedIndirect(drawArgs)
    } else {
      pass.drawIndexed(6, MAX_PARTICLES)
    }
    pass.end()

    device.queue.submit([encoder.finish()])

    // HUD — GPU가 정한 값을 20프레임에 한 번만 되짚는다 (리드백은 GPU 완료를 기다린다).
    if (++frame % 20 === 0 && !reading) {
      if (!indirect) {
        report(`직접 드로우 — 개수를 모르니 늘 최대치 ${MAX_PARTICLES}개를 그린다`)
        return
      }
      reading = true
      drawArgs
        .mapAsync(GPUMapMode.READ)
        .then((buffer: ArrayBuffer) => {
          const values = new Uint32Array(buffer)
          report(
            `GPU가 정한 인스턴스 ${values[1]}개 / 최대 ${MAX_PARTICLES}개 · ` +
              `워크그룹 ${Math.ceil(values[1] / WORKGROUP_SIZE)}개 — CPU는 넘기지 않았다`
          )
        })
        .catch((error: unknown) => report(`인자 리드백 실패: ${String(error)}`))
        .finally(() => {
          reading = false
        })
    }
  }
}

function GpuDrivenScene() {
  // `-altMode 1`이면 직접 드로우로 시작한다 (자동화 캡처용 — `bundle` 씬과 같은 규약).
  // Lynx가 불리언을 숫자로 옮겨 줄 수 있어 === 대신 truthy로 본다.
  const alt = !!(useInitData() as { altMode?: unknown } | undefined)?.altMode
  const [indirect, setIndirect] = useState(!alt)

  const indirectRef = useRef(!alt)
  indirectRef.current = indirect

  return (
    <DemoScene
      title="GPU-driven 렌더링"
      subtitle="컴퓨트가 개수를 정하고 간접 디스패치·드로우가 그 버퍼를 읽는다"
      setup={(scene) => setup(scene, indirectRef)}
      controls={
        <view className="controls">
          <text className="control-value">{indirect ? '개수를 GPU가 정함' : '최대치 고정'}</text>
          <text
            className={indirect ? 'control-button control-button-on' : 'control-button'}
            bindtap={() => setIndirect(true)}
          >
            간접
          </text>
          <text
            className={indirect ? 'control-button' : 'control-button control-button-on'}
            bindtap={() => setIndirect(false)}
          >
            직접
          </text>
        </view>
      }
    />
  )
}

root.render(<GpuDrivenScene />)
