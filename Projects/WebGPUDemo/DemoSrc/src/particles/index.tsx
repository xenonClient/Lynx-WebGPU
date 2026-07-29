import { root } from '@lynx-js/react'
import { DemoScene, type SceneContext } from '../scene.jsx'
import { GPUBufferUsage } from '../webgpu.js'

const PARTICLE_COUNT = 4096
const WORKGROUP_SIZE = 64

/** 입자 위치를 GPU에서 적분한다 — 스토리지 버퍼를 읽고 쓴다. */
const COMPUTE_SHADER = /* wgsl */ `
struct Particle {
  position: vec2f,
  velocity: vec2f,
};

struct Params {
  dt: f32,
  count: u32,
};

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn update(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= params.count) {
    return;
  }
  var p = particles[id.x];
  p.position = p.position + p.velocity * params.dt;

  // 화면 경계에서 반사
  if (p.position.x < -1.0 || p.position.x > 1.0) {
    p.velocity.x = -p.velocity.x;
  }
  if (p.position.y < -1.0 || p.position.y > 1.0) {
    p.velocity.y = -p.velocity.y;
  }
  p.position = clamp(p.position, vec2f(-1.0, -1.0), vec2f(1.0, 1.0));
  particles[id.x] = p;
}
`

/** 입자 1개당 사각형 1개를 인스턴싱으로 그린다 — 정점 버퍼 없이 스토리지 버퍼를 읽는다. */
const RENDER_SHADER = /* wgsl */ `
struct Particle {
  position: vec2f,
  velocity: vec2f,
};

struct RenderParams {
  aspect: f32,
  size: f32,
};

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: RenderParams;

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) offset: vec2f,
  @location(1) speed: f32,
};

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32,
           @builtin(instance_index) instanceIndex: u32) -> VertexOutput {
  var corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
    vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
  );
  let corner = corners[vertexIndex];
  let particle = particles[instanceIndex];
  let offset = vec2f(corner.x * params.size / params.aspect, corner.y * params.size);

  var out: VertexOutput;
  out.position = vec4f(particle.position + offset, 0.0, 1.0);
  out.offset = corner;
  out.speed = length(particle.velocity);
  return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
  // 사각형을 원형 글로우로 깎는다 (가산 블렌딩과 함께 쓴다).
  let falloff = clamp(1.0 - length(in.offset), 0.0, 1.0);
  let intensity = falloff * falloff;
  let hot = clamp(in.speed * 1.6, 0.0, 1.0);
  let color = mix(vec3f(0.25, 0.6, 1.0), vec3f(1.0, 0.55, 0.85), hot);
  return vec4f(color * intensity, intensity);
}
`

/** 스크린샷이 매번 같게 나오도록 결정적 난수를 쓴다. */
function makeRandom(seed: number) {
  let state = seed >>> 0
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0
    return state / 0x100000000
  }
}

function setup({ device, context, format }: SceneContext) {
  const random = makeRandom(20260729)
  const particles = new Float32Array(PARTICLE_COUNT * 4)
  for (let index = 0; index < PARTICLE_COUNT; index++) {
    const angle = random() * Math.PI * 2
    const radius = Math.sqrt(random()) * 0.85
    const speed = 0.12 + random() * 0.45
    const direction = random() * Math.PI * 2
    particles[index * 4 + 0] = Math.cos(angle) * radius
    particles[index * 4 + 1] = Math.sin(angle) * radius
    particles[index * 4 + 2] = Math.cos(direction) * speed
    particles[index * 4 + 3] = Math.sin(direction) * speed
  }

  const particleBuffer = device.createBuffer({
    size: particles.byteLength,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  })
  new Float32Array(particleBuffer.getMappedRange()).set(particles)
  particleBuffer.unmap()

  // 유니폼 구조체는 16바이트 정렬이다 (실제로 쓰는 건 앞의 8바이트).
  const computeParams = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })
  const renderParams = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })

  const computeModule = device.createShaderModule({ code: COMPUTE_SHADER, label: 'particles.update' })
  const renderModule = device.createShaderModule({ code: RENDER_SHADER, label: 'particles.render' })

  const computePipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module: computeModule, entryPoint: 'update' },
  })
  const computeBindGroup = device.createBindGroup({
    layout: computePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: particleBuffer } },
      { binding: 1, resource: { buffer: computeParams } },
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
        // 가산 블렌딩 — 입자가 겹칠수록 밝아진다.
        blend: {
          color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
          alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
        },
      }],
    },
    primitive: { topology: 'triangle-list' },
  })
  const renderBindGroup = device.createBindGroup({
    layout: renderPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: particleBuffer } },
      { binding: 1, resource: { buffer: renderParams } },
    ],
  })

  // dt(f32) + count(u32) 를 같은 16바이트 버퍼에 쓴다.
  const computeParamsData = new ArrayBuffer(16)
  const computeFloats = new Float32Array(computeParamsData)
  const computeUints = new Uint32Array(computeParamsData)
  computeUints[1] = PARTICLE_COUNT

  const renderParamsData = new Float32Array(4)
  const workgroups = Math.ceil(PARTICLE_COUNT / WORKGROUP_SIZE)

  return ({ delta, width, height }: { delta: number; width: number; height: number }) => {
    // 프레임이 튀어도 시뮬레이션이 폭발하지 않게 상한을 둔다.
    computeFloats[0] = Math.min(delta, 33) / 1000
    device.queue.writeBuffer(computeParams, 0, computeParamsData)

    renderParamsData[0] = width / height
    renderParamsData[1] = 0.02
    device.queue.writeBuffer(renderParams, 0, renderParamsData)

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
        clearValue: { r: 0.02, g: 0.03, b: 0.055, a: 1 },
      }],
    })
    pass.setPipeline(renderPipeline)
    pass.setBindGroup(0, renderBindGroup)
    pass.draw(6, PARTICLE_COUNT)
    pass.end()

    device.queue.submit([encoder.finish()])
  }
}

root.render(
  <DemoScene
    title={`입자 ${PARTICLE_COUNT}개`}
    subtitle="컴퓨트 셰이더 + 스토리지 버퍼 + 인스턴싱"
    setup={setup}
  />
)
