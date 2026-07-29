import { root } from '@lynx-js/react'
import { DemoScene, type SceneContext } from '../scene.jsx'
import { GPUBufferUsage } from '../webgpu.js'

/**
 * 같은 셰이더 모듈 하나를 `override` 값만 바꿔 세 파이프라인으로 만든다.
 * 값은 MSL 방출 시점에 상수로 박히므로, 셰이더 안에서 분기하는 것보다 빠르다.
 */
const SHADER = /* wgsl */ `
override sides: u32 = 3;
override hue: f32 = 0.0;

struct Uniforms {
  aspect: f32,
  scale: f32,
  spin: f32,
  _pad: f32,
  offset: vec2f,
};
@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) tint: vec3f,
};

/// 다각형을 삼각형 부채꼴로 만든다 — 정점 수는 sides * 3.
@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  let triangle = vertexIndex / 3u;
  let corner = vertexIndex % 3u;

  var local = vec2f(0.0, 0.0);
  if (corner > 0u) {
    let step = triangle + corner - 1u;
    let angle = f32(step) / f32(sides) * 6.28318530718 + uniforms.spin + 1.57079632679;
    local = vec2f(cos(angle), sin(angle));
  }

  var out: VertexOutput;
  // 도형만 종횡비 보정하고 배치 오프셋은 NDC 그대로 둔다 — 섞으면 화면 밖으로 나간다.
  out.position = vec4f(
    local.x * uniforms.scale / uniforms.aspect + uniforms.offset.x,
    local.y * uniforms.scale + uniforms.offset.y,
    0.0, 1.0
  );
  // hue를 세 채널로 흩어 색을 만든다 (파이프라인마다 다른 값이 박힌다).
  out.tint = vec3f(
    0.5 + 0.5 * cos(hue),
    0.5 + 0.5 * cos(hue + 2.09439510239),
    0.5 + 0.5 * cos(hue + 4.18879020479)
  ) * (0.45 + 0.55 * f32(corner > 0u));
  return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
  return vec4f(in.tint, 1.0);
}
`

const VARIANTS = [
  { sides: 3, hue: 0.0, slot: -1 },
  { sides: 5, hue: 2.1, slot: 0 },
  { sides: 8, hue: 4.2, slot: 1 },
]

function setup({ device, context, format, report }: SceneContext) {
  const module = device.createShaderModule({ code: SHADER, label: 'polygon' })

  const variants = VARIANTS.map((variant) => {
    const pipeline = device.createRenderPipeline({
      layout: 'auto',
      // 같은 module, 다른 constants → 별도 MSL로 방출·캐시된다.
      vertex: { module, entryPoint: 'vs_main', constants: { sides: variant.sides, hue: variant.hue } },
      fragment: { module, entryPoint: 'fs_main', constants: { hue: variant.hue }, targets: [{ format }] },
    })
    const uniformBuffer = device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
    })
    return { ...variant, pipeline, uniformBuffer, bindGroup, vertexCount: variant.sides * 3 }
  })

  report(`override sides = ${VARIANTS.map((v) => v.sides).join(' / ')} — 셰이더 소스는 하나`)

  const uniforms = new Float32Array(8)
  let spin = 0

  return ({ delta, width, height }: { delta: number; width: number; height: number }) => {
    spin += (delta / 1000) * 0.6
    const aspect = Math.max(width / height, 0.001)
    // 세로 화면이면 세로로, 가로 화면이면 가로로 3개를 늘어놓는다.
    const portrait = aspect < 1
    const scale = portrait ? 0.26 : 0.26 * Math.min(1, aspect / 1.8)

    for (const variant of variants) {
      uniforms[0] = aspect
      uniforms[1] = scale
      uniforms[2] = spin
      uniforms[3] = 0
      uniforms[4] = portrait ? 0 : variant.slot * 0.64      // offset.x
      uniforms[5] = portrait ? -variant.slot * 0.55 : 0     // offset.y
      device.queue.writeBuffer(variant.uniformBuffer, 0, uniforms)
    }

    const encoder = device.createCommandEncoder()
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: 0.043, g: 0.055, b: 0.08, a: 1 },
      }],
    })
    for (const variant of variants) {
      pass.setPipeline(variant.pipeline)
      pass.setBindGroup(0, variant.bindGroup)
      pass.draw(variant.vertexCount)
    }
    pass.end()
    device.queue.submit([encoder.finish()])
  }
}

root.render(
  <DemoScene title="파이프라인 상수" subtitle="같은 셰이더 · override 값만 다른 3개 파이프라인" setup={setup} />
)
