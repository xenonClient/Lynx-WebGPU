import { root } from '@lynx-js/react'
import { DemoScene, type SceneContext } from '../scene.jsx'
import { GPUBufferUsage, GPUShaderStage } from '../webgpu.js'

/**
 * WGSL 트랜스파일러를 건너뛰고 Metal Shading Language를 직접 넣는 탈출구.
 *
 * 이때는 바인딩 인덱스를 손으로 써야 하고(`[[buffer(0)]]`), `layout: 'auto'`를 쓸 수 없다 —
 * 셰이더에서 유도할 리플렉션 정보가 없기 때문이다 (docs/WGSL.md §5).
 */
const MSL = /* metal */ `
#include <metal_stdlib>
using namespace metal;

struct Uniforms {
    float time;
    float aspect;
};

struct VertexOutput {
    float4 position [[position]];
    float2 uv;
};

vertex VertexOutput vs_main(uint vertexIndex [[vertex_id]])
{
    float2 corners[3] = { float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0) };
    VertexOutput out;
    out.position = float4(corners[vertexIndex], 0.0, 1.0);
    out.uv = corners[vertexIndex] * 0.5 + 0.5;
    return out;
}

fragment float4 fs_main(VertexOutput in [[stage_in]], constant Uniforms& u [[buffer(0)]])
{
    float2 p = (in.uv - 0.5) * float2(u.aspect, 1.0) * 3.0;
    float r = length(p);
    float a = atan2(p.y, p.x);
    // 회전하는 나선 — MSL 경로가 실제로 도는지 눈으로 확인하기 좋은 무늬.
    float wave = sin(r * 6.0 - u.time * 2.0 + a * 3.0);
    float3 color = mix(float3(0.08, 0.12, 0.25), float3(0.35, 0.85, 0.95), wave * 0.5 + 0.5);
    color *= smoothstep(2.2, 0.2, r);
    return float4(color, 1.0);
}
`

function setup({ device, context, format, report }: SceneContext) {
  const module = device.createShaderModule({ code: MSL, language: 'msl', label: 'spiral.msl' })

  // MSL 모듈은 레이아웃을 명시해야 한다. 여기서 만든 순서가 그대로 Metal 인덱스가 된다
  // (그룹 → 바인딩 오름차순, 종류별 0부터 — docs/ARCHITECTURE.md §4).
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }],
  })
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] })

  const uniformBuffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })

  const pipeline = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: { module, entryPoint: 'vs_main' },
    fragment: { module, entryPoint: 'fs_main', targets: [{ format }] },
  })

  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
  })

  report('language: "msl" — WGSL 트랜스파일러를 거치지 않은 경로')

  const uniforms = new Float32Array(4)
  let time = 0

  return ({ delta, width, height }: { delta: number; width: number; height: number }) => {
    time += delta / 1000
    uniforms[0] = time
    uniforms[1] = width / height
    device.queue.writeBuffer(uniformBuffer, 0, uniforms)

    const encoder = device.createCommandEncoder()
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: 0.02, g: 0.03, b: 0.06, a: 1 },
      }],
    })
    pass.setPipeline(pipeline)
    pass.setBindGroup(0, bindGroup)
    pass.draw(3)
    pass.end()
    device.queue.submit([encoder.finish()])
  }
}

root.render(
  <DemoScene title="MSL 탈출구" subtitle="Metal 셰이더 직접 주입 + 명시적 파이프라인 레이아웃" setup={setup} />
)
