import { root } from '@lynx-js/react'
import { DemoScene, type SceneContext } from '../scene.jsx'
import { GPUBufferUsage } from '../webgpu.js'

const SHADER = /* wgsl */ `
struct Uniforms {
  angle: f32,
  aspect: f32,
};
@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) color: vec3f,
};

@vertex
fn vs_main(@location(0) position: vec2f, @location(1) color: vec3f) -> VertexOutput {
  let s = sin(uniforms.angle);
  let c = cos(uniforms.angle);
  let rotated = vec2f(position.x * c - position.y * s, position.x * s + position.y * c);

  // 종횡비 보정: 짧은 축에 맞춰 줄인다. 긴 축을 기준으로 늘리면 세로 화면에서 화면을 넘어간다.
  var scale = vec2f(1.0, 1.0);
  if (uniforms.aspect >= 1.0) {
    scale.x = 1.0 / uniforms.aspect;
  } else {
    scale.y = uniforms.aspect;
  }

  var out: VertexOutput;
  out.position = vec4f(rotated * scale, 0.0, 1.0);
  out.color = color;
  return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
  return vec4f(in.color, 1.0);
}
`

// 위치(x, y) + 색(r, g, b) 인터리브 — stride 20B
const VERTICES = new Float32Array([
  0.0, 0.9, 1.0, 0.35, 0.35,
  -0.85, -0.7, 0.35, 1.0, 0.5,
  0.85, -0.7, 0.4, 0.6, 1.0,
])

function setup({ device, context, format }: SceneContext) {
  const module = device.createShaderModule({ code: SHADER, label: 'triangle' })

  const vertexBuffer = device.createBuffer({
    size: VERTICES.byteLength,
    usage: GPUBufferUsage.VERTEX,
    mappedAtCreation: true,
  })
  new Float32Array(vertexBuffer.getMappedRange()).set(VERTICES)
  vertexBuffer.unmap()

  // f32 2개지만 유니폼 구조체는 16바이트 정렬이다.
  const uniformBuffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })

  const pipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: {
      module,
      entryPoint: 'vs_main',
      buffers: [{
        arrayStride: 20,
        attributes: [
          { format: 'float32x2', offset: 0, shaderLocation: 0 },
          { format: 'float32x3', offset: 8, shaderLocation: 1 },
        ],
      }],
    },
    fragment: { module, entryPoint: 'fs_main', targets: [{ format }] },
    primitive: { topology: 'triangle-list' },
  })

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
  })

  const uniforms = new Float32Array(4)
  let angle = 0

  return ({ delta, width, height }: { delta: number; width: number; height: number }) => {
    angle += (delta / 1000) * 1.1
    uniforms[0] = angle
    uniforms[1] = width / height
    device.queue.writeBuffer(uniformBuffer, 0, uniforms)

    const encoder = device.createCommandEncoder()
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: 0.043, g: 0.055, b: 0.08, a: 1 },
      }],
    })
    pass.setPipeline(pipeline)
    pass.setBindGroup(0, bindGroup)
    pass.setVertexBuffer(0, vertexBuffer)
    pass.draw(3)
    pass.end()
    device.queue.submit([encoder.finish()])
  }
}

root.render(
  <DemoScene title="회전 삼각형" subtitle="정점 버퍼 + 유니폼" setup={setup} />
)
