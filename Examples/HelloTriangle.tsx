/**
 * Hello Triangle — Lynx-WebGPU 최소 예제 (ReactLynx).
 *
 * 리소스는 한 번만 만들고 프레임마다 유니폼만 갱신한다. 커맨드는 `queue.submit()`에서
 * 한 번에 네이티브로 넘어가므로, 프레임당 브리지 왕복은 1회다.
 *
 * rspeedy 프로젝트에 `JS/webgpu.js`, `JS/webgpu.d.ts`, `JS/elements.d.ts`를 복사한 뒤 쓴다
 * (docs/JS-AUTHORING.md).
 */
import { useEffect, useRef } from '@lynx-js/react'
import gpu, { GPUBufferUsage, startFrameLoop } from './webgpu.js'
import './elements.d.ts'

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
  var out: VertexOutput;
  out.position = vec4f(rotated.x / uniforms.aspect, rotated.y, 0.0, 1.0);
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
  0.0, 0.6, 1.0, 0.2, 0.2,
  -0.6, -0.5, 0.2, 1.0, 0.2,
  0.6, -0.5, 0.2, 0.4, 1.0,
])

export function HelloTriangle() {
  const stopRef = useRef<(() => void) | null>(null)

  useEffect(() => {
    let disposed = false

    async function boot() {
      const adapter = await gpu.requestAdapter()
      if (!adapter) {
        console.error('WebGPU 어댑터를 얻지 못했다')
        return
      }
      const device = await adapter.requestDevice()
      device.onError((error) => console.error('WebGPU', error.kind, error.path, error.message))

      const context = gpu.getCanvasContext('main')
      const format = gpu.getPreferredCanvasFormat()
      context.configure({ device, format })

      const module = device.createShaderModule({ code: SHADER, label: 'triangle' })

      const vertexBuffer = device.createBuffer({
        size: VERTICES.byteLength,
        usage: GPUBufferUsage.VERTEX,
        mappedAtCreation: true,
      })
      new Float32Array(vertexBuffer.getMappedRange()).set(VERTICES)
      vertexBuffer.unmap()

      const uniformBuffer = device.createBuffer({
        size: 16, // f32 2개 + 16바이트 정렬 패딩
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

      const stop = startFrameLoop(({ delta }) => {
        if (disposed) return
        angle += (delta / 1000) * 1.2

        const { width, height } = context.getSize()
        if (width === 0 || height === 0) return

        uniforms[0] = angle
        uniforms[1] = width / height
        device.queue.writeBuffer(uniformBuffer, 0, uniforms)

        const encoder = device.createCommandEncoder()
        const pass = encoder.beginRenderPass({
          colorAttachments: [{
            view: context.getCurrentTexture().createView(),
            loadOp: 'clear',
            storeOp: 'store',
            clearValue: { r: 0.06, g: 0.07, b: 0.1, a: 1 },
          }],
        })
        pass.setPipeline(pipeline)
        pass.setBindGroup(0, bindGroup)
        pass.setVertexBuffer(0, vertexBuffer)
        pass.draw(3)
        pass.end()
        device.queue.submit([encoder.finish()])
      })

      stopRef.current = () => {
        stop()
        device.destroy()
      }
    }

    boot()
    return () => {
      disposed = true
      stopRef.current?.()
    }
  }, [])

  return <webgpu-canvas canvas-id="main" style={{ width: '100%', height: '100%' }} />
}
