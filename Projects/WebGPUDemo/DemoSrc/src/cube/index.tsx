import { root } from '@lynx-js/react'
import { DemoScene, type SceneContext } from '../scene.jsx'
import { GPUBufferUsage, GPUTextureUsage } from '../webgpu.js'

const SHADER = /* wgsl */ `
struct Uniforms {
  mvp: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) color: vec3f,
  @location(1) normal: vec3f,
};

@vertex
fn vs_main(@location(0) position: vec3f,
           @location(1) color: vec3f,
           @location(2) normal: vec3f) -> VertexOutput {
  var out: VertexOutput;
  out.position = uniforms.mvp * vec4f(position, 1.0);
  out.color = color;
  out.normal = normal;
  return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
  // 고정 방향광 — 면마다 밝기가 갈려 3D 형태가 눈에 들어온다.
  let light = normalize(vec3f(0.4, 0.8, 0.5));
  let diffuse = max(dot(normalize(in.normal), light), 0.0);
  return vec4f(in.color * (0.35 + 0.65 * diffuse), 1.0);
}
`

// ---------------------------------------------------------------------------
// 4x4 행렬 (열 우선 — WGSL mat4x4<f32>와 같은 배치)
// ---------------------------------------------------------------------------

function multiply(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(16)
  for (let col = 0; col < 4; col++) {
    for (let row = 0; row < 4; row++) {
      let sum = 0
      for (let k = 0; k < 4; k++) sum += a[k * 4 + row] * b[col * 4 + k]
      out[col * 4 + row] = sum
    }
  }
  return out
}

/** WebGPU 클립 공간(z 0~1, 오른손 좌표계, -Z를 바라봄). */
function perspective(fovY: number, aspect: number, near: number, far: number): Float32Array {
  const f = 1 / Math.tan(fovY / 2)
  const out = new Float32Array(16)
  out[0] = f / aspect
  out[5] = f
  out[10] = far / (near - far)
  out[11] = -1
  out[14] = (far * near) / (near - far)
  return out
}

function translation(x: number, y: number, z: number): Float32Array {
  const out = new Float32Array(16)
  out[0] = 1; out[5] = 1; out[10] = 1; out[15] = 1
  out[12] = x; out[13] = y; out[14] = z
  return out
}

function rotationY(angle: number): Float32Array {
  const c = Math.cos(angle)
  const s = Math.sin(angle)
  const out = new Float32Array(16)
  out[0] = c; out[2] = -s; out[5] = 1; out[8] = s; out[10] = c; out[15] = 1
  return out
}

function rotationX(angle: number): Float32Array {
  const c = Math.cos(angle)
  const s = Math.sin(angle)
  const out = new Float32Array(16)
  out[0] = 1; out[5] = c; out[6] = s; out[9] = -s; out[10] = c; out[15] = 1
  return out
}

// ---------------------------------------------------------------------------
// 큐브 (면당 정점 4개 = 24개, 인덱스 36개)
// ---------------------------------------------------------------------------

const FACES: Array<{ normal: number[]; color: number[]; corners: number[][] }> = [
  { normal: [0, 0, 1], color: [0.95, 0.35, 0.4], corners: [[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]] },
  { normal: [0, 0, -1], color: [0.35, 0.75, 0.95], corners: [[1, -1, -1], [-1, -1, -1], [-1, 1, -1], [1, 1, -1]] },
  { normal: [1, 0, 0], color: [0.4, 0.9, 0.55], corners: [[1, -1, 1], [1, -1, -1], [1, 1, -1], [1, 1, 1]] },
  { normal: [-1, 0, 0], color: [0.95, 0.8, 0.35], corners: [[-1, -1, -1], [-1, -1, 1], [-1, 1, 1], [-1, 1, -1]] },
  { normal: [0, 1, 0], color: [0.75, 0.55, 0.95], corners: [[-1, 1, 1], [1, 1, 1], [1, 1, -1], [-1, 1, -1]] },
  { normal: [0, -1, 0], color: [0.9, 0.6, 0.35], corners: [[-1, -1, -1], [1, -1, -1], [1, -1, 1], [-1, -1, 1]] },
]

function buildCube() {
  const vertices: number[] = []
  const indices: number[] = []
  FACES.forEach((face, faceIndex) => {
    for (const corner of face.corners) {
      vertices.push(corner[0] * 0.8, corner[1] * 0.8, corner[2] * 0.8)
      vertices.push(face.color[0], face.color[1], face.color[2])
      vertices.push(face.normal[0], face.normal[1], face.normal[2])
    }
    const base = faceIndex * 4
    indices.push(base, base + 1, base + 2, base, base + 2, base + 3)
  })
  return { vertices: new Float32Array(vertices), indices: new Uint16Array(indices) }
}

function setup({ device, context, format }: SceneContext) {
  const { vertices, indices } = buildCube()
  const module = device.createShaderModule({ code: SHADER, label: 'cube' })

  const vertexBuffer = device.createBuffer({
    size: vertices.byteLength,
    usage: GPUBufferUsage.VERTEX,
    mappedAtCreation: true,
  })
  new Float32Array(vertexBuffer.getMappedRange()).set(vertices)
  vertexBuffer.unmap()

  const indexBuffer = device.createBuffer({
    size: indices.byteLength,
    usage: GPUBufferUsage.INDEX,
    mappedAtCreation: true,
  })
  new Uint16Array(indexBuffer.getMappedRange()).set(indices)
  indexBuffer.unmap()

  const uniformBuffer = device.createBuffer({
    size: 64,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })

  const pipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: {
      module,
      entryPoint: 'vs_main',
      buffers: [{
        arrayStride: 36, // vec3 위치 + vec3 색 + vec3 법선
        attributes: [
          { format: 'float32x3', offset: 0, shaderLocation: 0 },
          { format: 'float32x3', offset: 12, shaderLocation: 1 },
          { format: 'float32x3', offset: 24, shaderLocation: 2 },
        ],
      }],
    },
    fragment: { module, entryPoint: 'fs_main', targets: [{ format }] },
    primitive: { topology: 'triangle-list', cullMode: 'back', frontFace: 'ccw' },
    depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' },
  })

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
  })

  // 깊이 텍스처는 캔버스 크기가 바뀔 때만 다시 만든다.
  let depthView: any = null
  let depthWidth = 0
  let depthHeight = 0
  function ensureDepth(width: number, height: number) {
    if (depthView && depthWidth === width && depthHeight === height) return
    const texture = device.createTexture({
      size: { width, height },
      format: 'depth32float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    })
    depthView = texture.createView()
    depthWidth = width
    depthHeight = height
  }

  let angle = 0

  return ({ delta, width, height }: { delta: number; width: number; height: number }) => {
    ensureDepth(width, height)
    angle += (delta / 1000) * 0.8

    const aspect = width / height
    const projection = perspective(Math.PI / 4, aspect, 0.1, 100)
    // 세로 화면은 수평 화각이 좁아져 물체가 크게 잡힌다 — 짧은 축에 맞춰 카메라를 뒤로 뺀다.
    const view = translation(0, 0, -4.5 / Math.min(1, aspect))
    const model = multiply(rotationY(angle), rotationX(angle * 0.55))
    const mvp = multiply(projection, multiply(view, model))
    device.queue.writeBuffer(uniformBuffer, 0, mvp)

    const encoder = device.createCommandEncoder()
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: 0.043, g: 0.055, b: 0.08, a: 1 },
      }],
      depthStencilAttachment: {
        view: depthView,
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    })
    pass.setPipeline(pipeline)
    pass.setBindGroup(0, bindGroup)
    pass.setVertexBuffer(0, vertexBuffer)
    pass.setIndexBuffer(indexBuffer, 'uint16')
    pass.drawIndexed(indices.length)
    pass.end()
    device.queue.submit([encoder.finish()])
  }
}

root.render(
  <DemoScene title="3D 큐브" subtitle="인덱스 드로우 + 깊이 테스트 + MVP" setup={setup} />
)
