import { root } from '@lynx-js/react'
import { DemoScene, type SceneContext } from '../scene.jsx'
import { GPUBufferUsage } from '../webgpu.js'

const COUNT = 3

const SHADER = /* wgsl */ `
struct Circle {
  center: vec2f,
  radius: f32,
  _pad: f32,
  color: vec4f,
};

struct Uniforms {
  circles: array<Circle, ${COUNT}>,
  aspect: f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) offset: vec2f,
  @location(1) color: vec4f,
};

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32,
           @builtin(instance_index) instanceIndex: u32) -> VertexOutput {
  var corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
    vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
  );
  let corner = corners[vertexIndex];
  let circle = uniforms.circles[instanceIndex];

  var out: VertexOutput;
  out.position = vec4f(
    circle.center + vec2f(corner.x * circle.radius / uniforms.aspect, corner.y * circle.radius),
    0.0, 1.0
  );
  out.offset = corner;
  out.color = circle.color;
  return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
  let distance = length(in.offset);
  if (distance > 1.0) {
    discard;
  }
  // Only the edge is softened — the alpha compositing reads clearly where they overlap.
  let edge = smoothstep(1.0, 0.9, distance);
  return vec4f(in.color.rgb * in.color.a * edge, in.color.a * edge);
}
`

function setup({ device, context, format }: SceneContext) {
  const module = device.createShaderModule({ code: SHADER, label: 'blending' })

  // Circle 32B × 3 + aspect(f32) → aligned to 16 → 112B
  const uniformBuffer = device.createBuffer({
    size: 112,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })

  const pipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module, entryPoint: 'vs_main' },
    fragment: {
      module,
      entryPoint: 'fs_main',
      targets: [{
        format,
        // Premultiplied alpha compositing — the fragment already multiplies alpha into RGB, so
        // srcFactor is `one` rather than `src-alpha`. The result is src·a + dst·(1−a).
        //
        // The canvas format is `bgra8unorm` (not sRGB), so compositing happens **on gamma-encoded values**.
        // It is the same behaviour as using a non-sRGB canvas in a browser, but the overlaps look muddier
        // than physically correct linear-space compositing. Use `bgra8unorm-srgb` if you want linear compositing.
        blend: {
          color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
        },
      }],
    },
  })

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
  })

  const data = new Float32Array(28) // 112 / 4
  const colors = [
    [1.0, 0.25, 0.3, 0.55],
    [0.25, 0.9, 0.45, 0.55],
    [0.3, 0.5, 1.0, 0.55],
  ]
  let time = 0

  return ({ delta, width, height }: { delta: number; width: number; height: number }) => {
    time += delta / 1000

    // The circles are sized against y — fitted to the short axis so they do not spread along x on a portrait screen.
    const aspect = Math.max(width / height, 0.001)
    const fit = Math.min(1, aspect)
    const radius = 0.42 * fit
    const orbit = 0.3 * fit

    for (let index = 0; index < COUNT; index++) {
      const angle = time * 0.7 + (index * Math.PI * 2) / COUNT
      const base = index * 8 // Circle = 32B = 8 f32s
      data[base + 0] = (Math.cos(angle) * orbit) / aspect   // center.x
      data[base + 1] = Math.sin(angle) * orbit              // center.y
      data[base + 2] = radius
      data[base + 3] = 0                                // _pad
      data[base + 4] = colors[index][0]
      data[base + 5] = colors[index][1]
      data[base + 6] = colors[index][2]
      data[base + 7] = colors[index][3]
    }
    data[24] = aspect                                     // aspect (offset 96B)
    device.queue.writeBuffer(uniformBuffer, 0, data)

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
    pass.draw(6, COUNT)
    pass.end()
    device.queue.submit([encoder.finish()])
  }
}

root.render(
  <DemoScene title="Alpha blending" subtitle="Overlapping translucent circles + discard + smoothstep" setup={setup} />
)
