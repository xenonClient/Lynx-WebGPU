import { root } from '@lynx-js/react'
import { DemoScene, type SceneContext } from '../scene.jsx'
import { GPUBufferUsage, GPUTextureUsage } from '../webgpu.js'

const SHADER = /* wgsl */ `
struct Uniforms {
  time: f32,
  aspect: f32,
};
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOutput {
  var corners = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var out: VertexOutput;
  out.position = vec4f(corners[index], 0.0, 1.0);
  out.uv = corners[index] * 0.5 + vec2f(0.5, 0.5);
  return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
  // The UV is rotated and scaled up, to see the sampler's repeat mode and filtering by eye.
  let centered = (in.uv - vec2f(0.5, 0.5)) * vec2f(uniforms.aspect, 1.0);
  let s = sin(uniforms.time * 0.25);
  let c = cos(uniforms.time * 0.25);
  let rotated = vec2f(centered.x * c - centered.y * s, centered.x * s + centered.y * c);
  let zoom = 2.0 + 1.5 * sin(uniforms.time * 0.4);
  return textureSample(tex, samp, rotated * zoom + vec2f(0.5, 0.5));
}
`

/** Builds an 8×8 checkerboard on the CPU and uploads it with writeTexture. */
function makeCheckerboard(size: number) {
  const pixels = new Uint8Array(size * size * 4)
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const index = (y * size + x) * 4
      const light = (x + y) % 2 === 0
      const tint = (x * 31 + y * 17) % 96
      pixels[index + 0] = light ? 235 : 40 + tint
      pixels[index + 1] = light ? 120 + tint : 60
      pixels[index + 2] = light ? 90 : 150 + tint
      pixels[index + 3] = 255
    }
  }
  return pixels
}

function setup({ device, context, format }: SceneContext) {
  const SIZE = 8
  const module = device.createShaderModule({ code: SHADER, label: 'texture' })

  const texture = device.createTexture({
    size: { width: SIZE, height: SIZE },
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    label: 'checkerboard',
  })
  device.queue.writeTexture(
    { texture },
    makeCheckerboard(SIZE),
    { bytesPerRow: SIZE * 4 },
    { width: SIZE, height: SIZE }
  )

  const sampler = device.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
    addressModeU: 'repeat',
    addressModeV: 'repeat',
  })

  const uniformBuffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })

  const pipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module, entryPoint: 'vs_main' },
    fragment: { module, entryPoint: 'fs_main', targets: [{ format }] },
  })

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: texture.createView() },
      { binding: 1, resource: sampler },
      { binding: 2, resource: { buffer: uniformBuffer } },
    ],
  })

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
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
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
  <DemoScene title="Texture · sampler" subtitle="writeTexture + a repeat sampler + textureSample" setup={setup} />
)
