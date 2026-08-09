import { root } from '@lynx-js/react'
import { DemoScene, type SceneContext } from '../scene.jsx'
import { GPUTextureUsage } from '../webgpu.js'

/**
 * Dynamic texture — a CPU-built plasma uploaded **with writeTexture every frame**.
 *
 * It was an impossible profile back when writeTexture waited for the GPU to finish on every call.
 * Now it rides the queue order via staging plus a blit (the staging is pooled and reused too), so a
 * 128×128 RGBA (64KB) upload every frame does not stall the frame. The bytes cross as an ArrayBuffer as they are.
 *
 * The HUD's live object count is the `objects` from the submit response — staying constant across frames is normal
 * (docs/JS-AUTHORING.md §8).
 */
const SHADER = /* wgsl */ `
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

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
  // Whether the upload really runs every frame is visible by eye — the texels themselves flow.
  return textureSample(tex, samp, in.uv);
}
`

const SIZE = 128

/** A 256-color palette — a gradient running deep sea → teal → amber. */
function makePalette() {
  const palette = new Uint8Array(256 * 4)
  for (let index = 0; index < 256; index++) {
    const angle = (index / 256) * Math.PI * 2
    palette[index * 4 + 0] = Math.round(128 + 110 * Math.sin(angle))
    palette[index * 4 + 1] = Math.round(96 + 90 * Math.sin(angle + 2.1))
    palette[index * 4 + 2] = Math.round(150 + 100 * Math.sin(angle + 4.2))
    palette[index * 4 + 3] = 255
  }
  return palette
}

/** A per-pixel sin loop is expensive on PrimJS — the waveforms are precomputed per axis and only combined. */
function makeWaveTable() {
  const table = new Uint8Array(512)
  for (let index = 0; index < 512; index++) {
    table[index] = Math.round(64 + 63 * Math.sin((index / 512) * Math.PI * 2))
  }
  return table
}

function setup({ device, context, format, report }: SceneContext) {
  const module = device.createShaderModule({ code: SHADER, label: 'dynamic' })
  const texture = device.createTexture({
    size: { width: SIZE, height: SIZE },
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    label: 'plasma',
  })
  const sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' })
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
    ],
  })

  const palette = makePalette()
  const wave = makeWaveTable()
  const pixels = new Uint8Array(SIZE * SIZE * 4) // reused — not allocated per frame
  const rowWave = new Uint16Array(SIZE)
  const colWave = new Uint16Array(SIZE)

  let time = 0
  let frames = 0

  return ({ delta }: { delta: number }) => {
    time += delta

    // The plasma: the per-axis waveforms are built first, and the pixel loop only adds and looks up the palette.
    const t1 = (time * 0.11) | 0
    const t2 = (time * 0.07) | 0
    const t3 = (time * 0.05) | 0
    for (let i = 0; i < SIZE; i++) {
      rowWave[i] = wave[(i * 3 + t1) & 511] + wave[(i * 7 + t2) & 511]
      colWave[i] = wave[(i * 5 + t2) & 511] + wave[(i * 2 + t3) & 511]
    }
    let offset = 0
    for (let y = 0; y < SIZE; y++) {
      const base = rowWave[y]
      for (let x = 0; x < SIZE; x++) {
        const paletteIndex = ((base + colWave[x] + wave[(x + y + t1) & 511]) & 255) * 4
        pixels[offset++] = palette[paletteIndex]
        pixels[offset++] = palette[paletteIndex + 1]
        pixels[offset++] = palette[paletteIndex + 2]
        pixels[offset++] = 255
      }
    }

    // The heart of this scene — a 64KB texture upload every frame.
    device.queue.writeTexture(
      { texture },
      pixels,
      { bytesPerRow: SIZE * 4 },
      { width: SIZE, height: SIZE }
    )

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
    const result = device.queue.submit([encoder.finish()])

    // A constant live object count means there is no missing destroy on the frame path.
    if (++frames % 120 === 0 && result && typeof result.objects === 'number') {
      report(`${result.objects} live GPU objects · ${(SIZE * SIZE * 4) / 1024}KB uploaded per frame`)
    }
  }
}

root.render(
  <DemoScene
    title="Dynamic texture"
    subtitle="A CPU plasma through writeTexture every frame — queue-ordered upload + a staging pool"
    setup={setup}
  />
)
