import { root } from '@lynx-js/react'
import { DemoScene, type SceneContext } from '../scene.jsx'
import { GPUTextureUsage } from '../webgpu.js'

/**
 * 동적 텍스처 — CPU가 만든 플라스마를 **매 프레임 writeTexture로** 올린다.
 *
 * writeTexture가 호출마다 GPU 완주를 기다리던 시절에는 불가능했던 프로파일이다.
 * 지금은 스테이징 + blit으로 큐 순서를 타므로(스테이징도 풀로 재사용) 128×128 RGBA(64KB)를
 * 매 프레임 올려도 프레임이 서지 않는다. 바이트열은 ArrayBuffer로 그대로 건너간다.
 *
 * HUD의 live 객체 수는 submit 응답의 `objects`다 — 프레임을 거듭해도 일정해야 정상이다
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
  // 업로드가 진짜 매 프레임 도는지는 눈으로 보인다 — 텍셀 자체가 흐른다.
  return textureSample(tex, samp, in.uv);
}
`

const SIZE = 128

/** 256색 팔레트 — 심해 → 청록 → 호박색으로 도는 그라데이션. */
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

/** sin 루프를 픽셀마다 돌리면 PrimJS에서 비싸다 — 파형을 축별로 미리 계산해 조합만 한다. */
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
  const pixels = new Uint8Array(SIZE * SIZE * 4) // 재사용 — 프레임마다 할당하지 않는다
  const rowWave = new Uint16Array(SIZE)
  const colWave = new Uint16Array(SIZE)

  let time = 0
  let frames = 0

  return ({ delta }: { delta: number }) => {
    time += delta

    // 플라스마: 축별 파형을 먼저 만들고 픽셀 루프는 덧셈 + 팔레트 조회만 한다.
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

    // 이 씬의 핵심 — 매 프레임 64KB 텍스처 업로드.
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

    // live 객체 수가 일정하면 프레임 경로에 destroy 누락이 없다는 뜻이다.
    if (++frames % 120 === 0 && result && typeof result.objects === 'number') {
      report(`live GPU 객체 ${result.objects}개 · 프레임당 ${(SIZE * SIZE * 4) / 1024}KB 업로드`)
    }
  }
}

root.render(
  <DemoScene
    title="동적 텍스처"
    subtitle="CPU 플라스마를 매 프레임 writeTexture로 — 큐 순서 업로드 + 스테이징 풀"
    setup={setup}
  />
)
