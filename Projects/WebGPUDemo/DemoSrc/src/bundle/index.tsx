import { root, useInitData, useRef, useState } from '@lynx-js/react'
import { DemoScene, type SceneContext } from '../scene.jsx'
import { GPUBufferUsage } from '../webgpu.js'

/**
 * 렌더 번들 — **JS가 매 프레임 같은 커맨드 배열을 다시 만들지 않게 한다.**
 *
 * 이 구현에서 번들의 이득은 브라우저와 다르다. Metal에는 대응 객체가 없어 네이티브가 명령
 * 목록을 되풀이할 뿐이라 GPU 쪽 비용은 같다. 아끼는 것은 **브리지를 건너는 커맨드 수**다.
 *
 * 그래서 HUD에 `submit()`이 돌려주는 `commandCount`를 그대로 띄운다 — 추정이 아니라
 * 네이티브가 실제로 받은 개수다. 타일이 움직이는 동안에도 번들은 그대로 재사용된다
 * (자세는 유니폼에서 오고 드로우 목록은 고정이기 때문).
 */
const COLUMNS = 10
const ROWS = 12
const TILES = COLUMNS * ROWS
/** 격자가 차지하는 NDC 범위 — 타일 크기는 여기서 나온 칸 간격에 비례한다. */
const SPAN_X = 1.76
const SPAN_Y = 1.72

const SHADER = /* wgsl */ `
struct Params {
  time: f32,
  _a: f32,
  _b: f32,
  _c: f32,
};
@group(0) @binding(0) var<uniform> params: Params;

struct Out {
  @builtin(position) position: vec4f,
  @location(0) color: vec3f,
};

@vertex
fn vs_main(@location(0) position: vec2f,
           @location(1) color: vec3f,
           @location(2) seed: f32) -> Out {
  // 타일마다 다른 위상으로 흔들린다 — 그림은 움직여도 드로우 목록은 그대로다.
  let sway = sin(params.time * 1.6 + seed * 6.28318) * 0.014;
  let lift = cos(params.time * 1.1 + seed * 3.14159) * 0.030;
  let pulse = 0.72 + 0.28 * sin(params.time * 2.2 + seed * 9.0);

  var out: Out;
  out.position = vec4f(position.x + sway, position.y + lift, 0.0, 1.0);
  out.color = color * pulse;
  return out;
}

@fragment
fn fs_main(in: Out) -> @location(0) vec4f {
  return vec4f(in.color, 1.0);
}
`

/**
 * 타일 하나를 정점 6개로 굽는다 — 위치·색·위상을 정점에 박아 드로우 인자를 없앤다.
 *
 * 타일 크기는 **칸 간격에 비례**한다. 화면비로 보정하면 세로 화면에서 격자가 좌우로
 * 넘치므로, 화면 모양을 그대로 따라가는 직사각형으로 둔다.
 */
function buildTiles() {
  const floats = new Float32Array(TILES * 6 * 6)
  const halfX = (SPAN_X / (COLUMNS - 1)) * 0.4
  const halfY = (SPAN_Y / (ROWS - 1)) * 0.4
  let cursor = 0

  for (let row = 0; row < ROWS; row++) {
    for (let column = 0; column < COLUMNS; column++) {
      const index = row * COLUMNS + column
      const centerX = -SPAN_X / 2 + (column / (COLUMNS - 1)) * SPAN_X
      const centerY = -SPAN_Y / 2 + (row / (ROWS - 1)) * SPAN_Y
      const hue = index / TILES
      const red = 0.35 + 0.6 * Math.abs(Math.sin(hue * 3.14159 + 0.4))
      const green = 0.3 + 0.55 * Math.abs(Math.sin(hue * 3.14159 + 2.1))
      const blue = 0.45 + 0.5 * Math.abs(Math.sin(hue * 3.14159 + 4.0))
      const seed = hue + row * 0.031

      const corners = [
        [-halfX, -halfY], [halfX, -halfY], [-halfX, halfY],
        [-halfX, halfY], [halfX, -halfY], [halfX, halfY],
      ]
      for (const [offsetX, offsetY] of corners) {
        floats[cursor++] = centerX + offsetX
        floats[cursor++] = centerY + offsetY
        floats[cursor++] = red
        floats[cursor++] = green
        floats[cursor++] = blue
        floats[cursor++] = seed
      }
    }
  }
  return floats
}

function setup({ device, context, format, report }: SceneContext, bundledRef: { current: boolean }) {
  const module = device.createShaderModule({ code: SHADER, label: 'bundle.tiles' })

  const tiles = buildTiles()
  const vertexBuffer = device.createBuffer({
    size: tiles.byteLength,
    usage: GPUBufferUsage.VERTEX,
    mappedAtCreation: true,
    label: 'tiles',
  })
  new Float32Array(vertexBuffer.getMappedRange()).set(tiles)
  vertexBuffer.unmap()

  const params = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })

  const pipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: {
      module,
      entryPoint: 'vs_main',
      buffers: [{
        arrayStride: 24,
        attributes: [
          { format: 'float32x2', offset: 0, shaderLocation: 0 },
          { format: 'float32x3', offset: 8, shaderLocation: 1 },
          { format: 'float32', offset: 20, shaderLocation: 2 },
        ],
      }],
    },
    fragment: { module, entryPoint: 'fs_main', targets: [{ format }] },
  })
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: params } }],
  })

  // 번들은 **한 번만** 기록한다. 이후 프레임은 핸들 하나로 이 전부를 되돌린다.
  const bundleEncoder = device.createRenderBundleEncoder({ colorFormats: [format], label: 'tiles' })
  bundleEncoder.setPipeline(pipeline)
  bundleEncoder.setBindGroup(0, bindGroup)
  bundleEncoder.setVertexBuffer(0, vertexBuffer)
  for (let tile = 0; tile < TILES; tile++) {
    bundleEncoder.draw(6, 1, tile * 6, 0)
  }
  const bundle = bundleEncoder.finish()

  const paramsData = new Float32Array(4)
  let time = 0
  let lastReported = -1

  return ({ delta }: { delta: number }) => {
    time += delta / 1000
    paramsData[0] = time
    device.queue.writeBuffer(params, 0, paramsData)

    const bundled = bundledRef.current
    const encoder = device.createCommandEncoder()
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: 0.03, g: 0.04, b: 0.06, a: 1 },
      }],
    })

    if (bundled) {
      pass.executeBundles([bundle])
    } else {
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, bindGroup)
      pass.setVertexBuffer(0, vertexBuffer)
      for (let tile = 0; tile < TILES; tile++) {
        pass.draw(6, 1, tile * 6, 0)
      }
    }

    pass.end()

    // submit이 돌려주는 것은 네이티브가 **실제로 받은** 커맨드 수다 (추정이 아니다).
    const result = device.queue.submit([encoder.finish()])
    const count = result && result.commandCount ? result.commandCount : 0
    if (count !== lastReported) {
      lastReported = count
      report(
        `타일 ${TILES}개 · 프레임당 커맨드 ${count}개` +
          (bundled ? ' — 번들 하나로 되돌린다' : ' — 드로우마다 한 줄씩 브리지를 건넌다')
      )
    }
  }
}

function BundleScene() {
  // `-altMode 1` 런치 인자로 시작 상태를 뒤집는다 — 시뮬레이터에는 터치 주입이 없어
  // 버튼을 누른 화면을 캡처하려면 이 경로가 필요하다 (`DemoViewController.initialData`).
  // Lynx가 불리언을 숫자로 옮겨 줄 수 있어 === 대신 truthy로 본다.
  const alt = !!(useInitData() as { altMode?: unknown } | undefined)?.altMode
  const [bundled, setBundled] = useState(!alt)

  const bundledRef = useRef(!alt)
  bundledRef.current = bundled

  return (
    <DemoScene
      title="렌더 번들"
      subtitle={`드로우 ${TILES}개를 한 번만 기록해 매 프레임 되돌린다`}
      setup={(scene) => setup(scene, bundledRef)}
      controls={
        <view className="controls">
          <text className="control-value">{bundled ? '번들 재사용' : '매 프레임 기록'}</text>
          <text
            className={bundled ? 'control-button control-button-on' : 'control-button'}
            bindtap={() => setBundled(true)}
          >
            번들
          </text>
          <text
            className={bundled ? 'control-button' : 'control-button control-button-on'}
            bindtap={() => setBundled(false)}
          >
            직접
          </text>
        </view>
      }
    />
  )
}

root.render(<BundleScene />)
