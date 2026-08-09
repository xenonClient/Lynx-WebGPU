import { root, useInitData, useRef, useState } from '@lynx-js/react'
import { DemoScene, type SceneContext } from '../scene.jsx'
import { GPUBufferUsage } from '../webgpu.js'

/**
 * Render bundles — **they keep JS from rebuilding the same command array every frame.**
 *
 * A bundle's benefit here differs from a browser's. Metal has no corresponding object, so native merely
 * replays the command list and the GPU-side cost is the same. What is saved is **the number of commands crossing the bridge**.
 *
 * So the HUD shows the `commandCount` `submit()` returns as is — not an estimate but the count native
 * actually received. The bundle is reused unchanged even while the tiles move
 * (the pose comes from a uniform and the draw list is fixed).
 */
const COLUMNS = 10
const ROWS = 12
const TILES = COLUMNS * ROWS
/** The NDC range the grid occupies — the tile size is proportional to the cell spacing derived from it. */
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
  // Each tile wobbles at a different phase — the picture moves while the draw list stays.
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
 * Bakes one tile out of 6 vertices — position, color and phase are baked into the vertices, removing draw arguments.
 *
 * The tile size is **proportional to the cell spacing**. Correcting by aspect would overflow the grid
 * sideways on a portrait screen, so it is left as a rectangle following the screen shape.
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

  // The bundle is recorded **once only**. Later frames replay all of it through a single handle.
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

    // What submit returns is the number of commands native **actually received** (not an estimate).
    const result = device.queue.submit([encoder.finish()])
    const count = result && result.commandCount ? result.commandCount : 0
    if (count !== lastReported) {
      lastReported = count
      report(
        `${TILES} tiles · ${count} commands per frame` +
          (bundled ? ' — replayed from one bundle' : ' — one bridge crossing per draw')
      )
    }
  }
}

function BundleScene() {
  // The `-altMode 1` launch argument flips the starting state — the simulator cannot inject touches, so
  // this path is needed to capture the screen with the button pressed (`DemoViewController.initialData`).
  // Lynx may move a boolean across as a number, so it is read as truthy rather than ===.
  const alt = !!(useInitData() as { altMode?: unknown } | undefined)?.altMode
  const [bundled, setBundled] = useState(!alt)

  const bundledRef = useRef(!alt)
  bundledRef.current = bundled

  return (
    <DemoScene
      title="Render bundle"
      subtitle={`${TILES} draws recorded once and replayed every frame`}
      setup={(scene) => setup(scene, bundledRef)}
      controls={
        <view className="controls">
          <text className="control-value">{bundled ? 'bundle reused' : 'recorded every frame'}</text>
          <text
            className={bundled ? 'control-button control-button-on' : 'control-button'}
            bindtap={() => setBundled(true)}
          >
            Bundle
          </text>
          <text
            className={bundled ? 'control-button' : 'control-button control-button-on'}
            bindtap={() => setBundled(false)}
          >
            Direct
          </text>
        </view>
      }
    />
  )
}

root.render(<BundleScene />)
