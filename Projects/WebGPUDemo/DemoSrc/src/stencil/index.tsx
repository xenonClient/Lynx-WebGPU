import { root, useInitData, useRef, useState } from '@lynx-js/react'
import { DemoScene, type SceneContext } from '../scene.jsx'
import { GPUBufferUsage, GPUTextureUsage } from '../webgpu.js'

/**
 * A stencil mask — a rotating star shape splits the screen into two regions.
 *
 * All three draws are **the same fullscreen triangle**. The only reason the picture splits is the stencil,
 * so if the stencil does not take, the screen is covered in a single color — instantly distinguishable by eye.
 *
 * It uses the standalone `stencil8` format with no depth. This combination used to kill the app with a
 * Metal assertion at pipeline creation (one of the bugs fixed alongside, in `docs/ROADMAP.md`).
 */
const SHADER = /* wgsl */ `
struct Uniforms {
  time: f32,
  aspect: f32,
  spikes: f32,
  style: f32,     // 0 = the dark side, 1 = the bright side
};
@group(0) @binding(0) var<uniform> u: Uniforms;

struct Out {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> Out {
  var corners = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var out: Out;
  out.position = vec4f(corners[index], 0.0, 1.0);
  out.uv = corners[index];
  return out;
}

// Corrects for aspect and moves into coordinates where **the short side is ±1**. The shape stays circular
// on a portrait screen without overflowing sideways.
fn fitted(uv: vec2f) -> vec2f {
  return vec2f(uv.x * u.aspect, uv.y) / min(u.aspect, 1.0);
}

// The mask pass — everything outside the star is discarded. The point is that **a discarded fragment does not write the stencil either**.
// Color is blocked by writeMask 0, so nothing is left on screen.
@fragment
fn fs_mask(in: Out) -> @location(0) vec4f {
  let p = fitted(in.uv);
  let angle = atan2(p.y, p.x) + u.time * 0.35;
  let radius = 0.62 + 0.22 * cos(angle * u.spikes);
  if (length(p) > radius) {
    discard;
  }
  return vec4f(1.0, 1.0, 1.0, 1.0);
}

// The fill pass — drawn twice, differing only in the stencil comparison (equal / not-equal).
@fragment
fn fs_fill(in: Out) -> @location(0) vec4f {
  let p = fitted(in.uv);
  let wave = sin(length(p) * 5.0 - u.time * 1.6) * 0.5 + 0.5;
  let bright = mix(vec3f(0.15, 0.55, 1.0), vec3f(1.0, 0.45, 0.75), wave);
  let dark = vec3f(0.05, 0.07, 0.11) + vec3f(0.04, 0.05, 0.07) * wave;
  return vec4f(mix(dark, bright, u.style), 1.0);
}
`

const SPIKES = 5

function setup({ device, context, format, report }: SceneContext, invertedRef: { current: boolean }) {
  const module = device.createShaderModule({ code: SHADER, label: 'stencil' })

  /** Pipelines differing only in stencil state. The color target and shaders are the same in all three. */
  // Per the spec, a `layout:"auto"` derived layout is **exclusive to that pipeline** — for the three to
  // share a bind group the layout must be explicit (Dawn and browsers refuse the reuse).
  const sharedBindLayout = device.createBindGroupLayout({
    entries: [{ binding: 0, visibility: 0x3 /* VERTEX|FRAGMENT */, buffer: {} }],
  })
  const sharedLayout = device.createPipelineLayout({ bindGroupLayouts: [sharedBindLayout] })

  function makePipeline(options: { compare: string; passOp?: string; writesColor: boolean }) {
    return device.createRenderPipeline({
      layout: sharedLayout,
      vertex: { module, entryPoint: 'vs_main' },
      fragment: {
        module,
        entryPoint: options.writesColor ? 'fs_fill' : 'fs_mask',
        // The masking pass blocks color and leaves only the stencil.
        targets: [{ format, writeMask: options.writesColor ? 0xf : 0 }],
      },
      depthStencil: {
        format: 'stencil8',
        stencilFront: { compare: options.compare, passOp: options.passOp ?? 'keep' },
        stencilBack: { compare: options.compare, passOp: options.passOp ?? 'keep' },
      },
    })
  }

  const maskPipeline = makePipeline({ compare: 'always', passOp: 'replace', writesColor: false })
  const insidePipeline = makePipeline({ compare: 'equal', writesColor: true })
  const outsidePipeline = makePipeline({ compare: 'not-equal', writesColor: true })

  // Being an explicit shared layout, the three pipelines can share a bind group.
  const layout = sharedBindLayout
  const brightUniforms = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    label: 'stencil.bright',
  })
  const darkUniforms = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    label: 'stencil.dark',
  })
  const brightGroup = device.createBindGroup({
    layout,
    entries: [{ binding: 0, resource: { buffer: brightUniforms } }],
  })
  const darkGroup = device.createBindGroup({
    layout,
    entries: [{ binding: 0, resource: { buffer: darkUniforms } }],
  })

  // The stencil attachment follows the canvas size — it is rebuilt on rotation and resize.
  let stencilTexture: any = null
  let stencilView: any = null
  let stencilWidth = 0
  let stencilHeight = 0

  function ensureStencil(width: number, height: number) {
    if (stencilTexture && stencilWidth === width && stencilHeight === height) return
    if (stencilTexture) stencilTexture.destroy()
    stencilTexture = device.createTexture({
      size: { width, height },
      format: 'stencil8',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
      label: 'stencil.mask',
    })
    stencilView = stencilTexture.createView()
    stencilWidth = width
    stencilHeight = height
    report(`stencil8 attachment ${width}×${height} — stencil only, no depth`)
  }

  const bright = new Float32Array([0, 1, SPIKES, 1])
  const dark = new Float32Array([0, 1, SPIKES, 0])
  let time = 0

  return ({ delta, width, height }: { delta: number; width: number; height: number }) => {
    time += delta / 1000
    ensureStencil(width, height)

    bright[0] = time
    bright[1] = width / height
    dark[0] = time
    dark[1] = width / height
    device.queue.writeBuffer(brightUniforms, 0, bright)
    device.queue.writeBuffer(darkUniforms, 0, dark)

  // Inverted, the **inside** of the star goes dark — it is just reading the same mask the other way.
    const inside = invertedRef.current ? darkGroup : brightGroup
    const outside = invertedRef.current ? brightGroup : darkGroup

    const encoder = device.createCommandEncoder()
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: 0.02, g: 0.03, b: 0.05, a: 1 },
      }],
      depthStencilAttachment: {
        view: stencilView,
        stencilClearValue: 0,
        stencilLoadOp: 'clear',
        stencilStoreOp: 'store',
      },
    })

    // 1) Leave a stencil 1 where the star shape is (no color left behind).
    pass.setStencilReference(1)
    pass.setPipeline(maskPipeline)
    pass.setBindGroup(0, brightGroup)
    pass.draw(3)

    // 2) Where the stencil is 1 / 3) where it is not — the same triangle, a different comparison.
    pass.setPipeline(insidePipeline)
    pass.setBindGroup(0, inside)
    pass.draw(3)

    pass.setPipeline(outsidePipeline)
    pass.setBindGroup(0, outside)
    pass.draw(3)

    pass.end()
    device.queue.submit([encoder.finish()])
  }
}

function StencilScene() {
  // With `-altMode 1` it starts with the mask inverted (for automated capture — the same convention as the `bundle` scene).
  // Lynx may move a boolean across as a number, so it is read as truthy rather than ===.
  const alt = !!(useInitData() as { altMode?: unknown } | undefined)?.altMode
  const [inverted, setInverted] = useState(alt)

  // The frame loop keeps using the closure from setup time — the latest value is handed over through a ref.
  const invertedRef = useRef(alt)
  invertedRef.current = inverted

  return (
    <DemoScene
      title="Stencil mask"
      subtitle="The same fullscreen triangle three times — the stencil is the only reason they differ"
      setup={(scene) => setup(scene, invertedRef)}
      controls={
        <view className="controls">
          <text className="control-value">
            {inverted ? 'compare: bright outside the star' : 'compare: bright inside the star'}
          </text>
          <text
            className={inverted ? 'control-button control-button-on' : 'control-button'}
            bindtap={() => setInverted((value) => !value)}
          >
            Invert mask
          </text>
        </view>
      }
    />
  )
}

root.render(<StencilScene />)
