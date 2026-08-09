import { root, useRef, useState } from '@lynx-js/react'
import { DemoScene, type SceneContext } from '../scene.jsx'
import { GPUBufferUsage, GPUTextureUsage, loadAsset } from '../webgpu.js'

/**
 * A scene that restores an HDR photo (Apple's gain map form), pushes it through `rgba16float`, and
 * actually sends those values all the way to the screen.
 *
 * The HDR HEIC an iPhone shoots has the structure "an SDR base + a gain map + headroom". The original
 * brightness is restored as
 *
 *     HDR_linear = SDR_linear × pow(headroom, gain)
 *
 * and this asset has headroom 4.89 with a gain map maximum of 248/255, so it **rises to 4.69×**.
 * A value an 8-bit UNORM texture cannot hold.
 *
 * The screen is split left and right with the same processing applied — the 8-bit original on the left,
 * the reconstructed values on the right. Drag the boundary with a finger to compare the same spot on both sides.
 *
 * There are three view modes:
 *   - **compare**  Reinhard tone mapping plus sRGB. Squeezed onto an SDR screen, the right side actually
 *                  looks brighter and more crushed — that is the essential limit of tone mapping.
 *   - **clipping** Only pixels whose original linear value exceeds 1.0 go red. Nothing on the left with
 *                  something on the right means the reconstruction really happened (judged independently of exposure).
 *   - **EDR**      The canvas is reconfigured to `rgba16float` + `toneMapping: 'extended'` and the linear
 *                  values go out as they are. Tone mapping becomes unnecessary and the display really does
 *                  render brighter, in the headroom above SDR white. **Only verifiable on a device.**
 */

/** The workgroup edge of the reconstruction pass. */
const WORKGROUP = 8

/** The view mode. The same values as the shader's `mode` uniform. */
const MODE_COMPARE = 0
const MODE_CLIPPING = 1
const MODE_EDR = 2

/**
 * Multiplies the gain map to make linear HDR values and writes them into an `rgba16float` storage texture.
 *
 * The gain map is half the base's resolution, so picking the nearest with `textureLoad` shows block boundaries.
 * It is read interpolated through a sampler — a compute shader must state the LOD, hence `textureSampleLevel`.
 */
const RECONSTRUCT_SHADER = /* wgsl */ `
struct Params {
  headroom: f32,
};

@group(0) @binding(0) var baseTex: texture_2d<f32>;
@group(0) @binding(1) var gainTex: texture_2d<f32>;
@group(0) @binding(2) var gainSampler: sampler;
@group(0) @binding(3) var hdrOut: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var<uniform> params: Params;

fn srgbToLinear(c: vec3f) -> vec3f {
  let lo = c / 12.92;
  let hi = pow((c + vec3f(0.055, 0.055, 0.055)) / 1.055, vec3f(2.4, 2.4, 2.4));
  return select(hi, lo, c <= vec3f(0.04045, 0.04045, 0.04045));
}

@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(hdrOut);
  if (id.x >= size.x || id.y >= size.y) {
    return;
  }
  let coord = vec2i(i32(id.x), i32(id.y));
  let uv = (vec2f(f32(id.x), f32(id.y)) + vec2f(0.5, 0.5)) / vec2f(f32(size.x), f32(size.y));

  let sdr = textureLoad(baseTex, coord, 0).rgb;
  let gain = textureSampleLevel(gainTex, gainSampler, uv, 0.0).r;

  // The gain map is an exponent normalized to 0..1. 1.0 means headroom times, 0 means unchanged.
  let scale = pow(params.headroom, gain);
  textureStore(hdrOut, coord, vec4f(srgbToLinear(sdr) * scale, 1.0));
}
`

/**
 * Draws both sides under identical conditions. The only difference between left and right is **whether the
 * input is 8-bit or 16-bit float**.
 */
const PRESENT_SHADER = /* wgsl */ `
struct Uniforms {
  exposure: f32,      // in stops
  wipe: f32,          // 0..1, the boundary position
  screenAspect: f32,  // the canvas width/height
  imageAspect: f32,   // the image width/height
  mode: f32,          // 0 compare · 1 clipping · 2 EDR
  peak: f32,          // this photo's measured maximum multiplier (for the clipping display intensity)
  pad0: f32,
  pad1: f32,
};

@group(0) @binding(0) var hdrTex: texture_2d<f32>;
@group(0) @binding(1) var baseTex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<uniform> u: Uniforms;

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOutput {
  var corners = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  let corner = corners[index];
  var out: VertexOutput;
  out.position = vec4f(corner, 0.0, 1.0);
  // Texture coordinates have 0 at the top, so y is flipped.
  out.uv = vec2f(corner.x * 0.5 + 0.5, 0.5 - corner.y * 0.5);
  return out;
}

fn srgbToLinear(c: vec3f) -> vec3f {
  let lo = c / 12.92;
  let hi = pow((c + vec3f(0.055, 0.055, 0.055)) / 1.055, vec3f(2.4, 2.4, 2.4));
  return select(hi, lo, c <= vec3f(0.04045, 0.04045, 0.04045));
}

fn linearToSrgb(c: vec3f) -> vec3f {
  let lo = c * 12.92;
  let hi = 1.055 * pow(c, vec3f(0.4166667, 0.4166667, 0.4166667)) - vec3f(0.055, 0.055, 0.055);
  return select(hi, lo, c <= vec3f(0.0031308, 0.0031308, 0.0031308));
}

/** Reinhard — squeezes values above 1.0 into 0..1. */
fn tonemap(c: vec3f) -> vec3f {
  return c / (vec3f(1.0, 1.0, 1.0) + c);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
  // Fitted inside the canvas so the whole image is visible (contain).
  var coverage = vec2f(1.0, 1.0);
  if (u.screenAspect > u.imageAspect) {
    coverage.x = u.imageAspect / u.screenAspect;
  } else {
    coverage.y = u.screenAspect / u.imageAspect;
  }
  let uv = (in.uv - vec2f(0.5, 0.5)) / coverage + vec2f(0.5, 0.5);

  // The samples are taken **before** the early returns (in uniform control flow) — a textureSample after a
  // varying branch or return is a uniformity violation and the spec validator (Dawn/browsers) rejects it.
  // The 8-bit original already has its highlights clipped at 1.0, while the gain map side goes far past 1.0.
  let baseLinear = srgbToLinear(textureSample(baseTex, samp, uv).rgb);
  let hdrLinear = textureSample(hdrTex, samp, uv).rgb;

  // The margins and boundary line go out without encoding. Low values are used so they do not blow out under EDR.
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
    return vec4f(0.008, 0.010, 0.016, 1.0);
  }
  if (abs(in.uv.x - u.wipe) < 0.0018) {
    return vec4f(1.0, 0.72, 0.15, 1.0);
  }

  let linear = select(hdrLinear, baseLinear, in.uv.x < u.wipe);

  // The clipping display — regardless of exposure, it only asks "does the original value exceed 1.0".
  if (u.mode > 0.5 && u.mode < 1.5) {
    let luma = dot(linear, vec3f(0.2126, 0.7152, 0.0722));
    if (luma > 1.0) {
      let over = min((luma - 1.0) / max(u.peak - 1.0, 0.001), 1.0);
      return vec4f(1.0, 0.72 - over * 0.66, 0.12, 1.0);
    }
    let grey = linearToSrgb(vec3f(luma, luma, luma) * 0.4);
    return vec4f(grey, 1.0);
  }

  let exposed = linear * pow(2.0, u.exposure);

  // EDR — the layer is an extended **linear** color space, so nothing is encoded. Values above 1.0 go out
  // as they are and the display renders them that much brighter. Tone mapping is unnecessary too.
  if (u.mode > 1.5) {
    return vec4f(exposed, 1.0);
  }

  return vec4f(linearToSrgb(tonemap(exposed)), 1.0);
}
`

/** The header `Tools/extract-hdr-asset.swift` wrote. */
function parseHeader(buffer: ArrayBuffer) {
  const view = new DataView(buffer)
  const magic = String.fromCharCode(
    view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3)
  )
  if (magic !== 'LWGH') throw new Error(`the asset magic differs: ${magic}`)

  return {
    version: view.getUint32(4, true),
    baseWidth: view.getUint32(8, true),
    baseHeight: view.getUint32(12, true),
    gainWidth: view.getUint32(16, true),
    gainHeight: view.getUint32(20, true),
    headroom: view.getFloat32(24, true),
    baseOffset: view.getUint32(28, true),
    baseLength: view.getUint32(32, true),
    gainOffset: view.getUint32(36, true),
    gainLength: view.getUint32(40, true),
  }
}

async function setup(
  { device, context, format, report, pointer }: SceneContext,
  exposureRef: { current: number },
  modeRef: { current: number }
) {
  const buffer = await loadAsset('hdr-sample.bin')
  const header = parseHeader(buffer)
  if (header.version !== 1) throw new Error(`unknown asset version: ${header.version}`)

  const basePixels = new Uint8Array(buffer, header.baseOffset, header.baseLength)
  const gainPixels = new Uint8Array(buffer, header.gainOffset, header.gainLength)

  // The photo's actual maximum multiplier is computed up front from the gain map's real maximum.
  let peakGain = 0
  for (let i = 0; i < gainPixels.length; i++) {
    if (gainPixels[i] > peakGain) peakGain = gainPixels[i]
  }
  const peakScale = Math.pow(header.headroom, peakGain / 255)

  // --- Textures ------------------------------------------------------------

  const baseTexture = device.createTexture({
    size: { width: header.baseWidth, height: header.baseHeight },
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    label: 'hdr.base',
  })
  device.queue.writeTexture(
    { texture: baseTexture },
    basePixels,
    { bytesPerRow: header.baseWidth * 4 },
    { width: header.baseWidth, height: header.baseHeight }
  )

  const gainTexture = device.createTexture({
    size: { width: header.gainWidth, height: header.gainHeight },
    format: 'r8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    label: 'hdr.gainmap',
  })
  device.queue.writeTexture(
    { texture: gainTexture },
    gainPixels,
    { bytesPerRow: header.gainWidth },
    { width: header.gainWidth, height: header.gainHeight }
  )

  /** The heart of this scene — the intermediate texture that holds values above 1.0. */
  const hdrTexture = device.createTexture({
    size: { width: header.baseWidth, height: header.baseHeight },
    format: 'rgba16float',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    label: 'hdr.linear',
  })

  const sampler = device.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
  })

  // --- The reconstruction pass ---------------------------------------------

  const reconstructModule = device.createShaderModule({
    code: RECONSTRUCT_SHADER,
    label: 'hdr.reconstruct',
  })
  const reconstructParams = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    label: 'hdr.params',
  })
  device.queue.writeBuffer(reconstructParams, 0, new Float32Array([header.headroom, 0, 0, 0]))

  const reconstructPipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module: reconstructModule, entryPoint: 'main' },
  })
  const reconstructBindGroup = device.createBindGroup({
    layout: reconstructPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: baseTexture.createView() },
      { binding: 1, resource: gainTexture.createView() },
      { binding: 2, resource: sampler },
      { binding: 3, resource: hdrTexture.createView() },
      { binding: 4, resource: { buffer: reconstructParams } },
    ],
  })

  // --- The display pass ----------------------------------------------------

  const presentModule = device.createShaderModule({ code: PRESENT_SHADER, label: 'hdr.present' })
  const presentUniforms = device.createBuffer({
    size: 32,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    label: 'hdr.present.uniforms',
  })

  /**
   * When the canvas format changes, the pipeline has to have been built for that format too.
   * Both are prepared up front so nothing is built each time EDR is toggled.
   */
  function makePresent(targetFormat: string) {
    const pipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module: presentModule, entryPoint: 'vs_main' },
      fragment: {
        module: presentModule,
        entryPoint: 'fs_main',
        targets: [{ format: targetFormat }],
      },
    })
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: hdrTexture.createView() },
        { binding: 1, resource: baseTexture.createView() },
        { binding: 2, resource: sampler },
        { binding: 3, resource: { buffer: presentUniforms } },
      ],
    })
    return { pipeline, bindGroup }
  }

  const sdrPresent = makePresent(format)
  const edrPresent = makePresent('rgba16float')

  report(
    `${header.baseWidth}×${header.baseHeight} · headroom ${header.headroom.toFixed(2)} · ` +
    `measured maximum ${peakScale.toFixed(2)}× · drag to move the boundary`
  )

  const workgroupsX = Math.ceil(header.baseWidth / WORKGROUP)
  const workgroupsY = Math.ceil(header.baseHeight / WORKGROUP)
  const imageAspect = header.baseWidth / header.baseHeight
  const uniforms = new Float32Array(8)

  // The gain map does not change, so the reconstruction runs once on the first frame only.
  let reconstructed = false
  let wipe = 0.5
  let configuredEdr = false
  let settleFrames = 0

  return ({ width, height }: { delta: number; width: number; height: number }) => {
    const mode = modeRef.current
    const wantEdr = mode === MODE_EDR

    if (wantEdr !== configuredEdr) {
      context.configure({
        device,
        format: wantEdr ? 'rgba16float' : format,
        colorSpace: 'srgb',
        toneMapping: { mode: wantEdr ? 'extended' : 'standard' },
      })
      configuredEdr = wantEdr
      // CAMetalLayer configuration goes to the main thread asynchronously (`WGPUMetalLayerSurface`).
      // A few frames of rest are needed until the drawable switches to the new format, so it does not disagree with the pipeline.
      settleFrames = 3
    }
    if (settleFrames > 0) {
      settleFrames -= 1
      return
    }

    // The boundary follows only while a finger is down. Released, it stays where it is.
    const point = pointer.current
    if (point) wipe = point.x

    uniforms[0] = exposureRef.current
    uniforms[1] = wipe
    uniforms[2] = width / height
    uniforms[3] = imageAspect
    uniforms[4] = mode
    uniforms[5] = peakScale
    device.queue.writeBuffer(presentUniforms, 0, uniforms)

    const encoder = device.createCommandEncoder()

    if (!reconstructed) {
      const compute = encoder.beginComputePass()
      compute.setPipeline(reconstructPipeline)
      compute.setBindGroup(0, reconstructBindGroup)
      compute.dispatchWorkgroups(workgroupsX, workgroupsY)
      compute.end()
      reconstructed = true
    }

    const present = wantEdr ? edrPresent : sdrPresent
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: 0.008, g: 0.01, b: 0.016, a: 1 },
      }],
    })
    pass.setPipeline(present.pipeline)
    pass.setBindGroup(0, present.bindGroup)
    pass.draw(3)
    pass.end()

    device.queue.submit([encoder.finish()])
  }
}

function HdrScene() {
  const [exposure, setExposure] = useState(0)
  const [mode, setMode] = useState(MODE_COMPARE)

  // The frame loop keeps using the closure from setup time — the latest value is handed over through a ref.
  const exposureRef = useRef(0)
  const modeRef = useRef(MODE_COMPARE)
  exposureRef.current = exposure
  modeRef.current = mode

  return (
    <DemoScene
      title="HDR gain map reconstruction"
      subtitle="rgba16float — the 4.7× highlights 8 bits cannot hold"
      setup={(scene) => setup(scene, exposureRef, modeRef)}
      controls={
        <view className="controls">
          <text
            className="control-button"
            bindtap={() => setExposure((value) => Math.max(value - 0.5, -8))}
          >
            −
          </text>
          <text className="control-value">
            {exposure > 0 ? '+' : ''}{exposure.toFixed(1)} stop
          </text>
          <text
            className="control-button"
            bindtap={() => setExposure((value) => Math.min(value + 0.5, 2))}
          >
            ＋
          </text>
          <text
            className={mode === MODE_CLIPPING ? 'control-button control-button-on' : 'control-button'}
            bindtap={() => setMode((m) => (m === MODE_CLIPPING ? MODE_COMPARE : MODE_CLIPPING))}
          >
            Clipping
          </text>
          <text
            className={mode === MODE_EDR ? 'control-button control-button-on' : 'control-button'}
            bindtap={() => setMode((m) => (m === MODE_EDR ? MODE_COMPARE : MODE_EDR))}
          >
            EDR
          </text>
        </view>
      }
    />
  )
}

root.render(<HdrScene />)
