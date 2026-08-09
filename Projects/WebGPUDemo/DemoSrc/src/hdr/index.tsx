import { root, useRef, useState } from '@lynx-js/react'
import { DemoScene, type SceneContext } from '../scene.jsx'
import { GPUBufferUsage, GPUTextureUsage, loadAsset } from '../webgpu.js'

/**
 * HDR 사진(Apple 게인맵 방식)을 되살려 `rgba16float`로 흘려보내고, 그 값을 실제로
 * 화면까지 내보내는 씬.
 *
 * iPhone이 찍는 HDR HEIC는 "SDR 베이스 + 게인맵 + headroom" 구조다. 원래 밝기는
 *
 *     HDR_linear = SDR_linear × pow(headroom, gain)
 *
 * 로 복원되는데, 이 애셋은 headroom 4.89에 게인맵 최대가 248/255라 **4.69배까지 올라간다**.
 * 8비트 UNORM 텍스처로는 담을 수 없는 값이다.
 *
 * 화면은 좌우로 갈라 같은 처리를 적용한다 — 왼쪽은 8비트 원본, 오른쪽은 재구성한 값.
 * 손가락으로 경계를 끌어 같은 지점을 양쪽으로 비교한다.
 *
 * 보기 방식이 셋이다:
 *   - **비교**   Reinhard 톤매핑 + sRGB. SDR 화면에 눌러 담으므로 오른쪽이 되레 더
 *                밝고 뭉쳐 보인다 — 이게 톤매핑의 본질적 한계다.
 *   - **클리핑** 원본 선형값이 1.0을 넘는 픽셀만 빨갛게. 왼쪽엔 뜨지 않고 오른쪽에만
 *                뜨면 재구성이 실제로 일어난 것이다 (노출과 무관하게 판정한다).
 *   - **EDR**    캔버스를 `rgba16float` + `toneMapping: 'extended'`로 다시 configure하고
 *                선형값을 그대로 내보낸다. 톤매핑이 필요 없어지고, 디스플레이가 SDR 흰색
 *                위쪽 여유 밝기로 실제로 더 밝게 낸다. **실기기에서만 확인된다.**
 */

/** 재구성 패스의 워크그룹 한 변. */
const WORKGROUP = 8

/** 보기 방식. 셰이더의 `mode` 유니폼과 같은 값이다. */
const MODE_COMPARE = 0
const MODE_CLIPPING = 1
const MODE_EDR = 2

/**
 * 게인맵을 곱해 선형 HDR 값을 만들어 `rgba16float` 스토리지 텍스처에 쓴다.
 *
 * 게인맵은 베이스의 절반 해상도라 `textureLoad`로 최근접을 집으면 블록 경계가 보인다.
 * 샘플러로 보간해 읽는다 — 컴퓨트에서는 LOD를 명시해야 하므로 `textureSampleLevel`이다.
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

  // 게인맵은 0..1로 정규화된 지수다. 1.0이면 headroom 배, 0이면 그대로.
  let scale = pow(params.headroom, gain);
  textureStore(hdrOut, coord, vec4f(srgbToLinear(sdr) * scale, 1.0));
}
`

/**
 * 좌우를 같은 조건으로 그린다. 왼쪽/오른쪽의 차이는 **입력이 8비트인가 16비트 float인가**
 * 하나뿐이다.
 */
const PRESENT_SHADER = /* wgsl */ `
struct Uniforms {
  exposure: f32,      // stop 단위
  wipe: f32,          // 0..1, 경계 위치
  screenAspect: f32,  // 캔버스 가로/세로
  imageAspect: f32,   // 이미지 가로/세로
  mode: f32,          // 0 비교 · 1 클리핑 · 2 EDR
  peak: f32,          // 이 사진의 실측 최대 배율 (클리핑 표시 강도용)
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
  // 텍스처 좌표는 위가 0이므로 y를 뒤집는다.
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

/** Reinhard — 1.0을 넘는 값을 0..1로 눌러 담는다. */
fn tonemap(c: vec3f) -> vec3f {
  return c / (vec3f(1.0, 1.0, 1.0) + c);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
  // 이미지 전체가 보이도록 캔버스 안에 맞춘다 (contain).
  var coverage = vec2f(1.0, 1.0);
  if (u.screenAspect > u.imageAspect) {
    coverage.x = u.imageAspect / u.screenAspect;
  } else {
    coverage.y = u.screenAspect / u.imageAspect;
  }
  let uv = (in.uv - vec2f(0.5, 0.5)) / coverage + vec2f(0.5, 0.5);

  // 샘플은 이른 return들 **앞**(균일 제어 흐름)에서 뜬다 — varying 분기·return 뒤의
  // textureSample은 uniformity 위반이라 명세 검증기(Dawn/브라우저)가 거부한다.
  // 8비트 원본은 하이라이트가 이미 1.0에서 잘려 있고, 게인맵 쪽은 1.0을 크게 넘는다.
  let baseLinear = srgbToLinear(textureSample(baseTex, samp, uv).rgb);
  let hdrLinear = textureSample(hdrTex, samp, uv).rgb;

  // 여백과 경계선은 인코딩을 거치지 않고 바로 낸다. EDR에서도 튀지 않도록 낮은 값을 쓴다.
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
    return vec4f(0.008, 0.010, 0.016, 1.0);
  }
  if (abs(in.uv.x - u.wipe) < 0.0018) {
    return vec4f(1.0, 0.72, 0.15, 1.0);
  }

  let linear = select(hdrLinear, baseLinear, in.uv.x < u.wipe);

  // 클리핑 표시 — 노출과 무관하게 "원본 값이 1.0을 넘는가"만 본다.
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

  // EDR — 레이어가 확장 **선형** 색공간이라 인코딩하지 않는다. 1.0을 넘는 값이 그대로
  // 나가고 디스플레이가 그만큼 밝게 낸다. 톤매핑도 필요 없다.
  if (u.mode > 1.5) {
    return vec4f(exposed, 1.0);
  }

  return vec4f(linearToSrgb(tonemap(exposed)), 1.0);
}
`

/** `Tools/extract-hdr-asset.swift`가 쓴 헤더. */
function parseHeader(buffer: ArrayBuffer) {
  const view = new DataView(buffer)
  const magic = String.fromCharCode(
    view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3)
  )
  if (magic !== 'LWGH') throw new Error(`애셋 매직이 다르다: ${magic}`)

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
  if (header.version !== 1) throw new Error(`모르는 애셋 버전: ${header.version}`)

  const basePixels = new Uint8Array(buffer, header.baseOffset, header.baseLength)
  const gainPixels = new Uint8Array(buffer, header.gainOffset, header.gainLength)

  // 게인맵의 실제 최대값으로 이 사진이 몇 배까지 올라가는지 계산해 둔다.
  let peakGain = 0
  for (let i = 0; i < gainPixels.length; i++) {
    if (gainPixels[i] > peakGain) peakGain = gainPixels[i]
  }
  const peakScale = Math.pow(header.headroom, peakGain / 255)

  // --- 텍스처 -------------------------------------------------------------

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

  /** 이 씬의 핵심 — 1.0을 넘는 값을 담는 중간 텍스처. */
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

  // --- 재구성 패스 ---------------------------------------------------------

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

  // --- 표시 패스 -----------------------------------------------------------

  const presentModule = device.createShaderModule({ code: PRESENT_SHADER, label: 'hdr.present' })
  const presentUniforms = device.createBuffer({
    size: 32,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    label: 'hdr.present.uniforms',
  })

  /**
   * 캔버스 포맷이 바뀌면 파이프라인도 그 포맷으로 만들어져 있어야 한다.
   * EDR을 켜고 끌 때마다 만들지 않도록 둘 다 미리 준비해 둔다.
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
    `실측 최대 ${peakScale.toFixed(2)}× · 드래그로 경계 이동`
  )

  const workgroupsX = Math.ceil(header.baseWidth / WORKGROUP)
  const workgroupsY = Math.ceil(header.baseHeight / WORKGROUP)
  const imageAspect = header.baseWidth / header.baseHeight
  const uniforms = new Float32Array(8)

  // 게인맵은 변하지 않으므로 재구성은 첫 프레임에 한 번만 돌린다.
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
      // CAMetalLayer 설정은 메인 스레드로 비동기로 넘어간다 (`WGPUMetalLayerSurface`).
      // 드로어블이 새 포맷으로 바뀔 때까지 몇 프레임 쉬어야 파이프라인과 어긋나지 않는다.
      settleFrames = 3
    }
    if (settleFrames > 0) {
      settleFrames -= 1
      return
    }

    // 손가락이 닿아 있는 동안만 경계가 따라온다. 놓으면 그 자리에 머문다.
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

  // 프레임 루프는 setup 시점의 클로저를 계속 쓴다 — 최신 값을 ref로 건넨다.
  const exposureRef = useRef(0)
  const modeRef = useRef(MODE_COMPARE)
  exposureRef.current = exposure
  modeRef.current = mode

  return (
    <DemoScene
      title="HDR 게인맵 재구성"
      subtitle="rgba16float — 8비트로는 못 담는 4.7배 하이라이트"
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
            클리핑
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
