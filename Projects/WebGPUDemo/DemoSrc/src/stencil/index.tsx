import { root, useInitData, useRef, useState } from '@lynx-js/react'
import { DemoScene, type SceneContext } from '../scene.jsx'
import { GPUBufferUsage, GPUTextureUsage } from '../webgpu.js'

/**
 * 스텐실 마스크 — 회전하는 별 모양으로 화면을 두 영역으로 가른다.
 *
 * 세 번의 드로우가 모두 **같은 풀스크린 삼각형**이다. 그림이 갈리는 이유는 오직 스텐실뿐이라,
 * 스텐실이 안 먹으면 화면이 한 가지 색으로 덮인다 — 눈으로 바로 구분된다.
 *
 * 깊이 없이 `stencil8` 단독 포맷을 쓴다. 예전에는 이 조합이 파이프라인 생성 단계에서
 * Metal 단언으로 앱을 죽였다 (`docs/ROADMAP.md`의 함께 고친 버그).
 */
const SHADER = /* wgsl */ `
struct Uniforms {
  time: f32,
  aspect: f32,
  spikes: f32,
  style: f32,     // 0 = 어두운 쪽, 1 = 밝은 쪽
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

// 화면비를 보정해 **짧은 쪽이 ±1이 되는** 좌표로 옮긴다. 세로 화면에서도 도형이 원형을
// 유지하면서 좌우로 넘치지 않는다.
fn fitted(uv: vec2f) -> vec2f {
  return vec2f(uv.x * u.aspect, uv.y) / min(u.aspect, 1.0);
}

// 마스크 패스 — 별 밖은 버린다. **버려진 프래그먼트는 스텐실도 쓰지 않는다**는 것이 요점이다.
// 컬러는 writeMask 0으로 막혀 있어 화면에는 아무것도 남기지 않는다.
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

// 채우기 패스 — 두 번 그리되 스텐실 비교만 다르다 (equal / not-equal).
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

  /** 스텐실 상태만 다른 파이프라인. 컬러 타깃과 셰이더는 셋 다 같다. */
  // 명세상 `layout:"auto"`의 파생 레이아웃은 **그 파이프라인 전용**이다 — 셋이 바인드
  // 그룹을 공유하려면 명시적 레이아웃이어야 한다 (Dawn/브라우저가 재사용을 거부한다).
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
        // 마스킹 패스는 컬러를 막고 스텐실만 남긴다.
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

  // 명시적 공유 레이아웃이라 세 파이프라인이 바인드 그룹을 공유할 수 있다.
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

  // 스텐실 어태치먼트는 캔버스 크기를 따라간다 — 회전·리사이즈 때 다시 만든다.
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
    report(`stencil8 어태치먼트 ${width}×${height} — 깊이 없이 스텐실만`)
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

    // 반전하면 별 **안쪽**이 어두워진다 — 같은 마스크를 반대로 읽는 것뿐이다.
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

    // 1) 별 모양이 있는 자리에 스텐실 1을 남긴다 (색은 안 남긴다).
    pass.setStencilReference(1)
    pass.setPipeline(maskPipeline)
    pass.setBindGroup(0, brightGroup)
    pass.draw(3)

    // 2) 스텐실이 1인 곳 / 3) 아닌 곳 — 같은 삼각형, 다른 비교.
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
  // `-altMode 1`이면 반전된 마스크로 시작한다 (자동화 캡처용 — `bundle` 씬과 같은 규약).
  // Lynx가 불리언을 숫자로 옮겨 줄 수 있어 === 대신 truthy로 본다.
  const alt = !!(useInitData() as { altMode?: unknown } | undefined)?.altMode
  const [inverted, setInverted] = useState(alt)

  // 프레임 루프는 setup 시점의 클로저를 계속 쓴다 — 최신 값을 ref로 건넨다.
  const invertedRef = useRef(alt)
  invertedRef.current = inverted

  return (
    <DemoScene
      title="스텐실 마스크"
      subtitle="같은 풀스크린 삼각형 3번 — 갈리는 이유는 스텐실뿐"
      setup={(scene) => setup(scene, invertedRef)}
      controls={
        <view className="controls">
          <text className="control-value">
            {inverted ? 'compare: 별 밖이 밝다' : 'compare: 별 안이 밝다'}
          </text>
          <text
            className={inverted ? 'control-button control-button-on' : 'control-button'}
            bindtap={() => setInverted((value) => !value)}
          >
            마스크 반전
          </text>
        </view>
      }
    />
  )
}

root.render(<StencilScene />)
