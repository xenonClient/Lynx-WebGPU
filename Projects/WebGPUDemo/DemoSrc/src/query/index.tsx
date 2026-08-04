import { root, useRef } from '@lynx-js/react'
import { DemoScene, type SceneContext } from '../scene.jsx'
import gpu, { GPUBufferUsage, GPUMapMode, GPUTextureUsage } from '../webgpu.js'

/**
 * GPU에게 되묻는 것들 — occlusion 쿼리 · 타임스탬프 · 오류 스코프.
 *
 * 셋 다 화면에 직접 그려지지 않는다. 그래서 **판정을 HUD에 숫자로 띄운다**:
 * 막대가 원을 가릴수록 살아남은 샘플 수가 줄고, 완전히 가려지면 정확히 0이 된다.
 *
 * 오류 스코프는 버튼 두 개로 대비를 보여 준다. 같은 잘못된 호출을 스코프 안에서 하면
 * 노란 줄에만 뜨고, 밖에서 하면 전역 핸들러를 타고 **빨간 줄**로 내려간다.
 */
const SHADER = /* wgsl */ `
struct Uniforms {
  time: f32,
  aspect: f32,
  depth: f32,
  center: f32,    // 막대의 가로 중심
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
  // 깊이를 유니폼에서 받는다 — 막대가 앞, 원이 뒤다.
  out.position = vec4f(corners[index], u.depth, 1.0);
  out.uv = corners[index];
  return out;
}

// 가리개 — 세로 막대가 좌우로 쓸고 지나간다.
@fragment
fn fs_bar(in: Out) -> @location(0) vec4f {
  let half = 0.24;
  let distance = abs(in.uv.x - u.center);
  if (distance > half) {
    discard;
  }
  let shade = 0.20 + 0.10 * smoothstep(0.0, half, distance);
  return vec4f(shade * 0.85, shade * 0.95, shade * 1.25, 1.0);
}

// 관측 대상 — 이 드로우가 통과시킨 샘플 수를 occlusion 쿼리가 센다.
// 좌표는 **짧은 쪽이 ±1이 되도록** 보정한다 (세로 화면에서 원이 좌우로 넘치지 않게).
@fragment
fn fs_target(in: Out) -> @location(0) vec4f {
  let p = vec2f(in.uv.x * u.aspect, in.uv.y) / min(u.aspect, 1.0);
  let radius = length(p);
  if (radius > 0.68) {
    discard;
  }
  let glow = 1.0 - radius / 0.68;
  let ripple = 0.85 + 0.15 * sin(radius * 14.0 - u.time * 2.4);
  return vec4f(mix(vec3f(0.2, 0.5, 1.0), vec3f(1.0, 0.85, 0.4), glow) * ripple, 1.0);
}
`

interface Actions {
  probe?: (inScope: boolean) => void
}

async function setup(
  { device, context, format, report }: SceneContext,
  actionsRef: { current: Actions }
) {
  const adapter = await gpu.requestAdapter()
  const hasTimestamp = !!adapter && adapter.features.has('timestamp-query')

  const module = device.createShaderModule({ code: SHADER, label: 'query' })

  function makePipeline(entryPoint: string) {
    return device.createRenderPipeline({
      layout: 'auto',
      vertex: { module, entryPoint: 'vs_main' },
      fragment: { module, entryPoint, targets: [{ format }] },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
    })
  }
  const barPipeline = makePipeline('fs_bar')
  const targetPipeline = makePipeline('fs_target')

  const layout = barPipeline.getBindGroupLayout(0)
  function makeUniforms(label: string) {
    const buffer = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label,
    })
    return {
      buffer,
      group: device.createBindGroup({ layout, entries: [{ binding: 0, resource: { buffer } }] }),
      data: new Float32Array(4),
    }
  }
  const bar = makeUniforms('query.bar')
  const target = makeUniforms('query.target')

  // occlusion 쿼리 1개 — 결과는 u64 하나(8바이트)다.
  const occlusionQuerySet = device.createQuerySet({ type: 'occlusion', count: 1 })
  const occlusionResults = device.createBuffer({
    size: 8,
    usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    label: 'occlusion.results',
  })
  // 리드백은 **전용 스테이징 버퍼**로 받는다 — MAP_READ는 COPY_DST와만 조합할 수 있고(명세),
  // 매핑 중인 버퍼는 큐 작업에서 거부되므로 resolve 대상을 직접 매핑하면 다음 프레임이 막힌다.
  const occlusionStaging = device.createBuffer({
    size: 8,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    label: 'occlusion.staging',
  })

  // 타임스탬프는 기기 조건이 붙는다 — 만들기 전에 adapter.features로 물어본다.
  const timestampQuerySet = hasTimestamp
    ? device.createQuerySet({ type: 'timestamp', count: 2 })
    : null
  const timestampResults = hasTimestamp
    ? device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
        label: 'timestamp.results',
      })
    : null
  const timestampStaging = hasTimestamp
    ? device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        label: 'timestamp.staging',
      })
    : null

  // 깊이 어태치먼트는 캔버스를 따라간다 — 막대가 원을 실제로 가리려면 깊이 테스트가 필요하다.
  let depthTexture: any = null
  let depthView: any = null
  let depthWidth = 0
  let depthHeight = 0

  function ensureDepth(width: number, height: number) {
    if (depthTexture && depthWidth === width && depthHeight === height) return
    if (depthTexture) depthTexture.destroy()
    depthTexture = device.createTexture({
      size: { width, height },
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
      label: 'query.depth',
    })
    depthView = depthTexture.createView()
    depthWidth = width
    depthHeight = height
  }

  /** 잘못된 파이프라인을 일부러 만든다 — 정점 포맷 철자가 명세에 없다. */
  function makeBadPipeline() {
    device.createRenderPipeline({
      layout: 'auto',
      vertex: {
        module,
        entryPoint: 'vs_main',
        buffers: [{
          arrayStride: 8,
          attributes: [{ format: 'float32x9', offset: 0, shaderLocation: 0 }],
        }],
      },
      fragment: { module, entryPoint: 'fs_target', targets: [{ format }] },
    })
  }

  actionsRef.current.probe = (inScope: boolean) => {
    if (!inScope) {
      makeBadPipeline()
      report('스코프 밖에서 실패시켰다 — 다음 제출에서 전역 핸들러(빨간 줄)로 내려간다')
      return
    }
    device.pushErrorScope('validation')
    makeBadPipeline()
    // popErrorScope는 결과를 받아야 하므로 스스로 제출한다 (mapAsync와 같은 성격).
    device
      .popErrorScope()
      .then((error: any) => {
        report(
          error
            ? `스코프가 잡았다 [${error.kind}] ${error.message} — 전역 핸들러로는 가지 않았다`
            : '스코프에 아무 오류도 잡히지 않았다'
        )
      })
      .catch((error: unknown) => report(`스코프 확인 실패: ${String(error)}`))
  }

  let time = 0
  let frame = 0
  let reading = false
  let peakSamples = 1

  return ({ delta, width, height }: { delta: number; width: number; height: number }) => {
    time += delta / 1000
    ensureDepth(width, height)

    const aspect = width / height
    // 막대는 앞(0.2), 원은 뒤(0.6) — 겹치는 곳에서 원이 깊이 테스트에 진다.
    bar.data[0] = time
    bar.data[1] = aspect
    bar.data[2] = 0.2
    bar.data[3] = Math.sin(time * 0.7)
    device.queue.writeBuffer(bar.buffer, 0, bar.data)

    target.data[0] = time
    target.data[1] = aspect
    target.data[2] = 0.6
    target.data[3] = 0
    device.queue.writeBuffer(target.buffer, 0, target.data)

    const encoder = device.createCommandEncoder()
    const passDescriptor: any = {
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: 0.02, g: 0.03, b: 0.05, a: 1 },
      }],
      depthStencilAttachment: {
        view: depthView,
        depthClearValue: 1,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
      // 쿼리는 **패스를 열 때만** 붙일 수 있다.
      occlusionQuerySet,
    }
    if (timestampQuerySet) {
      passDescriptor.timestampWrites = {
        querySet: timestampQuerySet,
        beginningOfPassWriteIndex: 0,
        endOfPassWriteIndex: 1,
      }
    }

    const pass = encoder.beginRenderPass(passDescriptor)

    pass.setPipeline(barPipeline)
    pass.setBindGroup(0, bar.group)
    pass.draw(3)

    // 이 드로우가 통과시킨 샘플만 센다.
    pass.beginOcclusionQuery(0)
    pass.setPipeline(targetPipeline)
    pass.setBindGroup(0, target.group)
    pass.draw(3)
    pass.endOcclusionQuery()

    pass.end()

    encoder.resolveQuerySet(occlusionQuerySet, 0, 1, occlusionResults, 0)
    if (timestampQuerySet && timestampResults) {
      encoder.resolveQuerySet(timestampQuerySet, 0, 2, timestampResults, 0)
    }

    // 20프레임에 한 번만 스테이징으로 복사한다 — 리드백은 GPU 완료를 기다리는 경로다.
    const wantsReadback = ++frame % 20 === 0 && !reading
    if (wantsReadback) {
      encoder.copyBufferToBuffer(occlusionResults, 0, occlusionStaging, 0, 8)
      if (timestampResults && timestampStaging) {
        encoder.copyBufferToBuffer(timestampResults, 0, timestampStaging, 0, 16)
      }
    }

    device.queue.submit([encoder.finish()])

    if (!wantsReadback) return
    reading = true

    const pending: Promise<ArrayBuffer>[] = [occlusionStaging.mapAsync(GPUMapMode.READ)]
    if (timestampStaging) pending.push(timestampStaging.mapAsync(GPUMapMode.READ))

    Promise.all(pending)
      .then(([occlusion, timestamps]: ArrayBuffer[]) => {
        // u64를 하위 32비트로 읽는다 — 샘플 수가 40억을 넘을 일은 없다.
        const samples = new Uint32Array(occlusion)[0]
        if (samples > peakSamples) peakSamples = samples
        const visible = Math.round((samples / peakSamples) * 100)

        let line = samples === 0
          ? 'occlusion 0 — 원이 완전히 가려졌다'
          : `occlusion ${samples} 샘플 · 보이는 비율 ${visible}%`

        if (timestamps) {
          // u64 두 개. hi·lo를 double로 합쳐도 2^53 안이라 차이가 정확하다.
          const parts = new Uint32Array(timestamps)
          const start = parts[1] * 4294967296 + parts[0]
          const end = parts[3] * 4294967296 + parts[2]
          line += ` · GPU 패스 ${((end - start) / 1e6).toFixed(3)}ms`
        } else {
          line += ' · 타임스탬프 미지원 기기'
        }
        report(line)
      })
      .catch((error: unknown) => report(`쿼리 리드백 실패: ${String(error)}`))
      .finally(() => {
        // 매핑을 풀어야 다음 주기의 복사가 이 버퍼를 다시 쓸 수 있다.
        occlusionStaging.unmap()
        if (timestampStaging) timestampStaging.unmap()
        reading = false
      })
  }
}

function QueryScene() {
  const actionsRef = useRef<Actions>({})

  return (
    <DemoScene
      title="쿼리 · 오류 스코프"
      subtitle="가려진 샘플 수 · GPU 패스 시간 · 잡은 오류 — 화면이 아니라 숫자로 나오는 것들"
      setup={(scene) => setup(scene, actionsRef)}
      controls={
        <view className="controls">
          <text
            className="control-button"
            bindtap={() => actionsRef.current.probe && actionsRef.current.probe(true)}
          >
            스코프 안에서 실패
          </text>
          <text
            className="control-button"
            bindtap={() => actionsRef.current.probe && actionsRef.current.probe(false)}
          >
            스코프 밖에서 실패
          </text>
        </view>
      }
    />
  )
}

root.render(<QueryScene />)
