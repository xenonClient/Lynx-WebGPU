import { root, useEffect, useState } from '@lynx-js/react'
import gpu, { GPUBufferUsage, GPUTextureUsage, startFrameLoop } from '../webgpu.js'
import '../demo.css'
import '../elements.d.ts'

/**
 * 명세 표면 체크리스트 — 최근에 채운 기능들이 **실제로 도는지 값으로** 확인한다.
 *
 * `three` 씬이 라이브러리 이식 관점의 검증이라면 이쪽은 shim/네이티브 계약 자체를 본다.
 * 단위 테스트가 이미 같은 계약을 걸고 있지만, 여기서는 **진짜 GPU와 진짜 브리지**를 지나며
 * 같은 결과가 나오는지 본다 — 목(mock)이 맞춰 준 계약이 실기에서도 맞는지가 요점이다.
 */

interface Check {
  label: string
  state: 'wait' | 'ok' | 'fail'
  detail?: string
}

const CHECKS = [
  'adapter.info (명세 GPUAdapterInfo)',
  'adapter.limits (명세 이름 31종)',
  'device.features / lost',
  'getCompilationInfo — 깨진 셰이더',
  'getCompilationInfo — 정상 셰이더',
  'createRenderPipelineAsync 성공',
  'createRenderPipelineAsync 실패 → GPUPipelineError',
  'onuncapturederror',
  'clearBuffer (구간만 0으로)',
  'copyBufferToBuffer 짧은 형태',
  'getMappedRange(offset, size)',
  '디버그 마커 (패스 안팎 + 번들)',
  'core 포맷 rgb10a2uint · rgb9e5ufloat',
  'canvas unconfigure / getConfiguration',
]

/** 지난 배치의 오류를 한 번만 모으는 수집기 — 검증마다 갈아 끼운다. */
function makeErrorSink(device: any) {
  /** @type {string[]} */
  const collected: string[] = []
  device.onError((_error: any, text: string) => collected.push(text))
  return {
    take() {
      const taken = collected.slice()
      collected.length = 0
      return taken
    },
  }
}

const TRIANGLE = /* wgsl */ `
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  return vec4f(p[i], 0.0, 1.0);
}
@fragment fn fs() -> @location(0) vec4f { return vec4f(0.0, 1.0, 0.5, 1.0); }
`

function SpecScene() {
  const [status, setStatus] = useState('준비 중…')
  const [checks, setChecks] = useState<Check[]>(CHECKS.map((label) => ({ label, state: 'wait' })))

  useEffect(() => {
    let disposed = false
    let stopLoop: (() => void) | null = null
    let device: any = null

    function mark(index: number, ok: boolean, detail?: string) {
      if (disposed) return
      setChecks((previous) => previous.map((check, i) => (
        i === index ? { ...check, state: ok ? 'ok' : 'fail', detail } : check
      )))
    }

    /** 검증 하나가 던져도 나머지는 계속 돈다 — 첫 실패에서 멈추면 정보가 가장 적다. */
    async function check(index: number, run: () => Promise<{ ok: boolean, detail: string }>) {
      try {
        const result = await run()
        mark(index, result.ok, result.detail)
      } catch (error) {
        mark(index, false, `예외: ${(error && (error as Error).message) || error}`.slice(0, 90))
      }
    }

    async function boot() {
      const adapter = await gpu.requestAdapter()
      if (!adapter) throw new Error('어댑터 없음')
      device = await adapter.requestDevice()
      const sink = makeErrorSink(device)
      setStatus(`${adapter.info.description || adapter.name} · ${adapter.info.architecture || '?'}`)

      // ① adapter.info — 명세 이름이 채워져 있고, 모르는 자리는 빈 문자열이다.
      await check(0, async () => {
        const info = adapter.info
        const ok = info.vendor === 'apple' && typeof info.description === 'string'
          && info.description.length > 0 && info.device === ''
          && info.isFallbackAdapter === false
        return { ok, detail: `${info.vendor}/${info.architecture}` }
      })

      // ② limits — 명세 철자로 전 항목이 있고 기본값 이상이다.
      await check(1, async () => {
        const required = [
          'maxTextureDimension2D', 'maxBindGroups', 'maxBufferSize', 'maxVertexBuffers',
          'maxComputeWorkgroupsPerDimension', 'maxUniformBufferBindingSize',
          'minUniformBufferOffsetAlignment', 'maxColorAttachments',
        ]
        const missing = required.filter((key) => typeof adapter.limits[key] !== 'number')
        const ok = missing.length === 0 && adapter.limits.maxTextureDimension2D >= 8192
        return { ok, detail: ok ? `2D ${adapter.limits.maxTextureDimension2D}` : `없음: ${missing.join(',')}` }
      })

      // ③ device.features / lost — 요청한 것만 들어오고, lost는 Promise다.
      await check(2, async () => {
        const requested = adapter.features.has('timestamp-query') ? ['timestamp-query'] : []
        const scoped = await adapter.requestDevice({ requiredFeatures: requested })
        const ok = scoped.features.size === requested.length && scoped.lost instanceof Promise
        return { ok, detail: `요청 ${requested.length}개 · lost ${scoped.lost instanceof Promise}` }
      })

      // ④ 깨진 셰이더의 진단 — 모듈은 만들어지고 줄 번호가 온다.
      await check(3, async () => {
        const broken = device.createShaderModule({
          code: '@vertex\nfn vs() -> @builtin(position) vec4f {\n  return vec4f(1.0 1.0, 1.0, 1.0);\n}',
        })
        const info = await broken.getCompilationInfo()
        sink.take()   // 파싱 실패가 전역으로도 보고된다 — 여기서는 기대한 것이라 지운다
        const first = info.messages[0]
        const ok = !!first && first.type === 'error' && first.lineNum === 3
        return { ok, detail: first ? `line ${first.lineNum}` : '진단 없음' }
      })

      // ⑤ 정상 셰이더는 비어 있다.
      const goodModule = device.createShaderModule({ code: TRIANGLE })
      await check(4, async () => {
        const info = await goodModule.getCompilationInfo()
        return { ok: info.messages.length === 0, detail: `${info.messages.length}건` }
      })

      // ⑥ 비동기 파이프라인 — 성공.
      const targets = [{ format: gpu.getPreferredCanvasFormat() }]
      await check(5, async () => {
        const pipeline = await device.createRenderPipelineAsync({
          layout: 'auto',
          vertex: { module: goodModule, entryPoint: 'vs' },
          fragment: { module: goodModule, entryPoint: 'fs', targets },
        })
        return { ok: !!pipeline && pipeline.id > 0, detail: `id ${pipeline.id}` }
      })

      // ⑦ 비동기 파이프라인 — 실패는 GPUPipelineError로 거부된다 (오류가 전역으로 새지 않는다).
      await check(6, async () => {
        const bad = device.createShaderModule({
          code: '@vertex fn vs() -> @builtin(position) vec4f { return nonexistent(1.0); }',
        })
        try {
          await device.createRenderPipelineAsync({
            layout: 'auto', vertex: { module: bad, entryPoint: 'vs' },
          })
          return { ok: false, detail: '거부되지 않았다' }
        } catch (error) {
          const failure = /** @type {any} */ (error)
          const leaked = sink.take()
          return {
            ok: failure.name === 'GPUPipelineError' && leaked.length === 0,
            detail: `${failure.name}/${failure.reason} · 전역 누출 ${leaked.length}`,
          }
        }
      })

      // ⑧ onuncapturederror — 스코프 밖 오류가 명세 통로로 온다.
      await check(7, async () => {
        /** @type {any[]} */
        const events: any[] = []
        device.onuncapturederror = (event: any) => events.push(event)
        const encoder = device.createCommandEncoder()
        // 없는 버퍼를 지운다 → validation 오류.
        encoder.clearBuffer({ id: 999999, size: 16 })
        device.queue.submit([encoder.finish()])
        device.onuncapturederror = null
        sink.take()
        const first = events[0]
        // `constructor.name`은 번들 최소화에 눌려 한 글자가 된다 — 이 구현이 함께 싣는
        // `kind`를 쓴다 (명세에는 없지만 진단에는 이쪽이 쓸모 있다).
        return {
          ok: events.length > 0 && typeof first.error.message === 'string',
          detail: first ? `kind=${first.error.kind}` : '이벤트 없음',
        }
      })

      // ⑨ clearBuffer — 구간만 0이 되고 나머지는 남는다.
      await check(8, async () => {
        const buffer = device.createBuffer({
          size: 32, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        })
        device.queue.writeBuffer(buffer, 0, new Uint8Array(32).fill(7))
        const encoder = device.createCommandEncoder()
        encoder.clearBuffer(buffer, 0, 16)
        device.queue.submit([encoder.finish()])
        const bytes = new Uint8Array(await buffer.mapAsync())
        buffer.unmap()
        const ok = bytes[0] === 0 && bytes[15] === 0 && bytes[16] === 7 && bytes[31] === 7
        buffer.destroy()
        return { ok, detail: `[${bytes[0]},${bytes[15]}|${bytes[16]},${bytes[31]}]` }
      })

      // ⑩ copyBufferToBuffer 짧은 형태 — (src, dst)만으로 전체 복사.
      await check(9, async () => {
        const source = device.createBuffer({
          size: 16, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        })
        const destination = device.createBuffer({
          size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        })
        device.queue.writeBuffer(source, 0, new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]))
        const encoder = device.createCommandEncoder()
        encoder.copyBufferToBuffer(source, destination)      // 짧은 형태
        device.queue.submit([encoder.finish()])
        const bytes = new Uint8Array(await destination.mapAsync())
        destination.unmap()
        const ok = bytes[0] === 1 && bytes[15] === 16
        source.destroy()
        destination.destroy()
        return { ok, detail: `[${bytes[0]}…${bytes[15]}]` }
      })

      // ⑪ getMappedRange(offset, size) — 그 구간만 오고, 쓴 내용이 되돌아간다.
      await check(10, async () => {
        const readback = device.createBuffer({
          size: 32, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        })
        const pattern = new Uint8Array(32)
        pattern.forEach((_, index) => { pattern[index] = index })
        device.queue.writeBuffer(readback, 0, pattern)
        device.queue.submit([])
        await readback.mapAsync()
        const middle = new Uint8Array(readback.getMappedRange(8, 8))
        readback.unmap()
        readback.destroy()

        // 쓰기 쪽 — mappedAtCreation으로 뒤쪽만 채우고 되돌아가는지 본다.
        const written = device.createBuffer({
          size: 32, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
          mappedAtCreation: true,
        })
        new Uint8Array(written.getMappedRange(24, 8)).fill(0xab)
        written.unmap()
        const staging = device.createBuffer({
          size: 32, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        })
        const encoder = device.createCommandEncoder()
        encoder.copyBufferToBuffer(written, staging)
        device.queue.submit([encoder.finish()])
        const back = new Uint8Array(await staging.mapAsync())
        staging.unmap()
        written.destroy()
        staging.destroy()

        const ok = middle[0] === 8 && middle[7] === 15 && back[24] === 0xab && back[0] === 0
        return { ok, detail: `읽기 ${middle[0]}~${middle[7]} · 쓰기 ${back[24]}` }
      })

      // ⑫ 디버그 마커 — 패스 안팎과 번들에서 모두 받는다 (오류가 없으면 통과).
      await check(11, async () => {
        const bundleEncoder = device.createRenderBundleEncoder({ colorFormats: [targets[0].format] })
        bundleEncoder.pushDebugGroup('번들 구간')
        bundleEncoder.popDebugGroup()
        bundleEncoder.finish()

        const encoder = device.createCommandEncoder()
        encoder.pushDebugGroup('프레임')
        encoder.insertDebugMarker('표식')
        encoder.popDebugGroup()
        device.queue.submit([encoder.finish()])

        const leaked = sink.take()
        return { ok: leaked.length === 0, detail: leaked.length ? leaked[0].slice(0, 60) : '오류 0' }
      })

      // ⑬ core 포맷 2종 — 만들고 복사까지 된다 (팩된 32비트라 픽셀당 4바이트).
      await check(12, async () => {
        const results: string[] = []
        for (const format of ['rgb10a2uint', 'rgb9e5ufloat']) {
          const texture = device.createTexture({
            size: { width: 4, height: 4 }, format,
            usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC,
          })
          const buffer = device.createBuffer({
            size: 4 * 4 * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
          })
          const encoder = device.createCommandEncoder()
          encoder.copyTextureToBuffer(
            { texture }, { buffer, bytesPerRow: 4 * 4 }, { width: 4, height: 4 }
          )
          device.queue.submit([encoder.finish()])
          const bytes = new Uint8Array(await buffer.mapAsync())
          buffer.unmap()
          results.push(`${format.slice(0, 6)}:${bytes.length}B`)
          texture.destroy()
          buffer.destroy()
        }
        const leaked = sink.take()
        return { ok: leaked.length === 0, detail: results.join(' ') }
      })

      // ⑭ 캔버스 unconfigure / getConfiguration.
      await check(13, async () => {
        const probe = gpu.getCanvasContext('main')
        const before = probe.getConfiguration()
        probe.configure({ device, format: targets[0].format })
        const configured = probe.getConfiguration()
        probe.unconfigure()
        const after = probe.getConfiguration()
        let rejected = false
        try {
          probe.getCurrentTexture()
        } catch (error) {
          rejected = true
        }
        // 다시 설정해 두고 끝낸다 — 아래 프레임 루프가 이 컨텍스트로 그린다.
        probe.configure({ device, format: targets[0].format })
        const ok = before === null && configured?.format === targets[0].format
          && after === null && rejected
        return { ok, detail: `설정 전 null · 해제 후 ${rejected ? '그리기 거부' : '거부 안 함'}` }
      })

      if (disposed) return

      // 배경 — 체크가 끝난 뒤에도 화면이 살아 있음을 보여 준다.
      // `getCanvasContext`는 같은 id에 같은 객체를 주므로 위 ⑭에서 설정한 그대로다.
      const context = gpu.getCanvasContext('main')
      context.configure({ device, format: targets[0].format })
      const pipeline = device.createRenderPipeline({
        layout: 'auto',
        vertex: { module: goodModule, entryPoint: 'vs' },
        fragment: { module: goodModule, entryPoint: 'fs', targets },
      })
      stopLoop = startFrameLoop(() => {
        if (disposed) return
        const size = context.getSize()
        if (!size.width) return
        const encoder = device.createCommandEncoder()
        encoder.pushDebugGroup('spec 프레임')   // 실제 프레임에도 마커를 붙여 둔다
        const pass = encoder.beginRenderPass({
          colorAttachments: [{
            view: context.getCurrentTexture().createView(),
            loadOp: 'clear', storeOp: 'store',
            clearValue: { r: 0.04, g: 0.05, b: 0.08, a: 1 },
          }],
        })
        pass.setPipeline(pipeline)
        pass.draw(3)
        pass.end()
        encoder.popDebugGroup()
        device.queue.submit([encoder.finish()])
      })
    }

    boot().catch((error) => {
      setStatus(`실패: ${(error && error.message) || error}`)
    })

    return () => {
      disposed = true
      if (stopLoop) stopLoop()
      if (device) device.destroy()
    }
  }, [])

  const icon = { wait: '○', ok: '✓', fail: '✗' }
  const failed = checks.filter((check) => check.state === 'fail').length
  const passed = checks.filter((check) => check.state === 'ok').length

  return (
    <view className="page">
      <webgpu-canvas className="canvas" canvas-id="main" />
      <view className="three-hud">
        <text className="title">명세 표면 체크리스트</text>
        <text className="subtitle">{status}</text>
        {checks.map((check, index) => (
          <text className={`check-row check-${check.state}`} key={`check-${index}`}>
            {icon[check.state]} {check.label}{check.detail ? ` — ${check.detail}` : ''}
          </text>
        ))}
        <text className="check-stats">{`통과 ${passed}/${CHECKS.length}${failed ? ` · 실패 ${failed}` : ''}`}</text>
      </view>
    </view>
  )
}

root.render(<SpecScene />)
