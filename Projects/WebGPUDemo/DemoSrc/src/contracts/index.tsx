import { root, useEffect, useState } from '@lynx-js/react'
import gpu, { GPUBufferUsage, GPUTextureUsage, startFrameLoop } from '../webgpu.js'
import '../demo.css'
import '../elements.d.ts'
import { ChecklistHud, type Check } from '../checklist-hud.jsx'

/**
 * 계약 점검 — 검토에서 지적된 자리들이 **실기에서** 어떻게 동작하는지.
 *
 * 여기 모인 것들의 공통점은 "지금까지 아무도 확인하지 않던 자리"다. 셋은 실제로 비어 있었고
 * (버퍼 복사의 기본값·범위 검증, 포맷 역방향 매핑), 둘은 이미 맞았지만 **그렇다는 증거가
 * 없었다** (번들의 인덱스 버퍼 격리, occlusion 쿼리 차단).
 *
 * 단위 테스트가 같은 계약을 걸고 있지만, 이 화면은 **진짜 GPU와 진짜 브리지**를 지난다 —
 * 목(mock)이 맞춰 준 계약이 실기에서도 맞는지가 요점이다.
 */

const CHECKS = [
  'copyBufferToBuffer — size 생략 = 원본 전부',
  'copyBufferToBuffer — sourceOffset 뒤 남은 전부',
  'copyBufferToBuffer — 범위를 넘으면 거부',
  'copyBufferToBuffer — 0바이트는 no-op',
  'vec3<i32> · vec3<u32> 유니폼 배치 (packed 정수 벡터)',
  '번들 실행 뒤 인덱스 버퍼가 무효화된다',
  '번들은 패스의 인덱스 버퍼를 물려받지 않는다',
  'occlusion 쿼리는 번들에 담을 수 없다',
  '드로어블 포맷 역방향 매핑 (캔버스 패스가 아는 이름)',
  '디바이스가 둘이어도 핸들이 겹치지 않는다',
]

/**
 * 정수 vec3의 배치를 값으로 확인하는 셰이더.
 *
 * WGSL `vec3<i32>`는 12바이트지만 MSL `int3`는 16바이트다. 방출기가 `packed_int3`를 쓰지
 * 않으면 뒤 필드가 4바이트씩 밀려 **오류 없이 엉뚱한 값**이 읽힌다 — 그래서 합을 색으로 낸다.
 */
const PACKED_SHADER = /* wgsl */ `
struct Counts {
  offsets: vec3<i32>,   // offset 0  (12B)
  total: i32,           // offset 12
  sizes: vec3<u32>,     // offset 16 (12B)
  stride: u32,          // offset 28
};
@group(0) @binding(0) var<uniform> counts: Counts;

@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  return vec4f(p[i], 0.0, 1.0);
}

@fragment fn fs() -> @location(0) vec4f {
  let a = counts.offsets.x + counts.offsets.y + counts.offsets.z + counts.total;
  let b = counts.sizes.x + counts.sizes.y + counts.sizes.z + counts.stride;
  let c = counts.offsets.z * 10 + i32(counts.sizes.y);
  return vec4f(f32(a) / 255.0, f32(b) / 255.0, f32(c) / 255.0, 1.0);
}
`

const TRIANGLE = /* wgsl */ `
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  return vec4f(p[i], 0.0, 1.0);
}
@fragment fn fs() -> @location(0) vec4f { return vec4f(0.2, 0.8, 1.0, 1.0); }
`

function ContractsScene() {
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
      setStatus(`${adapter.info.description || adapter.name} · ${adapter.info.architecture || '?'}`)

      const collected: string[] = []
      device.onError((_error: any, text: string) => collected.push(text))
      const takeErrors = () => {
        const taken = collected.slice()
        collected.length = 0
        return taken
      }

      /** 16바이트를 채운 원본 + 되읽을 대상 한 쌍. */
      function makeBufferPair(sourceSize = 16, destinationSize = 16) {
        const source = device.createBuffer({
          size: sourceSize, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        })
        const destination = device.createBuffer({
          size: destinationSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        })
        const pattern = new Uint8Array(sourceSize)
        pattern.forEach((_, index) => { pattern[index] = index + 1 })
        device.queue.writeBuffer(source, 0, pattern)
        return { source, destination }
      }

      // ① size 생략 — 명세의 짧은 형태 `copyBufferToBuffer(src, dst)`는 "원본 전부"다.
      //    JS shim이 채워 보내지만, 그 값이 실제로 원본 크기인지는 값으로만 알 수 있다.
      await check(0, async () => {
        const { source, destination } = makeBufferPair()
        const encoder = device.createCommandEncoder()
        encoder.copyBufferToBuffer(source, destination)
        device.queue.submit([encoder.finish()])
        const bytes = new Uint8Array(await destination.mapAsync())
        destination.unmap()
        const ok = bytes[0] === 1 && bytes[15] === 16
        source.destroy()
        destination.destroy()
        return { ok, detail: `[${bytes[0]}…${bytes[15]}] · ${bytes.length}B` }
      })

      // ② sourceOffset만 주면 그 **뒤로 남은 전부**다 — 원본 크기를 그대로 쓰면 범위를 넘는다.
      await check(1, async () => {
        const { source, destination } = makeBufferPair()
        const encoder = device.createCommandEncoder()
        encoder.copyBufferToBuffer(source, 8, destination, 0)   // size 생략
        device.queue.submit([encoder.finish()])
        const bytes = new Uint8Array(await destination.mapAsync())
        destination.unmap()
        // 원본 9~16번 바이트가 앞으로 온다. 뒤 8바이트는 건드리지 않는다.
        const ok = bytes[0] === 9 && bytes[7] === 16 && bytes[8] === 0 && takeErrors().length === 0
        source.destroy()
        destination.destroy()
        return { ok, detail: `[${bytes[0]}…${bytes[7]}|${bytes[8]}]` }
      })

      // ③ 범위 초과 — **Metal은 이것을 단언으로 죽인다.** 검증 오류로 와야 한다.
      await check(2, async () => {
        const results: string[] = []
        for (const [label, run] of [
          ['대상', (source: any, destination: any, encoder: any) => {
            encoder.copyBufferToBuffer(source, 0, destination, 0, 16)   // 대상은 8B뿐
          }],
          ['원본', (source: any, destination: any, encoder: any) => {
            encoder.copyBufferToBuffer(source, 12, destination, 0, 8)   // 원본 12+8 > 16
          }],
          ['음수', (source: any, destination: any, encoder: any) => {
            encoder.copyBufferToBuffer(source, 0, destination, 0, -4)
          }],
        ] as [string, (s: any, d: any, e: any) => void][]) {
          const { source, destination } = makeBufferPair(16, 8)
          const encoder = device.createCommandEncoder()
          run(source, destination, encoder)
          device.queue.submit([encoder.finish()])
          const errors = takeErrors()
          results.push(`${label}${errors.length ? '✓' : '✗'}`)
          source.destroy()
          destination.destroy()
        }
        return { ok: results.every((entry) => entry.endsWith('✓')), detail: results.join(' ') }
      })

      // ④ 0바이트는 no-op — Metal blit이 0바이트 복사를 거부하므로 그냥 넘기면 오류가 된다.
      await check(3, async () => {
        const { source, destination } = makeBufferPair()
        const encoder = device.createCommandEncoder()
        encoder.copyBufferToBuffer(source, 0, destination, 0, 0)
        device.queue.submit([encoder.finish()])
        const errors = takeErrors()
        source.destroy()
        destination.destroy()
        return { ok: errors.length === 0, detail: errors[0] ? errors[0].slice(0, 50) : '오류 0' }
      })

      // ⑤ 정수 vec3 배치 — 방출기가 packed를 안 쓰면 뒤 필드가 밀려 값이 달라진다.
      await check(4, async () => {
        const module = device.createShaderModule({ code: PACKED_SHADER })
        const pipeline = device.createRenderPipeline({
          layout: 'auto',
          vertex: { module, entryPoint: 'vs' },
          fragment: { module, entryPoint: 'fs', targets: [{ format: 'rgba8unorm' }] },
        })
        const uniform = device.createBuffer({
          size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        })
        // offsets(1,2,3) total 4 · sizes(5,6,7) stride 8 — WGSL 오프셋 그대로 채운다.
        const values = new Int32Array([1, 2, 3, 4, 5, 6, 7, 8])
        device.queue.writeBuffer(uniform, 0, values)
        const bindGroup = device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [{ binding: 0, resource: { buffer: uniform } }],
        })

        const pixel = await renderPixel(pipeline, (pass: any) => {
          pass.setBindGroup(0, bindGroup)
          pass.draw(3)
        })
        uniform.destroy()
        // r = 1+2+3+4 = 10 · g = 5+6+7+8 = 26 · b = 3*10 + 6 = 36
        const ok = pixel[0] === 10 && pixel[1] === 26 && pixel[2] === 36
        return { ok, detail: `rgb(${pixel.join(',')}) · 기대 rgb(10,26,36)` }
      })

      // --- 번들 격리 ---------------------------------------------------------

      const bundleModule = device.createShaderModule({ code: TRIANGLE })
      const bundlePipeline = device.createRenderPipeline({
        layout: 'auto',
        vertex: { module: bundleModule, entryPoint: 'vs' },
        fragment: { module: bundleModule, entryPoint: 'fs', targets: [{ format: 'rgba8unorm' }] },
      })
      const indices = device.createBuffer({
        size: 8, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      })
      device.queue.writeBuffer(indices, 0, new Uint16Array([0, 1, 2, 0]))

      // ⑥ 번들이 묶은 인덱스 버퍼가 패스로 새면, 이어지는 drawIndexed가 **실제로 그려진다**.
      await check(5, async () => {
        const bundleEncoder = device.createRenderBundleEncoder({ colorFormats: ['rgba8unorm'] })
        bundleEncoder.setPipeline(bundlePipeline)
        bundleEncoder.setIndexBuffer(indices, 'uint16')
        bundleEncoder.drawIndexed(3)
        const bundle = bundleEncoder.finish()

        await renderPixel(bundlePipeline, (pass: any) => {
          pass.executeBundles([bundle])
          pass.setPipeline(bundlePipeline)
          pass.drawIndexed(3)              // setIndexBuffer 없이
        }, { skipPipeline: true })
        const errors = takeErrors()
        return {
          // 백엔드마다 문구가 다르다 — Metal 엔진은 "setIndexBuffer", Dawn은 "Index buffer".
          ok: errors.some((text) => text.includes('setIndexBuffer') || text.includes('Index buffer')),
          detail: errors[0] ? errors[0].slice(0, 55) : '거부되지 않았다',
        }
      })

      // ⑦ 반대 방향 — 패스가 묶은 것을 번들이 물려받지 않는다.
      await check(6, async () => {
        const bundleEncoder = device.createRenderBundleEncoder({ colorFormats: ['rgba8unorm'] })
        bundleEncoder.setPipeline(bundlePipeline)
        bundleEncoder.drawIndexed(3)       // 번들 안에서는 묶은 적이 없다
        const bundle = bundleEncoder.finish()

        await renderPixel(bundlePipeline, (pass: any) => {
          pass.setIndexBuffer(indices, 'uint16')
          pass.executeBundles([bundle])
        }, { skipPipeline: true })
        const errors = takeErrors()
        return {
          // 명세는 이 검증을 번들 finish() 시점으로 정한다 — Dawn은 그때 "Index buffer was
          // not set"을 내고, Metal 엔진 재생은 실행 시점에 "setIndexBuffer"를 낸다. 둘 다 격리다.
          ok: errors.some((text) => text.includes('setIndexBuffer') || text.includes('Index buffer')),
          detail: errors[0] ? errors[0].slice(0, 55) : '거부되지 않았다',
        }
      })

      // ⑧ occlusion 쿼리는 렌더 **패스**의 것이다. 방어가 두 겹인지 본다:
      //    shim에는 메서드가 아예 없고(1겹), 밀어 넣어도 네이티브가 거부한다(2겹).
      await check(7, async () => {
        const bundleEncoder = device.createRenderBundleEncoder({ colorFormats: ['rgba8unorm'] })
        const shimHides = typeof (bundleEncoder as any).beginOcclusionQuery !== 'function'
        // 뒷문으로 밀어 넣는다 — 정상 경로가 아니라 **네이티브 방어선**을 보는 것이 요점이다.
        ;(bundleEncoder as any)._commands.push({ op: 'beginOcclusionQuery', queryIndex: 0 })
        bundleEncoder.finish()
        device.queue.submit([])
        const errors = takeErrors()
        return {
          ok: shimHides && errors.some((text) => text.includes('번들')),
          detail: `shim 미노출 ${shimHides ? '✓' : '✗'} · 네이티브 ${errors.length ? '거부' : '통과'}`,
        }
      })

      // ⑨ 드로어블 포맷 역방향 매핑 — 캔버스 패스가 자기 어태치먼트를 어떤 이름으로 아는가.
      //
      //    `MTLPixelFormat` → WebGPU 이름 표는 네이티브 안에만 있어 JS에서 직접 볼 수 없다.
      //    대신 **일부러 어긋난 번들**을 캔버스 패스에 넣으면, 거부 메시지가 패스의 실제
      //    포맷 이름을 실어 준다 — 그게 역방향 매핑의 출력이다.
      const context = gpu.getCanvasContext('main')
      const canvasFormat = gpu.getPreferredCanvasFormat()
      context.configure({ device, format: canvasFormat })
      await check(8, async () => {
        const wrong = canvasFormat === 'rgba8unorm' ? 'bgra8unorm' : 'rgba8unorm'
        const bundleEncoder = device.createRenderBundleEncoder({ colorFormats: [wrong] })
        const bundle = bundleEncoder.finish()

        const encoder = device.createCommandEncoder()
        const pass = encoder.beginRenderPass({
          colorAttachments: [{
            view: context.getCurrentTexture().createView(),
            loadOp: 'clear', storeOp: 'store',
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
          }],
        })
        pass.executeBundles([bundle])
        pass.end()
        device.queue.submit([encoder.finish()])

        const message = takeErrors()[0] || ''
        // "… 번들 rgba8unorm, 패스 bgra8unorm" — 뒤쪽이 드로어블에서 되돌린 이름이다.
        const reported = (message.match(/패스 ([a-z0-9-]+)/) || [])[1]
        // Metal 엔진은 거부 문구에 역방향 매핑된 이름을 싣는다. 네이티브 검증(Dawn)은
        // 자기 문구로 거부한다 — 그 경우 "어긋난 번들이 거부됐다"까지가 이 검사의 몫이다.
        const nativeRejected = /not compatible|Attachment state/i.test(message)
        return {
          ok: reported === canvasFormat || nativeRejected,
          detail: reported
            ? `패스가 아는 이름 ${reported} · configure ${canvasFormat}`
            : (message.slice(0, 55) || '거부되지 않았다'),
        }
      })

      // ⑩ 핸들 공간 — 네이티브 레지스트리는 컨텍스트당 하나이고 **핸들 정수만으로** 찾는다.
      //    카운터를 디바이스마다 두면 두 번째 디바이스가 1번부터 다시 내서, 첫 디바이스의
      //    객체를 조용히 덮어쓴다. 오류가 나지 않고 "내 버퍼에 남이 그리는" 증상만 남는다.
      await check(9, async () => {
        const second = await adapter.requestDevice()
        const a1 = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_SRC })
        const b1 = second.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_SRC })
        const a2 = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_SRC })
        const handles = [a1.id, b1.id, a2.id]
        a1.destroy()
        b1.destroy()
        a2.destroy()
        device.queue.submit([])
        return {
          ok: new Set(handles).size === handles.length,
          detail: `A ${a1.id} · B ${b1.id} · A ${a2.id}`,
        }
      })

      indices.destroy()
      if (disposed) return

      // 배경 — 체크가 끝난 뒤에도 화면이 살아 있음을 보여 준다.
      const screenPipeline = device.createRenderPipeline({
        layout: 'auto',
        vertex: { module: bundleModule, entryPoint: 'vs' },
        fragment: { module: bundleModule, entryPoint: 'fs', targets: [{ format: canvasFormat }] },
      })
      stopLoop = startFrameLoop(() => {
        if (disposed) return
        const size = context.getSize()
        if (!size.width) return
        const encoder = device.createCommandEncoder()
        const pass = encoder.beginRenderPass({
          colorAttachments: [{
            view: context.getCurrentTexture().createView(),
            loadOp: 'clear', storeOp: 'store',
            clearValue: { r: 0.04, g: 0.05, b: 0.08, a: 1 },
          }],
        })
        pass.setPipeline(screenPipeline)
        pass.draw(3)
        pass.end()
        device.queue.submit([encoder.finish()])
      })
    }

    /**
     * 8×8 오프스크린에 그리고 가운데 픽셀을 되읽는다.
     *
     * "오류가 없었다"가 아니라 **셰이더가 읽은 값**을 확인하려면 이 왕복이 필요하다.
     * 거부를 기대하는 검증에서는 픽셀을 보지 않고 오류 수집기만 쓴다.
     */
    async function renderPixel(
      pipeline: any,
      record: (pass: any) => void,
      options: { skipPipeline?: boolean } = {}
    ): Promise<number[]> {
      const target = device.createTexture({
        size: { width: 8, height: 8 }, format: 'rgba8unorm',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
      })
      // 텍스처→버퍼 복사의 bytesPerRow는 명세상 256의 배수여야 한다.
      const readback = device.createBuffer({
        size: 256 * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      })
      const encoder = device.createCommandEncoder()
      const pass = encoder.beginRenderPass({
        colorAttachments: [{
          view: target.createView(), loadOp: 'clear', storeOp: 'store',
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
        }],
      })
      if (!options.skipPipeline) pass.setPipeline(pipeline)
      record(pass)
      pass.end()
      encoder.copyTextureToBuffer(
        { texture: target }, { buffer: readback, bytesPerRow: 256 }, { width: 8, height: 8 }
      )
      device.queue.submit([encoder.finish()])

      const bytes = new Uint8Array(await readback.mapAsync())
      const offset = 4 * 256 + 4 * 4
      const pixel = [bytes[offset], bytes[offset + 1], bytes[offset + 2]]
      readback.unmap()
      readback.destroy()
      target.destroy()
      return pixel
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

  return (
    <view className="page">
      <webgpu-canvas className="canvas" canvas-id="main" />
      <ChecklistHud title="계약 점검" subtitle={status} checks={checks} />
    </view>
  )
}

root.render(<ContractsScene />)
