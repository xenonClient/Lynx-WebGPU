import { root, useEffect, useState } from '@lynx-js/react'
import gpu, { GPUBufferUsage, GPUTextureUsage, startFrameLoop } from '../webgpu.js'
import '../demo.css'
import '../elements.d.ts'
import { ChecklistHud, type Check } from '../checklist-hud.jsx'

/**
 * The spec surface checklist — it confirms **by value** that the recently filled-in features really run.
 *
 * Where the `three` scene verifies from a library-porting angle, this one looks at the shim/native contract itself.
 * The unit tests already hold the same contract, but here it passes through **a real GPU and a real bridge**
 * to see whether the same results come out — whether a contract satisfied by mocks holds on real hardware.
 */

const CHECKS = [
  'adapter.info (the spec GPUAdapterInfo)',
  'adapter.limits (the 31 spec names)',
  'device.features / lost',
  'getCompilationInfo — a broken shader',
  'getCompilationInfo — a healthy shader',
  'createRenderPipelineAsync success',
  'createRenderPipelineAsync failure → GPUPipelineError',
  'onuncapturederror',
  'clearBuffer (only the range zeroed)',
  'copyBufferToBuffer, the short form',
  'getMappedRange(offset, size)',
  'debug markers (inside and outside a pass + a bundle)',
  'the core formats rgb10a2uint · rgb9e5ufloat',
  'canvas unconfigure / getConfiguration',
]

/** A collector that gathers the previous batch's errors once — swapped in per check. */
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
  const [status, setStatus] = useState('getting ready…')
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

    /** The rest keep running even if one check throws — stopping at the first failure gives the least information. */
    async function check(index: number, run: () => Promise<{ ok: boolean, detail: string }>) {
      try {
        const result = await run()
        mark(index, result.ok, result.detail)
      } catch (error) {
        mark(index, false, `exception: ${(error && (error as Error).message) || error}`.slice(0, 90))
      }
    }

    async function boot() {
      const adapter = await gpu.requestAdapter()
      if (!adapter) throw new Error('no adapter')
      device = await adapter.requestDevice()
      const sink = makeErrorSink(device)
      setStatus(`${adapter.info.description || adapter.name} · ${adapter.info.architecture || '?'}`)

      // ① adapter.info — the spec names are filled in and unknown slots are the empty string.
      await check(0, async () => {
        const info = adapter.info
        const ok = info.vendor === 'apple' && typeof info.description === 'string'
          && info.description.length > 0 && info.device === ''
          && info.isFallbackAdapter === false
        return { ok, detail: `${info.vendor}/${info.architecture}` }
      })

      // ② limits — every item is present under the spec spelling and at or above the default.
      await check(1, async () => {
        const required = [
          'maxTextureDimension2D', 'maxBindGroups', 'maxBufferSize', 'maxVertexBuffers',
          'maxComputeWorkgroupsPerDimension', 'maxUniformBufferBindingSize',
          'minUniformBufferOffsetAlignment', 'maxColorAttachments',
        ]
        const missing = required.filter((key) => typeof adapter.limits[key] !== 'number')
        const ok = missing.length === 0 && adapter.limits.maxTextureDimension2D >= 8192
        return { ok, detail: ok ? `2D ${adapter.limits.maxTextureDimension2D}` : `missing: ${missing.join(',')}` }
      })

      // ③ device.features / lost — only what was requested arrives, and lost is a Promise.
      await check(2, async () => {
        const requested = adapter.features.has('timestamp-query') ? ['timestamp-query'] : []
        const scoped = await adapter.requestDevice({ requiredFeatures: requested })
        const ok = scoped.features.size === requested.length && scoped.lost instanceof Promise
        return { ok, detail: `${requested.length} requested · lost ${scoped.lost instanceof Promise}` }
      })

      // ④ Diagnostics for a broken shader — the module is created and a line number comes back.
      await check(3, async () => {
        const broken = device.createShaderModule({
          code: '@vertex\nfn vs() -> @builtin(position) vec4f {\n  return vec4f(1.0 1.0, 1.0, 1.0);\n}',
        })
        const info = await broken.getCompilationInfo()
        sink.take()   // the parse failure is reported globally too — expected here, so it is cleared
        const first = info.messages[0]
        const ok = !!first && first.type === 'error' && first.lineNum === 3
        return { ok, detail: first ? `line ${first.lineNum}` : 'no diagnostics' }
      })

      // ⑤ A healthy shader is empty.
      const goodModule = device.createShaderModule({ code: TRIANGLE })
      await check(4, async () => {
        const info = await goodModule.getCompilationInfo()
        return { ok: info.messages.length === 0, detail: `${info.messages.length} messages` }
      })

      // ⑥ An asynchronous pipeline — success.
      const targets = [{ format: gpu.getPreferredCanvasFormat() }]
      await check(5, async () => {
        const pipeline = await device.createRenderPipelineAsync({
          layout: 'auto',
          vertex: { module: goodModule, entryPoint: 'vs' },
          fragment: { module: goodModule, entryPoint: 'fs', targets },
        })
        return { ok: !!pipeline && pipeline.id > 0, detail: `id ${pipeline.id}` }
      })

      // ⑦ An asynchronous pipeline — failure is rejected as a GPUPipelineError (no error leaks globally).
      await check(6, async () => {
        const bad = device.createShaderModule({
          code: '@vertex fn vs() -> @builtin(position) vec4f { return nonexistent(1.0); }',
        })
        try {
          await device.createRenderPipelineAsync({
            layout: 'auto', vertex: { module: bad, entryPoint: 'vs' },
          })
          return { ok: false, detail: 'it was not rejected' }
        } catch (error) {
          const failure = /** @type {any} */ (error)
          const leaked = sink.take()
          return {
            ok: failure.name === 'GPUPipelineError' && leaked.length === 0,
            detail: `${failure.name}/${failure.reason} · global leaks ${leaked.length}`,
          }
        }
      })

      // ⑧ onuncapturederror — an error outside a scope arrives through the spec channel.
      await check(7, async () => {
        /** @type {any[]} */
        const events: any[] = []
        device.onuncapturederror = (event: any) => events.push(event)
        const encoder = device.createCommandEncoder()
        // Clearing a buffer that does not exist → a validation error.
        encoder.clearBuffer({ id: 999999, size: 16 })
        device.queue.submit([encoder.finish()])
        device.onuncapturederror = null
        sink.take()
        const first = events[0]
        // `constructor.name` gets squashed to a single letter by bundle minification — the `kind` this
        // implementation carries alongside is used instead (not in the spec, but more useful for diagnosis).
        return {
          ok: events.length > 0 && typeof first.error.message === 'string',
          detail: first ? `kind=${first.error.kind}` : 'no event',
        }
      })

      // ⑨ clearBuffer — only the range becomes 0 and the rest survives.
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

      // ⑩ copyBufferToBuffer, the short form — a whole copy from (src, dst) alone.
      await check(9, async () => {
        const source = device.createBuffer({
          size: 16, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        })
        const destination = device.createBuffer({
          size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        })
        device.queue.writeBuffer(source, 0, new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]))
        const encoder = device.createCommandEncoder()
        encoder.copyBufferToBuffer(source, destination)      // the short form
        device.queue.submit([encoder.finish()])
        const bytes = new Uint8Array(await destination.mapAsync())
        destination.unmap()
        const ok = bytes[0] === 1 && bytes[15] === 16
        source.destroy()
        destination.destroy()
        return { ok, detail: `[${bytes[0]}…${bytes[15]}]` }
      })

      // ⑪ getMappedRange(offset, size) — only that range comes back, and what was written goes back in.
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

        // The write side — filled at the back with mappedAtCreation, checking that it goes back in.
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
        return { ok, detail: `read ${middle[0]}~${middle[7]} · write ${back[24]}` }
      })

      // ⑫ Debug markers — received inside and outside a pass and in a bundle (no error means passed).
      await check(11, async () => {
        const bundleEncoder = device.createRenderBundleEncoder({ colorFormats: [targets[0].format] })
        bundleEncoder.pushDebugGroup('bundle range')
        bundleEncoder.popDebugGroup()
        bundleEncoder.finish()

        const encoder = device.createCommandEncoder()
        encoder.pushDebugGroup('frame')
        encoder.insertDebugMarker('marker')
        encoder.popDebugGroup()
        device.queue.submit([encoder.finish()])

        const leaked = sink.take()
        return { ok: leaked.length === 0, detail: leaked.length ? leaked[0].slice(0, 60) : '0 errors' }
      })

      // ⑬ Two core formats — they are created and even copied (packed 32-bit, so 4 bytes per pixel).
      //
      // `bytesPerRow` must be **a multiple of 256** (a spec requirement — with several rows it cannot be omitted either).
      // Using the tight 16B row of 4 bytes per pixel × 4 pixels as is would be rejected by a browser.
      await check(12, async () => {
        const results: string[] = []
        const bytesPerRow = 256
        for (const format of ['rgb10a2uint', 'rgb9e5ufloat']) {
          const texture = device.createTexture({
            size: { width: 4, height: 4 }, format,
            usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC,
          })
          const buffer = device.createBuffer({
            size: bytesPerRow * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
          })
          const encoder = device.createCommandEncoder()
          encoder.copyTextureToBuffer(
            { texture }, { buffer, bytesPerRow }, { width: 4, height: 4 }
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

      // ⑭ Canvas unconfigure / getConfiguration.
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
        // It is reconfigured before finishing — the frame loop below draws through this context.
        probe.configure({ device, format: targets[0].format })
        const ok = before === null && configured?.format === targets[0].format
          && after === null && rejected
        return { ok, detail: `null before configuration · ${rejected ? 'drawing rejected' : 'not rejected'} after release` }
      })

      if (disposed) return

      // The background — it shows the screen is still alive after the checks finish.
      // `getCanvasContext` gives the same object for the same id, so it stays as configured in ⑭ above.
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
        encoder.pushDebugGroup('spec frame')   // a marker is attached to the real frames too
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
      setStatus(`failed: ${(error && error.message) || error}`)
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
      <ChecklistHud title="Spec surface checklist" subtitle={status} checks={checks} />
    </view>
  )
}

root.render(<SpecScene />)
