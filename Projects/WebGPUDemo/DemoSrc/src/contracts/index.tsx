import { root, useEffect, useState } from '@lynx-js/react'
import gpu, { GPUBufferUsage, GPUTextureUsage, startFrameLoop } from '../webgpu.js'
import '../demo.css'
import '../elements.d.ts'
import { ChecklistHud, type Check } from '../checklist-hud.jsx'

/**
 * Contract checks — how the places flagged in review actually behave **on real hardware**.
 *
 * What these have in common is "places nobody had been checking". Three were genuinely empty (buffer copy
 * defaults and range validation, format reverse mapping), and two were already right but **had no evidence
 * that they were** (a bundle's index buffer isolation, occlusion query blocking).
 *
 * The unit tests hold the same contract, but this screen passes through **a real GPU and a real bridge** —
 * whether a contract satisfied by mocks holds on real hardware is the point.
 */

const CHECKS = [
  'copyBufferToBuffer — omitting size = the whole source',
  'copyBufferToBuffer — everything left after sourceOffset',
  'copyBufferToBuffer — going past the end is rejected',
  'copyBufferToBuffer — 0 bytes is a no-op',
  'vec3<i32> · vec3<u32> uniform layout (packed integer vectors)',
  'the index buffer is invalidated after a bundle runs',
  'a bundle does not inherit the pass index buffer',
  'an occlusion query cannot go in a bundle',
  'drawable format reverse mapping (the name the canvas pass knows)',
  'handles do not collide even with two devices',
]

/**
 * A shader that confirms integer vec3 layout by value.
 *
 * A WGSL `vec3<i32>` is 12 bytes while an MSL `int3` is 16. Without the emitter using `packed_int3`, the
 * following fields shift by 4 bytes and **the wrong value is read with no error** — hence the sum comes out as a color.
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
      setStatus(`${adapter.info.description || adapter.name} · ${adapter.info.architecture || '?'}`)

      const collected: string[] = []
      device.onError((_error: any, text: string) => collected.push(text))
      const takeErrors = () => {
        const taken = collected.slice()
        collected.length = 0
        return taken
      }

      /** A pair: a source filled with 16 bytes plus the destination to read back. */
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

      // ① Omitting size — the spec's short form `copyBufferToBuffer(src, dst)` means "the whole source".
      //    The JS shim fills it in, but whether that value really is the source size can only be told by value.
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

      // ② Given sourceOffset alone it is **everything left after it** — using the source size as is goes past the end.
      await check(1, async () => {
        const { source, destination } = makeBufferPair()
        const encoder = device.createCommandEncoder()
        encoder.copyBufferToBuffer(source, 8, destination, 0)   // size omitted
        device.queue.submit([encoder.finish()])
        const bytes = new Uint8Array(await destination.mapAsync())
        destination.unmap()
        // The source's bytes 9~16 come to the front. The last 8 bytes are untouched.
        const ok = bytes[0] === 9 && bytes[7] === 16 && bytes[8] === 0 && takeErrors().length === 0
        source.destroy()
        destination.destroy()
        return { ok, detail: `[${bytes[0]}…${bytes[7]}|${bytes[8]}]` }
      })

      // ③ Going past the end — **Metal kills this with an assertion.** It must come back as a validation error.
      await check(2, async () => {
        const results: string[] = []
        for (const [label, run] of [
          ['destination', (source: any, destination: any, encoder: any) => {
            encoder.copyBufferToBuffer(source, 0, destination, 0, 16)   // the destination is only 8B
          }],
          ['source', (source: any, destination: any, encoder: any) => {
            encoder.copyBufferToBuffer(source, 12, destination, 0, 8)   // the source is 12+8 > 16
          }],
          ['negative', (source: any, destination: any, encoder: any) => {
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

      // ④ 0 bytes is a no-op — a Metal blit rejects a 0-byte copy, so passing it straight through would be an error.
      await check(3, async () => {
        const { source, destination } = makeBufferPair()
        const encoder = device.createCommandEncoder()
        encoder.copyBufferToBuffer(source, 0, destination, 0, 0)
        device.queue.submit([encoder.finish()])
        const errors = takeErrors()
        source.destroy()
        destination.destroy()
        return { ok: errors.length === 0, detail: errors[0] ? errors[0].slice(0, 50) : '0 errors' }
      })

      // ⑤ Integer vec3 layout — without the emitter using packed, the later fields shift and the values change.
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
        // offsets(1,2,3) total 4 · sizes(5,6,7) stride 8 — filled at the WGSL offsets as they are.
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
        return { ok, detail: `rgb(${pixel.join(',')}) · expected rgb(10,26,36)` }
      })

      // --- Bundle isolation ---------------------------------------------------

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

      // ⑥ If the index buffer a bundle bound leaked into the pass, the drawIndexed that follows would **actually draw**.
      await check(5, async () => {
        const bundleEncoder = device.createRenderBundleEncoder({ colorFormats: ['rgba8unorm'] })
        bundleEncoder.setPipeline(bundlePipeline)
        bundleEncoder.setIndexBuffer(indices, 'uint16')
        bundleEncoder.drawIndexed(3)
        const bundle = bundleEncoder.finish()

        await renderPixel(bundlePipeline, (pass: any) => {
          pass.executeBundles([bundle])
          pass.setPipeline(bundlePipeline)
          pass.drawIndexed(3)              // with no setIndexBuffer
        }, { skipPipeline: true })
        const errors = takeErrors()
        return {
          // The wording differs per backend — the Metal engine says "setIndexBuffer", Dawn says "Index buffer".
          ok: errors.some((text) => text.includes('setIndexBuffer') || text.includes('Index buffer')),
          detail: errors[0] ? errors[0].slice(0, 55) : 'it was not rejected',
        }
      })

      // ⑦ The other direction — a bundle does not inherit what the pass bound.
      await check(6, async () => {
        const bundleEncoder = device.createRenderBundleEncoder({ colorFormats: ['rgba8unorm'] })
        bundleEncoder.setPipeline(bundlePipeline)
        bundleEncoder.drawIndexed(3)       // nothing was ever bound inside the bundle
        const bundle = bundleEncoder.finish()

        await renderPixel(bundlePipeline, (pass: any) => {
          pass.setIndexBuffer(indices, 'uint16')
          pass.executeBundles([bundle])
        }, { skipPipeline: true })
        const errors = takeErrors()
        return {
          // The spec puts this validation at the bundle's finish() — Dawn raises "Index buffer was
          // not set" then, while the Metal engine's replay raises "setIndexBuffer" at execution. Both are isolation.
          ok: errors.some((text) => text.includes('setIndexBuffer') || text.includes('Index buffer')),
          detail: errors[0] ? errors[0].slice(0, 55) : 'it was not rejected',
        }
      })

      // ⑧ An occlusion query belongs to a render **pass**. Whether the defence is two layers deep:
      //    the shim has no method at all (layer 1), and even forced in, native rejects it (layer 2).
      await check(7, async () => {
        const bundleEncoder = device.createRenderBundleEncoder({ colorFormats: ['rgba8unorm'] })
        const shimHides = typeof (bundleEncoder as any).beginOcclusionQuery !== 'function'
        // Forced in through the back door — the point is the **native line of defence**, not the normal path.
        ;(bundleEncoder as any)._commands.push({ op: 'beginOcclusionQuery', queryIndex: 0 })
        bundleEncoder.finish()
        device.queue.submit([])
        const errors = takeErrors()
        return {
          ok: shimHides && errors.some((text) => text.includes('render bundle')),
          detail: `shim hides it ${shimHides ? '✓' : '✗'} · native ${errors.length ? 'rejected' : 'passed'}`,
        }
      })

      // ⑨ Drawable format reverse mapping — by what name does a canvas pass know its own attachment.
      //
      //    The `MTLPixelFormat` → WebGPU name table lives only inside native and cannot be seen from JS.
      //    Instead, putting a **deliberately mismatched bundle** into a canvas pass makes the rejection
      //    message carry the pass's real format name — that is the reverse mapping's output.
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
        // "… bundle rgba8unorm, pass bgra8unorm" — the latter is the name mapped back from the drawable.
        const reported = (message.match(/pass ([a-z0-9-]+)/) || [])[1]
        // The Metal engine carries the reverse-mapped name in its rejection wording. Native validation (Dawn)
        // rejects in its own words — in that case this check only covers "the mismatched bundle was rejected".
        const nativeRejected = /not compatible|Attachment state/i.test(message)
        return {
          ok: reported === canvasFormat || nativeRejected,
          detail: reported
            ? `the name the pass knows ${reported} · configure ${canvasFormat}`
            : (message.slice(0, 55) || 'it was not rejected'),
        }
      })

      // ⑩ The handle space — the native registry is one per context and finds objects **by handle integer alone**.
      //    A per-device counter would make the second device start issuing from 1 again and silently
      //    overwrite the first device's objects. No error is raised, only the symptom of "someone else drawing into my buffer".
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

      // The background — it shows the screen is still alive after the checks finish.
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
     * Draws onto an 8×8 offscreen target and reads the center pixel back.
     *
     * This round trip is needed to confirm **the value the shader read**, rather than "no error occurred".
     * Checks that expect a rejection do not look at pixels and use only the error collector.
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
      // A texture→buffer copy's bytesPerRow must be a multiple of 256 per the spec.
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
      <ChecklistHud title="Contract checks" subtitle={status} checks={checks} />
    </view>
  )
}

root.render(<ContractsScene />)
