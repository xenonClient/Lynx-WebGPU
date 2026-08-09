import { root, useEffect, useState } from '@lynx-js/react'
import gpu, {
  GPUBufferUsage, GPUTextureUsage, createImageBitmap, startFrameLoop,
} from '../webgpu.js'
import '../demo.css'
import '../elements.d.ts'
import { ChecklistHud, type Check } from '../checklist-hud.jsx'

/**
 * The image path checklist — **block compressed textures** and **an external image → a texture**.
 *
 * Both are places where "wrong pixels with no error" comes easily (block arithmetic, channel order, vertical direction).
 * So the verdict here is not pass/fail but **reading the actual colors back** and checking the values.
 *
 * The compressed families differ per device — iOS devices always do ASTC and ETC2, and BC from the A14 on.
 * A missing family is shown as `–` rather than a failure (`adapter.features` is the answer).
 */

const CHECKS = [
  'adapter.features — the advertised compressed families',
  'ASTC 4x4 — sampling a single-color block',
  'ASTC 6x5 — a non-square block',
  'BC1 — an 8-byte block',
  'omitting bytesPerRow = the per-block default',
  'rejecting an origin off the block boundary',
  'rejecting a compressed texture as a render target',
  'createImageBitmap — the PNG size',
  'copyExternalImageToTexture — color and orientation',
  'flipY — top and bottom are flipped',
  'uploading only part of an image',
  'rejecting a non-4-byte format',
  'bitmap.close() — safe to call twice',
]

/** A 4x4 PNG — the top half red, the bottom half blue. It has to be asymmetric to show orientation. */
const PNG_BASE64
  = 'iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAYAAACp8Z5+AAAAFUlEQVR42mP4z8DwHxkzYAig8TEFACxQH+FE11LuAAAAAElFTkSuQmCC'

const SAMPLE_SHADER = /* wgsl */ `
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

struct Out {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

@vertex fn vs(@builtin(vertex_index) i: u32) -> Out {
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var out: Out;
  out.position = vec4f(p[i], 0.0, 1.0);
  // Texture coordinates have 0 at the top — y is flipped so the image's first row goes to the top of the screen.
  out.uv = vec2f(p[i].x * 0.5 + 0.5, 0.5 - p[i].y * 0.5);
  return out;
}

@fragment fn fs(in: Out) -> @location(0) vec4f {
  return textureSample(tex, samp, in.uv);
}
`

/** base64 → ArrayBuffer. The channel for baking an image into the bundle (no `loadAsset` needed). */
function decodeBase64(text: string): ArrayBuffer {
  const table = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'
  const clean = text.replace(/=+$/, '')
  const bytes = new Uint8Array((clean.length * 3) >> 2)
  let accumulator = 0
  let bits = 0
  let out = 0
  for (const character of clean) {
    accumulator = (accumulator << 6) | table.indexOf(character)
    bits += 6
    if (bits >= 8) {
      bits -= 8
      bytes[out++] = (accumulator >> bits) & 0xff
    }
  }
  return bytes.buffer
}

/**
 * An ASTC "void extent" block (16B) — the form that declares the whole block to be one color.
 * It is the only way to build deterministic compressed data with no encoder, and it is independent of block size.
 */
function astcVoidExtent(r: number, g: number, b: number): Uint8Array {
  const block = new Uint8Array(16)
  block.set([0xfc, 0xfd, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff])
  const channels = [r, g, b, 0xffff]
  channels.forEach((value, index) => {
    block[8 + index * 2] = value & 0xff
    block[9 + index * 2] = value >> 8
  })
  return block
}

/**
 * A BC1 block (8B) — with `color0 > color1` it is four-color mode and index 0 is `color0`.
 * Setting every index to 0 makes the whole 4×4 a single color.
 */
function bc1Block(rgb565: number): Uint8Array {
  const block = new Uint8Array(8)
  block[0] = rgb565 & 0xff
  block[1] = rgb565 >> 8
  return block
}

function ImagesScene() {
  const [status, setStatus] = useState('getting ready…')
  const [checks, setChecks] = useState<Check[]>(CHECKS.map((label) => ({ label, state: 'wait' })))

  useEffect(() => {
    let disposed = false
    let stopLoop: (() => void) | null = null
    let device: any = null

    function mark(index: number, state: Check['state'], detail?: string) {
      if (disposed) return
      setChecks((previous) => previous.map((check, i) => (
        i === index ? { ...check, state, detail } : check
      )))
    }

    /** The rest keep running even if one check throws — stopping at the first failure gives the least information. */
    async function check(
      index: number,
      run: () => Promise<{ ok: boolean, detail: string, skip?: boolean }>
    ) {
      try {
        const result = await run()
        mark(index, result.skip ? 'skip' : (result.ok ? 'ok' : 'fail'), result.detail)
      } catch (error) {
        mark(index, 'fail', `exception: ${(error && (error as Error).message) || error}`.slice(0, 90))
      }
    }

    async function boot() {
      const adapter = await gpu.requestAdapter()
      if (!adapter) throw new Error('no adapter')
      device = await adapter.requestDevice()

      /** A collector that gathers only the previous batch's errors — used by checks that expect a rejection. */
      const collected: string[] = []
      device.onError((_error: any, text: string) => collected.push(text))
      const takeErrors = () => {
        const taken = collected.slice()
        collected.length = 0
        return taken
      }

      const module = device.createShaderModule({ code: SAMPLE_SHADER })
      const sampler = device.createSampler()
      const pipeline = device.createRenderPipeline({
        layout: 'auto',
        vertex: { module, entryPoint: 'vs' },
        fragment: { module, entryPoint: 'fs', targets: [{ format: 'rgba8unorm' }] },
      })
      const layout = pipeline.getBindGroupLayout(0)

      /**
       * Draws the texture stretched onto an 8×8 offscreen target and reads the center pixel back.
       * The point is checking **the decoded color** rather than "it was created".
       */
      async function sampleCenter(texture: any, atRow = 4): Promise<number[]> {
        const target = device.createTexture({
          size: { width: 8, height: 8 }, format: 'rgba8unorm',
          usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
        })
        // A texture→buffer copy's bytesPerRow must be a multiple of 256 per the spec (Metal lets it slide
        // but browsers and Dawn reject it). Only the row stride widens; the pixel coordinates read stay the same.
        const readback = device.createBuffer({
          size: 256 * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        })
        const bindGroup = device.createBindGroup({
          layout,
          entries: [
            { binding: 0, resource: texture.createView() },
            { binding: 1, resource: sampler },
          ],
        })
        const encoder = device.createCommandEncoder()
        const pass = encoder.beginRenderPass({
          colorAttachments: [{
            view: target.createView(), loadOp: 'clear', storeOp: 'store',
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
          }],
        })
        pass.setPipeline(pipeline)
        pass.setBindGroup(0, bindGroup)
        pass.draw(3)
        pass.end()
        encoder.copyTextureToBuffer(
          { texture: target }, { buffer: readback, bytesPerRow: 256 }, { width: 8, height: 8 }
        )
        device.queue.submit([encoder.finish()])

        const bytes = new Uint8Array(await readback.mapAsync())
        const offset = atRow * 256 + 4 * 4
        const pixel = [bytes[offset], bytes[offset + 1], bytes[offset + 2]]
        readback.unmap()
        readback.destroy()
        target.destroy()
        return pixel
      }

      const near = (pixel: number[], want: number[]) =>
        pixel.every((value, index) => Math.abs(value - want[index]) <= 3)

      const features = adapter.features
      const hasASTC = features.has('texture-compression-astc')
      const hasBC = features.has('texture-compression-bc')
      setStatus(
        `${adapter.info.architecture || adapter.name} · `
        + `ASTC ${hasASTC ? 'present' : 'absent'} · BC ${hasBC ? 'present' : 'absent'}`
      )

      // ① The advertised families and what can actually be created must agree.
      await check(0, async () => {
        const advertised = ['astc', 'etc2', 'bc'].filter(
          (name) => features.has(`texture-compression-${name}`)
        )
        let created = true
        if (hasASTC) {
          const probe = device.createTexture({
            size: { width: 4, height: 4 }, format: 'astc-4x4-unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
          })
          device.queue.submit([])
          created = takeErrors().length === 0
          probe.destroy()
        }
        return {
          ok: advertised.length > 0 && created,
          detail: advertised.join(', ') || 'no advertised families',
        }
      })

      // ② ASTC 4x4 — a single-color block decodes to that color.
      await check(1, async () => {
        if (!hasASTC) return { ok: true, skip: true, detail: 'this device does not support ASTC' }
        const texture = device.createTexture({
          size: { width: 4, height: 4 }, format: 'astc-4x4-unorm',
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        })
        device.queue.writeTexture(
          { texture }, astcVoidExtent(0, 0xffff, 0xffff), { bytesPerRow: 16 },
          { width: 4, height: 4 }
        )
        const pixel = await sampleCenter(texture)
        texture.destroy()
        return { ok: near(pixel, [0, 255, 255]), detail: `rgb(${pixel.join(',')})` }
      })

      // ③ ASTC 6x5 — counting block rows as the height breaks right here.
      await check(2, async () => {
        if (!hasASTC) return { ok: true, skip: true, detail: 'this device does not support ASTC' }
        const texture = device.createTexture({
          size: { width: 6, height: 5 }, format: 'astc-6x5-unorm',
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        })
        device.queue.writeTexture(
          { texture }, astcVoidExtent(0xffff, 0x8000, 0), { bytesPerRow: 16 },
          { width: 6, height: 5 }
        )
        const pixel = await sampleCenter(texture)
        texture.destroy()
        return { ok: near(pixel, [255, 128, 0]), detail: `rgb(${pixel.join(',')})` }
      })

      // ④ BC1 — the 8-bytes-per-block family. RGB565's pure red is 255 in 8 bits.
      await check(3, async () => {
        if (!hasBC) return { ok: true, skip: true, detail: 'devices below the A14 have no BC' }
        const texture = device.createTexture({
          size: { width: 4, height: 4 }, format: 'bc1-rgba-unorm',
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        })
        device.queue.writeTexture(
          { texture }, bc1Block(0xf800), { bytesPerRow: 8 }, { width: 4, height: 4 }
        )
        const pixel = await sampleCenter(texture)
        texture.destroy()
        return { ok: near(pixel, [255, 0, 0]), detail: `rgb(${pixel.join(',')})` }
      })

      // ⑤ Omitting bytesPerRow — an 8x8 is 2x2 blocks, so a row is 32 bytes.
      await check(4, async () => {
        if (!hasASTC) return { ok: true, skip: true, detail: 'this device does not support ASTC' }
        const texture = device.createTexture({
          size: { width: 8, height: 8 }, format: 'astc-4x4-unorm',
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        })
        const blocks = new Uint8Array(64)
        for (let index = 0; index < 4; index += 1) {
          blocks.set(astcVoidExtent(0x4000, 0xffff, 0x4000), index * 16)
        }
        device.queue.writeTexture({ texture }, blocks, {}, { width: 8, height: 8 })
        const pixel = await sampleCenter(texture)
        texture.destroy()
        return { ok: near(pixel, [64, 255, 64]), detail: `rgb(${pixel.join(',')})` }
      })

      // ⑥ An origin off the block boundary — Metal dies on an assertion. It must come back as a validation error.
      await check(5, async () => {
        if (!hasASTC) return { ok: true, skip: true, detail: 'this device does not support ASTC' }
        const texture = device.createTexture({
          size: { width: 8, height: 8 }, format: 'astc-4x4-unorm',
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        })
        device.queue.writeTexture(
          { texture, origin: { x: 2, y: 0 } }, astcVoidExtent(0xffff, 0, 0),
          { bytesPerRow: 16 }, { width: 4, height: 4 }
        )
        device.queue.submit([])
        const errors = takeErrors()
        texture.destroy()
        return {
          ok: errors.length > 0 && errors[0].includes('block boundary'),
          detail: errors[0] ? errors[0].slice(0, 50) : 'it was not rejected',
        }
      })

      // ⑦ Compressed + RENDER_ATTACHMENT — the GPU has no block encoder.
      await check(6, async () => {
        if (!hasASTC) return { ok: true, skip: true, detail: 'this device does not support ASTC' }
        device.createTexture({
          size: { width: 4, height: 4 }, format: 'astc-4x4-unorm',
          usage: GPUTextureUsage.RENDER_ATTACHMENT,
        })
        device.queue.submit([])
        const errors = takeErrors()
        return {
          ok: errors.length > 0,
          detail: errors[0] ? errors[0].slice(0, 50) : 'it was not rejected',
        }
      })

      // ⑧ createImageBitmap — native unpacks the PNG and returns the size.
      const png = decodeBase64(PNG_BASE64)
      let bitmap: any = null
      await check(7, async () => {
        bitmap = await createImageBitmap(png)
        return {
          ok: bitmap.width === 4 && bitmap.height === 4,
          detail: `${bitmap.width}x${bitmap.height}`,
        }
      })

      // ⑨ Color and orientation — the first row must be red and the last row blue.
      //    A flipped channel order turns it blue; a flipped vertical swaps the positions.
      await check(8, async () => {
        const texture = device.createTexture({
          size: [bitmap.width, bitmap.height], format: 'rgba8unorm',
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        })
        device.queue.copyExternalImageToTexture({ source: bitmap }, { texture })
        const top = await sampleCenter(texture, 1)
        const bottom = await sampleCenter(texture, 6)
        texture.destroy()
        return {
          ok: near(top, [255, 0, 0]) && near(bottom, [0, 0, 255]),
          detail: `top rgb(${top.join(',')}) · bottom rgb(${bottom.join(',')})`,
        }
      })

      // ⑩ flipY — the same image arrives upside down.
      await check(9, async () => {
        const flipped = await createImageBitmap(png, { flipY: true })
        const texture = device.createTexture({
          size: [flipped.width, flipped.height], format: 'rgba8unorm',
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        })
        device.queue.copyExternalImageToTexture({ source: flipped }, { texture })
        const top = await sampleCenter(texture, 1)
        texture.destroy()
        flipped.close()
        return { ok: near(top, [0, 0, 255]), detail: `top rgb(${top.join(',')})` }
      })

      // ⑪ A partial copy — only 2x2 is cut from the bottom half (blue).
      await check(10, async () => {
        const texture = device.createTexture({
          size: [2, 2], format: 'rgba8unorm',
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        })
        device.queue.copyExternalImageToTexture(
          { source: bitmap, origin: { x: 2, y: 2 } }, { texture }, [2, 2]
        )
        const pixel = await sampleCenter(texture)
        texture.destroy()
        return { ok: near(pixel, [0, 0, 255]), detail: `rgb(${pixel.join(',')})` }
      })

      // ⑫ The decoded result is RGBA8, so a non-4-byte format is rejected rather than quietly mismatching.
      await check(11, async () => {
        const texture = device.createTexture({
          size: [4, 4], format: 'rgba16float',
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        })
        device.queue.copyExternalImageToTexture({ source: bitmap }, { texture })
        device.queue.submit([])
        const errors = takeErrors()
        texture.destroy()
        return {
          ok: errors.length > 0 && errors[0].includes('4-byte'),
          detail: errors[0] ? errors[0].slice(0, 50) : 'it was not rejected',
        }
      })

      // ⑬ close() releases the native pixels. Calling it twice must raise no error.
      await check(12, async () => {
        bitmap.close()
        bitmap.close()
        device.queue.submit([])
        const errors = takeErrors()
        return { ok: errors.length === 0, detail: errors[0] ? errors[0].slice(0, 50) : '0 errors' }
      })

      if (disposed) return

      // The background — it shows the screen is still alive after the checks finish.
      const context = gpu.getCanvasContext('main')
      const format = gpu.getPreferredCanvasFormat()
      context.configure({ device, format })
      const screenModule = device.createShaderModule({ code: SAMPLE_SHADER })
      const screenPipeline = device.createRenderPipeline({
        layout: 'auto',
        vertex: { module: screenModule, entryPoint: 'vs' },
        fragment: { module: screenModule, entryPoint: 'fs', targets: [{ format }] },
      })
      const screenBitmap = await createImageBitmap(png)
      const screenTexture = device.createTexture({
        size: [screenBitmap.width, screenBitmap.height], format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      })
      device.queue.copyExternalImageToTexture({ source: screenBitmap }, { texture: screenTexture })
      screenBitmap.close()
      const screenBind = device.createBindGroup({
        layout: screenPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: screenTexture.createView() },
          { binding: 1, resource: sampler },
        ],
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
        pass.setBindGroup(0, screenBind)
        pass.draw(3)
        pass.end()
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
      <ChecklistHud title="Images · compressed textures" subtitle={status} checks={checks} />
    </view>
  )
}

root.render(<ImagesScene />)
