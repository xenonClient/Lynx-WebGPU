// @ts-nocheck — three's type declarations assume the DOM lib and fight this project's tsconfig.
// The build is an SWC transpile so it has no effect, and this file's integration points are confirmed by runtime verification (the checklist).
import { root, useEffect, useState } from '@lynx-js/react'
import * as THREE from 'three/webgpu'
import gpu, {
  GPUBufferUsage, GPUTextureUsage, createImageBitmap, installAnimationFrame,
} from '../webgpu.js'
import '../demo.css'
import '../elements.d.ts'
import { ChecklistHud, type Check } from '../checklist-hud.jsx'

// three's internal Animation loop assumes rAF — PrimJS has none, so it has to be installed.
// (Without it, renderer.init() stalls forever with no error.) It is called at the top of the module
// so it can be installed before the import — safe even if three captures the globals during module initialization.
const uninstallAnimationFrame = installAnimationFrame()

// ---------------------------------------------------------------------------
// Command stream instrumentation — counts batch kinds (P = frame submission / I = internal submission) and errors
// ---------------------------------------------------------------------------

const streamStats = { frameBatches: 0, internalBatches: 0, errors: 0 }

function attachStreamCounter(device: any) {
  const recorder = device._recorder
  const originalFlush = recorder.flush.bind(recorder)
  let logged = 0
  // Note: **every argument** of flush is forwarded as is. Swallowing `present` disguises an internal
  // submission as a frame one, and swallowing the second argument (`presentOnly`) keeps the tick's wrap-up
  // batch from reaching native and freezes the screen — the place where instrumentation breaks what it observes.
  recorder.flush = (present?: boolean, ...rest: any[]) => {
    if (recorder.pending.length > 0) {
      if (present === false) streamStats.internalBatches += 1
      else streamStats.frameBatches += 1
      if (logged < 14) {
        const ops = recorder.pending.map((command: any) => command.op)
        console.log(`[3js-dump] #${logged}${present === false ? 'I' : 'P'} (${ops.length}) ${ops.join(' ')}`)
        logged += 1
      }
    }
    return originalFlush(present, ...rest)
  }
}

// ---------------------------------------------------------------------------
// The paths this branch newly opened — block compressed textures and external images
// ---------------------------------------------------------------------------

/**
 * An ASTC "void extent" block (16B) — the form that declares the whole block to be one color.
 *
 * It is the only way to build deterministic compressed data with no encoder, so **the real ASTC path** can
 * be exercised without putting an encoded asset in the bundle. The first 9 bits are the signature
 * (`0b111111100`) and the last 8 bytes are UNORM16 RGBA.
 * @param {number[]} rgb three UNORM16 channels
 */
function astcVoidExtent(rgb: number[]): Uint8Array {
  const block = new Uint8Array(16)
  block.set([0xfc, 0xfd, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff])
  const channels = [rgb[0], rgb[1], rgb[2], 0xffff]
  channels.forEach((value, index) => {
    block[8 + index * 2] = value & 0xff
    block[9 + index * 2] = value >> 8
  })
  return block
}

/** A 4×4 PNG — the top half red, the bottom half blue. It has to be asymmetric to show orientation. */
const PNG_BASE64
  = 'iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAYAAACp8Z5+AAAAFUlEQVR42mP4z8DwHxkzYAig8TEFACxQH+FE11LuAAAAAElFTkSuQmCC'

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
 * A 16×16 ASTC 4×4 texture — sixteen 4×4 blocks, each a single color, so it reads as **a grid of colors**.
 *
 * The cube spinning on screen uses this. It is where you see with your own eyes that **the compressed data
 * really is decoded onto the screen**, rather than "it was created" (256B — a quarter of uncompressed rgba8unorm).
 */
function makeCompressedGrid(): { texture: any, bytes: number, raw: number } {
  const blocksPerSide = 4
  const data = new Uint8Array(blocksPerSide * blocksPerSide * 16)
  for (let y = 0; y < blocksPerSide; y++) {
    for (let x = 0; x < blocksPerSide; x++) {
      // A color wheel running along the diagonal — the faces are distinguishable as the cube turns.
      const hue = ((x + y * blocksPerSide) / (blocksPerSide * blocksPerSide))
      const color = new THREE.Color().setHSL(hue, 0.75, 0.55)
      data.set(
        astcVoidExtent([
          Math.round(color.r * 0xffff), Math.round(color.g * 0xffff), Math.round(color.b * 0xffff),
        ]),
        (y * blocksPerSide + x) * 16
      )
    }
  }
  const side = blocksPerSide * 4
  const texture = new THREE.CompressedTexture(
    [{ data, width: side, height: side }], side, side,
    THREE.RGBA_ASTC_4x4_Format, THREE.UnsignedByteType
  )
  texture.minFilter = THREE.NearestFilter
  texture.magFilter = THREE.NearestFilter
  texture.needsUpdate = true
  return { texture, bytes: data.length, raw: side * side * 4 }
}

// ---------------------------------------------------------------------------
// Feature checks — draw into a render target and read the pixel values back to compare with the expected color
// ---------------------------------------------------------------------------

const CHECK_LABELS = [
  'bootstrap adapter→device→lost',
  'shim probe: buffer round trip',
  'shim probe: texture round trip',
  'clear color readback',
  'shader pipeline (a single-color quad)',
  'texture upload and sampling',
  'compressed texture (CompressedTexture · ASTC)',
  'external image (createImageBitmap → PNG decoding)',
  'lighting (Standard + Directional)',
  'depth test (front hiding back)',
  'instancing (InstancedMesh)',
  'mipmap generation (its own compute pass)',
  'alpha blending (the compositing formula)',
  'asynchronous pipeline (compileAsync)',
  'animation loop',
  'no command stream errors',
]
const CHECK_ANIMATION = 14
const CHECK_STREAM = 15

/**
 * A direct shim round trip that does not go through three — it tells which layer is at fault when a three check fails.
 * A: writeBuffer → mapAsync. B: writeTexture → copyTextureToBuffer → mapAsync.
 */
async function runShimProbes(
  device: any,
  mark: (index: number, state: 'ok' | 'fail', detail?: string) => void
) {
  {
    const buffer = device.createBuffer({
      size: 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    })
    device.queue.writeBuffer(buffer, 0, new Uint8Array([11, 22, 33, 44]))
    device.queue.submit([])
    const bytes = new Uint8Array(await buffer.mapAsync())
    buffer.unmap()
    const ok = bytes[0] === 11 && bytes[3] === 44
    mark(1, ok ? 'ok' : 'fail', `[${bytes.join(',')}]`)
    buffer.destroy()
  }
  {
    const texture = device.createTexture({
      size: { width: 1, height: 1 }, format: 'rgba8unorm',
      usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC | GPUTextureUsage.RENDER_ATTACHMENT,
    })
    device.queue.writeTexture(
      { texture }, new Uint8Array([255, 0, 255, 255]),
      { bytesPerRow: 4 }, { width: 1, height: 1 }
    )
    const buffer = device.createBuffer({
      size: 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    })
    const encoder = device.createCommandEncoder()
    encoder.copyTextureToBuffer(
      { texture }, { buffer, bytesPerRow: 256 }, { width: 1, height: 1 }
    )
    device.queue.submit([encoder.finish()])
    const bytes = new Uint8Array(await buffer.mapAsync())
    buffer.unmap()
    const ok = bytes[0] === 255 && bytes[1] === 0 && bytes[2] === 255
    mark(2, ok ? 'ok' : 'fail', `[${bytes.join(',')}]`)
    buffer.destroy()
    texture.destroy()
  }
}

/** One pixel [r, g, b] (0~255) of a render target. By default near the center of an 8×8 target. */
async function readPixel(
  renderer: any, target: any, x = 4, y = 4
): Promise<[number, number, number]> {
  const data = await renderer.readRenderTargetPixelsAsync(target, x, y, 1, 1)
  return [data[0], data[1], data[2]]
}

function formatRGB([r, g, b]: [number, number, number]) {
  return `(${r},${g},${b})`
}

/**
 * Pixel value verification — the items are chosen so each walks **a different GPU path**.
 *
 * Clear (pass initialization) → a single-color quad (node shader → WGSL translation → pipeline) →
 * texture (writeTexture + sampler) → lighting (lighting uniforms + BRDF) → depth (a depth attachment +
 * a comparison function) → instancing (an instance buffer + `@builtin(instance_index)`) → mipmaps (three's
 * own compute pass) → blending (fixed-function compositing) → asynchronous compilation (`createRenderPipelineAsync`).
 *
 * The readback itself is `copyTextureToBuffer` + `mapAsync`, so passing verifies that path along the way.
 */
async function runPixelChecks(
  renderer: any,
  mark: (index: number, state: 'ok' | 'fail', detail?: string) => void
) {
  const target = new THREE.RenderTarget(8, 8, { depthBuffer: true })
  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 10)
  camera.position.z = 1

  async function renderAndRead(
    scene: any, x = 4, y = 4, useCamera: any = camera
  ): Promise<[number, number, number]> {
    renderer.setRenderTarget(target)
    await renderer.renderAsync(scene, useCamera)
    renderer.setRenderTarget(null)
    return readPixel(renderer, target, x, y)
  }

  /** The rest keep running even if one check throws — stopping at the first failure gives the least information. */
  async function check(
    index: number,
    run: () => Promise<{ ok: boolean, detail: string }>
  ) {
    try {
      const result = await run()
      mark(index, result.ok ? 'ok' : 'fail', result.detail)
    } catch (error) {
      mark(index, 'fail', `exception: ${(error && (error as Error).message) || error}`.slice(0, 80))
    }
  }

  // ① Clear color — a scene with nothing but a red background.
  await check(3, async () => {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(1, 0, 0)
    const rgb = await renderAndRead(scene)
    return { ok: rgb[0] > 180 && rgb[1] < 60 && rgb[2] < 60, detail: formatRGB(rgb) }
  })

  // ② A single-color quad — a green MeshBasicMaterial covers the screen.
  await check(4, async () => {
    const scene = new THREE.Scene()
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshBasicMaterial({ color: 0x00ff00 })
    ))
    const rgb = await renderAndRead(scene)
    return { ok: rgb[1] > 180 && rgb[0] < 60 && rgb[2] < 60, detail: formatRGB(rgb) }
  })

  // ③ Texture — sampling an orange single-color 2×2 DataTexture.
  await check(5, async () => {
    const texels = new Uint8Array(16)
    for (let index = 0; index < 4; index++) texels.set([255, 128, 0, 255], index * 4)
    const texture = new THREE.DataTexture(texels, 2, 2, THREE.RGBAFormat, THREE.UnsignedByteType)
    texture.needsUpdate = true

    const scene = new THREE.Scene()
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshBasicMaterial({ map: texture })
    ))
    const rgb = await renderAndRead(scene)
    return {
      ok: rgb[0] > 180 && rgb[1] > 60 && rgb[1] < 220 && rgb[2] < 60,
      detail: formatRGB(rgb),
    }
  })

  // ③-1 Compressed texture — **a path three walks by itself**.
  //
  //  Seeing a `CompressedTexture`, three goes to `_copyCompressedBufferToTexture()` and calls `writeTexture`
  //  with `bytesPerRow = ceil(width/blockWidth) × blockBytes` and `rowsPerImage = ceil(height/blockHeight)` —
  //  exactly the same contract as the block arithmetic this branch added.
  //  Before, `adapter.features` carried no compressed families and three did not know this road existed.
  await check(6, async () => {
    const device = renderer.backend.device
    if (!device.features.has('texture-compression-astc')) {
      return { ok: false, detail: 'three never received the astc feature' }
    }
    // 8×8 = four 4×4 blocks. Each block is a single color, so the result is a 2×2 color grid.
    const blocks = new Uint8Array(4 * 16)
    const colors = [
      [0xffff, 0x2000, 0x2000],   // red
      [0x2000, 0xffff, 0x2000],   // green
      [0x2000, 0x2000, 0xffff],   // blue
      [0xffff, 0xffff, 0x2000],   // yellow
    ]
    colors.forEach((color, index) => blocks.set(astcVoidExtent(color), index * 16))

    const texture = new THREE.CompressedTexture(
      [{ data: blocks, width: 8, height: 8 }], 8, 8,
      THREE.RGBA_ASTC_4x4_Format, THREE.UnsignedByteType
    )
    texture.minFilter = THREE.NearestFilter
    texture.magFilter = THREE.NearestFilter
    texture.needsUpdate = true

    const scene = new THREE.Scene()
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshBasicMaterial({ map: texture })
    ))
    // Whether the four blocks come out **in different colors**. Looking at one block alone would pass even
    // with the arithmetic wrong, reading the same block four times.
    //
    // The texture v axis has its origin at the bottom (`CompressedTexture` has flipY false) — the bottom of
    // the screen is the first block row (red, green) and the top is the second (blue, yellow).
    const bottomLeft = await renderAndRead(scene, 2, 6)
    const bottomRight = await renderAndRead(scene, 6, 6)
    const topLeft = await renderAndRead(scene, 2, 2)
    const ok = bottomLeft[0] > 150 && bottomLeft[1] < 120        // red
      && bottomRight[1] > 150 && bottomRight[0] < 120            // green
      && topLeft[2] > 150 && topLeft[0] < 120                    // blue
    return {
      ok,
      detail: `${formatRGB(bottomLeft)}/${formatRGB(bottomRight)}/${formatRGB(topLeft)}`
        + ` · ${blocks.length}B (uncompressed ${8 * 8 * 4}B)`,
    }
  })

  // ③-2 External image — the path where three calls `queue.copyExternalImageToTexture()`.
  //
  //  Meeting a texture that is neither a DataTexture, compressed, nor a cube, three goes to
  //  `_copyImageToTexture()` and calls this API directly. Just putting the object `createImageBitmap()`
  //  returned into `image` makes it **literally the same shape** as the code used in a browser.
  await check(7, async () => {
    const bitmap = await createImageBitmap(decodeBase64(PNG_BASE64))
    const texture = new THREE.Texture(bitmap)
    texture.magFilter = THREE.NearestFilter
    texture.minFilter = THREE.NearestFilter
    texture.generateMipmaps = false
    // `flipY` is left alone — three's default is true. That value has to ride out as
    // `copyExternalImageToTexture`'s source option, per the spec, for the image to come out upright.
    texture.needsUpdate = true

    const scene = new THREE.Scene()
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshBasicMaterial({ map: texture })
    ))
    // The PNG's first row is red and its last row blue. three sets flipY, so **red must be on top on screen
    // too** — ignore it and it comes out flipped right here.
    const top = await renderAndRead(scene, 4, 2)
    const bottom = await renderAndRead(scene, 4, 6)
    bitmap.close()
    return {
      ok: top[0] > 150 && top[2] < 120 && bottom[2] > 150 && bottom[0] < 120,
      detail: `top ${formatRGB(top)} · bottom ${formatRGB(bottom)}`,
    }
  })

  // ④ Lighting — a white StandardMaterial plane with a head-on directional light. Dead lighting comes out black.
  await check(8, async () => {
    const scene = new THREE.Scene()
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 1, metalness: 0 })
    ))
    const light = new THREE.DirectionalLight(0xffffff, 3)
    light.position.set(0, 0, 1)
    scene.add(light)
    const rgb = await renderAndRead(scene)
    return { ok: rgb[0] > 80 && rgb[1] > 80 && rgb[2] > 80, detail: formatRGB(rgb) }
  })

  // ⑤ Depth test — blue in front, red behind. With the depth comparison dead, red wins by draw order.
  await check(9, async () => {
    const scene = new THREE.Scene()
    const back = new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial({ color: 0xff0000 })
    )
    back.position.z = -0.5
    const front = new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial({ color: 0x0000ff })
    )
    front.position.z = 0.5
    // The front one is drawn **first**, so nothing but a depth test can pass this.
    front.renderOrder = 0
    back.renderOrder = 1
    scene.add(front, back)

    const rgb = await renderAndRead(scene)
    return { ok: rgb[2] > 180 && rgb[0] < 60, detail: formatRGB(rgb) }
  })

  // ⑥ Instancing — a different position and color per instance. Without the instance buffer only one is drawn.
  await check(10, async () => {
    const mesh = new THREE.InstancedMesh(
      new THREE.PlaneGeometry(0.8, 2),
      new THREE.MeshBasicMaterial(),
      2
    )
    const matrix = new THREE.Matrix4()
    matrix.setPosition(-0.5, 0, 0)
    mesh.setMatrixAt(0, matrix)
    mesh.setColorAt(0, new THREE.Color(1, 0, 0))
    matrix.setPosition(0.5, 0, 0)
    mesh.setMatrixAt(1, matrix)
    mesh.setColorAt(1, new THREE.Color(0, 0, 1))
    mesh.instanceMatrix.needsUpdate = true
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true

    const scene = new THREE.Scene()
    scene.add(mesh)

    // The left (red) and right (blue) are read separately — with instance 1 undrawn, the right is black.
    const left = await renderAndRead(scene, 1, 4)
    const right = await renderAndRead(scene, 6, 4)
    return {
      ok: left[0] > 150 && left[2] < 80 && right[2] > 150 && right[0] < 80,
      detail: `L${formatRGB(left)} R${formatRGB(right)}`,
    }
  })

  // ⑦ Mipmaps — three builds the mips with its own compute pass. A lower mip is force-sampled to see
  //    whether the average of the two colors comes out (with mip generation dead it is the original color or black).
  await check(11, async () => {
    const size = 4
    const texels = new Uint8Array(size * size * 4)
    for (let index = 0; index < size * size; index++) {
      // Half red, half green — the average must be near (128, 128, 0).
      texels.set(index % 2 === 0 ? [255, 0, 0, 255] : [0, 255, 0, 255], index * 4)
    }
    const texture = new THREE.DataTexture(texels, size, size, THREE.RGBAFormat, THREE.UnsignedByteType)
    texture.generateMipmaps = true
    texture.minFilter = THREE.LinearMipmapLinearFilter
    texture.magFilter = THREE.LinearFilter
    texture.needsUpdate = true

    const material = new THREE.MeshBasicMaterial({ map: texture })
    const scene = new THREE.Scene()
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material)
    scene.add(mesh)

    const rgb = await renderAndRead(scene)
    // What is checked is **whether the mips were really built** — if they were, the lower mip mixes the two
    // colors and both channels stay alive. With generation dead, a single original texel (pure red or pure
    // green) comes out unchanged. A loose "passes if not black" would go green even with the whole mip
    // pipeline failing (an entry point resolution bug was very nearly missed that way).
    const ok = rgb[0] > 40 && rgb[1] > 40
    return { ok, detail: `${formatRGB(rgb)}${ok ? '' : ' not mixed'}` }
  })

  // ⑧ Alpha blending — 50% blue over opaque red. With compositing dead a pure color comes out.
  await check(12, async () => {
    const scene = new THREE.Scene()
    const back = new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial({ color: 0xff0000 })
    )
    back.position.z = -0.5
    const front = new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshBasicMaterial({ color: 0x0000ff, transparent: true, opacity: 0.5 })
    )
    front.position.z = 0.5
    scene.add(back, front)

    const rgb = await renderAndRead(scene)
    // Mixed half and half, both channels are mid-range — a pure color (255/0) means no compositing happened.
    return { ok: rgb[0] > 60 && rgb[0] < 210 && rgb[2] > 60 && rgb[2] < 210, detail: formatRGB(rgb) }
  })

  // ⑨ Asynchronous pipeline — three's compileAsync rides createRenderPipelineAsync.
  await check(13, async () => {
    const scene = new THREE.Scene()
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshBasicMaterial({ color: 0x00ffff })
    ))
    await renderer.compileAsync(scene, camera)
    const rgb = await renderAndRead(scene)
    return { ok: rgb[1] > 150 && rgb[2] > 150 && rgb[0] < 90, detail: formatRGB(rgb) }
  })

  target.dispose()
}

// ---------------------------------------------------------------------------
// The scene
// ---------------------------------------------------------------------------

/** The spinning cube to look at — a checker texture plus lighting, so any dead feature shows. */
/**
 * The scene you look at — **the two paths this branch opened really do make the picture.**
 *
 * - The spinning cube's surface is **ASTC block compressed data**. The GPU decodes the blocks and draws them.
 * - The panel behind is a texture uploaded from **a PNG decoded natively** (`createImageBitmap`).
 *   It rides the path where three calls `queue.copyExternalImageToTexture()` by itself.
 *
 * On a device with no compressed families the cube falls back to the old `DataTexture` checker — better
 * than an empty screen, and the HUD says which one it is.
 */
function buildSpinScene(aspect: number, options: { compressed: boolean, backdrop: any }) {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0x0b0e14)

  const camera = new THREE.PerspectiveCamera(50, aspect, 0.1, 20)
  camera.position.z = 4

  let surface: any
  let note: string
  if (options.compressed) {
    const grid = makeCompressedGrid()
    surface = grid.texture
    note = `ASTC 4x4 ${grid.bytes}B (uncompressed ${grid.raw}B)`
  } else {
    const size = 8
    const texels = new Uint8Array(size * size * 4)
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const even = (x + y) % 2 === 0
        texels.set(even ? [255, 176, 32, 255] : [24, 60, 116, 255], (y * size + x) * 4)
      }
    }
    surface = new THREE.DataTexture(texels, size, size, THREE.RGBAFormat, THREE.UnsignedByteType)
    surface.magFilter = THREE.NearestFilter
    surface.needsUpdate = true
    note = 'uncompressed (this device has no ASTC)'
  }

  const mesh = new THREE.Mesh(
    new THREE.BoxGeometry(1.6, 1.6, 1.6),
    new THREE.MeshStandardMaterial({ map: surface, roughness: 0.4, metalness: 0.1 })
  )
  scene.add(mesh)

  // The panel behind — a PNG decoded natively. Placed behind the cube so both appear in one frame.
  if (options.backdrop) {
    const backdrop = new THREE.Mesh(
      new THREE.PlaneGeometry(7, 7),
      new THREE.MeshBasicMaterial({ map: options.backdrop, opacity: 0.35, transparent: true })
    )
    backdrop.position.z = -2.5
    scene.add(backdrop)
  }

  const key = new THREE.DirectionalLight(0xffffff, 2.6)
  key.position.set(2, 3, 4)
  scene.add(key)
  scene.add(new THREE.AmbientLight(0xffffff, 0.35))

  return { scene, camera, mesh, note }
}

function ThreeScene() {
  const [status, setStatus] = useState('initializing three.js…')
  const [checks, setChecks] = useState<Check[]>(
    CHECK_LABELS.map((label) => ({ label, state: 'wait' }))
  )
  const [stats, setStats] = useState('')
  const [errorLines, setErrorLines] = useState<string[]>([])

  useEffect(() => {
    let disposed = false
    let renderer: any = null

    function mark(index: number, state: 'ok' | 'fail', detail?: string) {
      if (disposed) return
      setChecks((previous) =>
        previous.map((check, checkIndex) =>
          checkIndex === index ? { ...check, state, detail } : check
        )
      )
    }

    function refreshStats(fps: number | null) {
      const fpsText = fps === null ? '' : ` · ${fps}fps`
      setStats(
        `stream P ${streamStats.frameBatches} · I ${streamStats.internalBatches}`
          + ` · errors ${streamStats.errors}${fpsText}`
      )
    }

    async function boot() {
      const context = gpu.getCanvasContext('main')

      // Before layout the size is 0 — it waits briefly until it is ready.
      let size = context.getSize()
      for (let attempt = 0; attempt < 40 && (size.width === 0 || size.height === 0); attempt++) {
        await new Promise((resolve) => setTimeout(resolve, 50))
        if (disposed) return
        size = context.getSize()
      }
      if (size.width === 0) throw new Error('the canvas size never settled')

      // The minimal canvas surface three expects. setAttribute is deliberately left out
      // (the 'setAttribute' in domElement branch) so it does not take the DOM path.
      const fakeCanvas = {
        width: size.width,
        height: size.height,
        addEventListener() {},
        removeEventListener() {},
        dispatchEvent() {},
        getContext: () => context,
      }

      // No device is passed — three is made to walk the navigator.gpu.requestAdapter →
      // adapter.features → requestDevice({requiredFeatures}) → device.lost.then(...)
      // bootstrap **as is**, so the whole porting path is verified.
      renderer = new THREE.WebGPURenderer({
        canvas: fakeCanvas,
        context,
        antialias: false,
      })
      renderer.setPixelRatio(1)
      renderer.setSize(size.width, size.height, false)

      await renderer.init()
      if (disposed) return

      const device = renderer.backend.device
      const bootOk = !!(device && device.features && device.lost instanceof Promise)
      mark(0, bootOk ? 'ok' : 'fail', bootOk ? `${device.features.size} features requested` : undefined)
      setStatus(`${size.width}×${size.height} · r${THREE.REVISION}`)

      device.onError((_error: any, text: string) => {
        streamStats.errors += 1
        console.log(`[3js-error] ${text}`)
        setErrorLines((previous) => (previous.length < 5 ? [...previous, text] : previous))
        mark(CHECK_STREAM, 'fail', `${streamStats.errors} errors`)
      })
      attachStreamCounter(device)

      // There are paths where three reports a pipeline error only through console.warn/error — they are pulled onto the HUD.
      for (const level of ['warn', 'error'] as const) {
        const original = console[level].bind(console)
        console[level] = (...parts: any[]) => {
          original(...parts)
          const text = parts.map((part) => (part && part.message) || String(part)).join(' ')
          if (disposed) return
          setErrorLines((previous) => (
            previous.length < 5 ? [...previous, `console.${level}: ${text.slice(0, 200)}`] : previous
          ))
        }
      }

      // The direct shim probes → three's pixel checks (an offscreen render target, independent of the screen).
      await runShimProbes(device, mark)
      await runPixelChecks(renderer, mark)
      if (disposed) return

      // The spinning cube to look at, plus a frame counter.
      //
      // The cube's surface is ASTC compressed data and the panel behind is a natively decoded PNG — this is
      // where you see whether the two paths this branch opened **really reach the screen**.
      const compressed = device.features.has('texture-compression-astc')
      let backdrop: any = null
      try {
        const bitmap = await createImageBitmap(decodeBase64(PNG_BASE64))
        backdrop = new THREE.Texture(bitmap)
        backdrop.magFilter = THREE.LinearFilter
        backdrop.minFilter = THREE.LinearFilter
        backdrop.generateMipmaps = false
        backdrop.needsUpdate = true
      } catch (error) {
        console.log(`[3js-error] background image failed: ${error && (error as Error).message}`)
      }
      const spin = buildSpinScene(size.width / size.height, { compressed, backdrop })
      setStatus(`${size.width}×${size.height} · r${THREE.REVISION} · ${spin.note}`)
      let frames = 0
      let elapsedStart: number | null = null
      renderer.setAnimationLoop((time: number) => {
        spin.mesh.rotation.x = time / 1400
        spin.mesh.rotation.y = time / 900
        renderer.render(spin.scene, spin.camera)

        frames += 1
        if (elapsedStart === null) elapsedStart = time
        if (frames === 30) {
          const fps = Math.round(29000 / Math.max(time - elapsedStart, 1))
          mark(CHECK_ANIMATION, 'ok', `${fps}fps`)
          if (streamStats.errors === 0) mark(CHECK_STREAM, 'ok', '0 errors')
          refreshStats(fps)
        }
        // After that it refreshes only once every 2 seconds — so state updates do not re-render every frame.
        if (frames > 30 && frames % 120 === 0) {
          refreshStats(Math.round(((frames - 1) * 1000) / Math.max(time - (elapsedStart || 0), 1)))
        }
      })
      refreshStats(null)
    }

    boot().catch((error) => {
      console.log(`[3js-error] boot failed: ${error && error.message}`)
      setStatus(`boot failed: ${error && error.message}`)
      mark(0, 'fail', error && error.message)
    })

    return () => {
      disposed = true
      if (renderer) {
        renderer.setAnimationLoop(null)
        renderer.dispose()
      }
      // Any remaining rAF schedule is cut too, so the display link is definitely let go.
      uninstallAnimationFrame()
    }
  }, [])

  const icon = { wait: '○', ok: '✓', fail: '✗' }

  return (
    <view className="page">
      <webgpu-canvas className="canvas" canvas-id="main" />
      <ChecklistHud title="three.js WebGPURenderer" subtitle={status} checks={checks} summary={stats || undefined} errors={errorLines} />
    </view>
  )
}

root.render(<ThreeScene />)
