// @ts-nocheck — three's type declarations assume the DOM lib and fight this project's tsconfig.
// The build is an SWC transpile so it has no effect, and the integration points are confirmed by the runtime checklist.
import { root, useEffect, useState } from '@lynx-js/react'
import * as THREE from 'three/webgpu'
import {
  Fn, If, float, vec3, uniform, instanceIndex, instancedArray,
  positionLocal, positionWorld, uv, mx_noise_float, mix, sin, pass, smoothstep, fract,
} from 'three/tsl'
import { bloom } from 'three/addons/tsl/display/BloomNode.js'
import gpu, { installAnimationFrame } from '../webgpu.js'
import '../demo.css'
import '../elements.d.ts'
import { ChecklistHud, type Check } from '../checklist-hud.jsx'

// three's internal Animation loop assumes rAF — it is safest installed before the import.
const uninstallAnimationFrame = installAnimationFrame()

/**
 * A hard three.js combination — the things **running at once on one screen**:
 *
 * 1. A TSL procedural material (noise → color and roughness, vertex deformation)
 * 2. Shadow maps (`textureSampleCompare` — a depth-only pass plus a comparison sampler)
 * 3. GPU compute particles (a compute pass updates a storage buffer → the instances read it)
 * 4. Instancing (thousands of `InstancedMesh`)
 * 5. Post-processing (`PostProcessing` + `bloom` — a mip chain and multiple passes)
 *
 * They are turned on one at a time and confirmed by pixels and errors. **Isolated per step**, so how far it
 * gets is recorded — turned on in one lump, the first failure blacks the screen and the cause disappears.
 */

const CHECK_LABELS = [
  'TSL procedural material (noise → color)',
  'TSL vertex deformation (moving positionLocal)',
  'shadow map (textureSampleCompare)',
  'GPU compute → storage buffer',
  'the instances read the compute result',
  '4096 instances',
  'post-processing bloom (to a render target)',
  'post-processing to the canvas (in the composite loop)',
  'the full composite frame loop',
  'no command stream errors',
]
const CHECK_CANVAS_POST = 7
const CHECK_COMPOSITE = 8
const CHECK_STREAM = 9

const streamStats = { batches: 0, errors: 0 }

function formatRGB(rgb: number[]) {
  return `(${rgb[0]},${rgb[1]},${rgb[2]})`
}

/** Reads one render target pixel. three brings it down with `copyTextureToBuffer` + `mapAsync`. */
async function readPixel(renderer: any, target: any, x: number, y: number) {
  const buffer = await renderer.readRenderTargetPixelsAsync(target, x, y, 1, 1)
  return [
    Math.round(buffer[0] * (buffer instanceof Uint8Array ? 1 : 255)),
    Math.round(buffer[1] * (buffer instanceof Uint8Array ? 1 : 255)),
    Math.round(buffer[2] * (buffer instanceof Uint8Array ? 1 : 255)),
  ]
}

// ---------------------------------------------------------------------------
// Scene pieces — each built separately so failures stay isolated
// ---------------------------------------------------------------------------

/**
 * A TSL procedural material. Noise makes the color and shakes the roughness.
 *
 * The WGSL that comes out of this is unlike a hand-written shader — dozens of temporaries, deeply nested
 * functions, machine-generated names. It is the harshest input for the transpiler.
 */
function proceduralMaterial(scale: number) {
  const material = new THREE.MeshStandardNodeMaterial()
  const noise = mx_noise_float(positionWorld.mul(scale)).mul(0.5).add(0.5)
  const warm = vec3(1.0, 0.45, 0.15)
  const cool = vec3(0.1, 0.4, 0.95)
  material.colorNode = mix(cool, warm, smoothstep(0.35, 0.65, noise))
  material.roughnessNode = noise.mul(0.6).add(0.25)
  material.metalnessNode = float(0.1)
  return material
}

// ---------------------------------------------------------------------------

function ThreeLabScene() {
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
      setChecks((previous) => previous.map((check, i) => (
        i === index ? { ...check, state, detail } : check
      )))
    }

    async function boot() {
      const context = gpu.getCanvasContext('main')
      let size = context.getSize()
      for (let attempt = 0; attempt < 40 && size.width === 0; attempt++) {
        await new Promise((resolve) => setTimeout(resolve, 50))
        if (disposed) return
        size = context.getSize()
      }
      if (size.width === 0) throw new Error('the canvas size never settled')

      const fakeCanvas = {
        width: size.width,
        height: size.height,
        addEventListener() {}, removeEventListener() {}, dispatchEvent() {},
        getContext: () => context,
      }
      renderer = new THREE.WebGPURenderer({ canvas: fakeCanvas, context, antialias: false })
      renderer.setPixelRatio(1)
      renderer.setSize(size.width, size.height, false)
      // It has to be on to use shadows — the depth-only pass and comparison sampler attach here.
      renderer.shadowMap.enabled = true
      renderer.shadowMap.type = THREE.PCFShadowMap

      await renderer.init()
      if (disposed) return

      const device = renderer.backend.device
      setStatus(`${size.width}×${size.height} · r${THREE.REVISION}`)

      device.onError((_error: any, text: string) => {
        streamStats.errors += 1
        console.log(`[lab-error] ${text}`)
        setErrorLines((previous) => (previous.length < 6 ? [...previous, text.slice(0, 160)] : previous))
        mark(CHECK_STREAM, 'fail', `${streamStats.errors} errors`)
      })
      for (const level of ['warn', 'error'] as const) {
        const original = console[level].bind(console)
        console[level] = (...parts: any[]) => {
          original(...parts)
          const text = parts.map((part) => (part && part.message) || String(part)).join(' ')
          if (disposed) return
          setErrorLines((previous) => (
            previous.length < 6 ? [...previous, `console.${level}: ${text.slice(0, 160)}`] : previous
          ))
        }
      }

      // The failed batch and **the one just before it** are dumped once — a "no such handle" error only
      // reveals its cause once you see who created that handle and when.
      {
        const recorder = device._recorder
        const originalFlush = recorder.flush.bind(recorder)
        let previous: string[] = []
        let dumped = 0
        let batch = 0
        const created = new Map<number, number>()
        const drawables = new Set<number>()
        const viewSource = new Map<number, number>()
        const acquisitions: string[] = []
        const fromDrawable = new Set<number>()
        const destroyed = new Map<number, number>()
        // **Every argument is passed through.** Swallowing the second argument (`presentOnly`) keeps the
        // tick's wrap-up batch from reaching native, so neither the present nor the expiry happens — the
        // classic place where instrumentation breaks what it observes (a day really was spent here).
        recorder.flush = (present?: boolean, ...rest: any[]) => {
          const ops = recorder.pending.map((command: any) => {
            const id = command.id !== undefined ? `#${command.id}` : ''
            const view = command.colorAttachments ? `<${command.colorAttachments[0]?.view}` : ''
            const from = command.texture !== undefined ? `←${command.texture}` : ''
            return `${command.op}${id}${from}${view}`
          })
          batch += 1
          for (const command of recorder.pending) {
            // Mark whether the view came from a drawable (canvas) texture — that view is reclaimed by
            // native when the frame ends (present). Reusing it next frame is a "no such handle".
            if (command.op === 'getCurrentTexture') drawables.add(command.id)
            if (command.op === 'createTextureView') {
              created.set(command.id, batch)
              viewSource.set(command.id, command.texture)
              if (drawables.has(command.texture)) fromDrawable.add(command.id)
            }
            if (command.op === 'getCurrentTexture') acquisitions.push(`${command.id}@${batch}`)
            if (command.op === 'destroy') destroyed.set(command.id, batch)
          }
          const result = originalFlush(present, ...rest)
          if (result && result.ok === false && dumped < 1) {
            dumped += 1
            // Whether the handle in question was ever created, and who destroyed it and when.
            const failing = /GPUTextureView #(\d+)/.exec(
              (result.errors || []).map((e: any) => e.message).join(' ')
            )
            const id = failing ? Number(failing[1]) : -1
            setErrorLines((lines) => [
              ...lines,
              `#${id} created in batch ${created.get(id) ?? 'none'} · now ${batch}`
                + ` · source texture #${viewSource.get(id)}`
                + ` · from a drawable? ${fromDrawable.has(id) ? 'yes' : 'no'}`,
              `acquisition history (last 6): ${acquisitions.slice(-6).join(' ')}`,
              `failed(${ops.length}${present === false ? 'I' : 'P'}) ${ops.slice(0, 8).join(' ')}…`,
            ])
          }
          if (ops.length) previous = ops
          return result
        }
      }

      const target = new THREE.RenderTarget(16, 16, { depthBuffer: true })
      const ortho = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 10)
      ortho.position.z = 2

      async function renderAndRead(scene: any, camera: any = ortho, x = 8, y = 8) {
        renderer.setRenderTarget(target)
        await renderer.renderAsync(scene, camera)
        renderer.setRenderTarget(null)
        return readPixel(renderer, target, x, y)
      }

      async function check(index: number, run: () => Promise<{ ok: boolean, detail: string }>) {
        try {
          const result = await run()
          mark(index, result.ok ? 'ok' : 'fail', result.detail)
        } catch (error) {
          mark(index, 'fail', `exception: ${(error && (error as Error).message) || error}`.slice(0, 90))
        }
      }

      // ① The TSL procedural material — noise makes the color. Black or white means the graph is dead.
      await check(0, async () => {
        const scene = new THREE.Scene()
        const material = new THREE.MeshBasicNodeMaterial()
        // Whether the noise runs across the whole screen — it is uv-based, so it is independent of the camera.
        const noise = mx_noise_float(vec3(uv().mul(8.0), 0.0)).mul(0.5).add(0.5)
        material.colorNode = vec3(noise, noise.oneMinus(), 0.5)
        scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material))
        const rgb = await renderAndRead(scene)
        // Whatever the noise value, r + g ≈ 255 (one is the complement of the other).
        const sum = rgb[0] + rgb[1]
        return {
          ok: sum > 200 && sum < 310 && rgb[2] > 100 && rgb[2] < 160,
          detail: `${formatRGB(rgb)} · r+g=${sum}`,
        }
      })

      // ② TSL vertex deformation — positionLocal is pushed off screen. Unmoved, the color stays.
      await check(1, async () => {
        const scene = new THREE.Scene()
        scene.background = new THREE.Color(0, 0, 0)
        const material = new THREE.MeshBasicNodeMaterial()
        material.colorNode = vec3(1, 0, 0)
        // Pushing x by +5 puts it outside the screen (normalized coordinates -1..1).
        material.positionNode = positionLocal.add(vec3(5, 0, 0))
        scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material))
        const rgb = await renderAndRead(scene)
        return { ok: rgb[0] < 40, detail: `${formatRGB(rgb)} (black if it moved)` }
      })

      // ③ Shadow map — a light from above, a floor plane, and a box between them.
      //    Inside the shadow (right under the box) must be darker than outside.
      await check(2, async () => {
        const scene = new THREE.Scene()
        scene.background = new THREE.Color(0, 0, 0)

        const floor = new THREE.Mesh(
          new THREE.PlaneGeometry(10, 10),
          new THREE.MeshStandardNodeMaterial({ color: 0xffffff, roughness: 1 })
        )
        floor.rotation.x = -Math.PI / 2
        floor.receiveShadow = true
        scene.add(floor)

        // Placed thin and tall — the light is oblique, so the shadow falls **to the side** of the occluder
        // and the top-down camera sees the shadow and the lit floor on the same screen.
        // (With a wide occluder looked at from directly beneath, the camera only catches the occluder's top
        //  face and "inside" and "outside" both come out the same bright white — that was the first mistake.)
        const blocker = new THREE.Mesh(
          new THREE.BoxGeometry(0.8, 0.3, 4),
          new THREE.MeshStandardNodeMaterial({ color: 0xffffff })
        )
        blocker.position.y = 2.5
        blocker.castShadow = true
        scene.add(blocker)

        // The intensity is lowered — at 3 the floor saturates to white and inside and outside the shadow
        // **both come out 246** (that really happened on an earlier run: the check was measuring exposure rather than the feature).
        const light = new THREE.DirectionalLight(0xffffff, 1.1)
        light.position.set(4, 6, 0)
        light.castShadow = true
        light.shadow.mapSize.set(256, 256)
        light.shadow.camera.left = -5
        light.shadow.camera.right = 5
        light.shadow.camera.top = 5
        light.shadow.camera.bottom = -5
        scene.add(light)

        // The floor is viewed from above. The middle is inside the shadow, the corners outside.
        const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 30)
        camera.position.set(0, 8, 0.001)
        camera.lookAt(0, 0, 0)

        // Instead of computing exact pixel positions, **a horizontal line is scanned** and its darkest and
        // brightest points compared — the fact that "there is a shadow" survives small changes in the geometry.
        renderer.setRenderTarget(target)
        await renderer.renderAsync(scene, camera)
        renderer.setRenderTarget(null)
        const row = await renderer.readRenderTargetPixelsAsync(target, 0, 8, 16, 1)
        const scale = row instanceof Uint8Array ? 1 : 255
        let darkest = 255
        let brightest = 0
        for (let index = 0; index < row.length; index += 4) {
          const value = Math.round(row[index] * scale)
          darkest = Math.min(darkest, value)
          brightest = Math.max(brightest, value)
        }
        return {
          ok: brightest - darkest > 20 && brightest < 250,
          detail: `darkest on the line ${darkest} · brightest ${brightest}`,
        }
      })

      // ④ GPU compute → a storage buffer. The compute shader TSL built fills the values in.
      const COUNT = 4096
      const positions = instancedArray(COUNT, 'vec3')
      const velocities = instancedArray(COUNT, 'vec3')
      const seedTime = uniform(0)

      const initCompute = Fn(() => {
        const index = instanceIndex.toFloat()
        // A deterministic pseudo-random — the same input has to give the same arrangement for this to be checkable.
        const a = fract(sin(index.mul(12.9898)).mul(43758.5453))
        const b = fract(sin(index.mul(78.233)).mul(24634.6345))
        const c = fract(sin(index.mul(39.425)).mul(19873.1234))
        positions.element(instanceIndex).assign(
          vec3(a.sub(0.5).mul(8), b.mul(4).add(0.2), c.sub(0.5).mul(8))
        )
        velocities.element(instanceIndex).assign(
          vec3(a.sub(0.5).mul(0.4), b.mul(-0.5).sub(0.2), c.sub(0.5).mul(0.4))
        )
      })().compute(COUNT)

      const stepCompute = Fn(() => {
        const position = positions.element(instanceIndex)
        const velocity = velocities.element(instanceIndex)
        velocity.y.addAssign(float(-0.02))          // gravity
        position.addAssign(velocity.mul(0.016))
        // Bounced back up on hitting the floor — **the velocity has to be reversed too.**
        //
        // Reversing only the position lets `velocity.y` pile up by −0.02 every frame without bound, and a few
        // seconds later the particles streak past the screen (that is what it really looked like). All the
        // more so in a scene meant to show state accumulating, values that must not accumulate are cut off clearly.
        If(position.y.lessThan(0.0), () => {
          position.y.assign(4.0)
          velocity.y.assign(float(-0.2))
        })
      })().compute(COUNT)

      await check(3, async () => {
        await renderer.computeAsync(initCompute)
        // The storage buffer is read back to see whether it really got filled.
        const data = await renderer.getArrayBufferAsync(positions.value)
        const view = new Float32Array(data)
        let nonZero = 0
        for (let index = 0; index < Math.min(view.length, 4096); index++) {
          if (view[index] !== 0) nonZero += 1
        }
        return {
          ok: nonZero > 1000,
          detail: `${nonZero} non-zero out of ${view.length}`,
        }
      })

      // ⑤ Whether the render reads the compute result — running several steps must **change** the values.
      await check(4, async () => {
        const before = new Float32Array(await renderer.getArrayBufferAsync(positions.value)).slice(0, 64)
        for (let step = 0; step < 8; step++) await renderer.computeAsync(stepCompute)
        const after = new Float32Array(await renderer.getArrayBufferAsync(positions.value)).slice(0, 64)
        let moved = 0
        for (let index = 0; index < before.length; index++) {
          if (Math.abs(before[index] - after[index]) > 1e-5) moved += 1
        }
        return { ok: moved > 20, detail: `${moved} of 64 components moved` }
      })

      // ⑥ Instancing — the instances read and draw the positions the compute pass made.
      const particleMaterial = new THREE.MeshBasicNodeMaterial()
      particleMaterial.positionNode = positionLocal.add(positions.element(instanceIndex))
      particleMaterial.colorNode = vec3(1.0, 0.8, 0.3)
      const particles = new THREE.InstancedMesh(
        new THREE.SphereGeometry(0.06, 6, 4), particleMaterial, COUNT
      )
      particles.frustumCulled = false

      await check(5, async () => {
        const scene = new THREE.Scene()
        scene.background = new THREE.Color(0, 0, 0)
        scene.add(particles)
        const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 60)
        camera.position.set(0, 2, 12)
        camera.lookAt(0, 1.5, 0)
        // The particles are small and a few sample points would mostly miss — it is drawn once and **the whole face counted**.
        renderer.setRenderTarget(target)
        await renderer.renderAsync(scene, camera)
        renderer.setRenderTarget(null)
        const pixels = await renderer.readRenderTargetPixelsAsync(target, 0, 0, 16, 16)
        const scale = pixels instanceof Uint8Array ? 1 : 255
        let lit = 0
        for (let index = 0; index < pixels.length; index += 4) {
          if ((pixels[index] + pixels[index + 1] + pixels[index + 2]) * scale > 40) lit += 1
        }
        return { ok: lit > 12, detail: `particles on ${lit} of 256 pixels` }
      })

      // ⑦ Post-processing — bloom. A bright spot must bleed and brighten the surrounding pixels.
      let post: any = null
      await check(6, async () => {
        const scene = new THREE.Scene()
        scene.background = new THREE.Color(0, 0, 0)
        const glow = new THREE.Mesh(
          new THREE.PlaneGeometry(0.25, 0.25),
          new THREE.MeshBasicNodeMaterial({ color: 0xffffff })
        )
        scene.add(glow)

        const scenePass = pass(scene, ortho)
        post = new THREE.PostProcessing(renderer)
        post.outputNode = scenePass.add(bloom(scenePass, 1.2, 0.4, 0.0))

        renderer.setRenderTarget(target)
        await post.renderAsync()
        renderer.setRenderTarget(null)
        // Read **outside** the small white square. Without bloom it is black.
        const halo = await readPixel(renderer, target, 8, 11)
        post.dispose()
        post = null
        return { ok: halo[0] > 8, detail: `outside the bleed ${formatRGB(halo)}` }
      })

      if (disposed) return

      // --- The full composite: all five run together in one frame -----------
      const scene = new THREE.Scene()
      scene.background = new THREE.Color(0x05070d)

      const floor = new THREE.Mesh(
        new THREE.PlaneGeometry(24, 24),
        new THREE.MeshStandardNodeMaterial({ color: 0x223047, roughness: 0.9 })
      )
      floor.rotation.x = -Math.PI / 2
      floor.position.y = -0.2
      floor.receiveShadow = true
      scene.add(floor)

      const hero = new THREE.Mesh(new THREE.TorusKnotGeometry(1.1, 0.36, 128, 24), proceduralMaterial(0.6))
      hero.position.y = 2.2
      hero.castShadow = true
      scene.add(hero)

      scene.add(particles)

      const key = new THREE.DirectionalLight(0xffffff, 3.2)
      key.position.set(4, 9, 3)
      key.castShadow = true
      key.shadow.mapSize.set(512, 512)
      key.shadow.camera.left = -8
      key.shadow.camera.right = 8
      key.shadow.camera.top = 8
      key.shadow.camera.bottom = -8
      scene.add(key)
      scene.add(new THREE.AmbientLight(0x3355aa, 0.6))

      const camera = new THREE.PerspectiveCamera(55, size.width / size.height, 0.1, 60)
      camera.position.set(0, 3.4, 9)
      camera.lookAt(0, 2, 0)

      // It runs with post-processing on too — this path opened once the frame boundary moved to the tick's
      // end and `getCurrentTexture` was fixed to the spec (the same texture until expiry).
      const usePost = true
      const composite = new THREE.PostProcessing(renderer)
      const scenePass = pass(scene, camera)
      composite.outputNode = scenePass.add(bloom(scenePass, 0.6, 0.35, 0.65))

      let frames = 0
      let startTime: number | null = null
      // ⑧ runs **after the composite loop is up**. This probe deliberately steps on the place where three
      // reuses a canvas view across frames, and in doing so it pollutes three's render target descriptor
      // cache and **blacks out the composite screen that follows** — a diagnosis must not break what it observes.
      setStats('waiting for the loop to start…')
      renderer.setAnimationLoop((elapsed: number) => {
        if (disposed) return
        // A JS exception inside a frame **leaves no trace on screen** — it is not a WebGPU error but just a
        // throw, so the error collector never catches it and the loop simply stops.
        // It is caught once and lifted to the HUD.
        try {
          seedTime.value = elapsed / 1000
          renderer.compute(stepCompute)
          hero.rotation.x = elapsed / 2600
          hero.rotation.y = elapsed / 1700
          camera.position.x = Math.sin(elapsed / 4200) * 9
          camera.position.z = Math.cos(elapsed / 4200) * 9
          camera.lookAt(0, 1.8, 0)
          if (usePost) composite.render()
          else renderer.render(scene, camera)
        } catch (error) {
          mark(CHECK_COMPOSITE, 'fail', `exception: ${(error && (error as Error).message) || error}`.slice(0, 90))
          setErrorLines((lines) => (lines.length < 8
            ? [...lines, `frame exception: ${String((error && (error as Error).stack) || error).slice(0, 240)}`]
            : lines))
          renderer.setAnimationLoop(null)
          return
        }

        frames += 1
        if (startTime === null) startTime = elapsed
        if (frames === 40) {
          const fps = Math.round(39000 / Math.max(elapsed - startTime, 1))
          mark(CHECK_COMPOSITE, 'ok', `${fps}fps · ${COUNT} particles`)
          // ⑧ Post-processing **straight to the canvas** — the composite loop is that verification.
          //
          //   Several passes run in one frame (scene → bloom mip chain → output). When the frame boundary
          //   was `submit()`, the first submission sent the drawable out and expired the view, and the
          //   remaining passes were rejected wholesale. Now it presents once at **the tick's end**, per the spec.
          mark(CHECK_CANVAS_POST, streamStats.errors === 0 ? 'ok' : 'fail',
            streamStats.errors === 0 ? `${frames} frames · 0 errors` : `${streamStats.errors} errors`)
          if (streamStats.errors === 0) mark(CHECK_STREAM, 'ok', '0 errors')
          setStats(`composite frames ${frames} · errors ${streamStats.errors}`)
        }
        if (frames > 40 && frames % 120 === 0) {
          const fps = Math.round(((frames - 1) * 1000) / Math.max(elapsed - (startTime || 0), 1))
          setStats(`composite frames ${frames} · ${fps}fps · errors ${streamStats.errors}`)
        }
      })

    }

    boot().catch((error) => {
      console.log(`[lab-error] boot failed: ${error && error.message}`)
      setStatus(`boot failed: ${(error && error.message) || error}`)
      setErrorLines((previous) => [...previous, String((error && error.stack) || error).slice(0, 300)])
    })

    return () => {
      disposed = true
      if (renderer) {
        renderer.setAnimationLoop(null)
        renderer.dispose()
      }
      uninstallAnimationFrame()
    }
  }, [])

  return (
    <view className="page">
      <webgpu-canvas className="canvas" canvas-id="main" />
      <ChecklistHud title="A hard three.js combination" subtitle={status} checks={checks} summary={stats || undefined} errors={errorLines} />
    </view>
  )
}

root.render(<ThreeLabScene />)
