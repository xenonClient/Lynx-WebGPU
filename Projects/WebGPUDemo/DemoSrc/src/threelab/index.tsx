// @ts-nocheck — three의 타입 선언이 DOM lib을 전제해서 이 프로젝트 tsconfig와 싸운다.
// 빌드는 SWC 트랜스파일이라 영향이 없고, 통합 지점은 런타임 체크리스트로 확인한다.
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

// three의 내부 Animation 루프가 rAF를 전제한다 — import 전에 깔려야 안전하다.
const uninstallAnimationFrame = installAnimationFrame()

/**
 * three.js 고난도 조합 — **한 화면에서 동시에 도는** 것들:
 *
 * 1. TSL 절차적 머티리얼 (노이즈 → 색·거칠기, 정점 변형)
 * 2. 그림자 맵 (`textureSampleCompare` — 깊이 전용 패스 + 비교 샘플러)
 * 3. GPU 컴퓨트 파티클 (스토리지 버퍼를 컴퓨트가 갱신 → 인스턴스가 읽는다)
 * 4. 인스턴싱 (`InstancedMesh` 수천 개)
 * 5. 포스트프로세싱 (`PostProcessing` + `bloom` — 밉 체인·다중 패스)
 *
 * 하나씩 켜면서 픽셀·오류로 확인한다. **단계마다 격리해서** 어디까지 되는지 남긴다 —
 * 한 덩어리로 켜면 첫 실패에서 화면이 검게 되고 원인이 사라진다.
 */

interface Check {
  label: string
  state: 'wait' | 'ok' | 'fail'
  detail?: string
}

const CHECK_LABELS = [
  'TSL 절차적 머티리얼 (노이즈 → 색)',
  'TSL 정점 변형 (positionLocal 이동)',
  '그림자 맵 (textureSampleCompare)',
  'GPU 컴퓨트 → 스토리지 버퍼',
  '컴퓨트 결과를 인스턴스가 읽는다',
  '인스턴싱 4096개',
  '포스트프로세싱 bloom (렌더 타깃으로)',
  '포스트프로세싱을 캔버스로 (알려진 한계)',
  '전체 합성 프레임 루프',
  '커맨드 스트림 무오류',
]
const CHECK_CANVAS_POST = 7
const CHECK_COMPOSITE = 8
const CHECK_STREAM = 9

const streamStats = { batches: 0, errors: 0 }

function formatRGB(rgb: number[]) {
  return `(${rgb[0]},${rgb[1]},${rgb[2]})`
}

/** 렌더 타깃 픽셀 하나를 읽는다. three가 `copyTextureToBuffer` + `mapAsync`로 내려 준다. */
async function readPixel(renderer: any, target: any, x: number, y: number) {
  const buffer = await renderer.readRenderTargetPixelsAsync(target, x, y, 1, 1)
  return [
    Math.round(buffer[0] * (buffer instanceof Uint8Array ? 1 : 255)),
    Math.round(buffer[1] * (buffer instanceof Uint8Array ? 1 : 255)),
    Math.round(buffer[2] * (buffer instanceof Uint8Array ? 1 : 255)),
  ]
}

// ---------------------------------------------------------------------------
// 씬 조각들 — 각각 따로 세워서 실패를 격리한다
// ---------------------------------------------------------------------------

/**
 * TSL 절차적 머티리얼. 노이즈로 색을 만들고 거칠기를 흔든다.
 *
 * 여기서 나오는 WGSL은 사람이 쓴 셰이더와 성격이 다르다 — 임시 변수가 수십 개고,
 * 함수가 깊게 중첩되며, 이름이 기계 생성이다. 트랜스파일러에 가장 가혹한 입력이다.
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
  const [status, setStatus] = useState('three.js 초기화 중…')
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
      if (size.width === 0) throw new Error('캔버스 크기가 잡히지 않았다')

      const fakeCanvas = {
        width: size.width,
        height: size.height,
        addEventListener() {}, removeEventListener() {}, dispatchEvent() {},
        getContext: () => context,
      }
      renderer = new THREE.WebGPURenderer({ canvas: fakeCanvas, context, antialias: false })
      renderer.setPixelRatio(1)
      renderer.setSize(size.width, size.height, false)
      // 그림자를 쓰려면 켜야 한다 — 깊이 전용 패스와 비교 샘플러가 여기서 붙는다.
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
        mark(CHECK_STREAM, 'fail', `${streamStats.errors}건`)
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

      // 실패한 배치와 **직전 배치**를 한 번만 찍는다 — "핸들이 없다"는 오류는 그 핸들을
      // 누가 언제 만들었는지 봐야 원인이 나온다.
      {
        const recorder = device._recorder
        const originalFlush = recorder.flush.bind(recorder)
        let previous: string[] = []
        let dumped = 0
        let batch = 0
        const created = new Map<number, number>()
        const drawables = new Set<number>()
        const fromDrawable = new Set<number>()
        const destroyed = new Map<number, number>()
        recorder.flush = (present?: boolean) => {
          const ops = recorder.pending.map((command: any) => {
            const id = command.id !== undefined ? `#${command.id}` : ''
            const view = command.colorAttachments ? `<${command.colorAttachments[0]?.view}` : ''
            const from = command.texture !== undefined ? `←${command.texture}` : ''
            return `${command.op}${id}${from}${view}`
          })
          batch += 1
          for (const command of recorder.pending) {
            // 드로어블(캔버스) 텍스처에서 나온 뷰인지 표시해 둔다 — 그 뷰는 프레임이
            // 끝나면(present) 네이티브가 회수한다. 다음 프레임에 다시 쓰면 "없는 핸들"이다.
            if (command.op === 'getCurrentTexture') drawables.add(command.id)
            if (command.op === 'createTextureView') {
              created.set(command.id, batch)
              if (drawables.has(command.texture)) fromDrawable.add(command.id)
            }
            if (command.op === 'destroy') destroyed.set(command.id, batch)
          }
          const result = originalFlush(present)
          if (result && result.ok === false && dumped < 1) {
            dumped += 1
            // 문제의 핸들이 만들어진 적은 있는지, 누가 언제 destroy 했는지.
            const failing = /GPUTextureView #(\d+)/.exec(
              (result.errors || []).map((e: any) => e.message).join(' ')
            )
            const id = failing ? Number(failing[1]) : -1
            setErrorLines((lines) => [
              ...lines,
              `#${id} 생성 배치 ${created.get(id) ?? '없음'}`
                + ` · destroy ${destroyed.get(id) ?? '없음'}`
                + ` · 지금 ${batch}`
                + ` · 드로어블 뷰? ${fromDrawable.has(id) ? '예' : '아니오'}`,
              `실패(${ops.length}${present === false ? 'I' : 'P'}) ${ops.slice(0, 8).join(' ')}…`,
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
          mark(index, 'fail', `예외: ${(error && (error as Error).message) || error}`.slice(0, 90))
        }
      }

      // ① TSL 절차적 머티리얼 — 노이즈가 색을 만든다. 검거나 흰색이면 그래프가 죽은 것이다.
      await check(0, async () => {
        const scene = new THREE.Scene()
        const material = new THREE.MeshBasicNodeMaterial()
        // 화면 전체에서 노이즈가 도는지 — uv 기반이라 카메라와 무관하다.
        const noise = mx_noise_float(vec3(uv().mul(8.0), 0.0)).mul(0.5).add(0.5)
        material.colorNode = vec3(noise, noise.oneMinus(), 0.5)
        scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material))
        const rgb = await renderAndRead(scene)
        // 노이즈 값이 무엇이든 r + g ≈ 255여야 한다 (한쪽이 다른 쪽의 여집합이다).
        const sum = rgb[0] + rgb[1]
        return {
          ok: sum > 200 && sum < 310 && rgb[2] > 100 && rgb[2] < 160,
          detail: `${formatRGB(rgb)} · r+g=${sum}`,
        }
      })

      // ② TSL 정점 변형 — positionLocal을 밀어 화면 밖으로 보낸다. 안 밀리면 색이 남는다.
      await check(1, async () => {
        const scene = new THREE.Scene()
        scene.background = new THREE.Color(0, 0, 0)
        const material = new THREE.MeshBasicNodeMaterial()
        material.colorNode = vec3(1, 0, 0)
        // x를 +5 밀면 화면(정규 좌표 -1..1) 밖이다.
        material.positionNode = positionLocal.add(vec3(5, 0, 0))
        scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material))
        const rgb = await renderAndRead(scene)
        return { ok: rgb[0] < 40, detail: `${formatRGB(rgb)} (밀렸으면 검정)` }
      })

      // ③ 그림자 맵 — 위에서 비추는 빛, 바닥판, 그 사이의 상자.
      //    그림자 안(상자 바로 아래)이 밖보다 어두워야 한다.
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

        // 가늘고 높게 둔다 — 빛이 비스듬하므로 그림자가 가림막 **옆으로** 떨어져,
        // 위에서 내려다보는 카메라가 그림자와 밝은 바닥을 같은 화면에서 본다.
        // (가림막을 넓게 두고 바로 아래를 보면 카메라에 가림막 윗면만 잡혀,
        //  "안"과 "밖"이 둘 다 밝은 흰색으로 같게 나온다 — 처음에 그렇게 틀렸다.)
        const blocker = new THREE.Mesh(
          new THREE.BoxGeometry(0.8, 0.3, 4),
          new THREE.MeshStandardNodeMaterial({ color: 0xffffff })
        )
        blocker.position.y = 2.5
        blocker.castShadow = true
        scene.add(blocker)

        // 강도를 낮춘다 — 3이면 바닥이 흰색으로 포화돼 그림자 안팎이 **둘 다 246**이 된다
        // (앞선 실행에서 실제로 그랬다: 체크가 기능이 아니라 노출을 재고 있었다).
        const light = new THREE.DirectionalLight(0xffffff, 1.1)
        light.position.set(4, 6, 0)
        light.castShadow = true
        light.shadow.mapSize.set(256, 256)
        light.shadow.camera.left = -5
        light.shadow.camera.right = 5
        light.shadow.camera.top = 5
        light.shadow.camera.bottom = -5
        scene.add(light)

        // 바닥을 위에서 내려다본다. 가운데는 그림자 안, 모서리는 밖이다.
        const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 30)
        camera.position.set(0, 8, 0.001)
        camera.lookAt(0, 0, 0)

        // 정확한 픽셀 자리를 계산하는 대신 **가로 한 줄을 훑어** 가장 어두운 곳과 밝은 곳을
        // 비교한다 — 기하 배치가 조금 달라져도 "그림자가 있다"는 사실은 그대로 잡힌다.
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
          detail: `한 줄에서 어두운 곳 ${darkest} · 밝은 곳 ${brightest}`,
        }
      })

      // ④ GPU 컴퓨트 → 스토리지 버퍼. TSL이 만든 컴퓨트 셰이더가 값을 채운다.
      const COUNT = 4096
      const positions = instancedArray(COUNT, 'vec3')
      const velocities = instancedArray(COUNT, 'vec3')
      const seedTime = uniform(0)

      const initCompute = Fn(() => {
        const index = instanceIndex.toFloat()
        // 결정적인 의사 난수 — 같은 입력이면 같은 배치가 나와야 확인이 가능하다.
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
        position.addAssign(velocity.mul(0.016))
        // 바닥에 닿으면 위로 되돌린다 — 상태가 실제로 누적되는지 눈으로 보이게 한다.
        If(position.y.lessThan(0.0), () => {
          position.y.assign(4.0)
        })
        velocity.y.addAssign(float(-0.02))
      })().compute(COUNT)

      await check(3, async () => {
        await renderer.computeAsync(initCompute)
        // 스토리지 버퍼를 되읽어 실제로 채워졌는지 본다.
        const data = await renderer.getArrayBufferAsync(positions.value)
        const view = new Float32Array(data)
        let nonZero = 0
        for (let index = 0; index < Math.min(view.length, 4096); index++) {
          if (view[index] !== 0) nonZero += 1
        }
        return {
          ok: nonZero > 1000,
          detail: `${view.length}개 중 0이 아닌 값 ${nonZero}개`,
        }
      })

      // ⑤ 컴퓨트 결과를 렌더가 읽는가 — 스텝을 여러 번 돌리면 값이 **달라져야** 한다.
      await check(4, async () => {
        const before = new Float32Array(await renderer.getArrayBufferAsync(positions.value)).slice(0, 64)
        for (let step = 0; step < 8; step++) await renderer.computeAsync(stepCompute)
        const after = new Float32Array(await renderer.getArrayBufferAsync(positions.value)).slice(0, 64)
        let moved = 0
        for (let index = 0; index < before.length; index++) {
          if (Math.abs(before[index] - after[index]) > 1e-5) moved += 1
        }
        return { ok: moved > 20, detail: `64개 성분 중 ${moved}개가 움직였다` }
      })

      // ⑥ 인스턴싱 — 컴퓨트가 만든 위치를 인스턴스가 읽어 그린다.
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
        // 입자가 작아 점 몇 개로는 대부분 빗나간다 — 한 번 그리고 **면 전체를 세어** 판단한다.
        renderer.setRenderTarget(target)
        await renderer.renderAsync(scene, camera)
        renderer.setRenderTarget(null)
        const pixels = await renderer.readRenderTargetPixelsAsync(target, 0, 0, 16, 16)
        const scale = pixels instanceof Uint8Array ? 1 : 255
        let lit = 0
        for (let index = 0; index < pixels.length; index += 4) {
          if ((pixels[index] + pixels[index + 1] + pixels[index + 2]) * scale > 40) lit += 1
        }
        return { ok: lit > 12, detail: `256픽셀 중 ${lit}개에 입자` }
      })

      // ⑦ 포스트프로세싱 — bloom. 밝은 곳이 번져 주변 픽셀까지 밝아져야 한다.
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
        // 작은 흰 사각형 **바깥**을 읽는다. bloom이 없으면 검정이다.
        const halo = await readPixel(renderer, target, 8, 11)
        post.dispose()
        post = null
        return { ok: halo[0] > 8, detail: `번짐 밖 ${formatRGB(halo)}` }
      })

      // ⑧ 포스트프로세싱을 **캔버스로** 내보내면 깨진다 — 이 씬이 찾아낸 한계다.
      //
      //   three는 렌더 타깃 디스크립터에 **텍스처 뷰를 캐시**한다
      //   (`_getRenderPassDescriptor`, 크기·샘플 수가 바뀔 때만 무효화). 브라우저에서는 한
      //   프레임의 여러 `submit()`이 present를 일으키지 않아 그 캐시가 유효하지만, 여기서는
      //   `submit()`마다 present하고 그때 드로어블 뷰가 만료된다 — 다음 프레임에 캐시된
      //   뷰를 다시 쓰면 "없는 핸들"이 된다.
      //
      //   렌더 타깃으로 내보내는 경로(⑦)는 멀쩡하다. 캔버스로 직접 내보낼 때만 걸린다.
      await check(CHECK_CANVAS_POST, async () => {
        const probe = new THREE.Scene()
        probe.background = new THREE.Color(0, 0, 0)
        probe.add(new THREE.Mesh(
          new THREE.PlaneGeometry(0.5, 0.5),
          new THREE.MeshBasicNodeMaterial({ color: 0xffffff })
        ))
        const probePass = pass(probe, ortho)
        const probePost = new THREE.PostProcessing(renderer)
        probePost.outputNode = probePass.add(bloom(probePass, 1.0, 0.4, 0.0))

        const before = streamStats.errors
        await probePost.renderAsync()   // 렌더 타깃을 지정하지 않으면 캔버스로 간다
        await probePost.renderAsync()   // 두 번째 프레임에서 캐시된 뷰가 되살아난다
        probePost.dispose()
        const leaked = streamStats.errors - before
        return {
          ok: leaked === 0,
          detail: leaked === 0 ? '오류 0' : `오류 ${leaked}건 — 드로어블 뷰 캐시 (아래 참고)`,
        }
      })

      if (disposed) return

      // --- 전체 합성: 다섯 가지가 한 프레임에 같이 돈다 --------------------
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

      // 합성 화면은 포스트프로세싱을 **끄고** 돈다 — 캔버스 출력 경로가 위 ⑧의 한계에
      // 걸리기 때문이다. 나머지(절차적 머티리얼·그림자·컴퓨트·인스턴싱)는 그대로 돈다.
      const usePost = false
      const composite = new THREE.PostProcessing(renderer)
      const scenePass = pass(scene, camera)
      composite.outputNode = scenePass.add(bloom(scenePass, 0.6, 0.35, 0.65))

      let frames = 0
      let startTime: number | null = null
      renderer.setAnimationLoop((elapsed: number) => {
        if (disposed) return
        seedTime.value = elapsed / 1000
        renderer.compute(stepCompute)
        hero.rotation.x = elapsed / 2600
        hero.rotation.y = elapsed / 1700
        camera.position.x = Math.sin(elapsed / 4200) * 9
        camera.position.z = Math.cos(elapsed / 4200) * 9
        camera.lookAt(0, 1.8, 0)
        if (usePost) composite.render()
        else renderer.render(scene, camera)

        frames += 1
        if (startTime === null) startTime = elapsed
        if (frames === 40) {
          const fps = Math.round(39000 / Math.max(elapsed - startTime, 1))
          mark(CHECK_COMPOSITE, 'ok', `${fps}fps · 입자 ${COUNT}개`)
          if (streamStats.errors === 0) mark(CHECK_STREAM, 'ok', '0건')
          setStats(`합성 프레임 ${frames} · 오류 ${streamStats.errors}`)
        }
        if (frames > 40 && frames % 120 === 0) {
          const fps = Math.round(((frames - 1) * 1000) / Math.max(elapsed - (startTime || 0), 1))
          setStats(`합성 프레임 ${frames} · ${fps}fps · 오류 ${streamStats.errors}`)
        }
      })
    }

    boot().catch((error) => {
      console.log(`[lab-error] boot 실패: ${error && error.message}`)
      setStatus(`boot 실패: ${(error && error.message) || error}`)
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

  const icon = { wait: '○', ok: '✓', fail: '✗' }
  const passed = checks.filter((check) => check.state === 'ok').length

  return (
    <view className="page">
      <webgpu-canvas className="canvas" canvas-id="main" />
      <view className="three-hud">
        <text className="title">three.js 고난도 조합</text>
        <text className="subtitle">{status}</text>
        {checks.map((check, index) => (
          <text className={`check-row check-${check.state}`} key={`check-${index}`}>
            {icon[check.state]} {check.label}{check.detail ? ` — ${check.detail}` : ''}
          </text>
        ))}
        <text className="check-stats">{`통과 ${passed}/${CHECK_LABELS.length}${stats ? ` · ${stats}` : ''}`}</text>
        {errorLines.map((line, index) => (
          <text className="check-row check-fail" key={`err-${index}`}>{line}</text>
        ))}
      </view>
    </view>
  )
}

root.render(<ThreeLabScene />)
