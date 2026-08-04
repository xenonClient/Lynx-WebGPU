// @ts-nocheck — three의 타입 선언이 DOM lib을 전제해서 이 프로젝트 tsconfig와 싸운다.
// 빌드는 SWC 트랜스파일이라 영향이 없고, 이 파일의 통합 지점은 런타임 검증(체크리스트)으로 확인한다.
import { root, useEffect, useState } from '@lynx-js/react'
import * as THREE from 'three/webgpu'
import gpu, { GPUBufferUsage, GPUTextureUsage, installAnimationFrame } from '../webgpu.js'
import '../demo.css'
import '../elements.d.ts'

// three의 내부 Animation 루프가 rAF를 전제한다 — PrimJS에는 없어서 깔아 줘야 한다.
// (안 깔면 renderer.init()이 오류 없이 영구 정지한다.) import 전에 깔릴 수 있도록
// 모듈 최상단에서 부른다 — three가 모듈 초기화에서 전역을 캡처해도 안전하다.
const uninstallAnimationFrame = installAnimationFrame()

// ---------------------------------------------------------------------------
// 커맨드 스트림 계측 — 배치 종류(P=프레임 제출/I=내부 제출)와 오류를 센다
// ---------------------------------------------------------------------------

const streamStats = { frameBatches: 0, internalBatches: 0, errors: 0 }

function attachStreamCounter(device: any) {
  const recorder = device._recorder
  const originalFlush = recorder.flush.bind(recorder)
  let logged = 0
  // 주의: flush의 인자(present)를 반드시 그대로 전달한다 — 삼키면 내부 제출이 프레임
  // 제출로 둔갑해, 프레임 중간 present 버그를 계측이 다시 만든다.
  recorder.flush = (present?: boolean) => {
    if (recorder.pending.length > 0) {
      if (present === false) streamStats.internalBatches += 1
      else streamStats.frameBatches += 1
      if (logged < 14) {
        const ops = recorder.pending.map((command: any) => command.op)
        console.log(`[3js-dump] #${logged}${present === false ? 'I' : 'P'} (${ops.length}) ${ops.join(' ')}`)
        logged += 1
      }
    }
    return originalFlush(present)
  }
}

// ---------------------------------------------------------------------------
// 기능 체크 — 렌더 타깃에 그리고 픽셀 값을 읽어 기대색과 비교한다
// ---------------------------------------------------------------------------

interface Check {
  label: string
  state: 'wait' | 'ok' | 'fail'
  detail?: string
}

const CHECK_LABELS = [
  '부트스트랩 adapter→device→lost',
  'shim 프로브: 버퍼 왕복',
  'shim 프로브: 텍스처 왕복',
  '클리어 색 리드백',
  '셰이더 파이프라인 (단색 쿼드)',
  '텍스처 업로드·샘플링',
  '조명 (Standard + Directional)',
  '깊이 테스트 (앞뒤 가림)',
  '인스턴싱 (InstancedMesh)',
  '밉맵 생성 (자체 컴퓨트 패스)',
  '알파 블렌딩 (합성 공식)',
  '비동기 파이프라인 (compileAsync)',
  '애니메이션 루프',
  '커맨드 스트림 무오류',
]
const CHECK_ANIMATION = 12
const CHECK_STREAM = 13

/**
 * three를 거치지 않는 shim 직접 왕복 — three 검증이 실패할 때 어느 층인지 가른다.
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

/** 렌더 타깃의 한 픽셀 [r, g, b] (0~255). 기본은 8×8 타깃의 중앙 근처. */
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
 * 픽셀 값 검증 — 각 항목이 **서로 다른 GPU 경로**를 밟도록 고른 것들이다.
 *
 * 클리어(패스 초기화) → 단색 쿼드(노드 셰이더 → WGSL 변환 → 파이프라인) →
 * 텍스처(writeTexture + 샘플러) → 조명(라이팅 유니폼 + BRDF) → 깊이(깊이 어태치먼트 +
 * 비교 함수) → 인스턴싱(인스턴스 버퍼 + `@builtin(instance_index)`) → 밉맵(three가 도는
 * 자체 컴퓨트 패스) → 블렌딩(고정 함수 합성) → 비동기 컴파일(`createRenderPipelineAsync`).
 *
 * 리드백 자체가 `copyTextureToBuffer` + `mapAsync`라, 통과하면 그 경로도 함께 검증된다.
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

  /** 검증 하나가 던져도 나머지는 계속 돈다 — 첫 실패에서 멈추면 정보가 가장 적다. */
  async function check(
    index: number,
    run: () => Promise<{ ok: boolean, detail: string }>
  ) {
    try {
      const result = await run()
      mark(index, result.ok ? 'ok' : 'fail', result.detail)
    } catch (error) {
      mark(index, 'fail', `예외: ${(error && (error as Error).message) || error}`.slice(0, 80))
    }
  }

  // ① 클리어 색 — 빨강 배경만 있는 씬.
  await check(3, async () => {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(1, 0, 0)
    const rgb = await renderAndRead(scene)
    return { ok: rgb[0] > 180 && rgb[1] < 60 && rgb[2] < 60, detail: formatRGB(rgb) }
  })

  // ② 단색 쿼드 — 초록 MeshBasicMaterial이 화면을 덮는다.
  await check(4, async () => {
    const scene = new THREE.Scene()
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshBasicMaterial({ color: 0x00ff00 })
    ))
    const rgb = await renderAndRead(scene)
    return { ok: rgb[1] > 180 && rgb[0] < 60 && rgb[2] < 60, detail: formatRGB(rgb) }
  })

  // ③ 텍스처 — 주황 단색 2×2 DataTexture를 샘플링한다.
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

  // ④ 조명 — 흰 StandardMaterial 플레인에 정면 직사광. 라이팅이 죽었으면 검게 나온다.
  await check(6, async () => {
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

  // ⑤ 깊이 테스트 — 파랑이 앞, 빨강이 뒤. 깊이 비교가 죽으면 그리는 순서대로 빨강이 이긴다.
  await check(7, async () => {
    const scene = new THREE.Scene()
    const back = new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial({ color: 0xff0000 })
    )
    back.position.z = -0.5
    const front = new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial({ color: 0x0000ff })
    )
    front.position.z = 0.5
    // 앞의 것을 **먼저** 그리게 해서 깊이 테스트가 아니면 통과할 수 없게 만든다.
    front.renderOrder = 0
    back.renderOrder = 1
    scene.add(front, back)

    const rgb = await renderAndRead(scene)
    return { ok: rgb[2] > 180 && rgb[0] < 60, detail: formatRGB(rgb) }
  })

  // ⑥ 인스턴싱 — 인스턴스마다 다른 위치·색. 인스턴스 버퍼가 안 오면 하나만 그려진다.
  await check(8, async () => {
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

    // 왼쪽(빨강)과 오른쪽(파랑)을 각각 본다 — 인스턴스 1이 안 그려지면 오른쪽이 검다.
    const left = await renderAndRead(scene, 1, 4)
    const right = await renderAndRead(scene, 6, 4)
    return {
      ok: left[0] > 150 && left[2] < 80 && right[2] > 150 && right[0] < 80,
      detail: `L${formatRGB(left)} R${formatRGB(right)}`,
    }
  })

  // ⑦ 밉맵 — three가 자체 컴퓨트 패스로 밉을 만든다. 낮은 밉을 강제로 샘플링해
  //    두 색의 평균이 나오는지 본다 (밉 생성이 죽으면 원본 색 그대로거나 검다).
  await check(9, async () => {
    const size = 4
    const texels = new Uint8Array(size * size * 4)
    for (let index = 0; index < size * size; index++) {
      // 절반은 빨강, 절반은 초록 — 평균은 (128, 128, 0) 근처여야 한다.
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
    // **밉을 실제로 만들었는지**를 본다 — 만들었으면 아래 밉에서 두 색이 섞여 양쪽 채널이
    // 함께 살아 있다. 생성이 죽으면 원본 텍셀 하나(순빨강 또는 순초록)가 그대로 나온다.
    // 느슨하게 "검지 않으면 통과"로 두면, 밉 파이프라인이 통째로 실패해도 초록불이 켜진다
    // (실제로 그렇게 진입점 해석 버그를 놓칠 뻔했다).
    const ok = rgb[0] > 40 && rgb[1] > 40
    return { ok, detail: `${formatRGB(rgb)}${ok ? '' : ' 섞이지 않음'}` }
  })

  // ⑧ 알파 블렌딩 — 불투명 빨강 위에 50% 파랑. 합성이 죽으면 순색이 나온다.
  await check(10, async () => {
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
    // 0.5씩 섞이면 양쪽 채널이 모두 중간값이다 — 순색(255/0)이면 합성이 안 된 것이다.
    return { ok: rgb[0] > 60 && rgb[0] < 210 && rgb[2] > 60 && rgb[2] < 210, detail: formatRGB(rgb) }
  })

  // ⑨ 비동기 파이프라인 — three의 compileAsync가 createRenderPipelineAsync를 탄다.
  await check(11, async () => {
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
// 씬
// ---------------------------------------------------------------------------

/** 눈으로 볼 회전 큐브 — 체커 텍스처 + 조명이라 어느 기능이 죽어도 티가 난다. */
function buildSpinScene(aspect: number) {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0x0b0e14)

  const camera = new THREE.PerspectiveCamera(50, aspect, 0.1, 20)
  camera.position.z = 4

  const size = 8
  const texels = new Uint8Array(size * size * 4)
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const even = (x + y) % 2 === 0
      texels.set(even ? [255, 176, 32, 255] : [24, 60, 116, 255], (y * size + x) * 4)
    }
  }
  const checker = new THREE.DataTexture(texels, size, size, THREE.RGBAFormat, THREE.UnsignedByteType)
  checker.magFilter = THREE.NearestFilter
  checker.needsUpdate = true

  const mesh = new THREE.Mesh(
    new THREE.BoxGeometry(1.6, 1.6, 1.6),
    new THREE.MeshStandardMaterial({ map: checker, roughness: 0.4, metalness: 0.1 })
  )
  scene.add(mesh)

  const key = new THREE.DirectionalLight(0xffffff, 2.6)
  key.position.set(2, 3, 4)
  scene.add(key)
  scene.add(new THREE.AmbientLight(0xffffff, 0.35))

  return { scene, camera, mesh }
}

function ThreeScene() {
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
      setChecks((previous) =>
        previous.map((check, checkIndex) =>
          checkIndex === index ? { ...check, state, detail } : check
        )
      )
    }

    function refreshStats(fps: number | null) {
      const fpsText = fps === null ? '' : ` · ${fps}fps`
      setStats(
        `스트림 P ${streamStats.frameBatches} · I ${streamStats.internalBatches}`
          + ` · 오류 ${streamStats.errors}${fpsText}`
      )
    }

    async function boot() {
      const context = gpu.getCanvasContext('main')

      // 레이아웃 전에는 크기가 0이다 — 준비될 때까지 짧게 기다린다.
      let size = context.getSize()
      for (let attempt = 0; attempt < 40 && (size.width === 0 || size.height === 0); attempt++) {
        await new Promise((resolve) => setTimeout(resolve, 50))
        if (disposed) return
        size = context.getSize()
      }
      if (size.width === 0) throw new Error('캔버스 크기가 잡히지 않았다')

      // three가 기대하는 최소한의 캔버스 표면. setAttribute를 일부러 빼서
      // ('setAttribute' in domElement 분기) DOM 경로를 타지 않게 한다.
      const fakeCanvas = {
        width: size.width,
        height: size.height,
        addEventListener() {},
        removeEventListener() {},
        dispatchEvent() {},
        getContext: () => context,
      }

      // device를 넘기지 않는다 — three가 navigator.gpu.requestAdapter →
      // adapter.features → requestDevice({requiredFeatures}) → device.lost.then(...)
      // 부트스트랩을 **그대로** 밟게 해서 이식 경로 전체를 검증한다.
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
      mark(0, bootOk ? 'ok' : 'fail', bootOk ? `기능 ${device.features.size}개 요청됨` : undefined)
      setStatus(`${size.width}×${size.height} · r${THREE.REVISION}`)

      device.onError((_error: any, text: string) => {
        streamStats.errors += 1
        console.log(`[3js-error] ${text}`)
        setErrorLines((previous) => (previous.length < 5 ? [...previous, text] : previous))
        mark(CHECK_STREAM, 'fail', `${streamStats.errors}건`)
      })
      attachStreamCounter(device)

      // three가 파이프라인 오류를 console.warn/error로만 알리는 경로가 있다 — HUD로 끌어온다.
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

      // shim 직접 프로브 → three 픽셀 검증 (오프스크린 렌더 타깃이라 화면과 독립이다).
      await runShimProbes(device, mark)
      await runPixelChecks(renderer, mark)
      if (disposed) return

      // 눈으로 볼 회전 큐브 + 프레임 카운터.
      const spin = buildSpinScene(size.width / size.height)
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
          if (streamStats.errors === 0) mark(CHECK_STREAM, 'ok', '0건')
          refreshStats(fps)
        }
        // 이후에는 2초에 한 번만 갱신한다 — 상태 갱신이 프레임마다 리렌더를 만들지 않게.
        if (frames > 30 && frames % 120 === 0) {
          refreshStats(Math.round(((frames - 1) * 1000) / Math.max(time - (elapsedStart || 0), 1)))
        }
      })
      refreshStats(null)
    }

    boot().catch((error) => {
      console.log(`[3js-error] boot 실패: ${error && error.message}`)
      setStatus(`boot 실패: ${error && error.message}`)
      mark(0, 'fail', error && error.message)
    })

    return () => {
      disposed = true
      if (renderer) {
        renderer.setAnimationLoop(null)
        renderer.dispose()
      }
      // 남은 rAF 예약까지 끊어 디스플레이 링크를 확실히 놓는다.
      uninstallAnimationFrame()
    }
  }, [])

  const icon = { wait: '○', ok: '✓', fail: '✗' }

  return (
    <view className="page">
      <webgpu-canvas className="canvas" canvas-id="main" />
      <view className="three-hud">
        <text className="title">three.js WebGPURenderer</text>
        <text className="subtitle">{status}</text>
        {checks.map((check, index) => (
          <text className={`check-row check-${check.state}`} key={`check-${index}`}>
            {icon[check.state]} {check.label}{check.detail ? ` — ${check.detail}` : ''}
          </text>
        ))}
        {stats !== '' && <text className="check-stats">{stats}</text>}
        {errorLines.map((line, index) => (
          <text className="check-row check-fail" key={`err-${index}`}>{line}</text>
        ))}
      </view>
    </view>
  )
}

root.render(<ThreeScene />)
