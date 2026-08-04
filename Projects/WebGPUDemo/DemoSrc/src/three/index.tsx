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
  '애니메이션 루프',
  '커맨드 스트림 무오류',
]
const CHECK_ANIMATION = 7
const CHECK_STREAM = 8

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

/** 렌더 타깃 중앙 픽셀 [r, g, b] (0~255). */
async function readCenter(renderer: any, target: any): Promise<[number, number, number]> {
  const data = await renderer.readRenderTargetPixelsAsync(target, 4, 4, 1, 1)
  return [data[0], data[1], data[2]]
}

function formatRGB([r, g, b]: [number, number, number]) {
  return `(${r},${g},${b})`
}

/**
 * 픽셀 값 검증 4종. 각각이 서로 다른 경로를 밟는다:
 * 클리어(패스 초기화) → 단색 쿼드(노드 셰이더 → WGSL 변환 → 파이프라인) →
 * 텍스처 쿼드(writeTexture 업로드 + 샘플러) → 조명 플레인(라이팅 유니폼 + BRDF).
 * 리드백 자체가 copyTextureToBuffer + mapAsync라, 통과하면 그 경로까지 함께 검증된다.
 */
async function runPixelChecks(
  renderer: any,
  mark: (index: number, state: 'ok' | 'fail', detail?: string) => void
) {
  const target = new THREE.RenderTarget(8, 8)
  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 10)
  camera.position.z = 1

  async function renderAndRead(scene: any): Promise<[number, number, number]> {
    renderer.setRenderTarget(target)
    await renderer.renderAsync(scene, camera)
    renderer.setRenderTarget(null)
    return readCenter(renderer, target)
  }

  // ① 클리어 색 — 빨강 배경만 있는 씬.
  {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(1, 0, 0)
    const rgb = await renderAndRead(scene)
    const ok = rgb[0] > 180 && rgb[1] < 60 && rgb[2] < 60
    mark(3, ok ? 'ok' : 'fail', formatRGB(rgb))
  }

  // ② 단색 쿼드 — 초록 MeshBasicMaterial이 화면을 덮는다.
  {
    const scene = new THREE.Scene()
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshBasicMaterial({ color: 0x00ff00 })
    ))
    const rgb = await renderAndRead(scene)
    const ok = rgb[1] > 180 && rgb[0] < 60 && rgb[2] < 60
    mark(4, ok ? 'ok' : 'fail', formatRGB(rgb))
  }

  // ③ 텍스처 — 주황 단색 2×2 DataTexture를 샘플링한다.
  {
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
    const ok = rgb[0] > 180 && rgb[1] > 60 && rgb[1] < 220 && rgb[2] < 60
    mark(5, ok ? 'ok' : 'fail', formatRGB(rgb))
  }

  // ④ 조명 — 흰 StandardMaterial 플레인에 정면 직사광. 라이팅이 죽었으면 검게 나온다.
  {
    const scene = new THREE.Scene()
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 1, metalness: 0 })
    ))
    const light = new THREE.DirectionalLight(0xffffff, 3)
    light.position.set(0, 0, 1)
    scene.add(light)
    const rgb = await renderAndRead(scene)
    const ok = rgb[0] > 80 && rgb[1] > 80 && rgb[2] > 80
    mark(6, ok ? 'ok' : 'fail', formatRGB(rgb))
  }

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
  const [lastError, setLastError] = useState('')

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
        setLastError(text)
        mark(CHECK_STREAM, 'fail', `${streamStats.errors}건`)
      })
      attachStreamCounter(device)

      // three가 파이프라인 오류를 console.warn/error로만 알리는 경로가 있다 — HUD로 끌어온다.
      for (const level of ['warn', 'error'] as const) {
        const original = console[level].bind(console)
        console[level] = (...parts: any[]) => {
          original(...parts)
          const text = parts.map((part) => (part && part.message) || String(part)).join(' ')
          if (!disposed) setLastError(`console.${level}: ${text.slice(0, 160)}`)
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
        {lastError !== '' && <text className="check-row check-fail">{lastError}</text>}
      </view>
    </view>
  )
}

root.render(<ThreeScene />)
