// @ts-nocheck — three의 타입 선언이 DOM lib을 전제해서 이 프로젝트 tsconfig와 싸운다.
// 빌드는 SWC 트랜스파일이라 영향이 없고, 이 파일의 통합 지점은 런타임 검증(HUD)으로 확인한다.
import { root, useEffect, useState } from '@lynx-js/react'
import * as THREE from 'three/webgpu'
import gpu, { startFrameLoop } from '../webgpu.js'
import '../demo.css'
import '../elements.d.ts'

// ---------------------------------------------------------------------------
// requestAnimationFrame 심 — three의 내부 Animation 루프용
// ---------------------------------------------------------------------------
// renderer.init()이 무조건 _animation.start() → self.requestAnimationFrame을 부른다.
// define(self → globalThis)으로 컨텍스트는 잡히지만 rAF 자체는 PrimJS에 없다.
// **멤버 접근**(this._context.requestAnimationFrame)이라 globalThis에 대입하면 보인다 —
// bare 식별자 해석만 깨져 있는 것과 구분되는 지점이다 (docs/JS-AUTHORING.md §10).

let rafCallbacks: Array<(time: number) => void> = []
let rafStopLoop: (() => void) | null = null

function pumpFrames() {
  if (rafStopLoop) return
  rafStopLoop = startFrameLoop(({ timestamp }) => {
    const callbacks = rafCallbacks
    rafCallbacks = []
    for (const callback of callbacks) callback(timestamp)
    // 이번 틱에서 아무도 다음 프레임을 예약하지 않았으면 디스플레이 링크를 놓아 준다.
    if (rafCallbacks.length === 0 && rafStopLoop) {
      rafStopLoop()
      rafStopLoop = null
    }
  })
}

globalThis.requestAnimationFrame = (callback) => {
  rafCallbacks.push(callback)
  pumpFrames()
  return rafCallbacks.length
}
globalThis.cancelAnimationFrame = () => {
  // 이 씬에는 루프가 하나뿐이다 — 전부 비우는 것으로 충분하다.
  rafCallbacks = []
}

// ---------------------------------------------------------------------------
// 커맨드 스트림 덤프 — 리뷰에서 확정하지 못한 "프레임 중간 flush" 를 눈으로 잡는다
// ---------------------------------------------------------------------------

const DUMP_BATCHES = 14

/** 오류가 어느 배치에서 났는지 짝지을 수 있게 마지막 flush 배치 번호를 공유한다. */
const dumpState = { current: -1 }

function attachDump(device: any, log: (line: string) => void) {
  const recorder = device._recorder
  const originalFlush = recorder.flush.bind(recorder)
  let batch = 0
  // 주의: flush의 인자(present)를 반드시 그대로 전달한다 — 삼키면 내부 제출이 프레임
  // 제출로 둔갑해, 이 덤프가 잡으려는 바로 그 버그를 덤프가 다시 만든다.
  recorder.flush = (present?: boolean) => {
    if (recorder.pending.length > 0) {
      dumpState.current = batch
      if (batch < DUMP_BATCHES) {
        const ops = recorder.pending.map((command: any) => command.op)
        const kind = present === false ? 'I' : 'P'   // I = 내부 제출, P = 프레임 제출
        const line = `#${batch}${kind} (${ops.length}) ${ops.join(' ')}`
        console.log(`[3js-dump] ${line}`)
        log(line)
      }
      batch += 1
    }
    return originalFlush(present)
  }
}

// ---------------------------------------------------------------------------
// 씬
// ---------------------------------------------------------------------------

function ThreeScene() {
  const [status, setStatus] = useState('three.js 초기화 중…')
  const [errors, setErrors] = useState<string[]>([])
  const [batches, setBatches] = useState<string[]>([])

  useEffect(() => {
    let disposed = false
    let renderer: any = null

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
      // 부트스트랩을 **그대로** 밟게 해서 리뷰가 짚은 경로 전체를 검증한다.
      renderer = new THREE.WebGPURenderer({
        canvas: fakeCanvas,
        context,
        antialias: false,
      })
      renderer.setPixelRatio(1)
      renderer.setSize(size.width, size.height, false)

      await renderer.init()
      if (disposed) return
      setStatus(`init 완료 — ${size.width}×${size.height}`)

      const device = renderer.backend.device
      device.onError((_error: any, text: string) => {
        const line = `batch#${dumpState.current} ${text}`
        console.log(`[3js-error] ${line}`)
        setErrors((previous) => (previous.length < 6 ? [...previous, line] : previous))
      })
      attachDump(device, (line) => {
        // 실패는 시작 구간에서 난다 — 첫 배치들을 고정 표시한다 (마지막 N이 아니라).
        setBatches((previous) => (previous.length < DUMP_BATCHES ? [...previous, line] : previous))
      })

      const scene = new THREE.Scene()
      scene.background = new THREE.Color(0x101020)
      const camera = new THREE.PerspectiveCamera(50, size.width / size.height, 0.1, 20)
      camera.position.z = 4

      const mesh = new THREE.Mesh(
        new THREE.BoxGeometry(1.4, 1.4, 1.4),
        new THREE.MeshStandardMaterial({ color: 0x3aa1ff, roughness: 0.35, metalness: 0.2 })
      )
      scene.add(mesh)

      const light = new THREE.DirectionalLight(0xffffff, 2.4)
      light.position.set(2, 3, 4)
      scene.add(light)
      scene.add(new THREE.AmbientLight(0xffffff, 0.4))

      renderer.setAnimationLoop((time: number) => {
        mesh.rotation.x = time / 1400
        mesh.rotation.y = time / 900
        renderer.render(scene, camera)
      })
    }

    boot().catch((error) => {
      console.log(`[3js-error] boot 실패: ${error && error.message}`)
      setStatus(`boot 실패: ${error && error.message}`)
    })

    return () => {
      disposed = true
      if (renderer) {
        renderer.setAnimationLoop(null)
        renderer.dispose()
      }
      if (rafStopLoop) {
        rafStopLoop()
        rafStopLoop = null
      }
      rafCallbacks = []
    }
  }, [])

  return (
    <view className="page">
      <webgpu-canvas className="canvas" canvas-id="main" />
      <view className="hud">
        <text className="title">three.js WebGPURenderer</text>
        <text className="subtitle">{status}</text>
        {errors.map((line, index) => (
          <text className="status" key={`error-${index}`}>{line}</text>
        ))}
        {batches.map((line, index) => (
          <text className="subtitle" key={`batch-${index}`}>{line}</text>
        ))}
      </view>
    </view>
  )
}

root.render(<ThreeScene />)
