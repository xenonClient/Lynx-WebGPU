import { root } from '@lynx-js/react'
import { DemoScene, type SceneContext } from '../scene.jsx'
import { GPUBufferUsage, GPUMapMode } from '../webgpu.js'

/**
 * ArrayBuffer 브리징 스모크 — base64를 거치지 않고 바이너리를 네이티브로 넘길 수 있는지 본다.
 *
 * 검증 대상은 **Lynx의 값 변환기**다. 커맨드 스트림은
 *   `execute({ commands: [ { op: 'writeBuffer', …, data: <ArrayBuffer> } ] })`
 * 처럼 바이너리가 **최상위 인자가 아니라 배열 안 객체의 필드**에 들어간다. Lynx가 이 중첩
 * 위치까지 재귀적으로 훑어 ArrayBuffer를 `NSData`로 바꿔 주어야 `WGPUValueReader.requiredData`의
 * `raw as? Data` 분기가 성립한다.
 *
 * 판정: 알려진 바이트열을 써 넣고 되읽어 **바이트가 그대로인지** 비교한다.
 * 오류만 안 나는 것으로는 부족하다 — 변환이 어긋나면 조용히 다른 바이트가 들어갈 수 있다.
 *
 * 되읽기(`mapAsync`)는 기존 base64 경로를 그대로 쓴다. 이 씬이 재는 것은 JS → 네이티브 방향뿐이다.
 */

const SMALL = 256
const LARGE = 64 * 1024 // 동적 텍스처 한 장(128×128 RGBA)과 같은 크기

/** 자리마다 값이 다른 패턴 — 오프셋이 밀리면 바로 티가 난다. */
function makePattern(length: number): Uint8Array {
  const bytes = new Uint8Array(length)
  for (let index = 0; index < length; index += 1) bytes[index] = (index * 31 + 7) & 0xff
  return bytes
}

/**
 * TypedArray → ArrayBuffer.
 *
 * **`view.buffer`를 그냥 쓰면 안 된다** — 그건 뷰가 아니라 백킹 버퍼 전체다.
 * 그리고 Lynx는 TypedArray를 ArrayBuffer로 보지 않으므로(평범한 객체로 취급해
 * `{"0":1,…}` 로 만들어 버린다) 반드시 풀어서 넘겨야 한다.
 */
function toArrayBuffer(view: Uint8Array): ArrayBuffer {
  // `buffer`의 정적 타입은 ArrayBufferLike(SharedArrayBuffer 포함)라 좁혀 준다.
  // 여기서 만드는 뷰는 항상 평범한 ArrayBuffer 위에 있다.
  const backing = view.buffer as ArrayBuffer
  return view.byteOffset === 0 && view.byteLength === backing.byteLength
    ? backing
    : backing.slice(view.byteOffset, view.byteOffset + view.byteLength)
}

function bytesEqual(a: Uint8Array, b: Uint8Array): number {
  if (a.length !== b.length) return -1
  for (let index = 0; index < a.length; index += 1) {
    if (a[index] !== b[index]) return index
  }
  return -2 // 전부 같음
}

type Outcome = { ok: boolean; detail: string }

/**
 * 바이트열을 버퍼에 올렸다가 되읽어 비교한다.
 *
 * @param raw true면 ArrayBuffer를 커맨드에 그대로 싣는다(검증 대상),
 *            false면 셰임의 기존 base64 경로를 탄다(대조군).
 */
async function roundTrip(device: any, bytes: Uint8Array, raw: boolean): Promise<Outcome> {
  const buffer = device.createBuffer({
    size: bytes.length,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | GPUBufferUsage.MAP_READ,
    label: raw ? 'smoke.arraybuffer' : 'smoke.base64',
  })

  if (raw) {
    // 셰임의 writeBuffer는 base64로 인코딩하므로 우회해서 커맨드를 직접 넣는다.
    // 스트림에 그대로 실리는 것 말고는 평소 경로와 같다.
    device._recorder.push({
      op: 'writeBuffer',
      buffer: buffer.id,
      bufferOffset: 0,
      data: toArrayBuffer(bytes),
    })
  } else {
    device.queue.writeBuffer(buffer, 0, bytes)
  }
  device.queue.submit([])

  try {
    const read = new Uint8Array(await buffer.mapAsync(GPUMapMode.READ))
    const diff = bytesEqual(bytes, read)
    buffer.destroy()
    if (diff === -2) return { ok: true, detail: `${bytes.length}B 일치` }
    if (diff === -1) return { ok: false, detail: `길이 다름 (${read.length}B 왔음)` }
    return { ok: false, detail: `${diff}번째 바이트부터 다름` }
  } catch (error) {
    buffer.destroy()
    return { ok: false, detail: String(error) }
  }
}

function setup({ device, context, report }: SceneContext) {
  // 판정 전 회색 → 통과 초록 / 실패 빨강.
  let clearValue = { r: 0.05, g: 0.06, b: 0.09, a: 1 }

  async function probe() {
    const lines: string[] = []
    let allOk = true

    const control = await roundTrip(device, makePattern(SMALL), false)
    lines.push(`base64 대조군 ${control.ok ? '✓' : '✗'} ${control.detail}`)
    allOk = allOk && control.ok

    const small = await roundTrip(device, makePattern(SMALL), true)
    lines.push(`ArrayBuffer ${SMALL}B ${small.ok ? '✓' : '✗'} ${small.detail}`)
    allOk = allOk && small.ok

    const large = await roundTrip(device, makePattern(LARGE), true)
    lines.push(`ArrayBuffer ${LARGE / 1024}KB ${large.ok ? '✓' : '✗'} ${large.detail}`)
    allOk = allOk && large.ok

    clearValue = allOk
      ? { r: 0.05, g: 0.35, b: 0.16, a: 1 }
      : { r: 0.45, g: 0.08, b: 0.1, a: 1 }
    report(`${allOk ? 'PASS' : 'FAIL'} · ${lines.join(' · ')}`)
  }

  probe().catch((error: unknown) => {
    clearValue = { r: 0.45, g: 0.08, b: 0.1, a: 1 }
    report(`FAIL · 스모크 자체가 던짐: ${String(error)}`)
  })

  return () => {
    const encoder = device.createCommandEncoder()
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          loadOp: 'clear',
          storeOp: 'store',
          clearValue,
        },
      ],
    })
    pass.end()
    device.queue.submit([encoder.finish()])
  }
}

root.render(
  <DemoScene
    title="ArrayBuffer 브리징"
    subtitle="base64 없이 바이너리를 커맨드에 실어 왕복 — 초록이면 통과"
    setup={setup}
  />
)
