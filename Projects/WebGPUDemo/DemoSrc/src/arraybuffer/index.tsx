import { root } from '@lynx-js/react'
import { DemoScene, type SceneContext } from '../scene.jsx'
import { GPUBufferUsage, GPUMapMode } from '../webgpu.js'

/**
 * 바이너리 브리징 스모크 — base64 없이 바이트가 **양방향으로** 오가는지 본다.
 *
 * 검증 대상은 Lynx의 값 변환기다. 두 방향 모두 확인한다:
 *
 * - **JS → 네이티브**: 커맨드 스트림의 중첩 위치(`commands[i].data`)에 실린 `ArrayBuffer`가
 *   `NSData`로 바뀌어 `WGPUValueReader.requiredData`의 `raw as? Data` 분기에 걸리는가.
 * - **네이티브 → JS**: `LynxWebGPUContext.readBuffer`가 실은 `Data`가 JS에서 `ArrayBuffer`로
 *   오는가 (`mapAsync`가 디코딩 없이 그대로 돌려준다).
 *
 * 판정은 오류 유무가 아니라 **바이트 대조**다. 변환이 어긋나면 오류 없이 다른 바이트가
 * 들어갈 수 있으므로, 자리마다 값이 다른 패턴을 써 넣고 되읽어 비교한다.
 *
 * 타입도 함께 단언한다 — 뷰(TypedArray)가 새면 Lynx가 `{"0":1,…}` 객체로 바꿔
 * **조용히** 깨지기 때문이다.
 */

const SMALL = 256
const LARGE = 64 * 1024 // 동적 텍스처 한 장(128×128 RGBA)과 같은 크기

/** 자리마다 값이 다른 패턴 — 오프셋이 밀리면 바로 티가 난다. */
function makePattern(length: number): Uint8Array {
  const bytes = new Uint8Array(length)
  for (let index = 0; index < length; index += 1) bytes[index] = (index * 31 + 7) & 0xff
  return bytes
}

/** 다른 첫 바이트 위치. -1이면 길이가 다름, -2면 전부 같음. */
function firstDifference(a: Uint8Array, b: Uint8Array): number {
  if (a.length !== b.length) return -1
  for (let index = 0; index < a.length; index += 1) {
    if (a[index] !== b[index]) return index
  }
  return -2
}

type Outcome = { ok: boolean; detail: string }

/** 바이트열을 버퍼에 올렸다가 되읽어 비교한다 — 올리기와 내리기를 한 번에 밟는다. */
async function roundTrip(device: any, bytes: Uint8Array): Promise<Outcome> {
  // MAP_READ는 COPY_DST와만 조합할 수 있다(명세). 왕복에는 그 둘이면 충분하다.
  const buffer = device.createBuffer({
    size: bytes.length,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    label: 'smoke.roundtrip',
  })

  device.queue.writeBuffer(buffer, 0, bytes)
  device.queue.submit([])

  try {
    const mapped = await buffer.mapAsync(GPUMapMode.READ)
    if (!(mapped instanceof ArrayBuffer)) {
      buffer.destroy()
      return { ok: false, detail: `리드백이 ArrayBuffer가 아님 (${typeof mapped})` }
    }
    const diff = firstDifference(bytes, new Uint8Array(mapped))
    buffer.destroy()
    if (diff === -2) return { ok: true, detail: `${bytes.length}B 일치` }
    if (diff === -1) return { ok: false, detail: `길이 다름 (${mapped.byteLength}B 왔음)` }
    return { ok: false, detail: `${diff}번째 바이트부터 다름` }
  } catch (error) {
    buffer.destroy()
    return { ok: false, detail: String(error) }
  }
}

/**
 * 셰임이 커맨드에 싣는 `data`가 진짜 `ArrayBuffer`인지 본다.
 *
 * 왕복이 맞더라도 뷰가 실려 나가면 Lynx가 객체로 바꿔 버리므로, 타입을 직접 확인한다.
 * 레코더를 훔쳐보는 것 말고 방법이 없어 내부 필드를 쓴다 (이 씬만).
 */
function checkPayloadType(device: any): Outcome {
  const buffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST })
  device.queue.writeBuffer(buffer, 0, new Float32Array([1, 2, 3, 4]))
  const pending = device._recorder.pending
  const command = pending[pending.length - 1]
  const ok = command && command.op === 'writeBuffer' && command.data instanceof ArrayBuffer
  device.queue.submit([])
  buffer.destroy()
  return {
    ok,
    detail: ok ? 'ArrayBuffer' : `뷰가 샜다 (${Object.prototype.toString.call(command && command.data)})`,
  }
}

function setup({ device, context, report }: SceneContext) {
  // 판정 전 회색 → 통과 초록 / 실패 빨강.
  let clearValue = { r: 0.05, g: 0.06, b: 0.09, a: 1 }

  async function probe() {
    const lines: string[] = []
    let allOk = true

    const type = checkPayloadType(device)
    lines.push(`페이로드 타입 ${type.ok ? '✓' : '✗'} ${type.detail}`)
    allOk = allOk && type.ok

    const small = await roundTrip(device, makePattern(SMALL))
    lines.push(`왕복 ${SMALL}B ${small.ok ? '✓' : '✗'} ${small.detail}`)
    allOk = allOk && small.ok

    const large = await roundTrip(device, makePattern(LARGE))
    lines.push(`왕복 ${LARGE / 1024}KB ${large.ok ? '✓' : '✗'} ${large.detail}`)
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
    title="바이너리 브리징"
    subtitle="ArrayBuffer로 양방향 왕복 — 초록이면 통과"
    setup={setup}
  />
)
