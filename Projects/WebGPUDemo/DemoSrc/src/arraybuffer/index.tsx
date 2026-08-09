import { root } from '@lynx-js/react'
import { DemoScene, type SceneContext } from '../scene.jsx'
import { GPUBufferUsage, GPUMapMode } from '../webgpu.js'

/**
 * A binary bridging smoke test — whether bytes travel **both ways** without base64.
 *
 * What is under test is Lynx's value converter. Both directions are checked:
 *
 * - **JS → native**: whether an `ArrayBuffer` put on a nested position of the command stream
 *   (`commands[i].data`) becomes `NSData` and lands in `WGPUValueReader.requiredData`'s `raw as? Data` branch.
 * - **native → JS**: whether the `Data` `LynxWebGPUContext.readBuffer` puts on arrives in JS as an
 *   `ArrayBuffer` (`mapAsync` returns it undecoded).
 *
 * The verdict is **a byte comparison**, not the presence of an error. A mismatched conversion could put
 * different bytes in with no error, so a pattern whose value differs per position is written and read back for comparison.
 *
 * The type is asserted alongside — a leaked view (TypedArray) gets turned into a `{"0":1,…}` object by Lynx
 * and breaks **silently**.
 */

const SMALL = 256
const LARGE = 64 * 1024 // the same size as one dynamic texture (128×128 RGBA)

/** A pattern whose value differs per position — a shifted offset shows immediately. */
function makePattern(length: number): Uint8Array {
  const bytes = new Uint8Array(length)
  for (let index = 0; index < length; index += 1) bytes[index] = (index * 31 + 7) & 0xff
  return bytes
}

/** The position of the first differing byte. -1 means the lengths differ, -2 means all equal. */
function firstDifference(a: Uint8Array, b: Uint8Array): number {
  if (a.length !== b.length) return -1
  for (let index = 0; index < a.length; index += 1) {
    if (a[index] !== b[index]) return index
  }
  return -2
}

type Outcome = { ok: boolean; detail: string }

/** Puts a byte sequence into a buffer and reads it back for comparison — upload and download in one step. */
async function roundTrip(device: any, bytes: Uint8Array): Promise<Outcome> {
  // MAP_READ can only combine with COPY_DST (spec). Those two are enough for a round trip.
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
      return { ok: false, detail: `the readback is not an ArrayBuffer (${typeof mapped})` }
    }
    const diff = firstDifference(bytes, new Uint8Array(mapped))
    buffer.destroy()
    if (diff === -2) return { ok: true, detail: `${bytes.length}B matched` }
    if (diff === -1) return { ok: false, detail: `length differs (${mapped.byteLength}B came back)` }
    return { ok: false, detail: `differs from byte ${diff}` }
  } catch (error) {
    buffer.destroy()
    return { ok: false, detail: String(error) }
  }
}

/**
 * Checks whether the `data` the shim puts on a command is a real `ArrayBuffer`.
 *
 * Even with a correct round trip, a view riding out would be turned into an object by Lynx, so the type is checked directly.
 * There is no way but to peek at the recorder, so an internal field is used (in this scene only).
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
    detail: ok ? 'ArrayBuffer' : `a view leaked (${Object.prototype.toString.call(command && command.data)})`,
  }
}

function setup({ device, context, report }: SceneContext) {
  // Grey before the verdict → green on pass / red on failure.
  let clearValue = { r: 0.05, g: 0.06, b: 0.09, a: 1 }

  async function probe() {
    const lines: string[] = []
    let allOk = true

    const type = checkPayloadType(device)
    lines.push(`payload type ${type.ok ? '✓' : '✗'} ${type.detail}`)
    allOk = allOk && type.ok

    const small = await roundTrip(device, makePattern(SMALL))
    lines.push(`round trip ${SMALL}B ${small.ok ? '✓' : '✗'} ${small.detail}`)
    allOk = allOk && small.ok

    const large = await roundTrip(device, makePattern(LARGE))
    lines.push(`round trip ${LARGE / 1024}KB ${large.ok ? '✓' : '✗'} ${large.detail}`)
    allOk = allOk && large.ok

    clearValue = allOk
      ? { r: 0.05, g: 0.35, b: 0.16, a: 1 }
      : { r: 0.45, g: 0.08, b: 0.1, a: 1 }
    report(`${allOk ? 'PASS' : 'FAIL'} · ${lines.join(' · ')}`)
  }

  probe().catch((error: unknown) => {
    clearValue = { r: 0.45, g: 0.08, b: 0.1, a: 1 }
    report(`FAIL · the smoke test itself threw: ${String(error)}`)
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
    title="Binary bridging"
    subtitle="A two-way round trip as ArrayBuffer — green means passed"
    setup={setup}
  />
)
