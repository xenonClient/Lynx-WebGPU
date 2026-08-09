import { root, useRef } from '@lynx-js/react'
import { DemoScene, type SceneContext } from '../scene.jsx'
import gpu, { GPUBufferUsage, GPUMapMode, GPUTextureUsage } from '../webgpu.js'

/**
 * The things you ask the GPU back about — occlusion queries, timestamps, error scopes.
 *
 * None of the three draw directly to the screen. So **the verdict is shown as numbers on the HUD**:
 * the more the bar hides the circle, the fewer samples survive, and fully hidden it is exactly 0.
 *
 * Error scopes show the contrast through two buttons. The same bad call made inside a scope shows only on
 * the yellow line; made outside, it rides the global handler down to **the red line**.
 */
const SHADER = /* wgsl */ `
struct Uniforms {
  time: f32,
  aspect: f32,
  depth: f32,
  center: f32,    // the bar's horizontal center
};
@group(0) @binding(0) var<uniform> u: Uniforms;

struct Out {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> Out {
  var corners = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var out: Out;
  // Depth comes from a uniform — the bar in front, the circle behind.
  out.position = vec4f(corners[index], u.depth, 1.0);
  out.uv = corners[index];
  return out;
}

// The occluder — a vertical bar sweeping left and right.
@fragment
fn fs_bar(in: Out) -> @location(0) vec4f {
  let half = 0.24;
  let distance = abs(in.uv.x - u.center);
  if (distance > half) {
    discard;
  }
  let shade = 0.20 + 0.10 * smoothstep(0.0, half, distance);
  return vec4f(shade * 0.85, shade * 0.95, shade * 1.25, 1.0);
}

// The observed object — the occlusion query counts the samples this draw let through.
// The coordinates are corrected so that **the short side is ±1** (so the circle does not overflow sideways on a portrait screen).
@fragment
fn fs_target(in: Out) -> @location(0) vec4f {
  let p = vec2f(in.uv.x * u.aspect, in.uv.y) / min(u.aspect, 1.0);
  let radius = length(p);
  if (radius > 0.68) {
    discard;
  }
  let glow = 1.0 - radius / 0.68;
  let ripple = 0.85 + 0.15 * sin(radius * 14.0 - u.time * 2.4);
  return vec4f(mix(vec3f(0.2, 0.5, 1.0), vec3f(1.0, 0.85, 0.4), glow) * ripple, 1.0);
}
`

interface Actions {
  probe?: (inScope: boolean) => void
}

async function setup(
  { device, context, format, report }: SceneContext,
  actionsRef: { current: Actions }
) {
  const adapter = await gpu.requestAdapter()
  const hasTimestamp = !!adapter && adapter.features.has('timestamp-query')

  const module = device.createShaderModule({ code: SHADER, label: 'query' })

  // Per the spec an auto derived layout is pipeline-exclusive — sharing requires an explicit layout.
  const sharedBindLayout = device.createBindGroupLayout({
    entries: [{ binding: 0, visibility: 0x3 /* VERTEX|FRAGMENT */, buffer: {} }],
  })
  const sharedLayout = device.createPipelineLayout({ bindGroupLayouts: [sharedBindLayout] })

  function makePipeline(entryPoint: string) {
    return device.createRenderPipeline({
      layout: sharedLayout,
      vertex: { module, entryPoint: 'vs_main' },
      fragment: { module, entryPoint, targets: [{ format }] },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
    })
  }
  const barPipeline = makePipeline('fs_bar')
  const targetPipeline = makePipeline('fs_target')

  const layout = sharedBindLayout
  function makeUniforms(label: string) {
    const buffer = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label,
    })
    return {
      buffer,
      group: device.createBindGroup({ layout, entries: [{ binding: 0, resource: { buffer } }] }),
      data: new Float32Array(4),
    }
  }
  const bar = makeUniforms('query.bar')
  const target = makeUniforms('query.target')

  // One occlusion query — the result is a single u64 (8 bytes).
  const occlusionQuerySet = device.createQuerySet({ type: 'occlusion', count: 1 })
  const occlusionResults = device.createBuffer({
    size: 8,
    usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    label: 'occlusion.results',
  })
  // The readback goes into a **dedicated staging buffer** — MAP_READ can only combine with COPY_DST (spec),
  // and a buffer being mapped is rejected by queue operations, so mapping the resolve target directly would block the next frame.
  const occlusionStaging = device.createBuffer({
    size: 8,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    label: 'occlusion.staging',
  })

  // Timestamps come with device conditions — the adapter.features are asked before creating one.
  const timestampQuerySet = hasTimestamp
    ? device.createQuerySet({ type: 'timestamp', count: 2 })
    : null
  const timestampResults = hasTimestamp
    ? device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
        label: 'timestamp.results',
      })
    : null
  const timestampStaging = hasTimestamp
    ? device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        label: 'timestamp.staging',
      })
    : null

  // The depth attachment follows the canvas — the bar needs a depth test to actually hide the circle.
  let depthTexture: any = null
  let depthView: any = null
  let depthWidth = 0
  let depthHeight = 0

  function ensureDepth(width: number, height: number) {
    if (depthTexture && depthWidth === width && depthHeight === height) return
    if (depthTexture) depthTexture.destroy()
    depthTexture = device.createTexture({
      size: { width, height },
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
      label: 'query.depth',
    })
    depthView = depthTexture.createView()
    depthWidth = width
    depthHeight = height
  }

  /** Deliberately builds a bad pipeline — the vertex format spelling is not in the spec. */
  function makeBadPipeline() {
    device.createRenderPipeline({
      layout: 'auto',
      vertex: {
        module,
        entryPoint: 'vs_main',
        buffers: [{
          arrayStride: 8,
          attributes: [{ format: 'float32x9', offset: 0, shaderLocation: 0 }],
        }],
      },
      fragment: { module, entryPoint: 'fs_target', targets: [{ format }] },
    })
  }

  actionsRef.current.probe = (inScope: boolean) => {
    if (!inScope) {
      makeBadPipeline()
      report('failed outside a scope — it rides down to the global handler (the red line) on the next submission')
      return
    }
    device.pushErrorScope('validation')
    makeBadPipeline()
    // popErrorScope has to get a result, so it submits itself (the same nature as mapAsync).
    device
      .popErrorScope()
      .then((error: any) => {
        report(
          error
            ? `the scope caught it [${error.kind}] ${error.message} — it never went to the global handler`
            : 'the scope caught no error at all'
        )
      })
      .catch((error: unknown) => report(`scope check failed: ${String(error)}`))
  }

  let time = 0
  let frame = 0
  let reading = false
  let peakSamples = 1

  return ({ delta, width, height }: { delta: number; width: number; height: number }) => {
    time += delta / 1000
    ensureDepth(width, height)

    const aspect = width / height
    // The bar in front (0.2), the circle behind (0.6) — where they overlap the circle loses the depth test.
    bar.data[0] = time
    bar.data[1] = aspect
    bar.data[2] = 0.2
    bar.data[3] = Math.sin(time * 0.7)
    device.queue.writeBuffer(bar.buffer, 0, bar.data)

    target.data[0] = time
    target.data[1] = aspect
    target.data[2] = 0.6
    target.data[3] = 0
    device.queue.writeBuffer(target.buffer, 0, target.data)

    const encoder = device.createCommandEncoder()
    const passDescriptor: any = {
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: 0.02, g: 0.03, b: 0.05, a: 1 },
      }],
      depthStencilAttachment: {
        view: depthView,
        depthClearValue: 1,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
      // A query can only be attached **when opening a pass**.
      occlusionQuerySet,
    }
    if (timestampQuerySet) {
      passDescriptor.timestampWrites = {
        querySet: timestampQuerySet,
        beginningOfPassWriteIndex: 0,
        endOfPassWriteIndex: 1,
      }
    }

    const pass = encoder.beginRenderPass(passDescriptor)

    pass.setPipeline(barPipeline)
    pass.setBindGroup(0, bar.group)
    pass.draw(3)

    // Only the samples this draw let through are counted.
    pass.beginOcclusionQuery(0)
    pass.setPipeline(targetPipeline)
    pass.setBindGroup(0, target.group)
    pass.draw(3)
    pass.endOcclusionQuery()

    pass.end()

    encoder.resolveQuerySet(occlusionQuerySet, 0, 1, occlusionResults, 0)
    if (timestampQuerySet && timestampResults) {
      encoder.resolveQuerySet(timestampQuerySet, 0, 2, timestampResults, 0)
    }

    // Copied to staging only once every 20 frames — a readback waits for GPU completion.
    const wantsReadback = ++frame % 20 === 0 && !reading
    if (wantsReadback) {
      encoder.copyBufferToBuffer(occlusionResults, 0, occlusionStaging, 0, 8)
      if (timestampResults && timestampStaging) {
        encoder.copyBufferToBuffer(timestampResults, 0, timestampStaging, 0, 16)
      }
    }

    device.queue.submit([encoder.finish()])

    if (!wantsReadback) return
    reading = true

    const pending: Promise<ArrayBuffer>[] = [occlusionStaging.mapAsync(GPUMapMode.READ)]
    if (timestampStaging) pending.push(timestampStaging.mapAsync(GPUMapMode.READ))

    Promise.all(pending)
      .then(([occlusion, timestamps]: ArrayBuffer[]) => {
        // The u64 is read as its low 32 bits — the sample count will never pass 4 billion.
        const samples = new Uint32Array(occlusion)[0]
        if (samples > peakSamples) peakSamples = samples
        const visible = Math.round((samples / peakSamples) * 100)

        let line = samples === 0
          ? 'occlusion 0 — the circle is fully hidden'
          : `occlusion ${samples} samples · ${visible}% visible`

        if (timestamps) {
          // Two u64s. Combining hi and lo as a double stays inside 2^53, so the difference is exact.
          const parts = new Uint32Array(timestamps)
          const start = parts[1] * 4294967296 + parts[0]
          const end = parts[3] * 4294967296 + parts[2]
          line += ` · GPU pass ${((end - start) / 1e6).toFixed(3)}ms`
        } else {
          line += ' · this device does not support timestamps'
        }
        report(line)
      })
      .catch((error: unknown) => report(`query readback failed: ${String(error)}`))
      .finally(() => {
        // The mapping must be released for the next cycle's copy to use this buffer again.
        occlusionStaging.unmap()
        if (timestampStaging) timestampStaging.unmap()
        reading = false
      })
  }
}

function QueryScene() {
  const actionsRef = useRef<Actions>({})

  return (
    <DemoScene
      title="Queries · error scopes"
      subtitle="Occluded sample count · GPU pass time · caught errors — the things that come out as numbers, not on screen"
      setup={(scene) => setup(scene, actionsRef)}
      controls={
        <view className="controls">
          <text
            className="control-button"
            bindtap={() => actionsRef.current.probe && actionsRef.current.probe(true)}
          >
            Fail inside a scope
          </text>
          <text
            className="control-button"
            bindtap={() => actionsRef.current.probe && actionsRef.current.probe(false)}
          >
            Fail outside a scope
          </text>
        </view>
      }
    />
  )
}

root.render(<QueryScene />)
