# Lynx-WebGPU

[한국어](README.md) · **English**

An SPM library that gives JS in a [Lynx](https://lynxjs.org) bundle **GPU access shaped like WebGPU**.
It maps the [W3C WebGPU](https://www.w3.org/TR/webgpu/) object model and [WGSL](https://www.w3.org/TR/WGSL/) onto Metal.

```tsx
<webgpu-canvas canvas-id="main" style={{ width: '100%', height: '100%' }} />
```

```js
import gpu, { startFrameLoop } from './webgpu.js'

const adapter = await gpu.requestAdapter()
const device  = await adapter.requestDevice()
const context = gpu.getCanvasContext('main')
context.configure({ device, format: gpu.getPreferredCanvasFormat() })

const pipeline = device.createRenderPipeline({ layout: 'auto', vertex: { module, entryPoint: 'vs_main' }, … })

startFrameLoop(() => {
  const encoder = device.createCommandEncoder()
  const pass = encoder.beginRenderPass({ colorAttachments: [{ view: context.getCurrentTexture().createView(), … }] })
  pass.setPipeline(pipeline)
  pass.draw(3)
  pass.end()
  device.queue.submit([encoder.finish()])    // ← the whole frame crosses the bridge exactly once
})
```

Browser WebGPU code and WGSL shaders port over almost verbatim.

## Getting started

```swift
// Package.swift
.package(url: "https://github.com/xenonClient/Lynx-WebGPU", from: "0.3.1")
```

That gives you the **engine** (`LynxWebGPU`). Types that touch Lynx — such as `LynxWebGPUHost` below —
come from the **four files in `Sources/LynxWebGPUBridge/`, which you add to a target of your own**,
so that your app picks the Lynx SDK's version and distribution channel
(see `docs/LYNX-INTEGRATION.md` §2; the demo app is the worked example).

Wiring up the host app takes three steps:

```swift
let host = try LynxWebGPUHost()
let lynxView = LynxView { builder in
    let config = LynxConfig(provider: provider)
    LynxWebGPU.register(in: config, host: host)   // NativeModules.WebGPU + <webgpu-canvas>
    builder.config = config
}
host.attach(to: lynxView)
```

On the bundle side, copy `JS/webgpu.js`, `JS/webgpu.d.ts`, and `JS/elements.d.ts` into your rspeedy project's `src/` and you're done.
Full instructions: [docs/LYNX-INTEGRATION.md](docs/LYNX-INTEGRATION.md).

## Documentation

- [CLAUDE.md](CLAUDE.md) — build/test commands, module graph, conventions
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — design (command stream, shader pipeline, layout problem, concurrency)
- [docs/WEBGPU-API.md](docs/WEBGPU-API.md) — supported API reference and what's missing
- [docs/WGSL.md](docs/WGSL.md) — WGSL subset grammar · MSL mapping · constraints
- [docs/JS-AUTHORING.md](docs/JS-AUTHORING.md) — bundle (JS) authoring guide, performance rules
- [docs/LYNX-INTEGRATION.md](docs/LYNX-INTEGRATION.md) — host app integration
- [docs/TESTING.md](docs/TESTING.md) — test strategy/harnesses/conventions
- [docs/ROADMAP.md](docs/ROADMAP.md) — what's next (web-library port gaps → image pipeline polish)
- [Examples/HelloTriangle.tsx](Examples/HelloTriangle.tsx) — minimal ReactLynx example

All docs except this file are written in Korean — they translate well with any LLM.

## Modules

| Module | Role |
|---|---|
| `LynxWebGPUCore` | WebGPU enums · descriptors · errors · handle registry (Metal-free) |
| `LynxWebGPUShader` | WGSL lexer/parser/reflection/MSL emitter (pure Swift) |
| `LynxWebGPU` | Metal backend + canvas surfaces + command stream interpreter |
| `LynxWebGPUBridge` | Lynx NativeModule + `<webgpu-canvas>` element (iOS only) — **sources, not an SPM target** |

**This package has zero external dependencies.** The only SPM product is the engine (`LynxWebGPU`);
the Lynx SDK's version and distribution channel are **the app's choice** — SPM, an existing CocoaPods
setup, or an in-house build all work. The Lynx integration layer ships as sources wrapped in
`#if canImport(Lynx)`, so it switches on wherever Lynx is visible (a dedicated bridge target, or the
app target itself — see `docs/LYNX-INTEGRATION.md` §2).

The `LynxWebGPU` product works **without Lynx** — use it to run WebGPU command streams straight from a
Swift app, or purely as a WGSL-to-MSL translator.

## Design highlights

**Command stream** — WebGPU calls are only recorded in JS and handed over in one batch at `queue.submit()`.
Handles are issued by JS, so object creation never waits on a native round trip. The bridge is crossed exactly
once per frame. (Same idea as Chrome talking to its GPU process over the Dawn wire.)

**WGSL → MSL** — lexer through emitter, implemented from scratch. MSL has no mutable globals, so uniforms and
textures are threaded down the call graph as function arguments, and stage I/O attributes only appear on
entry-point wrapper structs. The `vec3` size mismatch (12 bytes in WGSL vs 16 in MSL) is reproduced by computing
the layout with `packed_float3`/padding — **JS fills uniforms by WGSL rules and it just works.**

**Errors are collected, not thrown** — a bad call never kills the process. Errors carry paths like
`commands[3].vertex.buffers[0].format`, and shader compile failures include the full generated MSL.

**Nothing blocks on the frame path** — canvas size reads a cache refreshed by submit responses, keeping bridge
round trips at exactly one per frame. `writeBuffer`/`writeTexture` go through pooled staging + blit in queue
order, so nothing waits for the GPU to finish — **dynamic textures can be uploaded every frame.**
If the GPU falls behind, a 3-frames-in-flight cap makes the frame ticker skip, so `nextDrawable()` blocking
never spills into the JS thread (touches and timers included).

**Lifetimes by hand, watched automatically** — handles are integers, so JS GC knows nothing about native
lifetimes. `submit()` returns a live-object count (`objects`) and the native side warns past 4096 objects to
catch missing destroys; when the engine supports `FinalizationRegistry`, a GC-driven auto-release safety net
kicks in.

## Compatibility with real-world WebGPU shaders

We ran all 68 WGSL shaders from the official [webgpu-samples](https://github.com/webgpu/webgpu-samples)
**unmodified** through translation plus an actual Metal compile:

| Result | Count |
|---|---|
| Pass as-is | **60 / 67 (89%)** |
| Work once the host supplies `constants` | 4 |
| Not standalone files in the corpus itself | 3 |

**The remaining 3 are not transpiler gaps** — they are fragments whose declarations live in other `.wgsl` files
the sample concatenates, or that contain JS string-substitution slots like
`texture_storage_2d<{OUTPUT_FORMAT}, write>`. Runtime-sized arrays (`arrayLength`), external textures
(`textureSampleBaseClampToEdge`), and untyped constant expressions (`vec3(1)`) are all supported.

The measurement harness ships in the repo, so the numbers can be re-taken anytime
([docs/TESTING.md](docs/TESTING.md) §7).

## Verification

125 Swift + 20 JS tests run in seconds — no simulator, no device.

- Transpiler tests push the generated MSL through the **actual Metal compiler**.
- Render tests draw into offscreen textures and **assert pixel values** (triangle, uniforms, indexed draw,
  alpha blending, compute + readback, texture sampling, edge-clamp sampling, depth testing).
  `rgba16float` surfaces are read back and checked that **values above 1.0 and below 0 survive** — the
  evidence that HDR results are never crushed to 8 bits.
- Command stream contracts are asserted too — that `writeTexture` executes **after** an earlier render pass
  in the same batch (queue order), that the staging pool stops growing across frames, that the in-flight
  counter rises and falls with commit/completion.
- Values the runtime slips into shaders behind the scenes (like `arrayLength()`) are read back from the GPU
  and **asserted as numbers**.
- The JS shim is verified with node's built-in runner — binary-path types and bytes are asserted, and a mock
  counts that the bridge is crossed exactly once per frame.

```zsh
swift test           # engine · transpiler · GPU render
cd JS && npm test    # JS shim (zero dependencies)
```

## Demo app

`Projects/WebGPUDemo` contains a Tuist demo host app and **22 Lynx bundles**. The app opens with a scene list,
and each scene maps 1:1 to a feature the offscreen harness verifies automatically — spinning triangle, 3D cube
(depth testing), 4096 particles (compute + instancing), texture & sampler, **dynamic texture (CPU plasma via
`writeTexture` every frame)**, alpha blending, **stencil masking (standalone `stencil8` format)**,
**GPU-driven rendering (compute picks the count, indirect draw reads it)**, **render bundles (120 draws in
6 commands)**, **queries & error scopes (occlusion sample counts · timestamps · `pushErrorScope`)**,
compute readback (`mapAsync`), pipeline constants (`override`),
MSL escape hatch, holographic card (touch → 3D pose → foil), WGSL compatibility (`arrayLength` · external
textures · untyped constant expressions), binary bridging, bridge cost benchmark, **HDR gain-map
reconstruction (`rgba16float` intermediate texture → EDR output)**, and scroll pass-through
(`passthrough-touches` verification with a canvas over a `<scroll-view>`).
Everything runs at 60 fps with Lynx `<text>` HUDs composited over the canvas.

```zsh
mise exec -- tuist generate --no-open
# Run WebGPUDemo from Xcode, or:
xcrun simctl launch <device> org.lynxwebgpu.demo                  # scene list
xcrun simctl launch <device> org.lynxwebgpu.demo -demo particles  # jump straight in
```

## Requirements

Xcode 26.2 / Swift 6.2 · iOS 17.0+ · macOS 14.0+ (for the dev loop)

**This package does not pull in the Lynx SDK** — the app supplies it (zero external dependencies).
The demo app links [xenonClient/Lynx-XCFramework](https://github.com/xenonClient/Lynx-XCFramework)
directly (device + simulator slices included); that is the **demo's** choice, not a library requirement.
Point it at a different repo or version by editing `Projects/WebGPUDemo/Project.swift`.
