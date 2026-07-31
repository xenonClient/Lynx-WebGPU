# 지원 WebGPU API

브라우저 WebGPU 명세 중 이 구현이 지원하는 범위. 표기는 JS 클라이언트(`JS/webgpu.js`) 기준이며,
괄호 안은 커맨드 스트림의 `op` 이름이다 (네이티브로 넘어가는 실제 형태 — `docs/ARCHITECTURE.md` §3).

## 1. 진입점

```js
import gpu, { GPUBufferUsage, GPUTextureUsage, GPUShaderStage, startFrameLoop } from './webgpu.js'

const adapter = await gpu.requestAdapter()          // adapterInfo (동기 네이티브 호출)
const device  = await adapter.requestDevice()
const context = gpu.getCanvasContext('main')        // <webgpu-canvas canvas-id="main">
const format  = gpu.getPreferredCanvasFormat()      // "bgra8unorm"
```

`adapter.limits`는 Metal 인자 테이블에서 오는 실제 한계값이다:

| 키 | 뜻 |
|---|---|
| `maxVertexBuffers` | 8 — 정점 버퍼 슬롯 |
| `maxBindGroupBuffers` | 22 — 바인드 그룹이 쓸 수 있는 버퍼 (정점 버퍼 8슬롯 + `arrayLength()` 크기 표 1슬롯 예약분 제외) |
| `maxTexturesPerStage` / `maxSamplersPerStage` | 31 / 16 |
| `maxBufferSize` | `MTLDevice.maxBufferLength` |
| `maxThreadsPerThreadgroup` | 컴퓨트 워크그룹 상한 |

## 2. 캔버스 (`GPUCanvasContext`)

```js
context.configure({ device, format, alphaMode: 'opaque' })   // configureCanvas
const texture = context.getCurrentTexture()                  // getCurrentTexture
const view = texture.createView()                            // createTextureView
const { width, height } = context.getSize()                  // 캐시 (execute 응답으로 갱신)
```

- `configure`는 `CAMetalLayer` 설정을 메인 스레드에 **비동기**로 반영한다. `getPreferredCanvasFormat()`
  (= `bgra8unorm`)을 쓰면 엘리먼트 기본값과 같아 첫 프레임부터 일치한다.
- `getSize()`는 **제출 응답의 `canvases`로 갱신되는 캐시**를 읽는다. 동기 네이티브 조회는
  캐시가 빌 때(= `configure` 직후) 한 번뿐이므로 프레임 안에서 불러도 왕복이 없다.
- `getCurrentTexture()`가 돌려준 텍스처와 그 뷰는 **그 프레임 안에서만** 유효하다.
- present는 자동이다 — 배치가 끝날 때 획득한 드로어블이 화면에 올라간다.

## 3. 리소스

### 버퍼

```js
// (1) 초기 데이터와 함께
const vertexBuffer = device.createBuffer({ size, usage: GPUBufferUsage.VERTEX, mappedAtCreation: true })
new Float32Array(vertexBuffer.getMappedRange()).set(data)
vertexBuffer.unmap()                                  // 여기서 createBuffer 명령이 기록된다

// (2) 나중에 쓰기
device.queue.writeBuffer(buffer, 0, float32Array)      // writeBuffer

// (3) 읽기 — GPU 완료를 기다리므로 비동기
const bytes = await buffer.mapAsync(GPUMapMode.READ)   // readBuffer (콜백형 네이티브 호출)
```

`usage` 플래그: `MAP_READ` `MAP_WRITE` `COPY_SRC` `COPY_DST` `INDEX` `VERTEX` `UNIFORM` `STORAGE`
(`INDIRECT`/`QUERY_RESOLVE`는 값만 있고 기능은 미지원).

> `writeBuffer`는 스테이징 버퍼 + blit으로 큐에 **순서를 태워** 올라간다. 직접 memcpy 하면
> 직전 프레임 GPU 작업과 경쟁하기 때문이다. 스테이징 버퍼는 네이티브가 풀로 재사용하므로
> (프레임 완료 시 회수, 총량 상한) 매 프레임 불러도 할당이 쌓이지 않는다.

### 텍스처 · 샘플러

```js
const texture = device.createTexture({                 // createTexture
  size: { width: 256, height: 256 },
  format: 'rgba8unorm',
  usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
})
device.queue.writeTexture({ texture }, pixels, { bytesPerRow: 256 * 4 }, { width: 256, height: 256 })
const sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' })   // createSampler
```

> `writeTexture`도 `writeBuffer`처럼 스테이징 + blit으로 큐에 **순서를 태워** 올라간다 —
> 호출마다 GPU 완주를 기다리지 않으므로 동적 텍스처를 매 프레임 갱신할 수 있고, 같은 배치에서
> 앞선 렌더 패스가 그린 내용과의 순서도 스트림 그대로 보존된다. 배열 텍스처는
> `depthOrArrayLayers`만큼 슬라이스별로 복사된다.

지원 포맷: 8/16/32비트 컬러 전 계열(`r8unorm` … `rgba32float`), `bgra8unorm(-srgb)`, `rgb10a2unorm`,
`rg11b10ufloat`, 깊이/스텐실(`depth16unorm` `depth24plus` `depth32float` `stencil8` + `-stencil8` 조합).
**블록 압축(BC/ETC/ASTC)은 미지원.**

> `depth24plus`는 Apple GPU에 24비트 깊이 포맷이 없어 `depth32Float`으로 올려 준다 (명세가 "24비트 이상"을 요구하므로 적법).

`createSampler` 옵션: `addressModeU/V/W` `magFilter` `minFilter` `mipmapFilter` `lodMinClamp`
`lodMaxClamp` `compare` `maxAnisotropy`.

### 셰이더 모듈

```js
const module = device.createShaderModule({ code: WGSL_SOURCE })              // createShaderModule
const raw    = device.createShaderModule({ code: MSL_SOURCE, language: 'msl' })  // 탈출구
```

`language: 'msl'`은 WGSL 트랜스파일러를 건너뛴다. 이때 바인딩 인덱스를 MSL에 직접 써야 하고
`layout: 'auto'`를 쓸 수 없다 (`GPUPipelineLayout`을 명시할 것).

## 4. 바인딩

```js
// 자동 유도 — 셰이더의 @group/@binding 선언에서 만든다
const bindGroup = device.createBindGroup({                    // getBindGroupLayout + createBindGroup
  layout: pipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: uniformBuffer, offset: 0 } },
    { binding: 1, resource: textureView },
    { binding: 2, resource: sampler },
  ],
})

// 셰이더가 `array<T>`(런타임 크기)를 쓰면 길이는 **바인딩된 크기**에서 나온다.
// `size`를 주면 그만큼만, 안 주면 버퍼 끝까지다 — `arrayLength()`가 그 값을 본다.
{ binding: 0, resource: { buffer: particles, offset: 0, size: 48 } }   // arrayLength → 48 / sizeof(T)

// 명시적 레이아웃
const layout = device.createBindGroupLayout({                 // createBindGroupLayout
  entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } }],
})
const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] })  // createPipelineLayout
```

`hasDynamicOffset: true`인 바인딩은 `pass.setBindGroup(0, group, [offset])`로 오프셋을 준다.
오프셋은 **바인딩 번호 오름차순**으로 소비된다.

## 5. 파이프라인

```js
const pipeline = device.createRenderPipeline({                // createRenderPipeline
  layout: 'auto',                     // 또는 GPUPipelineLayout
  vertex: {
    module, entryPoint: 'vs_main',
    buffers: [{
      arrayStride: 20,
      stepMode: 'vertex',             // 'instance'
      attributes: [
        { format: 'float32x2', offset: 0, shaderLocation: 0 },
        { format: 'float32x3', offset: 8, shaderLocation: 1 },
      ],
    }],
  },
  fragment: {
    module, entryPoint: 'fs_main',
    targets: [{
      format,
      writeMask: GPUColorWrite.ALL,
      blend: {
        color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
        alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
      },
    }],
  },
  primitive: { topology: 'triangle-list', cullMode: 'back', frontFace: 'ccw' },
  depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' },
  multisample: { count: 4 },
})

const compute = device.createComputePipeline({                // createComputePipeline
  layout: 'auto',
  compute: { module, entryPoint: 'main' },
})
```

- `vertex` / `fragment` / `compute` 각각 `constants: { name: value }` 로 셰이더의 `override` 값을 준다
  (`docs/WGSL.md` §2-2). 같은 셰이더라도 상수가 다르면 별도로 컴파일·캐시된다.
- 정점 버퍼 슬롯은 **최대 8개**, `arrayStride: 0`은 미지원.
- 컴퓨트 워크그룹 크기는 WGSL의 `@workgroup_size`에서 리플렉션으로 가져온다 (MSL 모듈은 (1,1,1)).
- 스텐실 상태(`stencilFront`/`stencilBack`)는 미지원 — 깊이만 반영된다.

## 6. 커맨드 인코딩

```js
const encoder = device.createCommandEncoder()

const pass = encoder.beginRenderPass({                       // beginRenderPass
  colorAttachments: [{ view, loadOp: 'clear', storeOp: 'store', clearValue: { r:0,g:0,b:0,a:1 } }],
  depthStencilAttachment: { view: depthView, depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
})
pass.setPipeline(pipeline)          // setPipeline
pass.setBindGroup(0, bindGroup)     // setBindGroup
pass.setVertexBuffer(0, buffer)     // setVertexBuffer
pass.setIndexBuffer(buffer, 'uint16')  // setIndexBuffer
pass.setViewport(0, 0, w, h, 0, 1)  // setViewport
pass.setScissorRect(0, 0, w, h)     // setScissorRect
pass.setBlendConstant(color)        // setBlendConstant
pass.setStencilReference(1)         // setStencilReference
pass.draw(3, 1, 0, 0)               // draw
pass.drawIndexed(6, 1, 0, 0, 0)     // drawIndexed
pass.end()                          // endPass

const computePass = encoder.beginComputePass()               // beginComputePass
computePass.setPipeline(compute)
computePass.setBindGroup(0, group)
computePass.dispatchWorkgroups(8, 8)                         // dispatchWorkgroups
computePass.end()

encoder.copyBufferToBuffer(src, 0, dst, 0, size)             // copyBufferToBuffer
encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow }, { width, height })
encoder.copyBufferToTexture({ buffer, bytesPerRow }, { texture }, { width, height })
encoder.copyTextureToTexture({ texture: a }, { texture: b }, { width, height })

device.queue.submit([encoder.finish()])   // ← 여기서 한 번에 네이티브로 넘어간다
```

- `beginRenderPass`는 멀티샘플 `resolveTarget`을 지원한다 (store op이 `multisampleResolve`로 바뀐다).
- 복사 명령은 패스 **밖에서만** 쓸 수 있다.

## 7. 오류 처리

`queue.submit()`은 `{ ok, errors, canvases }`를 돌려준다. 오류는 실행을 멈추지 않고 누적된다.

```js
device.onError((error, text) => console.error(text))
// error = { kind: 'validation' | 'unsupported' | 'out-of-memory' | 'backend',
//           message: '…', path: 'commands[3].vertex.buffers[0].format' }
```

핸들러를 등록하지 않으면 `console.error`로 나간다.

| kind | 뜻 |
|---|---|
| `validation` | 잘못된 인자·상태. 대부분 호출 측 버그다. |
| `unsupported` | 명세상 유효하지만 이 구현이 아직 안 하는 것. |
| `out-of-memory` | 리소스 생성 실패. |
| `backend` | Metal/셰이더 컴파일 오류. 셰이더 실패 시 **생성된 MSL이 메시지에 포함**된다. |

## 8. 미지원 목록

| 기능 | 상태 |
|---|---|
| `GPUQuerySet` / 타임스탬프 / occlusion 쿼리 | 미지원 |
| `drawIndirect` / `dispatchWorkgroupsIndirect` | 미지원 |
| `GPURenderBundle` | 미지원 |
| 스텐실 테스트 상태 | 미지원 (깊이만) |
| 블록 압축 텍스처 (BC/ETC/ASTC) | 미지원 |
| `GPUExternalTexture` (`importExternalTexture`) | 미지원 — Lynx에 비디오 엘리먼트 핸들이 없다. 다만 WGSL `texture_external` + `textureSampleBaseClampToEdge`는 **지원**하므로, 프레임을 텍스처로 올려 그 자리에 `GPUTextureView`를 묶으면 된다 |
| `pushErrorScope` / `popErrorScope` | `submit()` 반환의 `errors` 배열로 대체 |
| 파이프라인 상수 (`override` / `constants`) | **지원** (§5) |
| `writeTimestamp`, `resolveQuerySet` | 미지원 |

새 명령을 추가하는 절차는 `.claude/skills/webgpu-command/SKILL.md` 참고.
