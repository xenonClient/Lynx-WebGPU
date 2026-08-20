# 커맨드 스트림 명세

JS(`JS/webgpu.js`)와 네이티브 런타임(`WebGPURuntime`) 사이의 **유일한 계약**이다.

이 문서가 있는 이유는 둘이다:

1. **런타임을 갈아끼울 수 있게 하려고.** 스트림은 순수 데이터라 반대편 구현이 무엇이든 상관없다
   (`docs/ARCHITECTURE.md` §3-1). 다른 백엔드를 만드는 사람에게는 이 문서가 사양서다.
2. **필드 이름의 단일 출처.** JS는 `Record<string, any>`로 싣고 Swift는 문자열 키로 읽으므로,
   이름이 어긋나도 **양쪽 다 컴파일된다.** 타입 검사가 잡아 주지 않는 자리라 글로 못 박는다.

Swift 쪽 디코딩은 전부 `LynxWebGPUCore`에 있다 — 여기 적힌 이름과 1:1이다:
`WGPUDescriptors.swift`(명세 `GPU*Descriptor`) · `WGPUCommands.swift`(op 인자) ·
`WGPUCommand.swift`(op 이름 → 케이스 **디스패치 표**). 디스패치와 명세 검증은
`WGPUBackendEngine`(Core)의 exhaustive switch 한 곳이 실행하고, 백엔드는 `WGPUBackend`
동사 프로토콜만 구현하므로, §4에 op을 더하면 컴파일러가 모든 백엔드의 누락을 잡는다.

---

## 1. 요청

```json
{ "commands": [ { "op": "…", … }, … ], "present": true }
```

| 필드 | 타입 | 뜻 |
|---|---|---|
| `commands` | 객체 배열 | 순서대로 실행한다. 빈 배열도 유효하다 |
| `present` | bool (기본 `true`) | `false`면 프레임 **중간**의 내부 제출이다 — 커밋만 하고 드로어블 present와 프레임 스코프 핸들 만료를 미룬다 |

### `present: false`가 필요한 이유

shim은 `popErrorScope`·`mapAsync`처럼 **결과를 받아야 하는** 호출에서 프레임 도중에도 배치를
흘려보낸다. 그 배치가 `writeBuffer` 하나로라도 커맨드 버퍼를 만들면, 획득해 둔 드로어블이
그리기도 전에 present되고 핸들이 만료되어 **그 프레임의 남은 패스가 통째로 거부된다.**
Three.js의 지연 파이프라인 생성이 정확히 이 모양이었다.

## 2. 응답

```json
{ "ok": true, "commandCount": 42, "objects": 137,
  "errors": [ … ], "canvases": { … }, "errorScopes": [ … ] }
```

| 필드 | 언제 | 뜻 |
|---|---|---|
| `ok` | 항상 | 스코프에 잡히지 **않은** 오류가 하나도 없었는가 |
| `commandCount` | 항상 | 받은 명령 수 (되돌려 주는 확인값) |
| `objects` | 항상 | 살아 있는 네이티브 객체 수 — JS가 `destroy` 누락(레지스트리 증식)을 감시한다 |
| `errors` | 있을 때만 | `{kind, message, path?, line?}` 배열. **아래 §2-1** |
| `canvases` | 건드린 캔버스가 있을 때 | `{ "<id>": {width, height} }` — JS가 크기 캐시를 갱신한다 |
| `errorScopes` | pop이 있었을 때 | pop **순서 그대로**. `null`(깨끗) · 오류 객체(잡힘) · `{rejected: true}`(짝 없음) |

### 2-1. 오류

```json
{ "kind": "validation", "message": "…", "path": "commands[3].vertex.buffers[0].format", "line": 12 }
```

| `kind` | 명세 대응 | 뜻 |
|---|---|---|
| `validation` | `GPUValidationError` | 잘못된 인자·상태. 대부분 호출 측 버그 |
| `out-of-memory` | `GPUOutOfMemoryError` | 리소스 생성 실패 |
| `unsupported` | (`validation` 스코프가 잡는다) | 명세상 유효하지만 이 구현이 아직 안 하는 것 |
| `backend` | `GPUInternalError` | 백엔드 내부 오류 (셰이더 컴파일 실패 등) |

**오류는 실행을 멈추지 않는다.** 명령 하나가 실패해도 나머지를 계속 실행하고 오류를 모아
돌려준다 — 잘못된 호출 하나가 프레임 전체를 죽이지 않는다는 WebGPU의 계약이다.

`path`는 **선택이지만 강하게 권한다.** 이것이 있으면 JS 쪽에서 어느 인자가 문제인지 바로 안다.
`line`은 셰이더 오류에만 붙고, `getCompilationInfo()`의 `lineNum`이 이 값을 그대로 쓴다.

**배치가 끝난 뒤에 드러나는 실패는 다음 배치에 실어 보낸다.** GPU 실행 실패(메모리 부족·
타임아웃·디바이스 제거 — Dawn이라면 uncaptured error)는 배치가 결과를 돌려준 뒤에야 도착하고
따로 콜백 통로가 없다. 런타임은 모아 두었다가 **다음 `execute` 응답의 `errors`**로 흘려보낸다
(열린 오류 스코프가 있으면 그쪽이 먼저 잡는다). Swift 쪽 자리는 `WGPUDeferredErrorQueue`(Core)다.

응답 키의 철자와 생략 규칙은 `WGPUBatchResult`(Core)가 조립한다 — 백엔드가 이 모양을 손으로
다시 만들면 철자 하나로 어긋난다. 적합성 검사 `error-accumulation`이 판정하는 것이 이 모양이다.

## 3. 스트림 전체에 걸리는 규칙

### 3-1. 핸들은 **클라이언트가 발급한다**

모든 생성 op은 명세에 없는 `id` 필드를 싣는다. JS가 정수를 발급하고 네이티브는 그 번호로
객체를 등록한다 — `createBuffer`가 네이티브 왕복을 기다리지 않으므로 생성과 사용을 한 배치에
이어서 기록할 수 있다 (Dawn wire와 같은 모델).

발급기는 **shim 모듈 전체에 하나**여야 한다. 디바이스마다 카운터를 두면 두 번째 디바이스가
1번부터 다시 발급해 첫 디바이스의 객체를 **조용히 덮어쓴다.** 런타임은 겹침을 세어 로그로
남기는 것이 좋다 (`WGPUObjectRegistry.displacedHandleCount`).

### 3-2. 인자는 **기록 시점 값으로 고정된다**

shim의 `snapshotValue`가 기록 시점에 깊은 복사를 한다. 브라우저가 호출 시점에 직렬화하는 것과
같은 계약이다 — 호출 뒤 디스크립터를 재사용·리셋하는 코드(three.js의 싱글턴 디스크립터
패턴)가 flush를 기다리는 명령을 오염시키지 못한다.

### 3-3. 값 표기

| 명세 타입 | 받는 표기 |
|---|---|
| `GPUColor` | `{r,g,b,a}` 또는 `[r,g,b,a]` (a 기본 1) |
| `GPUExtent3D` | `{width,height?,depthOrArrayLayers?}` 또는 `[w,h?,d?]` (기본 1) |
| `GPUOrigin3D` | `{x?,y?,z?}` 또는 `[x?,y?,z?]` (기본 0) |
| 플래그 (`usage`·`visibility`) | 정수 비트마스크 |
| 열거형 | **명세 철자 문자열 그대로** (`"rgba8unorm"`, `"triangle-list"`) |
| 바이너리 (`data`) | `ArrayBuffer` / base64 문자열 / 바이트 배열 |
| 핸들 | 정수. `null`/누락은 "없음" |

`null`은 **누락과 같게** 다룬다 (`WGPUValueReader`가 `NSNull`을 없는 값으로 본다). 단
`colorFormats` 배열 안의 `null`은 "이 슬롯에는 어태치먼트가 없다"는 뜻으로 보존된다.

### 3-4. 패스 상태

- 패스는 `beginRenderPass`/`beginComputePass`로 열고 `endPass`로 닫는다. 인코더 객체가
  스트림에 없으므로 **한 번에 하나만** 열린다.
- `executeBundles`는 앞뒤로 **파이프라인·바인드 그룹·정점/인덱스 버퍼 바인딩을 무효화**한다
  (명세의 "Reset the render pass binding state"). 뷰포트·시저·블렌드 상수·스텐실 참조는 남는다.
- 드로우/디스패치 전에 파이프라인 레이아웃이 요구하는 바인드 그룹과 정점 버퍼가 **전부**
  바인드되어 있어야 한다.

---

## 4. op 목록 (51개)

`id`는 **JS가 발급하는 새 객체의 핸들**이다. 표의 "기본값" 칸이 비면 필수 필드다.

### 4-1. 리소스 생성

| op | 필드 |
|---|---|
| `createBuffer` | `id` · `size`(생략 시 `data` 길이) · `usage`(플래그) · `mappedAtCreation`(false) · `data`(초기값) · `label` |
| `createTexture` | `id` · `size`(extent) · `format` · `usage`(플래그) · `dimension`(`"2d"`) · `mipLevelCount`(1) · `sampleCount`(1) · `label` |
| `createTextureView` | `id` · `texture` · `format` · `dimension` · `aspect`(`"all"`) · `baseMipLevel`(0) · `mipLevelCount` · `baseArrayLayer`(0) · `arrayLayerCount` · `label` |
| `createSampler` | `id` · `addressModeU/V/W`(`"clamp-to-edge"`) · `magFilter`/`minFilter`/`mipmapFilter`(`"nearest"`) · `lodMinClamp`(0) · `lodMaxClamp`(32) · `compare` · `maxAnisotropy`(1) · `label` |
| `createShaderModule` | `id` · `code` · `language`(`"wgsl"` \| `"msl"`) · `label` |
| `createQuerySet` | `id` · `type`(`"occlusion"`\|`"timestamp"`) · `count` · `label` |
| `destroy` | `id` — 그 핸들의 객체를 버린다 |

`language: "msl"`은 **선택 기능**이다 — Metal 백엔드의 탈출구라 와이어에 남는다. 지원하지 않는
런타임은 그 `createShaderModule`을 `unsupported`로 **깨끗이 거부**해야 한다 (크래시·조용한
무시 금지). 적합성 검사 `msl-optional`이 이 계약을 판정한다 — 이 op을 쓰는 번들은 백엔드를
갈아끼우면 그 모듈만 거부된다는 것을 감수하는 것이다.

### 4-2. 바인딩 · 파이프라인

| op | 필드 |
|---|---|
| `createBindGroupLayout` | `id` · `entries[]` · `label` |
| `createPipelineLayout` | `id` · `bindGroupLayouts[]`(핸들 배열) · `label` |
| `createBindGroup` | `id` · `layout` · `entries[]` · `label` |
| `createRenderPipeline` | `id` · `layout`(핸들 또는 `"auto"`) · `vertex` · `primitive` · `depthStencil` · `multisample` · `fragment` · `label` |
| `createComputePipeline` | `id` · `layout` · `compute:{module, entryPoint?, constants?}` · `label` |
| `getBindGroupLayout` | `id` · `pipeline` · `index` — `layout:"auto"`로 유도된 레이아웃을 핸들로 꺼낸다 |
| `createRenderBundle` | `id` · `commands[]` · `colorFormats[]` · `depthStencilFormat` · `sampleCount`(1) · `depthReadOnly`(false) · `stencilReadOnly`(false) · `label` |

**`entries[]` (바인드 그룹 레이아웃)** — `binding` · `visibility`(플래그) + 다음 넷 중 **정확히 하나**:

| 키 | 안쪽 필드 |
|---|---|
| `buffer` | `type`(`"uniform"`) · `hasDynamicOffset`(false) · `minBindingSize`(0) |
| `sampler` | `type`(`"filtering"`) |
| `texture` | `sampleType`(`"float"`) · `viewDimension`(`"2d"`) · `multisampled`(false) |
| `storageTexture` | `access`(`"write-only"`) · `format` · `viewDimension`(`"2d"`) |

**`entries[]` (바인드 그룹)** — `binding` · `resource:{…}`, 안에 셋 중 하나:
`{buffer, offset?, size?}` · `{sampler}` · `{textureView}`.

**`vertex`** — `module` · `entryPoint`(생략 시 그 스테이지의 유일한 진입점) · `buffers[]` · `constants{}`.
`buffers[]` 항목은 `arrayStride` · `stepMode`(`"vertex"`) · `attributes[{format, offset(0), shaderLocation}]`.

**`fragment`** — `module` · `entryPoint` · `targets[{format, blend?, writeMask(all)}]` · `constants{}`.
`blend`는 `{color, alpha}`이고 각각 `{operation("add"), srcFactor("one"), dstFactor("zero")}`.

**`depthStencil`** — `format` · `depthWriteEnabled`(false) · `depthCompare`(`"always"`) ·
`depthBias`(0) · `depthBiasSlopeScale`(0) · `depthBiasClamp`(0) · `stencilFront`/`stencilBack`
(`{compare("always"), failOp/depthFailOp/passOp("keep")}`) · `stencilReadMask`/`stencilWriteMask`(0xFFFFFFFF).

**`multisample`** — `count`(1) · `mask`(0xFFFFFFFF) · `alphaToCoverageEnabled`(false).

### 4-3. 큐 작업

| op | 필드 |
|---|---|
| `writeBuffer` | `buffer` · `data` · `bufferOffset`(0). 대상은 `COPY_DST`여야 하고 offset·크기는 **4의 배수**여야 한다 |
| `writeTexture` | `texture` · `data` · `mipLevel`(0) · `origin` · `size`(extent) · `bytesPerRow`(포맷에서 유도) · `rowsPerImage`(블록 행 수) |
| `copyExternalImageToTexture` | `source:{source(ImageBitmap 핸들), origin?, flipY?}` · `destination:{texture, mipLevel?, origin?}` · `copySize`(생략 시 이미지의 남은 전부) |
| `unmapBuffer` | `buffer` — `mapAsync` 상태를 푼다 |

`bytesPerRow`·`rowsPerImage`는 **블록 단위**다 (명세 `GPUTexelCopyBufferLayout`). 비압축
포맷은 블록이 1×1이라 픽셀 행과 같다.

### 4-4. 복사

| op | 필드 |
|---|---|
| `copyBufferToBuffer` | `source` · `sourceOffset`(0) · `destination` · `destinationOffset`(0) · `size`(생략 시 원본의 남은 전부). 원본은 `COPY_SRC`, 대상은 `COPY_DST`여야 하고 오프셋·크기는 전부 **4의 배수**여야 한다 |
| `clearBuffer` | `buffer` · `offset`(0) · `size`(생략 시 버퍼 끝까지). 둘 다 4의 배수여야 한다 |
| `copyTextureToBuffer` | `source:{texture, mipLevel?, origin?}` · `destination:{buffer, offset?, bytesPerRow?, rowsPerImage?}` · `copySize` |
| `copyBufferToTexture` | `source:{buffer, offset?, bytesPerRow?, rowsPerImage?}` · `destination:{texture, mipLevel?, origin?}` · `copySize` |
| `copyTextureToTexture` | `source:{texture, mipLevel?, origin?}` · `destination:{…}` · `copySize` |

버퍼가 한쪽 끝인 두 복사(`copyTextureToBuffer`·`copyBufferToTexture`)에서 `bytesPerRow`는
**256의 배수**여야 하고, 복사가 여러 블록 행·레이어에 걸치면 **생략할 수 없다**.
`queue.writeTexture`에는 이 제약이 없다 — 명세가 큐 업로드와 인코더 복사를 다르게 정한다.
Metal은 둘 다 느슨해서, 안 막으면 브라우저·Dawn에서만 거부되는 코드가 나온다
(데모 씬 둘이 실제로 이 자리에서 32를 쓰고 있었다).

### 4-5. 캔버스

| op | 필드 |
|---|---|
| `configureCanvas` | `canvas`(문자열 id) · `format`(`"bgra8unorm"`) · `usage`(`RENDER_ATTACHMENT`) · `alphaMode`(`"opaque"`) · `colorSpace`(`"srgb"`) · `toneMapping:{mode}`(`"standard"`) |
| `getCurrentTexture` | `id` · `canvas` — 얻은 텍스처와 그 뷰의 핸들은 **프레임 스코프**다 |

캔버스는 **문자열 id**로 지목한다 (`<webgpu-canvas canvas-id="…">`). 핸들이 아닌 이유는
캔버스의 수명이 JS가 아니라 UI 트리에 달려 있기 때문이다.

### 4-6. 렌더 패스

| op | 필드 |
|---|---|
| `beginRenderPass` | `colorAttachments[]` · `depthStencilAttachment` · `occlusionQuerySet` · `timestampWrites` · `label` |
| `setPipeline` | `pipeline` (렌더/컴퓨트 공용 — 열린 패스가 어느 쪽인지 정한다) |
| `setBindGroup` | `index` · `bindGroup` · `dynamicOffsets[]` |
| `setVertexBuffer` | `slot` · `buffer` · `offset`(0) |
| `setIndexBuffer` | `buffer` · `format`(`"uint16"`\|`"uint32"`) · `offset`(0) |
| `setViewport` | `x`(0) · `y`(0) · `width` · `height` · `minDepth`(0) · `maxDepth`(1) |
| `setScissorRect` | `x`(0) · `y`(0) · `width` · `height` |
| `setBlendConstant` | `color` |
| `setStencilReference` | `reference`(0) — **WebIDL의 `u32` 변환(modulo)을 적용한다** |
| `draw` | `vertexCount` · `instanceCount`(1) · `firstVertex`(0) · `firstInstance`(0) |
| `drawIndexed` | `indexCount` · `instanceCount`(1) · `firstIndex`(0) · `baseVertex`(0) · `firstInstance`(0) |
| `drawIndirect` | `indirectBuffer` · `indirectOffset`(0) — 인자 16B |
| `drawIndexedIndirect` | `indirectBuffer` · `indirectOffset`(0) — 인자 20B |
| `executeBundles` | `bundles[]` |
| `endPass` | (없음) — 열려 있는 인코더를 닫는다 |

**`colorAttachments[]`** — `view` · `resolveTarget` · `clearValue`(투명) · `loadOp`(`"clear"`) · `storeOp`(`"store"`).
후행 `null` 슬롯은 허용된다 (번들 호환성 비교에서 무시된다).

**`depthStencilAttachment`** — `view` · `depthClearValue`(1) · `depthLoadOp` · `depthStoreOp` ·
`depthReadOnly`(false) · `stencilClearValue`(0) · `stencilLoadOp` · `stencilStoreOp` · `stencilReadOnly`(false).

**`timestampWrites`** — `querySet` · `beginningOfPassWriteIndex` · `endOfPassWriteIndex`.
둘 중 최소 하나가 있어야 하고 서로 달라야 한다 — 둘 다 없으면 **조용한 no-op**이 되어
앱이 GPU 시간을 0ns로 읽는다.

### 4-7. 컴퓨트 패스

| op | 필드 |
|---|---|
| `beginComputePass` | `timestampWrites` · `label` |
| `dispatchWorkgroups` | `x`(1) · `y`(1) · `z`(1) — 각각 **최소 1로 올린다** |
| `dispatchWorkgroupsIndirect` | `indirectBuffer` · `indirectOffset`(0) — 인자 12B |

### 4-8. 쿼리

| op | 필드 |
|---|---|
| `beginOcclusionQuery` | `queryIndex` — `beginRenderPass`에 `occlusionQuerySet`이 있어야 한다 |
| `endOcclusionQuery` | (없음) |
| `resolveQuerySet` | `querySet` · `firstQuery`(0) · `queryCount`(생략 시 남은 전부) · `destination` · `destinationOffset`(0, **256의 배수**) |

occlusion 쿼리는 중첩할 수 없고, 한 패스에서 같은 인덱스를 두 번 쓸 수 없다.

### 4-9. 오류 스코프 · 디버그

| op | 필드 |
|---|---|
| `pushErrorScope` | `filter`(`"validation"`\|`"out-of-memory"`\|`"internal"`) |
| `popErrorScope` | (없음) — 결과가 응답의 `errorScopes`에 pop 순서로 실린다 |
| `pushDebugGroup` | `groupLabel` |
| `popDebugGroup` | (없음) |
| `insertDebugMarker` | `markerLabel` |

**오류 스코프는 배치를 넘어 살아 있다.** 디바이스 상태이고 `push`와 `pop` 사이에 `submit`이
몇 번이든 들어갈 수 있다. 필터 파싱이 실패해도 **스택 깊이는 맞춰야** 한다 — 안 그러면
이후 `pop`이 바깥 스코프를 가져가서, 앱이 안쪽 구간의 결과라고 믿는 값이 실제로는 바깥
구간의 결과가 된다.

---

## 5. 커맨드 스트림 밖 (NativeModule 직접 호출)

프레임 배치에 실을 수 없는 것들 — 결과를 기다려야 하거나(비동기), 프레임과 무관하다.
`JS/lynx-env.d.ts`와 `WebGPUNativeModule.methodLookup`이 짝이 맞아야 한다.

| 메서드 | 뜻 |
|---|---|
| `execute(payload)` | 위의 전부 |
| `adapterInfo()` | `requestAdapter()`의 `info`·`limits`·`features`. **키는 명세 철자 그대로** |
| `shaderCompilationInfo({module})` | `getCompilationInfo()` |
| `canvasInfo({canvas})` | `{ok, width, height, format}` |
| `readBuffer({buffer, offset?, size?}, cb)` | `mapAsync` + `getMappedRange` |
| `loadAsset({name}, cb)` | 브라우저 `fetch()` 자리. 해석은 `WGPUAssetProvider`가 한다 |
| `decodeImage({id, data?/name?, flipY?, premultiplyAlpha?, resizeWidth?, resizeHeight?}, cb)` | `createImageBitmap()` |
| `startFrameLoop({fps})` / `stopFrameLoop()` | `webgpu:frame` 전역 이벤트 |
| `reset()` | 모든 GPU 객체를 버린다 |

### 5-1. 스레딩·수명 규약

원본은 `WebGPURuntime`(Sources/LynxWebGPUCore/WebGPURuntime.swift)의 문서 주석이다. 요약:

| 멤버 | 스레드 | 비고 |
|---|---|---|
| `execute` | JS 백그라운드 | 구현이 직렬화한다 |
| `attachCanvas` · `attachOffscreenCanvas` · `resizeCanvas` | 메인 | 레이어 초기 속성(`pixelFormat` 등)은 **런타임이** 정한다 — 엘리먼트는 레이어를 넘기기만 한다 |
| `detachCanvas` | **임의** | 엘리먼트 deinit이 부른다 — 표면 등록부는 락 필수 |
| `reset` | 메인 | `execute`와 동시 진입 가능 |
| `readBuffer` · `decodeImage` 콜백 | 임의 · **동기 가능** | 이미 끝난 작업이면 호출 스레드에서 즉시 와도 계약 위반이 아니다 |
| `isReadyForNextFrame` · `processEvents` | 메인 (틱마다) | 저비용·논블로킹. 펌프는 준비 게이트 **앞**에서 불리고 **`execute`와 동시에** 불린다 — 백엔드가 스레드 안전하지 않으면 구현이 직렬화할 것 (적합성 `pump-concurrency`가 판정). 직렬화하면서 논블로킹이려면 **락을 시도만 하고, 잡혀 있으면 그 틱을 거른다** |

- `configureCanvas`의 레이어 반영은 **비동기여도 된다** — 첫 프레임이 이전 설정으로 나갈 수
  있고, 그래서 `getCurrentTexture`는 캐시가 아니라 실제 드로어블의 포맷을 보고해야 한다.
- `processEvents()`는 명시적 이벤트 처리를 요구하는 백엔드(Dawn의
  `wgpuInstanceProcessEvents`)의 펌프 자리다. 프레임 티커가 틱을 건너뛸 때도 디스플레이
  주기마다 부르고, 적합성 하네스도 콜백 대기 중에 부른다. 기본 구현은 no-op이다.
- **`execute`에 넘긴 페이로드는 런타임이 붙들지 않는다.** 진입 즉시 순수 Swift 컨테이너로
  옮기므로(`WGPUPayload`), 호출이 끝나면 호스트가 넘긴 `NSDictionary`/`NSArray`를 더는
  참조하지 않는다. 호스트가 변환 버퍼를 재사용해도 안전하다는 뜻이고, 반대로 **런타임 구현이
  이 실체화를 건너뛰면 안 된다는 뜻이기도 하다** — 아래 "페이로드 소유권" 참고.

### 5-1-1. 페이로드 소유권

`NSDictionary`를 `[String: Any]`로 받으면 Swift는 **최상위 한 겹만** 네이티브로 옮긴다.
그 안의 배열·딕셔너리는 여전히 호스트 소유 ObjC 객체를 가리키는 창(window)이고, 값을 읽을
때마다 그 객체를 다시 만진다.

엔진은 중첩 디스크립터(`colorAttachments`, `entries`, `vertex.buffers` …)를 배치 시작이 아니라
**해당 명령을 실행할 때** 읽는다. 그래서 호스트가 변환 결과를 호출이 끝나기 전에 재사용하면
배치 후반의 명령이 이미 다른 데이터를 읽는다. 실제로 이 모양의 크래시가 있었다 — JS 스레드에서
커맨드 리더를 복사하다 `objc_retain`이 쓰레기 isa를 읽고 죽었고, 그 자리에는 객체 헤더가 아니라
부동소수 페이로드가 들어 있었다.

그래서 `WGPUBackendEngine.execute(_:)`는 락을 잡기 전에 `WGPUPayload.materialize(_:)`로
트리 전체를 훑어 네이티브 컨테이너로 옮긴다. 잎(`NSString`·`NSNumber`·`NSData`)은 **참조만**
옮긴다 — 수명을 지키는 데 필요한 것은 강한 참조 하나이지 바이트 복사가 아니고, 텍스처 업로드
`NSData`를 프레임마다 딥카피하면 수 MB를 매번 다시 쓰게 된다. 부수 효과로 읽기 경로는 오히려
빨라진다: 지연 브리지에서는 필드마다 `objc_msgSend` + 값 브리징이 붙지만 실체화 뒤에는
네이티브 해시 조회 한 번이다.

### 5-2. `adapterInfo` 확장 키 등급

명세 철자 키(`info`·`limits`·`features`·`preferredCanvasFormat`)는 필수다. 그 밖의 키:

| 등급 | 키 | 소비처 |
|---|---|---|
| 확장 — 권장 | `name` (표시명) | 데모 씬. 없으면 shim이 `info.description`으로 메꾼다 |
| 확장 — 선택 | `backend` · `hasUnifiedMemory` | shim이 `GPUAdapter` 필드로 실을 뿐 분기하지 않는다. 없으면 `''`/`false` |
| Metal 런타임 전용 — **계약 아님** | `supportsFamilyApple7` | 소비처 없음. 다른 런타임은 채우지 않는다 |

---

## 6. 알려진 차이

명세와 어긋나거나 op 사이에 일관되지 않은 자리. **모르고 밟는 것보다 적어 두는 편이 낫다.**

| 자리 | 지금 |
|---|---|
| `copyTextureToBuffer` / `copyBufferToTexture`의 `rowsPerImage` | **읽지만 쓰지 않는다.** 한 번에 한 슬라이스만 복사하므로 이미지 스트라이드가 필요 없다. 3D/배열 텍스처를 여러 레이어에 걸쳐 복사하게 되면 그때 반영해야 한다 (`writeTexture`는 이미 존중한다) |
| `GPURenderPassColorAttachment.depthSlice` | 읽지 않는다 — 항상 0번 슬라이스에 그린다 |
| `GPURenderPassDescriptor.maxDrawCount` | 무시한다 (드라이버 힌트) |
| `setBindGroup`의 `Uint32Array` 오버로드 | 없다 — 배열 형태와 결과가 같고 브리지를 건널 때 어차피 배열로 펴진다 |
| `setBindGroup`의 `dynamicOffsets` 파싱 실패 | **빈 배열로 본다.** 실제로 필요했다면 바인드 그룹 적용에서 "개수가 부족하다"로 잡힌다 |

미지원 API 전체 목록은 `docs/WEBGPU-API.md` §8에 있다.

---

## 7. op을 더할 때

`.claude/skills/webgpu-command/SKILL.md`의 순서를 따른다. 이 문서 기준으로는:

1. `LynxWebGPUCore/WGPUCommands.swift`에 디코딩 구조체를 더한다 (**이름의 출처**).
2. `LynxWebGPUCore/WGPUCommand.swift`에 케이스 + 디코더 분기 + `opName`을 더한다 —
   여기부터는 컴파일러가 안내한다: 케이스가 늘면 **엔진의 exhaustive switch가 깨져서**
   누락이 조용히 지나가지 않는다. `WGPUCommandDecodeTests`의 픽스처(수 단언 51 포함)도 한 줄.
3. `WGPUBackendEngine.dispatch`에 케이스 한 줄 — 명세 검증(범위·usage·상태)은 엔진에,
   인코딩은 `WGPUBackend`에 동사로 더한다. **동사가 늘면 모든 백엔드가 컴파일에서 깨진다** —
   Metal(`WGPUMetalBackend`)과 Dawn(`DawnBackend`)을 함께 채울 것.
4. JS shim이 같은 이름으로 싣는지 확인한다.
5. **§4의 표에 행을 더한다.** 여기 없는 op은 다른 런타임에서 구현되지 않는다.
6. 적합성 검사(`Sources/LynxWebGPUConformance`)나 계약 테스트(`Tests/LynxWebGPUTests`)를
   더한다 — 런타임을 갈아끼워도 같은 결과가 나오는지 보는 자리다.
