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

`adapter.limits`는 **명세 `GPUSupportedLimits`의 전 항목**을 명세 철자 그대로 싣는다 —
웹 라이브러리가 이 이름으로 읽고 자기 예산을 정하기 때문이다. 값은 가능한 한 Metal 디바이스에서
실제로 읽고, 런타임 조회가 없는 것만 Metal 기능 집합 표의 보장값을 쓴다.

| 키 | 값 | 근거 |
|---|---|---|
| `maxTextureDimension1D` / `2D` | 16384 (구형 8192) | Apple family 3+ / Mac2 |
| `maxTextureDimension3D` / `maxTextureArrayLayers` | 2048 / 2048 | Metal 보장값 |
| `maxVertexBuffers` | 8 | 우리 인자 테이블 배정 규칙 |
| `maxSampledTexturesPerShaderStage` / `maxSamplersPerShaderStage` | 31 / 16 | Metal 인자 테이블 |
| `maxStorageBuffersPerShaderStage` / `maxUniformBuffersPerShaderStage` | 22 | 정점 버퍼 8슬롯 + `arrayLength()` 크기 표 1슬롯 예약분 제외 |
| `maxBufferSize` / `maxStorageBufferBindingSize` | `MTLDevice.maxBufferLength` | 실제 조회 |
| `minUniformBufferOffsetAlignment` / `minStorage…` | 256 | **명세 기본값을 쓴다** — Metal 요구(32B)보다 크므로 이걸 지키면 Metal도 만족한다. 32를 보고하면 브라우저에서만 깨지는 코드가 나온다 |
| `maxComputeWorkgroupSizeX/Y/Z` · `maxComputeWorkgroupStorageSize` | 디바이스 조회 | `maxThreadsPerThreadgroup`, `maxThreadgroupMemoryLength` |
| `maxComputeWorkgroupsPerDimension` | 65535 | Metal에 조회가 없다 (Dawn과 같은 보수적 값) |

나머지(`maxBindGroups` 4, `maxColorAttachments` 8 …)도 전부 실려 있다. 값이 명세 기본값보다
낮지 않은지는 `CommandInterpreterTests`가 못 박는다 — 낮으면 브라우저에서 되는 코드가
여기서만 거부되고 앱은 이유를 알 수 없다.

`adapter.features`는 기기마다 갈리는 기능을 알려 준다 (웹과 같은 `has()` 인터페이스):

```js
if (adapter.features.has('timestamp-query')) { /* 타임스탬프 쿼리셋을 만들 수 있다 */ }
```

`adapter.info`(= `device.adapterInfo`)는 명세 `GPUAdapterInfo`다 — GPU 종류로 분기하는 코드가
읽는 표준 이름이다:

```js
adapter.info.vendor          // 'apple'
adapter.info.architecture    // 'apple-8' 등 — Metal의 supportsFamily로 알아낸 만큼
adapter.info.description     // MTLDevice.name ("Apple M3")
adapter.info.device          // '' — Metal에 벤더 식별자 개념이 없다
```

**모르는 자리는 빈 문자열이다** (명세 규칙). 지어내면 그 값으로 분기하는 코드가 잘못된
우회로를 탄다. 이 구현만의 `adapter.name`·`backend`·`hasUnifiedMemory`도 그대로 남아 있다.

`device.features`에는 명세대로 **`requestDevice({ requiredFeatures })`로 요청한 것만** 들어간다 —
어댑터가 지원해도 요청하지 않았으면 `has()`는 false다. 지원하지 않는 기능을 요구하면
`requestDevice`가 거부한다. `device.lost`는 존재하되 영원히 pending이다 (§8).

## 2. 캔버스 (`GPUCanvasContext`)

```js
context.configure({ device, format, alphaMode: 'opaque' })   // configureCanvas
const texture = context.getCurrentTexture()                  // getCurrentTexture
const view = texture.createView()                            // createTextureView
const { width, height } = context.getSize()                  // 캐시 (execute 응답으로 갱신)

context.getConfiguration()   // 마지막 설정 (없으면 null)
context.unconfigure()        // 다시 configure하기 전까지 그릴 수 없다
```

- `configure`는 `CAMetalLayer` 설정을 메인 스레드에 **비동기**로 반영한다. `getPreferredCanvasFormat()`
  (= `bgra8unorm`)을 쓰면 엘리먼트 기본값과 같아 첫 프레임부터 일치한다.
- `getSize()`는 **제출 응답의 `canvases`로 갱신되는 캐시**를 읽는다. 동기 네이티브 조회는
  캐시가 빌 때(= `configure` 직후) 한 번뿐이므로 프레임 안에서 불러도 왕복이 없다.
- `unconfigure()`는 포맷을 바꿔 재구성하는 코드(HDR 토글 등)가 밟는 자리다. 이후
  `getCurrentTexture()`는 거부된다. **이미 화면에 나간 프레임을 지우지는 않는다** — 브라우저는
  캔버스를 투명 검정으로 비우지만, 여기서 그러려면 표면을 클리어해 present해야 하고
  설정을 푸는 호출이 프레임을 하나 소비하는 편이 더 놀랍다고 봤다.
- `getCurrentTexture()`가 돌려준 텍스처와 그 뷰는 **그 프레임 안에서만** 유효하다.
- present는 자동이다 — 배치가 끝날 때 획득한 드로어블이 화면에 올라간다.

### HDR 출력 (EDR)

```js
context.configure({
  device,
  format: 'rgba16float',              // 1.0을 넘는 값을 담을 수 있어야 한다
  colorSpace: 'srgb',                 // 또는 'display-p3'
  toneMapping: { mode: 'extended' },  // 기본값 'standard'
})
```

`toneMapping.mode: 'extended'`는 `CAMetalLayer`의 `wantsExtendedDynamicRangeContent`를 켜고
색공간을 **확장 선형**으로 바꾼다. 1.0을 넘는 값이 잘리지 않고 디스플레이가 SDR 흰색보다
밝게 낼 수 있는 여유(EDR)로 그대로 나간다.

세 가지가 **함께** 맞아야 실제로 밝아진다. 하나라도 어긋나면 조용히 SDR로 보인다:

1. `format`이 1.0 초과를 담는 `rgba16float` — UNORM 포맷은 0~1로 클램프된다
2. `toneMapping.mode`가 `'extended'`
3. 셰이더가 **선형 값을 그대로** 출력 — 확장 색공간이 선형이므로 sRGB 인코딩을 하면 안 되고,
   톤매핑도 하면 안 된다 (톤매핑이 하는 일이 정확히 "1.0 초과를 0~1로 눌러 담기"다)

- **시뮬레이터에서는 확인할 수 없다.** EDR은 실기기 디스플레이 기능이다.
- 포맷이 바뀌는 `configure`는 레이어 설정이 메인 스레드에 비동기로 반영되므로, 파이프라인을
  새 포맷으로 갈아탈 때 몇 프레임 여유를 둬야 한다 (`hdr` 데모 씬 참고).

## 3. 리소스

### 버퍼

```js
// (1) 초기 데이터와 함께
const vertexBuffer = device.createBuffer({ size, usage: GPUBufferUsage.VERTEX, mappedAtCreation: true })
new Float32Array(vertexBuffer.getMappedRange()).set(data)
vertexBuffer.unmap()                                  // 여기서 createBuffer 명령이 기록된다

// (2) 나중에 쓰기
device.queue.writeBuffer(buffer, 0, float32Array)      // writeBuffer

// (3) 읽기 — GPU 완료를 기다리므로 비동기. 다 읽었으면 unmap()
const bytes = await staging.mapAsync(GPUMapMode.READ)  // readBuffer (콜백형 네이티브 호출)
const part  = staging.getMappedRange(8, 16)            // 부분만 (offset 8의 배수, size 4의 배수)
staging.unmap()
```

`usage` 플래그: `MAP_READ` `MAP_WRITE` `COPY_SRC` `COPY_DST` `INDEX` `VERTEX` `UNIFORM` `STORAGE`
`INDIRECT` `QUERY_RESOLVE`.

버퍼·텍스처는 명세의 **읽기 전용 속성**을 그대로 갖는다 — 객체를 받아 스스로 판단하는
코드(라이브러리)가 이 이름으로 읽는다:

```js
buffer.size · buffer.usage · buffer.mapState        // 'unmapped' | 'pending' | 'mapped'
texture.width · height · depthOrArrayLayers · mipLevelCount · sampleCount
texture.dimension · format · usage · textureBindingViewDimension
```

`textureBindingViewDimension`은 생략하면 `dimension`과 레이어 수에서 정해진다
(2d + 레이어 2 이상 → `'2d-array'`).

- **`MAP_READ`는 `COPY_DST`와만, `MAP_WRITE`는 `COPY_SRC`와만** 조합할 수 있다 (명세 요구).
  Metal은 `.storageModeShared` 하나로 전부 되지만, 안 막으면 브라우저에서만 깨진다.
- 그래서 계산 결과를 읽는 정석은 **2단**이다 — 결과 버퍼(`STORAGE|COPY_SRC`)를
  `copyBufferToBuffer`로 스테이징 버퍼(`COPY_DST|MAP_READ`)에 옮기고 그쪽을 매핑한다.

> **`getMappedRange`가 돌려주는 것은 사본이다.** JS에서는 `ArrayBuffer`가 다른 `ArrayBuffer`의
> 일부를 가리킬 수 없어, 구간을 복사해 주고 `unmap()`에서 되돌려 쓴다 — 쓴 내용은 사라지지
> 않지만 **반환된 버퍼는 `unmap()` 전까지만** 의미가 있다 (브라우저에서 detach되는 시점과 같다).
> 전체 구간을 처음 요청하면 복사 없이 매핑 자체를 준다. 구간끼리 **겹칠 수 없고**, `offset`은
> 8의 배수·`size`는 4의 배수여야 한다 (명세 규칙, `OperationError`).
>
> 정렬 요구는 **명시한 값에만** 적용한다 — 매핑 크기가 곧 네이티브 버퍼 크기라 3바이트짜리도
> 정상인데, 생략한 `size`까지 4의 배수를 요구하면 그런 버퍼를 아예 못 읽는다.

> **매핑 중인 버퍼는 큐 작업에 쓸 수 없다.** `mapAsync`는 명세대로 버퍼를 "unavailable"로
> 만들고, `unmap()`을 부를 때까지 쓰기·복사·resolve·드로우 바인딩을 전부 거부한다.
> 이 구현은 스테이징 없이 `.storageModeShared` 버퍼를 직접 읽으므로, 이 상태가 없으면
> 리드백이 GPU 완료를 기다리는 사이 다음 프레임의 쓰기가 같은 메모리에 겹쳐
> **받은 값이 어느 프레임 것인지 보장되지 않는다.** 읽고 나면 반드시 `unmap()`할 것.

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

지원 포맷: 8/16/32비트 컬러 전 계열(`r8unorm` … `rgba32float`), `bgra8unorm(-srgb)`,
팩된 32비트(`rgb10a2unorm` `rgb10a2uint` `rg11b10ufloat` `rgb9e5ufloat`),
깊이/스텐실(`depth16unorm` `depth24plus` `depth32float` `stencil8` + `-stencil8` 조합),
**블록 압축 52종**(아래). 16비트 unorm/snorm 6종은 명세에서도 선택 기능이라 없다 (§8).

#### 블록 압축 (BC · ETC2 · ASTC)

셋 다 명세의 **선택 기능**이라 쓰기 전에 `adapter.features`를 확인한다. 기기가 못 하는 계열로
텍스처를 만들면 검증 오류로 거부된다 — 그냥 넘기면 Metal이 프로세스를 죽이기 때문이다.

| feature | 포맷 | Apple GPU |
|---|---|---|
| `texture-compression-astc` | `astc-4x4-unorm` … `astc-12x12-unorm-srgb` (28종) | 전 iOS 기기 + Apple Silicon Mac |
| `texture-compression-etc2` | `etc2-rgb8unorm` … `eac-rg11snorm` (10종) | 전 iOS 기기 + Apple Silicon Mac |
| `texture-compression-bc` | `bc1-rgba-unorm` … `bc7-rgba-unorm-srgb` (14종) | A14/M1 이상 (그리고 Intel·AMD Mac) |

```js
if (adapter.features.has('texture-compression-astc')) {
  const texture = device.createTexture({
    size: { width: 256, height: 256 }, format: 'astc-8x8-unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  })
  // bytesPerRow는 **블록** 단위다 — 256px / 8 = 32블록 × 16B. 생략하면 이 값이 기본이다.
  device.queue.writeTexture({ texture }, blocks, { bytesPerRow: 32 * 16 }, { width: 256, height: 256 })
}
```

지켜야 할 것 (전부 검증 오류로 돌려준다):

- **행은 픽셀이 아니라 블록으로 센다.** `bytesPerRow` = `ceil(width / 블록너비) × 블록바이트`,
  `rowsPerImage`도 **블록 행** 수다. 생략하면 이 식이 기본값이 된다.
- **렌더 타깃·스토리지 텍스처가 될 수 없다** — GPU에 디코더는 있어도 인코더가 없다.
  멀티샘플·3D도 안 된다.
- 복사의 `origin`은 **블록 경계**여야 하고, 크기는 블록 배수이거나 **밉 레벨의 끝에 닿아야**
  한다 (가장자리 블록은 잘려 있으므로 예외다).
- 인코딩은 이 라이브러리가 하지 않는다. 이미 압축된 바이트를 올리는 통로다
  (`loadAsset`으로 `.ktx2`/`.astc` 파일을 읽어 헤더를 벗기고 넘기면 된다).

#### 이미지 파일에서 텍스처로 (`createImageBitmap`)

PNG·JPEG·HEIC를 풀어 텍스처로 올린다. 웹의 `createImageBitmap()` +
`queue.copyExternalImageToTexture()`와 같은 모양이고, 디코딩은 네이티브(ImageIO)가 한다.

```js
import { createImageBitmap, loadAsset } from './webgpu.js'

const bitmap = await createImageBitmap(await loadAsset('photo.jpg'))
// 애셋 이름을 바로 줘도 된다 — 바이트가 JS를 거치지 않는다.
// const bitmap = await createImageBitmap('photo.jpg')

const texture = device.createTexture({
  size: [bitmap.width, bitmap.height], format: 'rgba8unorm',
  usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
})
device.queue.copyExternalImageToTexture({ source: bitmap }, { texture },
  [bitmap.width, bitmap.height])
bitmap.close()   // 네이티브 픽셀을 놓아 준다
```

> **픽셀은 네이티브에 남는다.** 브리지를 건너는 것은 핸들과 크기뿐이라, 4K 사진도
> `ArrayBuffer`로 실어 나르지 않는다. JS에서 PNG를 손으로 푸는 것보다 훨씬 빠르고
> HEIC처럼 JS 디코더가 없는 형식도 열린다.

`createImageBitmap(source, options)`:

| 옵션 | 뜻 |
|---|---|
| `flipY` | 위아래를 뒤집는다. 기본은 웹과 같은 좌상단 기준이다 |
| `premultiplyAlpha` | `'premultiply'`면 색에 알파를 곱해 둔다. 기본(`'none'`)은 곱하지 않은 값 — 다만 디코딩 자체가 곱해진 값을 거치므로 **반투명 영역은 완전히 복원되지 않는다** |
| `resizeWidth` / `resizeHeight` | 둘 다 주면 그 크기로 그린다 |

소스는 `ArrayBuffer`/`TypedArray`(이미지 파일 바이트) 또는 **애셋 이름 문자열**이다.
문자열을 주면 `loadAsset`과 같은 규칙으로 호스트가 해석한다 (`WGPUAssetProvider`).

`copyExternalImageToTexture(source, destination, copySize)`:

- `source`: `{ source: bitmap, origin?: {x, y}, flipY?: boolean }` — 이미지의 일부만 잘라
  올릴 수 있고, `flipY`는 **복사 시점**에 위아래를 뒤집는다.
  `createImageBitmap`의 `flipY`(디코딩 시점)와는 다른 층이라 둘 다 켜면 제자리로 돌아온다.
  **웹 라이브러리는 이쪽을 쓴다** — three.js의 `Texture.flipY`가 기본 `true`다.
- `destination`: `{ texture, mipLevel?, origin? }`.
- 대상은 **4바이트 컬러 포맷**이어야 한다 (`rgba8unorm` `bgra8unorm` `rgba8unorm-srgb` …).
  디코딩 결과가 RGBA8이므로, 그 밖은 조용히 어긋나느니 검증 오류로 거부한다.
  압축 텍스처도 같은 이유로 안 된다 — GPU에는 블록 인코더가 없다.
- `copySize`를 생략하면 `origin`부터 이미지 끝까지다.

> `rgb9e5ufloat`는 채널마다 가수 9비트 + 공통 지수 5비트를 쓰는 공유 지수 HDR 포맷이다 —
> `rgba16float`의 **절반 크기로 같은 동적 범위**를 담으므로 읽기 전용 HDR 소스(환경맵 등)에 맞다.
> 팩된 포맷은 `readPixels`가 채널로 펴 주지 않는다 (`data`를 직접 해석할 것 — `docs/TESTING.md` §4-1).

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

```js
const { messages } = await module.getCompilationInfo()   // shaderCompilationInfo
// messages[] = { message, type: 'error', lineNum, linePos, offset, length }
```

- **셰이더 모듈은 컴파일에 실패해도 만들어진다** (명세 모델). 실패는 이 목록과 파이프라인 생성
  실패로 드러나므로, `createShaderModule()`이 돌아온 뒤에도 확인할 값이 있다.
  핸들이 아예 없으면 이후 명령이 전부 "존재하지 않는다"로만 깨져 **진짜 원인이 사라진다.**
- `lineNum`은 WGSL 소스의 줄 번호(1부터)다. `linePos`·`offset`·`length`는 이 구현이 알지 못해
  **0으로 둔다** — 모르는 값을 지어내면 편집기가 엉뚱한 곳에 밑줄을 긋는다.
- Metal 런타임 API는 경고를 따로 주지 않으므로 `type`은 항상 `'error'`다.
- 왕복이 하나 붙는다. 진단 경로에서만 쓸 것.

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

- **`entryPoint`는 생략할 수 있다.** 그러면 그 스테이지의 **유일한** 진입점을 쓴다 (명세의
  "get the entry point"). 후보가 없거나 둘 이상이면 이름을 알려 주는 validation 오류다.
- `vertex` / `fragment` / `compute` 각각 `constants: { name: value }` 로 셰이더의 `override` 값을 준다
  (`docs/WGSL.md` §2-2). 같은 셰이더라도 상수가 다르면 별도로 컴파일·캐시된다.
- 정점 버퍼 슬롯은 **최대 8개**, `arrayStride: 0`은 미지원.
- 컴퓨트 워크그룹 크기는 WGSL의 `@workgroup_size`에서 리플렉션으로 가져온다 (MSL 모듈은 (1,1,1)).

### 비동기 생성 — 실패를 그 자리에서 안다

```js
try {
  const pipeline = await device.createRenderPipelineAsync(descriptor)   // createComputePipelineAsync 도 같다
} catch (error) {
  // error.name === 'GPUPipelineError', error.reason === 'validation' | 'internal'
  console.error(error.message)     // 경로 + 사유 (셰이더 실패면 생성된 MSL까지)
}
```

동기 판은 명령만 기록하므로 실패가 **다음 `submit()`의 오류 배열로 늦게** 온다. 비동기 판은
생성을 오류 스코프로 감싸 즉시 제출하고 결과로 Promise를 푼다 — 명세대로 **오류를 디바이스로
보내지 않고**(전역 `onError`에 안 뜬다) `GPUPipelineError`로 거부한다. `reason`은 셰이더
번역·컴파일 실패면 `'internal'`, 그 밖은 `'validation'`이다.

대가는 왕복 하나다. 프레임 루프가 아니라 **초기화 경로에서** 쓸 것 (§5의 `popErrorScope`와 같은 이유).

### 스텐실 상태

```js
depthStencil: {
  format: 'stencil8',                 // 또는 'depth24plus-stencil8' · 'depth32float-stencil8'
  stencilFront: { compare: 'equal', failOp: 'keep', depthFailOp: 'keep', passOp: 'replace' },
  stencilBack:  { compare: 'equal', passOp: 'replace' },
  stencilReadMask: 0xffffffff,        // 비교 전에 양쪽 값을 가리는 마스크
  stencilWriteMask: 0xffffffff,       // 갱신할 비트
}
```

- 연산 8종: `keep` `zero` `replace` `invert` `increment-clamp` `decrement-clamp`
  `increment-wrap` `decrement-wrap`. `replace`가 쓰는 값은 `pass.setStencilReference(n)`이다.
- 기본값은 "아무것도 하지 않음"이다 — `compare: 'always'` + 세 연산 모두 `'keep'`.
  그래서 `stencilFront`/`stencilBack`을 주지 않으면 스텐실이 결과에 영향을 주지 않는다.
- 비교는 `(reference & readMask)` ↔ `(저장된 값 & readMask)` 사이에서 일어난다.
- 세 연산의 구분: `failOp`는 스텐실 테스트에 졌을 때, `depthFailOp`는 스텐실은 통과하고
  **깊이 테스트에 졌을 때**, `passOp`는 둘 다 통과했을 때다.
- 마킹 패스는 `targets: [{ format, writeMask: 0 }]`으로 색을 막고 스텐실만 남긴다.
- 깊이가 없는 `stencil8` 단독 포맷도 쓸 수 있다 (렌더 패스에 깊이 어태치먼트를 두지 않는다).

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
pass.drawIndirect(argsBuffer, 0)        // drawIndirect
pass.drawIndexedIndirect(argsBuffer, 0) // drawIndexedIndirect
pass.end()                          // endPass

const computePass = encoder.beginComputePass()               // beginComputePass
computePass.setPipeline(compute)
computePass.setBindGroup(0, group)
computePass.dispatchWorkgroups(8, 8)                         // dispatchWorkgroups
computePass.dispatchWorkgroupsIndirect(argsBuffer, 0)        // dispatchWorkgroupsIndirect
computePass.end()

encoder.copyBufferToBuffer(src, dst)                         // copyBufferToBuffer — 원본 전체
encoder.copyBufferToBuffer(src, dst, size)                   // 앞에서 size 바이트
encoder.copyBufferToBuffer(src, 0, dst, 0, size)             // 오프셋까지 지정 (size 생략 가능)
encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow }, { width, height })
encoder.copyBufferToTexture({ buffer, bytesPerRow }, { texture }, { width, height })
encoder.copyTextureToTexture({ texture: a }, { texture: b }, { width, height })
encoder.clearBuffer(buffer, offset, size)                    // clearBuffer — 0으로 채운다

// 디버그 마커 — 커맨드 인코더와 패스·번들 인코더 모두에 있다
encoder.pushDebugGroup('프레임')                              // pushDebugGroup
pass.insertDebugMarker('그림자 직전')                          // insertDebugMarker
encoder.popDebugGroup()                                      // popDebugGroup

device.queue.submit([encoder.finish()])   // ← 여기서 한 번에 네이티브로 넘어간다
```

- `beginRenderPass`는 멀티샘플 `resolveTarget`을 지원한다 (store op이 `multisampleResolve`로 바뀐다).
- 복사 명령은 패스 **밖에서만** 쓸 수 있다.
- **디버그 마커는 Xcode GPU 캡처에 구간 이름으로 뜬다** (Metal `pushDebugGroup`/`insertDebugSignpost`).
  없으면 캡처가 이름 없는 드로우 나열이라 어느 패스가 무엇인지 알 수 없다. 스코프는 두 층이다 —
  **패스 안**에서 열면 그 패스 구간, **패스 밖**에서 열면 프레임 구간이다 (`writeBuffer`·복사가
  속으로 여는 blit 인코더는 스코프가 아니다 — 그것까지 세면 프레임 구간이 거기서 닫히려 한다).
  `push`와 `pop`의 짝이 맞지 않거나 연 채로 패스가 끝나면 **Metal이 단언으로 프로세스를 죽이므로**,
  네이티브가 깊이를 세어 막고 validation 오류로 알린다 (닫아 주고 계속 간다).
- `clearBuffer`는 `writeBuffer`로 0 배열을 미는 것과 결과가 같지만 **CPU에서 그 배열을 만들어
  브리지로 실어 보내지 않는다.** 큰 스토리지 버퍼를 프레임마다 초기화할 때 차이가 크다.
  `offset`·`size`는 4의 배수여야 하고 버퍼는 `COPY_DST`여야 한다 (명세 규칙). `size` 생략 시 끝까지.
- `depthStencilAttachment`의 `depthReadOnly` / `stencilReadOnly`는 "이 패스는 그쪽을 쓰지
  않는다"는 선언이다. 선언해 두면 **실제로 강제된다** — 깊이를 쓰는 파이프라인
  (`depthWriteEnabled: true`)이나 스텐실을 쓰는 파이프라인(`failOp`·`depthFailOp`·`passOp` 중
  하나라도 `"keep"`이 아닌 것)을 `setPipeline`에서 거부한다. Metal은 그냥 써 버리므로
  여기서 안 막으면 read-only라고 적어 둔 버퍼가 조용히 변조된다.
  `readOnly`인 쪽에는 `loadOp`/`storeOp`을 **함께 줄 수 없다** (명세 요구 — 모순이다).

### 쿼리 (occlusion · 타임스탬프)

```js
const querySet = device.createQuerySet({ type: 'occlusion', count: 2 })   // createQuerySet
const results  = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC })
// MAP_READ는 COPY_DST와만 조합할 수 있다 — 리드백은 스테이징 버퍼로 받는다.
const staging  = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })

const pass = encoder.beginRenderPass({ colorAttachments, occlusionQuerySet: querySet })
pass.beginOcclusionQuery(0)        // beginOcclusionQuery
pass.draw(3)
pass.endOcclusionQuery()           // endOcclusionQuery
pass.end()

encoder.resolveQuerySet(querySet, 0, 2, results, 0)   // resolveQuerySet
encoder.copyBufferToBuffer(results, 0, staging, 0, 16)
device.queue.submit([encoder.finish()])
const counts = new BigUint64Array(await staging.mapAsync(GPUMapMode.READ))
staging.unmap()                    // 다음 주기의 복사가 이 버퍼를 다시 쓸 수 있게
```

- 결과 하나는 `u64` 8바이트다. `occlusion`은 **통과한 샘플 수**이므로 0이면 완전히 가려진 것이다.
- `count`는 **1 이상 4096 이하**다 (명세 상한 — Metal은 더 큰 것도 받지만 여기서 막는다).
- 쿼리셋은 **패스를 열 때만** 붙일 수 있고(`occlusionQuerySet`), occlusion 쿼리는 중첩할 수 없다.
- occlusion 쿼리는 **패스를 닫기 전에 `endOcclusionQuery`로 닫아야** 하고, 같은 인덱스를
  **한 패스에서 두 번 쓸 수 없다**. 둘 다 Metal은 그냥 넘어가지만(값까지 정상으로 보인다)
  브라우저에서는 패스가 통째로 무효화되므로 여기서 직접 막는다.
- `timestampWrites`는 `beginningOfPassWriteIndex`·`endOfPassWriteIndex` 중 **최소 하나**를 줘야 하고,
  둘 다 주면 **서로 달라야** 한다. 둘 다 빠지면 아무것도 찍히지 않은 채 0ns로 읽히고,
  같으면 끝 샘플이 시작 샘플을 덮는다.
- 목적지 버퍼는 `GPUBufferUsage.QUERY_RESOLVE`로 만들어야 하고 `destinationOffset`은
  **256의 배수**여야 한다 (명세 요구 — Metal은 더 느슨해서 여기서 직접 막는다).

타임스탬프는 패스 경계에서 찍는다. **기기 조건이 붙으므로 만들기 전에 확인할 것**:

```js
if (adapter.features.has('timestamp-query')) {
  const stamps = device.createQuerySet({ type: 'timestamp', count: 2 })
  encoder.beginRenderPass({
    colorAttachments,
    timestampWrites: { querySet: stamps, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 },
  })
  // 컴퓨트 패스도 같은 모양: encoder.beginComputePass({ timestampWrites })
}
```

- 값은 GPU 시계 눈금이다. **차이만 의미가 있고 절대값은 의미가 없다.**
- 지원하지 않는 기기에서 `type: 'timestamp'`로 만들면 `unsupported` 오류가 난다.
- **컴퓨트 패스 타임스탬프는 Apple GPU에서 신뢰할 수 없다.** 샘플 자체는 찍히지만
  `resolveQuerySet`이 쓰는 GPU 측 resolve 경로가 같은 코드에서도 0을 돌려줄 때가 있다
  (드라이버 사정이라 이 구현이 고칠 수 없다). 프레임 계측이 목적이면 **렌더 패스 쪽을 쓸 것** —
  그쪽은 안정적이다.

### 렌더 번들

매 프레임 똑같은 드로우 묶음을 다시 기록하지 않도록 한 번 기록해 두고 재사용한다.

```js
const bundleEncoder = device.createRenderBundleEncoder({
  colorFormats: [format],          // 이 번들을 **실행할 패스의 모양**
  depthStencilFormat: 'depth32float',   // 패스에 깊이가 없으면 생략
  sampleCount: 1,
})
bundleEncoder.setPipeline(pipeline)
bundleEncoder.setBindGroup(0, bindGroup)
bundleEncoder.draw(3)
const bundle = bundleEncoder.finish()            // createRenderBundle

// 매 프레임
pass.executeBundles([bundle])                    // executeBundles
```

- 번들에 담을 수 있는 것은 `setPipeline` `setBindGroup` `setVertexBuffer` `setIndexBuffer`
  `draw` `drawIndexed` `drawIndirect` `drawIndexedIndirect` **여덟 개뿐**이다. 뷰포트·시저·
  블렌드 상수·스텐실 참조·중첩 번들·복사는 담을 수 없다 — 번들 인코더에 그 메서드가 아예 없다.
- **상태는 양방향으로 격리된다.** 파이프라인·바인드 그룹·정점 버퍼·인덱스 버퍼 네 가지가
  대상이다. 번들은 패스가 지정해 둔 것을 물려받지 않고, 번들 실행이 끝나면 패스 쪽 바인딩도
  무효화된다. 이어서 그리려면 `setPipeline`·`setBindGroup`·`setVertexBuffer`를 **다시 해야
  한다** (이전 값으로 **복원되는 것이 아니다** — 명세 계약). 뷰포트·시저·블렌드 상수·스텐실
  참조는 그대로 남는다.
  > Metal 인코더에는 "바인딩 해제"가 없어서, 격리는 이 구현이 **드로우 직전에 직접 확인**해
  > 성립시킨다 — 파이프라인이 요구하는 바인드 그룹·정점 버퍼 슬롯이 빠져 있으면 그 드로우를
  > 거부한다. 그러지 않으면 브라우저에서 무효인 코드가 여기서만 조용히 그려진다.
- `colorFormats`(와 `depthStencilFormat`·`sampleCount`)가 실제 패스와 어긋나면
  `executeBundles`에서 오류다. `colorFormats`의 **후행 `null`은 무시**하고 비교한다
  (`['bgra8unorm', null]`은 컬러 1개짜리 패스와 맞는다 — 명세의 레이아웃 동치 규칙).
- 어태치먼트가 **최소 하나** 있어야 한다 — `colorFormats`에 non-null 하나이거나
  `depthStencilFormat`이거나. 둘 다 없으면 번들을 만들 때 거부한다.
- `depthReadOnly` / `stencilReadOnly` 패스에서 실행하려면 번들도 같은 플래그를 `true`로 두고
  만들어야 한다. 반대(쓰기 가능 패스에 read-only 번들)는 제약이 없다.
- `finish()`는 **한 번만** 부를 수 있다. 두 번째 호출은 오류를 내고 무효한 번들을 돌려준다.
- 번들은 자기가 쓰는 리소스 래퍼를 붙잡는다 — 초기화 함수가 번들만 반환해도 안전하다
  (`docs/JS-AUTHORING.md` §8). 단 **명시적 `destroy()`는 별개다.**

> **이 구현에서 번들이 무엇을 아껴 주나.** 브라우저는 드라이버 명령을 미리 만들어 두지만,
> 여기서는 Metal에 대응 객체가 없어 명령 목록을 저장했다가 되풀이한다. 그래서 이득은
> GPU 쪽이 아니라 **JS 쪽**이다 — 매 프레임 같은 커맨드 배열을 다시 만들고 브리지로
> 실어 보내는 비용이 핸들 하나로 줄어든다.

### 간접 드로우 · 디스패치

드로우 인자를 CPU가 아니라 **GPU 버퍼에서** 읽는다. 컴퓨트가 "몇 개를 그릴지"를 계산해
그대로 넘길 수 있다 — 개수를 알려고 CPU로 되읽는 왕복이 사라진다.

```js
const args = device.createBuffer({
  size: 16,
  usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE,   // 컴퓨트가 쓰고 드로우가 읽는다
})
// 같은 배치 안에서 컴퓨트 → 드로우 순서가 그대로 실행 순서다.
computePass.dispatchWorkgroups(1)     // args에 [vertexCount, instanceCount, …]를 쓴다
pass.drawIndirect(args, 0)
```

버퍼에 들어가는 `u32` 배열은 명세 그대로다:

| 명령 | 인자 (`u32`) | 크기 |
|---|---|---|
| `drawIndirect` | `vertexCount, instanceCount, firstVertex, firstInstance` | 16B |
| `drawIndexedIndirect` | `indexCount, instanceCount, firstIndex, baseVertex`(부호 있는 `i32`)`, firstInstance` | 20B |
| `dispatchWorkgroupsIndirect` | `x, y, z` | 12B |

- 버퍼는 `GPUBufferUsage.INDIRECT`로 만들어야 한다. Metal에는 이 개념이 없어 확인해 주지
  않으므로 이 구현이 직접 막는다 — 안 막으면 여기서만 돌고 브라우저에서 깨진다.
- `indirectOffset`은 **4의 배수**여야 하고, 인자 크기만큼이 버퍼 안에 들어가야 한다.
- `drawIndexedIndirect`의 `firstIndex`는 인자 버퍼 안에 있으므로,
  `setIndexBuffer(buffer, format, offset)`의 오프셋과 **더해지지 않고 따로** 적용된다.
- **`firstInstance`를 0이 아닌 값으로 쓰려면 `indirect-first-instance` 기능이 필요하다.**
  이 구현은 Metal이 `baseInstance`를 그대로 존중하므로 그 기능을 항상 보고한다
  (`adapter.features.has('indirect-first-instance')` → `true`). 하지만 브라우저에서는
  기능을 **요청하지 않으면 그 드로우가 통째로 no-op**이 되므로, 옮길 코드라면
  `requiredFeatures`에 넣어 두거나 `firstInstance`를 0으로 둘 것. 인자 값이 GPU 버퍼 안에
  있어 인코딩 시점에는 검사할 수 없다 — 실패가 "아무것도 안 그려짐"으로만 나타난다.

## 7. 오류 처리

`queue.submit()`은 `{ ok, errors, canvases, objects }`를 돌려준다. 오류는 실행을 멈추지 않고
누적된다. `objects`는 네이티브에 살아 있는 GPU 객체 수 — 프레임마다 늘고 있다면 어딘가에서
`destroy()`를 빼먹은 것이다 (`docs/JS-AUTHORING.md` §8).

```js
device.onError((error, text) => console.error(text))
// error = { kind: 'validation' | 'unsupported' | 'out-of-memory' | 'backend',
//           message: '…', path: 'commands[3].vertex.buffers[0].format' }
```

명세의 통로도 그대로 쓸 수 있다 — 웹 코드를 옮길 때 이름을 안 바꿔도 된다:

```js
device.onuncapturederror = (event) => console.error(event.error.message)
device.addEventListener('uncapturederror', (event) => { /* … */ })   // 같은 이벤트
```

- `event.error`는 `GPUValidationError` · `GPUOutOfMemoryError` · `GPUInternalError` 중 하나다
  (`instanceof`로 갈린다). `unsupported`는 `GPUValidationError`로 접힌다 — 스코프 필터와 같은 규칙.
- 명세에 없는 `kind`·`path`도 함께 실린다. 커맨드 스트림은 "몇 번째 명령의 어느 필드"까지
  알고 있고, 그걸 버리면 진단이 크게 나빠진다.
- **두 통로는 함께 동작한다.** 둘 다 등록하면 둘 다 받고, **아무도 안 듣고 있을 때만**
  `console.error`로 떨어진다 (조용히 사라지는 오류도, 두 번 찍히는 로그도 없게).
- 리스너가 던져도 나머지 리스너와 다음 오류는 계속 간다.

| kind | 뜻 |
|---|---|
| `validation` | 잘못된 인자·상태. 대부분 호출 측 버그다. |
| `unsupported` | 명세상 유효하지만 이 구현이 아직 안 하는 것. |
| `out-of-memory` | 리소스 생성 실패. |
| `backend` | Metal/셰이더 컴파일 오류. 셰이더 실패 시 **생성된 MSL이 메시지에 포함**된다. |

### 오류 스코프

특정 구간의 오류만 따로 받고 싶을 때 쓴다 — "이 셰이더가 컴파일되는지"처럼 **실패를 예상하고
분기하는** 코드에 필요하다.

```js
device.pushErrorScope('validation')                 // pushErrorScope
const pipeline = device.createRenderPipeline(descriptor)
const error = await device.popErrorScope()          // popErrorScope
if (error) fallBackToSimplePipeline()
```

- 필터는 `'validation'` · `'out-of-memory'` · `'internal'` 셋이다. 스코프는 **중첩**할 수 있고,
  오류는 **가장 안쪽의 맞는 스코프**가 가져간다. 필터가 안 맞으면 바깥으로 흘러간다.
- 스코프가 가져간 오류는 `submit()` 반환의 `errors`에도, `device.onError`에도 **가지 않는다**.
- 스코프가 돌려주는 것은 **처음 잡힌 오류 하나**다 (명세와 같다).
- 이 구현의 오류 분류는 넷이라 명세의 셋에 이렇게 붙인다:
  `unsupported`는 `validation`이 잡고(브라우저에서 같은 코드는 유효하거나 validation이다),
  `backend`(Metal/셰이더 컴파일 실패)는 `internal`이 잡는다.
- **스코프는 `submit()` 경계를 넘어 이어진다** — 디바이스 상태이기 때문이다.
  `push`와 `pop` 사이에 프레임이 몇 개 들어가도 된다.
- `popErrorScope()`는 `mapAsync`처럼 **즉시 제출한다** (안 그러면 다음 `submit()`까지 Promise가
  안 풀린다). 그래서 프레임 루프 안에서 부르면 왕복이 하나 는다 — 초기화·진단용 API다.
  왕복이 느는 것뿐이고 프레임을 깨지는 않는다 (`getCurrentTexture()`로 얻은 뷰는 `present`
  전까지 유효하다).
- **스코프는 `submit()`으로 넘어간 명령만 관측한다.** `GPUCommandEncoder`의 메서드는 명령을
  자기 배열에 모으고 `queue.submit()`에서야 네이티브로 넘어가므로, 인코더 명령을 감싸려면
  `pop` **전에** `submit()`을 부를 것. 디바이스 레벨 호출(`createRenderPipeline` 등)은
  기록 시점에 실행되므로 이 제약이 없다.
- `push`와 짝이 맞지 않는 `pop`은 **Promise가 `OperationError`로 reject된다** (명세와 같다).
  이 실패는 GPUError가 아니므로 `onError`로 가지 않는다 — 그래야 "스코프가 깨끗했다(`null`)"와
  "짝이 안 맞았다"를 구분할 수 있다.
- 알 수 없는 필터는 **동기 `TypeError`**다 (브라우저의 WebIDL enum 변환과 같은 자리).

## 8. 미지원 목록

| 기능 | 왜 없나 |
|---|---|
| `GPUExternalTexture` (`importExternalTexture`) | Lynx에 **비디오** 엘리먼트 핸들이 없다. 다만 WGSL `texture_external` + `textureSampleBaseClampToEdge`는 **지원**하므로, 프레임을 텍스처로 올려 그 자리에 `GPUTextureView`를 묶으면 된다. 정지 이미지는 `createImageBitmap()`이 그 길이다 (§3) |
| `writeTimestamp` | Metal은 패스 경계에서만 카운터를 샘플링한다 — `timestampWrites`(§6)를 쓸 것 |
| `device.lost`의 **유실 통지** | iOS/macOS에는 디바이스 손실에 해당하는 사건이 사실상 없다. **속성 자체는 있다** — 웹 코드(`Three.js WebGPUBackend.init()` 등)가 `device.lost.then(...)`을 걸어도 TypeError가 나지 않도록 영원히 pending인 Promise를 준다. **GPU 실행 자체의 실패**(`.outOfMemory`·`.timeout` 등)는 다음 `submit()` 응답에 `backend` 오류로 실려 나온다 |
| `setBindGroup`의 `Uint32Array` 오버로드 | 동적 오프셋을 `(data, start, length)`로 넘기는 형태. 배열 형태(`setBindGroup(i, group, [o1, o2])`)와 결과가 같고, 브리지를 건널 때 어차피 배열로 펴진다 — 두 경로를 두면 검증만 두 배가 된다 |
| `GPURenderPassColorAttachment.depthSlice` | 3D 텍스처의 한 슬라이스를 렌더 타깃으로 삼는 값. 지금은 **읽지 않으므로 항상 0번 슬라이스**에 그린다. 3D 렌더 타깃을 쓰게 되면 그때 넣는다 (2D 배열은 `baseArrayLayer`로 이미 된다) |
| `GPURenderPassDescriptor.maxDrawCount` | 드라이버에 주는 검증 힌트다. 무시해도 그림은 같고, 넘겼을 때의 거부는 명세상 선택이다 |
| `GPUCanvasContext.canvas` | 명세는 `HTMLCanvasElement`를 돌려준다 — Lynx에는 대응하는 엘리먼트 객체가 없다. `context.canvasId`(문자열)가 그 자리다 |
| `setImmediates` (immediate data) | 명세에 최근 들어온 기능이고 `maxImmediateSize` 한계도 따라온다. 유니폼 버퍼로 대체할 수 있어 미룬다 |

### 선택 기능 (`adapter.features`에 광고하지 않는다)

명세가 **선택**으로 둔 것들이라 없어도 적합하다. 쓰려면 `adapter.features.has(...)`로 확인하는
웹 코드가 알아서 다른 길로 간다 — 문제는 "왜 없는지"가 안 적혀 있는 것이라 여기 남긴다.

| 기능 | 상태 |
|---|---|
| `texture-formats-tier1` / `tier2` | 16비트 unorm/snorm 6종(`r16unorm` `rgba16snorm` …)과 추가 스토리지 텍스처 조합이 여기 묶여 있다. Metal에는 포맷이 있으므로 **필요해지면 매핑만 추가하면 된다** — 지금은 쓰는 곳이 없어 광고하지 않는다 |
| `depth-clip-control` (`unclippedDepth`) | Metal의 `depthClipMode`로 대응할 수 있다. 그림자 볼륨처럼 깊이 클리핑을 끄는 기법을 쓰게 되면 넣는다 |
| `shader-f16` | WGSL `f16`. 트랜스파일러는 `f16`→`half`를 이미 옮기지만(`enable f16` 포함), 명세가 요구하는 기능 선언·한계값 검증을 하지 않아 광고하지 않는다 |
| `float32-filterable` / `float32-blendable` | `r32float` 계열의 필터링·블렌딩. Apple GPU는 지원하지만 광고 전에 확인 경로를 만들어야 한다 |
| `bgra8unorm-storage` | `bgra8unorm`에 `STORAGE_BINDING`. 이미지 처리에서 쓸 만해 우선순위가 있는 편이다 |
| `dual-source-blending` · `clip-distances` · `subgroups` · `subgroup-size-control` · `primitive-index` · `texture-component-swizzle` | WGSL 확장이 함께 필요하다 (`@blend_src`, `clip_distances`, 서브그룹 내장 함수 …). 트랜스파일러 작업이 붙으므로 실사용 요구가 생길 때 본다 |
| `rg11b10ufloat-renderable` | `rg11b10ufloat`을 렌더 타깃으로. 포맷 자체는 지원한다 |

### 프레임 경계 — present는 `submit()`이 아니라 **프레임 루프 콜백의 끝**이다

명세는 현재 텍스처의 만료를 "updating the rendering of a WebGPU canvas"에 건다 — 즉
**프레임의 끝**이다. `queue.submit()`은 present하지 않는다. 그래서 한 프레임 안에서 패스를
여러 개 도는 코드(three.js `PostProcessing`: 씬 패스 → bloom 밉 체인 → 출력 패스)가
드로어블 텍스처를 그 프레임 내내 공유할 수 있다.

여기서도 같다:

- **프레임 루프 안**(`startFrameLoop` · `installAnimationFrame`의 rAF)에서는 `submit()`이
  명령만 보내고 present는 **틱의 끝에 한 번**만 한다. 콜백이 던져도 present는 나간다 —
  안 그러면 화면이 그 프레임에서 멈춘다.
- **루프 밖**에서는 `submit()`이 그대로 present한다. 미룰 경계가 없기 때문이다
  (오프스크린 렌더·테스트 하네스가 이 경로다).

`getCurrentTexture()`도 명세대로다 — **만료 전까지 같은 텍스처를 돌려준다.**
만료는 present · `configure()` · 캔버스 리사이즈에서 일어난다. 설정 전에 부르면
`InvalidStateError`다.

이 둘을 맞추고 나서 three.js의 `PostProcessing`을 **캔버스로 직접** 내보내는 경로가 열렸다
(`threelab` 데모 씬 — 씬 패스 → bloom 밉 체인 → 출력 패스가 한 프레임에 돌고 60fps·오류 0).
그전에는 첫 제출이 드로어블을 내보내고 뷰를 만료시켜 남은 패스가 통째로 거부됐다.

> 프레임 티커는 **참조로 센다.** 한 씬이 잠깐 `startFrameLoop`을 열었다 닫아도 rAF 펌프까지
> 같이 멈추지 않는다 — 그러면 화면이 조용히 정지한다.

### 기기에 따라 갈리는 것

| 기능 | 조건 |
|---|---|
| 타임스탬프 쿼리 | `adapter.features.has('timestamp-query')`. **컴퓨트 패스 쪽 값은 신뢰할 수 없다** (§6) |
| **간접 드로우·디스패치 자체** | Metal이 **Apple GPU family 3 이상**을 요구한다. 실기기는 iOS 17 최소 사양(A12 = family 5)이라 **항상 지원**하지만, **iOS 시뮬레이터는 family 2로 보고해 빠진다.** 지원하지 않으면 `unsupported` 오류로 거부하고 `adapter.features`의 `indirect-first-instance`도 감춘다 — 그대로 Metal에 넘기면 `MTLValidateFeatureSupport … failed assertion`으로 **프로세스가 죽는다** |
| 간접 드로우의 `firstInstance ≠ 0` | 지원 기기에서는 항상 되고 `adapter.features`에도 `indirect-first-instance`로 보고된다. **브라우저에서는 기능을 요청해야** 한다 (§6) |
| 캔버스 EDR 출력 (`toneMapping: 'extended'`) | 실기기 디스플레이 기능 — 시뮬레이터에서는 확인되지 않는다 (§2) |
| 블록 압축 텍스처 | `texture-compression-astc` / `-etc2` / `-bc`. iOS 기기는 ASTC·ETC2를 항상 하고 BC는 A14부터, Intel Mac은 그 반대다 (§3) |

새 명령을 추가하는 절차는 `.claude/skills/webgpu-command/SKILL.md` 참고.
