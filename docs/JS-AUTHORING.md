# Lynx 번들(JS)에서 WebGPU 쓰기

## 1. 설치

`JS/` 아래 세 파일을 rspeedy 프로젝트의 `src/`로 복사한다:

```
src/
├── webgpu.js       — 클라이언트 shim (런타임)
├── webgpu.d.ts     — 타입 선언 (shim의 JSDoc에서 생성된다 — 손으로 고치지 말 것)
└── elements.d.ts   — <webgpu-canvas> TSX 선언
```

`JS/lynx-env.d.ts`와 `JS/tsconfig.json`은 **복사하지 않는다.** 이 라이브러리 자신의 타입 검사에만
쓰는 파일이고, 번들 쪽 프로젝트에 넣으면 `@lynx-js/types`의 전역 선언과 겹칠 수 있다.

```tsx
import gpu, { GPUBufferUsage, GPUTextureUsage, GPUShaderStage, startFrameLoop } from './webgpu.js'
import './elements.d.ts'
```

호스트 앱이 `LynxWebGPU.register(in:host:)` + `host.attach(to:)`를 해 두어야 한다
(`docs/LYNX-INTEGRATION.md`). 안 되어 있으면 첫 호출에서
`NativeModules.WebGPU 를 찾을 수 없다` 오류가 난다.

## 2. 최소 골격

```tsx
<webgpu-canvas canvas-id="main" style={{ width: '100%', height: '100%' }} />
```

```js
const adapter = await gpu.requestAdapter()
const device  = await adapter.requestDevice()
device.onError((e, text) => console.error(text))     // 켜 두면 디버깅이 훨씬 빠르다

const context = gpu.getCanvasContext('main')
const format  = gpu.getPreferredCanvasFormat()
context.configure({ device, format })
```

이후는 브라우저 WebGPU와 같다. 전체 예제는 `Examples/HelloTriangle.tsx`.

## 3. 브라우저 WebGPU와 다른 점

| 항목 | 브라우저 | 여기 |
|---|---|---|
| 캔버스 얻기 | `canvas.getContext('webgpu')` | `gpu.getCanvasContext('main')` — `<webgpu-canvas canvas-id>` 이름 |
| 프레임 루프 | `requestAnimationFrame` | `startFrameLoop(handler)` (§4) |
| 버퍼 읽기 | `mapAsync` + `getMappedRange` | `await buffer.mapAsync()` 가 ArrayBuffer를 바로 돌려준다 |
| 오류 | `pushErrorScope` / `uncapturederror` | `device.onError()` + `submit()` 반환의 `errors` |
| 캔버스 크기 | `canvas.width/height` | `context.getSize()` (제출 응답으로 캐시 갱신) 또는 `bindcanvasresize` |
| 애셋 가져오기 | `fetch()` / `<img>` | `await loadAsset(name)` → `ArrayBuffer` (등록 이름·파일 경로·번들 상대 경로 — 해석은 호스트의 `assetProvider`가 정한다) |
| HDR 출력 | `toneMapping: { mode }` | 같음 — `configure`에 넘긴다 (`docs/WEBGPU-API.md` §2) |
| 미지원 기능 | — | `docs/WEBGPU-API.md` §8 |

셰이더(WGSL)는 `docs/WGSL.md`의 서브셋 안이면 그대로 옮겨진다.

## 4. 프레임 루프

`setInterval`은 화면 갱신과 어긋나 프레임이 뭉치거나 버려진다. 네이티브 `CADisplayLink`가
몰아 주는 `startFrameLoop`를 쓴다.

```js
const stop = startFrameLoop(({ timestamp, delta }) => {
  // delta: 직전 프레임과의 간격(ms)
  render(delta)
}, { fps: 60 })

// 페이지를 떠날 때 반드시
stop()
```

`stop()`을 부르지 않으면 디스플레이 링크가 계속 돌며 배터리를 먹는다.
ReactLynx라면 `useEffect`의 정리 함수에서 부를 것.

## 5. 성능 — 프레임당 브리지 왕복 1회

이 구현의 핵심 계약은 **`queue.submit()`이 한 프레임의 모든 명령을 한 번에 넘긴다**는 것이다.
그 계약을 깨는 호출은 프레임 안에서 피한다:

| 프레임 안에서 피할 것 | 이유 | 대안 |
|---|---|---|
| `buffer.mapAsync()` | GPU 완료를 기다린다 | 결과가 필요한 프레임에만 |
| `gpu.requestAdapter()` | 동기 네이티브 호출 | 초기화에서 1회 |
| `device.createRenderPipeline()` | 셰이더 컴파일이 붙는다 | 초기화에서 1회 |

반대로 **얼마든지 해도 되는 것**: `setPipeline`, `setBindGroup`, `draw`, `writeBuffer`,
`drawIndirect`/`drawIndexedIndirect`/`dispatchWorkgroupsIndirect`(핸들과 오프셋만 싣는다 —
드로우 인자는 GPU 버퍼 안에 있으므로 CPU가 읽지 않는다),
그리고 `context.getSize()`/`getCurrentTexture()` — 앞의 것들은 JS 배열에 push만 하고
submit에서 한 번에 나가고, 크기는 **제출 응답으로 갱신되는 캐시**를 읽는다
(동기 조회는 `configure` 시점 1회뿐이다). 리사이즈는 다음 제출 응답에 반영되므로
한 프레임 늦을 수 있다 — 즉시성이 필요하면 `bindcanvasresize`를 쓴다.

### 바이트열 올리기·내리기

바이트열은 **`ArrayBuffer`로 그대로** 브리지를 건넌다 (Lynx가 `NSData`로 바꿔 준다).
문자열 인코딩이 없으므로 팽창도 인코딩 루프도 없다 — `writeBuffer`/`writeTexture`/`mapAsync`
모두 같다. `writeTexture`/`writeBuffer`는 GPU 완주를 기다리지 않고 큐 순서를 탄다.

데모 `bench` 씬으로 잰 업로드 비용 (Apple Silicon, 페이로드를 만드는 비용 + `execute` 왕복):

| 페이로드 | 비용 | 60fps 프레임 예산(16.7ms) 대비 |
|---|---|---|
| 유니폼 1KB | 0.112 ms | 0.7% |
| 동적 텍스처 128×128 RGBA (64KB) | 0.148 ms | 0.9% |
| 텍스처 256×256 RGBA (256KB) | 0.188 ms | 1.1% |
| 정점 스트림 1MB | 0.250 ms | 1.5% |

**크기에 거의 비례하지 않는다** — 1KB와 1MB의 차이가 두 배 남짓이다. 브리지 왕복 자체가
비용의 대부분이고 바이트 전송은 memcpy라서다. 그래서 업로드 크기보다 **왕복 횟수**를
줄이는 것이 여전히 중요하다 (그래서 프레임당 1회 계약이 있다).

수 MB를 매 프레임 올리는 것도 이제 프레임 예산 안에 들어오지만, 그 정도라면 애초에
스토리지 버퍼 + 컴퓨트 셰이더로 GPU 안에서 갱신하는 편이 낫다 — 브리지를 아예 안 건넌다.

`TypedArray`를 손으로 커맨드에 실으면 안 된다 — Lynx는 진짜 `ArrayBuffer`만 알아보고 뷰는
평범한 객체로 바꿔 **오류 없이 조용히** 깨진다. 셰임의 `writeBuffer`/`writeTexture`에 넘기면
알아서 풀어 주므로 신경 쓸 일은 없다.

## 6. 캔버스 크기 다루기

드로어블 크기는 **화면 배율이 곱해진 픽셀 값**이다 (CSS px이 아니다).

```tsx
<webgpu-canvas
  canvas-id="main"
  pixel-ratio={1}                 /* 부하가 크면 1로 낮춘다 (3배 → 1배면 픽셀 수 1/9) */
  bindcanvasresize={(e) => {
    const { width, height, pixelRatio } = e.detail
    aspect = width / height        // 투영행렬 갱신
  }}
/>
```

크기가 0인 동안(레이아웃 전) `getCurrentTexture()`는 오류를 낸다 — `getSize()`가 0을 주면
그 프레임은 그냥 건너뛰는 것이 맞다.

## 7. 유니폼 패킹

WGSL 배치 규칙대로 채우면 된다. 네이티브가 MSL 구조체를 WGSL 오프셋에 맞춰 패딩하므로,
브라우저에서 쓰던 패킹 코드가 그대로 동작한다 (`docs/WGSL.md` §3).

주의할 것은 WGSL 자체의 규칙이다:

```wgsl
struct Uniforms {
  mvp: mat4x4<f32>,   // offset 0,  size 64
  tint: vec4f,        // offset 64, size 16
  time: f32,          // offset 80
};                    // size 96 (16 정렬)
```

```js
const data = new Float32Array(24)   // 96 / 4
data.set(mvp, 0)
data.set(tint, 16)
data[20] = time
device.queue.writeBuffer(uniformBuffer, 0, data)
```

- `vec3f`는 정렬 16이다. `vec3f` 다음 `f32`는 offset 12에 붙지만, `vec3f` 다음 `vec3f`는 16으로 밀린다.
- uniform 주소 공간의 배열 원소는 스트라이드가 **16의 배수**여야 한다 — `array<f32, N>` 대신 `array<vec4f, N>`을 쓸 것.

## 8. 리소스 수명

핸들은 정수라 **JS GC가 네이티브 객체의 수명을 모른다.** 브라우저에서는 GC가 처리해 주던
"버리면 알아서 사라진다"가 여기서는 성립하지 않으므로, 규칙은 두 가지다:

- **초기화에서 만든 것은 페이지를 떠날 때** — `device.destroy()`가 전부 정리한다.
- **프레임마다 만드는 것은 만들지 말 것.** 매 프레임 `createView`/`createBindGroup`을 새로
  만드는 웹 관용구를 그대로 옮기면 네이티브 레지스트리가 무한히 자란다. 초기화 때 한 번
  만들어 재사용하거나, 정말 필요하면 프레임 끝에 `destroy()`를 부른다.
  (예외: `getCurrentTexture()`와 그 뷰는 프레임 스코프라 네이티브가 알아서 회수한다.)

새는지 감시하는 장치가 둘 있다:

- `submit()` 반환값의 `objects` — 네이티브에 살아 있는 객체 수다. 프레임을 거듭해도
  일정해야 정상이고, 계속 늘면 어딘가에서 destroy를 빼먹은 것이다.
- 네이티브 로그 — 4096개를 넘으면 (이후 두 배마다) `Registry` 카테고리로 경고를 남긴다.

엔진이 `FinalizationRegistry`를 지원하면 GC로 사라진 래퍼의 destroy를 자동으로 끼워 넣는
안전망이 켜진다. 단 PrimJS 지원 여부에 기대지 말 것 — **명시적 destroy가 정답**이고,
자동 해제는 놓친 것을 주워 담는 보조 장치다.

## 9. 디버깅

```js
device.onError((error, text) => {
  console.error(text)      // "[WebGPU:validation] commands[3].vertex.buffers[0].format — 알 수 없는 값 …"
})
```

- `path`가 커맨드 스트림 상 위치를 정확히 짚어 준다.
- 셰이더 컴파일 실패(`kind: 'backend'`)는 **생성된 MSL 전문**이 메시지에 들어 있다.
  WGSL이 어떻게 번역됐는지 그대로 볼 수 있다.
- 화면이 검게만 나오면: (1) `getSize()`가 0인지, (2) `configure` 했는지,
  (3) `submit()`을 불렀는지 순서로 확인한다.
