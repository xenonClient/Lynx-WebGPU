# Lynx-WebGPU 아키텍처

Lynx 렌더 엔진 위에 W3C WebGPU 모델을 얹어, Lynx 번들의 JS가 Metal GPU에 직접 접근하게 하는 SPM 라이브러리의 설계 문서.

참고 명세:
- [W3C WebGPU](https://www.w3.org/TR/webgpu/) · [WGSL](https://www.w3.org/TR/WGSL/)
- [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- [Lynx — Integrate with Existing Apps (iOS)](https://lynxjs.org/guide/start/integrate-with-existing-apps?platform=ios)

## 1. 문제 정의

Lynx는 `<view>`/`<text>`/`<image>` 같은 선언적 엘리먼트를 렌더한다. Lynx로 만든 화면에서
파티클, 3D, 이미지 필터, GPGPU처럼 **픽셀·연산 단위로 GPU를 직접 몰아야 하는 일**을 하려면
이 모델로는 부족하다.

브라우저는 같은 문제를 WebGPU로 푼다. 그래서 이 라이브러리는 **브라우저 WebGPU와 같은 API 모양**을
Lynx 위에 제공한다 — 웹에서 쓰던 셰이더와 렌더 코드를 거의 그대로 옮길 수 있게.

## 2. 개념 매핑 (WebGPU → Metal)

| WebGPU | 이 구현 | Metal |
|---|---|---|
| `GPUAdapter` / `GPUDevice` | `LynxWebGPUContext` | `MTLDevice` |
| `GPUQueue` | 컨텍스트가 소유 | `MTLCommandQueue` |
| `GPUBuffer` | `WGPUBufferObject` | `MTLBuffer` (`.storageModeShared`) |
| `GPUTexture` / `GPUTextureView` | `WGPUTextureObject` / `WGPUTextureViewObject` | `MTLTexture` (+ `makeTextureView`) |
| `GPUSampler` | `WGPUSamplerObject` | `MTLSamplerState` |
| `GPUShaderModule` | `WGPUShaderModuleObject` | `MTLLibrary` (파이프라인마다 방출) |
| `GPUBindGroupLayout` / `GPUPipelineLayout` | `WGPUBindGroupLayoutObject` / `WGPUPipelineLayoutObject` | 인자 인덱스 배정표 |
| `GPUBindGroup` | `WGPUBindGroupObject` | `setVertexBuffer`/`setFragmentTexture` … 묶음 |
| `GPURenderPipeline` | `WGPURenderPipelineObject` | `MTLRenderPipelineState` + `MTLDepthStencilState` |
| `GPUComputePipeline` | `WGPUComputePipelineObject` | `MTLComputePipelineState` |
| `GPUCommandEncoder` | 커맨드 스트림 (JS 측) | `MTLCommandBuffer` |
| `GPURenderPassEncoder` | 스트림 명령들 | `MTLRenderCommandEncoder` |
| `GPUCanvasContext` | `WGPUSurface` | `CAMetalLayer` / 오프스크린 `MTLTexture` |
| WGSL | `LynxWebGPUShader` | MSL |

## 3. 커맨드 스트림 — 이 설계의 중심

### 문제

WebGPU는 객체 지향 API다. 한 프레임에 `setPipeline`, `setBindGroup`, `setVertexBuffer`, `draw`가
수십~수백 번 불린다. 이걸 Lynx NativeModule 호출로 1:1 매핑하면 **프레임당 브리지 왕복이 수백 번**이 되어
GPU가 아니라 브리지가 병목이 된다.

데모 `bench` 씬으로 재 보면 **왕복 비용이 페이로드 크기에 거의 비례하지 않는다** — 1KB 업로드가
0.112ms, 1MB 업로드가 0.250ms로 두 배 남짓이다. 비용의 대부분이 왕복 자체에 있다는 뜻이고,
그래서 **크기를 줄이는 것보다 횟수를 줄이는 것**이 설계의 중심이 된다.

### 해법

JS 쪽에 **WebGPU 객체 그래프를 그대로 두되, 호출을 명령으로 기록만** 한다. `queue.submit()` 시점에
쌓인 명령 배열을 **한 번의 네이티브 호출**로 넘긴다. 네이티브는 그 배열을 순서대로 해석해 Metal로 인코딩한다.

```
JS                                             Native
──────────────────────────────────────────────────────────────────
device.createBuffer(…)      → [createBuffer id=1]
pass.setPipeline(p)         → [setPipeline …]        (아직 안 보냄)
pass.draw(3)                → [draw …]
queue.submit([cb])          ────── execute({commands}) ──────▶ 해석 → Metal 인코딩 → commit
                            ◀───── {ok, errors, canvases} ────
```

핵심은 **핸들을 JS가 발급한다**는 것이다. `createBuffer`가 네이티브 왕복을 기다리지 않으므로
생성과 사용을 한 배치에 이어서 기록할 수 있다. Chrome의 WebGPU가 Dawn wire로 GPU 프로세스와
통신하는 구조와 같은 아이디어다.

부수 효과로 얻는 것:
- **오류 누적** — 명령 하나가 실패해도 나머지를 계속 실행하고, 오류를 배열로 모아 돌려준다.
  WebGPU가 잘못된 호출로 컨텍스트를 죽이지 않는 것과 같은 계약이다.
- **경로 진단** — 오류에 `commands[3].vertex.buffers[0].format` 같은 위치가 붙는다.
- **결정적 재현** — 커맨드 배열은 순수 데이터라 로그로 남기고 그대로 재생할 수 있다.

### 스레딩

Lynx는 NativeModule 메서드를 **JS 백그라운드 스레드**에서 부른다. 커맨드 해석과 Metal 인코딩은
그 스레드에서 그대로 한다 — Metal은 메인 스레드를 요구하지 않고, 메인으로 넘기면 UI 작업과 경쟁해
프레임만 늦어진다.

예외는 `CAMetalLayer` 프로퍼티(포맷·크기·불투명도)로, 메인 스레드 전용이다. JS 스레드에서
`DispatchQueue.main.sync`를 걸면 메인이 Lynx 런타임을 기다리는 순간 교착이 나므로 **비동기로** 넘기고,
`getCurrentTexture`는 캐시된 설정 대신 **실제 드로어블 텍스처의 포맷**을 보고한다.

## 4. 셰이더 파이프라인 (WGSL → MSL)

```
WGSL 소스
   │  WGSLLexer            토큰
   │  WGSLParser           구문 트리 (타입 추론기가 아니라 구문 번역기다)
   │  WGSLReflectionBuilder 진입점 / 바인딩 / 전역 사용 관계(호출 그래프 고정점)
   ▼
WGSLShaderModule           ← createShaderModule 이 만드는 것 (여기까지가 Metal-free)
   │
   │  파이프라인 생성 시점: 레이아웃 → WGSLBindingAssignment
   │  MSLEmitter
   ▼
MSL 소스 → MTLLibrary → MTLFunction
```

### 왜 MSL 방출을 파이프라인 생성까지 미루나

WGSL은 `@group(g) @binding(b)`라는 2차원 좌표를 쓰고, Metal은 스테이지별 1차원 인덱스
(`[[buffer(n)]]`, `[[texture(n)]]`, `[[sampler(n)]]`)를 쓴다. 좌표 → 인덱스 변환은 **파이프라인 레이아웃**이
정해져야 결정된다. 그래서 셰이더 모듈은 파싱 결과만 들고 있다가, 파이프라인이 만들어질 때
그 레이아웃으로 MSL을 방출하고 결과를 캐시한다 (`(진입점 조합, 배정 서명)` 키).

인덱스 배정 규칙 (`WGSLMetalLimits`):
- 바인드 그룹 리소스: 그룹 → 바인딩 오름차순으로 종류별 카운터를 0부터 올린다.
- **정점 버퍼는 테이블 위쪽(30번)부터 역순.** 둘이 만나기 전까지 충돌하지 않는다 (Dawn과 같은 규칙).

### 두 가지 구조적 변환

**(1) 리소스 스레딩** — MSL에는 가변 전역이 없다. WGSL의 모듈 스코프 변수(유니폼/스토리지/텍스처/
샘플러/`var<workgroup>`/`var<private>`)를 진입점 인자로 받아 **호출 그래프를 따라 함수 인자로 내려보낸다.**
어느 함수가 무엇을 쓰는지는 호출 그래프 고정점 계산으로 구하되, 수집이 **스코프를 따진다** —
매개변수·지역 선언에 가려진 이름은 전역 사용이 아니다 (Three.js처럼 같은 이름을 모듈과 지역에
기계 생성하는 코드가 흔하다). 전역이 정당하게 주입된 함수 안에서 같은 이름의 지역 선언이 나오면
(WGSL은 합법, C++에서는 매개변수 재정의) 방출기가 지역을 `wgpu_shadow_*`로 리네임한다.

```wgsl
@group(0) @binding(0) var<uniform> u: Uniforms;
fn tint(c: vec3f) -> vec3f { return c * u.color.rgb; }   // 헬퍼가 유니폼을 읽는다
```
```metal
float3 tint(float3 c, constant Uniforms& u) { return c * u.color.rgb; }
vertex … vs_main(…, constant Uniforms& u [[buffer(0)]]) { … tint(color, u) … }
```

**(2) 진입점 래핑** — WGSL 진입점의 시그니처를 그대로 둔 `…_inner` 함수를 만들고, 스테이지 I/O 속성
(`[[attribute]]`/`[[user(locn0)]]`/`[[position]]`/`[[color(0)]]`)은 **전용 래퍼 구조체**에만 붙인다.
같은 구조체를 유니폼 버퍼와 정점 I/O에 동시에 쓰는 셰이더에서 속성이 버퍼 레이아웃을 오염시키지 않게 하기 위해서다.

### vec3 배치 문제

WGSL `vec3<f32>`는 정렬 16 / **크기 12**, MSL `float3`는 크기 **16**이다. 그래서

```wgsl
struct Light { direction: vec3f, intensity: f32 }   // WGSL: 16바이트
```
를 그대로 `struct { float3 direction; float intensity; }`로 옮기면 32바이트가 되어, JS가 WGSL 규칙으로
채운 유니폼 버퍼를 잘못 읽는다.

`WGSLLayout`이 WGSL 명세대로 오프셋을 계산하고, 방출기가
- 뒤 멤버가 12~15바이트 구간을 쓰면 → `packed_float3`(크기 12)
- 아니면 → `float3` + 명시적 `char` 패딩

으로 **WGSL 배치를 정확히 재현**한다. uniform 주소 공간에서 구조체/배열 정렬이 16으로 올라가는 규칙도 반영한다.

## 5. 캔버스 표면

`WGPUSurface` 프로토콜 하나로 두 구현을 감춘다:

- `WGPUMetalLayerSurface` — `<webgpu-canvas>` 엘리먼트의 `CAMetalLayer`. 화면 출력.
- `WGPUOffscreenSurface` — 텍스처. **테스트 하네스가 렌더 결과를 픽셀로 읽는 용도.**

JS는 문자열 id(`canvas-id` prop)로 표면을 지목한다. 엘리먼트가 화면에 붙을 때 컨텍스트에 등록하고,
사라질 때 해제한다.

`present`는 명시적 명령이 아니다. WebGPU와 마찬가지로, 한 배치 안에서 `getCurrentTexture()`로 얻은
드로어블은 **배치가 끝날 때 자동으로 present** 된다. 드로어블 텍스처와 그 뷰의 핸들은 프레임이 끝나면
회수된다 (브라우저와 같은 규칙 — 프레임 밖에서는 유효하지 않다).

## 6. 프레임 루프

JS `setInterval`로 프레임을 돌리면 화면 갱신과 어긋나 프레임이 뭉치거나 버려진다.
네이티브가 `CADisplayLink`로 몰아 주고 `webgpu:frame` 전역 이벤트로 JS를 깨우는 편이 훨씬 고르다.
`startFrameLoop(handler)`가 이 경로를 쓰고, `GlobalEventEmitter`가 없는 환경에서는 타이머로 폴백한다.

### 백프레셔 — in-flight 프레임 제한

GPU가 프레임을 소화하지 못하고 밀리면 `CAMetalLayer`의 드로어블 풀(3개)이 고갈되고,
`nextDrawable()`이 **JS 스레드 전체를 최대 1초까지 세운다** — 캔버스뿐 아니라 그 페이지의
터치 핸들러·타이머·네트워크 콜백까지 함께 멈추는 최악의 백프레셔다.

그래서 표면마다 in-flight 카운터를 둔다. 드로어블을 실은 커맨드 버퍼가 커밋될 때 올리고
완료 핸들러에서 내린다 (`WGPUMetalLayerSurface.maxFramesInFlight = 3`). 카운터가 상한에
닿은 표면이 있으면 **프레임 티커가 그 틱을 통째로 건너뛴다** — JS는 깨어나지도 않으므로
블록될 일이 없고, GPU가 완료를 돌려주면 다음 틱부터 자연히 재개된다. 화면에는 프레임 드랍으로
보인다 (밀린 프레임을 기다렸다 몰아서 그리는 것보다 낫다).

티커 없이 직접 프레임을 만드는 코드는 이 게이트를 지나지 않는다 — 그 경우의 백프레셔는
이전과 같이 `nextDrawable()` 블로킹이다.

## 7. 동시성

Swift 5 언어 모드를 쓴다. GPU 객체 그래프(`MTLBuffer`, `LynxUI` 등)는 Sendable이 아닌 참조 타입을
JS 스레드와 메인 스레드가 공유하므로, 컴파일러 격리 대신 **명시적 락**으로 보장한다:

| 자원 | 보호 | 접근 스레드 |
|---|---|---|
| `WGPUObjectRegistry` (핸들 → 객체) | `NSLock` | JS(생성/조회), 메인(해제) |
| `LynxWebGPUContext.surfaces` | `NSLock` | JS(조회), 메인(등록/해제) |
| 커맨드 실행 | `executionLock` — 한 번에 하나 | JS |
| `WGPUMetalLayerSurface.cachedSize` | `NSLock` | 메인(쓰기), JS(읽기) |
| `WGPUShaderModuleObject.libraryCache` | `NSLock` | JS |

락 구간에서는 딕셔너리 조작만 하고 GPU 작업은 하지 않는다.

## 8. 의도적으로 하지 않은 것

- **런타임 검증 전부** — WebGPU 명세의 사용 플래그·포맷 호환성 검사를 다 하지는 않는다.
  핸들 존재/타입, 범위, 파이프라인 상태처럼 **크래시로 이어지는 것**은 막고, 나머지는 Metal 검증 레이어에 맡긴다.
- **WGSL 타입 추론기** — 구문 번역기다. MSL이 C++ 기반이라 대부분의 연산자·생성자 의미가 그대로 옮겨진다.
  옮길 수 없는 것은 조용히 틀리게 만들지 않고 명시적으로 거부한다 (`docs/WGSL.md` §4).
- **간접 드로우 / 쿼리 / 타임스탬프** — 필요해지면 커맨드 몇 개를 더하면 된다 (`.claude/skills/webgpu-command`).
