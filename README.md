# Lynx-WebGPU

**한국어** · [English](README.en.md)

[Lynx](https://lynxjs.org) 번들의 JS에서 **WebGPU 모양으로 GPU에 접근**하게 해 주는 SPM 라이브러리.
[W3C WebGPU](https://www.w3.org/TR/webgpu/)의 객체 모델과 [WGSL](https://www.w3.org/TR/WGSL/)을 Metal로 옮긴다.

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
  device.queue.submit([encoder.finish()])    // ← 프레임 전체가 브리지를 한 번만 건넌다
})
```

브라우저 WebGPU 코드와 셰이더(WGSL)를 거의 그대로 옮길 수 있다.

## 시작하기

```swift
// Package.swift
.package(url: "https://github.com/xenonClient/Lynx-WebGPU", from: "0.5.0")
```

이것으로 **엔진**(`LynxWebGPU`)이 들어온다. 아래 `LynxWebGPUHost`처럼 Lynx와 맞물리는 타입은
`Sources/LynxWebGPUBridge/`의 **네 파일을 앱 쪽 타깃에 넣어** 쓴다 — Lynx SDK의 버전·배포처를
앱이 정할 수 있게 하기 위해서다 (`docs/LYNX-INTEGRATION.md` §2, 데모가 그 본보기다).

호스트 앱 연동은 세 단계다. **런타임(GPU 백엔드)은 앱이 넣는다** — 브리지는
`WebGPURuntime` 프로토콜만 알아서, 다른 백엔드로 갈아끼워도 브리지와 JS 번들은 그대로다:

```swift
let host = LynxWebGPUHost(runtime: try LynxWebGPUContext())   // 기본 엔진 (Metal)
let lynxView = LynxView { builder in
    let config = LynxConfig(provider: provider)
    LynxWebGPU.register(in: config, host: host)   // NativeModules.WebGPU + <webgpu-canvas>
    builder.config = config
}
host.attach(to: lynxView)
```

번들 쪽은 `JS/webgpu.js`, `JS/webgpu.d.ts`, `JS/elements.d.ts`를 rspeedy 프로젝트의 `src/`로 복사하면 끝이다.
전체 절차는 [docs/LYNX-INTEGRATION.md](docs/LYNX-INTEGRATION.md).

## 문서

- [CLAUDE.md](CLAUDE.md) — 빌드/테스트 명령, 모듈 그래프, 컨벤션
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — 설계 (커맨드 스트림, 셰이더 파이프라인, 배치 문제, 동시성)
- [docs/WEBGPU-API.md](docs/WEBGPU-API.md) — 지원 API 레퍼런스와 미지원 목록
- [docs/WGSL.md](docs/WGSL.md) — WGSL 서브셋 문법 · MSL 매핑 · 제약
- [docs/JS-AUTHORING.md](docs/JS-AUTHORING.md) — 번들(JS) 작성 가이드, 성능 규칙
- [docs/LYNX-INTEGRATION.md](docs/LYNX-INTEGRATION.md) — 호스트 앱 연동
- [docs/COMMAND-STREAM.md](docs/COMMAND-STREAM.md) — 커맨드 스트림 와이어 명세 (**다른 백엔드를 만들 때의 사양서**)
- [docs/TESTING.md](docs/TESTING.md) — 테스트 전략/하네스/컨벤션
- [docs/ROADMAP.md](docs/ROADMAP.md) — 다음 기능 계획 (웹 라이브러리 이식 갭 → 이미지 처리 경로)
- [Examples/HelloTriangle.tsx](Examples/HelloTriangle.tsx) — ReactLynx 최소 예제

## 모듈

| 모듈 | 역할 |
|---|---|
| `LynxWebGPUCore` | WebGPU 열거형·디스크립터·오류·핸들 레지스트리 + **커맨드 스트림 엔진**(`WGPUBackendEngine` — 디코딩·검증·오류 스코프·프레임 수명)과 백엔드 동사 프로토콜(`WGPUBackend`). Metal-free |
| `LynxWebGPUShader` | WGSL 렉서/파서/리플렉션/MSL 방출기 (순수 Swift) |
| `LynxWebGPU` | 엔진 위의 **Metal 백엔드**(인코딩) + 캔버스 표면 + 기본 런타임 `LynxWebGPUContext` |
| `LynxWebGPUConformance` | 런타임 무관 **적합성 스위트**(29검사) — 저장소 밖에서 만든 백엔드가 같은 계약을 지키는지 픽셀로 판정한다 ([docs/TESTING.md](docs/TESTING.md) §2-1) |
| `LynxWebGPUBridge` | Lynx NativeModule + `<webgpu-canvas>` 엘리먼트 (iOS 전용) — **SPM 타깃이 아니라 소스**다 |

**이 패키지는 외부 의존성이 0이다.** SPM product는 셋(`LynxWebGPU` 엔진 · `LynxWebGPUCore`
계약 · `LynxWebGPUConformance` 적합성 스위트)이고, Lynx SDK의
버전·배포처는 **앱이 정한다** — SPM으로 받든, CocoaPods로 이미 쓰고 있든, 사내 배포본을 물리든
그대로 붙는다. Lynx 연동 레이어는 `#if canImport(Lynx)`로 감싼 소스로 제공되어, Lynx가 보이는
타깃에서 컴파일하면 켜진다 (별도 브리지 타깃 · 앱 타깃 직접 — `docs/LYNX-INTEGRATION.md` §2).

`LynxWebGPU` product는 **Lynx 없이도** 쓸 수 있다 — Swift 앱에서 WebGPU 커맨드 스트림을 직접 실행하거나,
WGSL을 MSL로 번역하는 용도로만 가져다 써도 된다.

## 설계 요점

**커맨드 스트림** — WebGPU 호출을 JS에서 기록만 하고 `queue.submit()`에서 한 번에 넘긴다.
핸들은 JS가 발급하므로 객체 생성이 네이티브 왕복을 기다리지 않는다. 프레임당 브리지 왕복이 1회로 고정된다.
(Chrome이 Dawn wire로 GPU 프로세스와 통신하는 구조와 같은 아이디어)

**WGSL → MSL** — 렉서부터 방출기까지 직접 구현했다. MSL에 가변 전역이 없으므로 유니폼/텍스처는
호출 그래프를 따라 함수 인자로 내려보내고, 스테이지 I/O 속성은 진입점 전용 래퍼 구조체에만 붙인다.
`vec3`의 크기 차이(WGSL 12 vs MSL 16)는 배치를 계산해 `packed_float3`/패딩으로 재현하므로,
**JS는 WGSL 규칙대로 유니폼을 채우면 된다.**

**오류는 모아서 돌려준다** — 잘못된 호출이 프로세스를 죽이지 않는다. `commands[3].vertex.buffers[0].format`
같은 경로가 붙고, 셰이더 컴파일 실패에는 생성된 MSL 전문이 함께 온다.
메시지는 **영문**이다 (0.5.0에서 전부 옮겼다 — 문구로 분기하는 코드가 있다면 확인할 것).

**프레임 경로에서 기다리지 않는다** — 캔버스 크기는 제출 응답으로 갱신되는 캐시를 읽어
프레임당 브리지 왕복이 정확히 1회다. `writeBuffer`/`writeTexture`는 스테이징(풀로 재사용) + blit으로
큐 순서를 타므로 GPU 완주를 기다리지 않는다 — **동적 텍스처를 매 프레임 올릴 수 있다.**
GPU가 밀리면 in-flight 3프레임 제한이 프레임 티커를 건너뛰어, `nextDrawable()` 블로킹이
JS 스레드(터치·타이머 포함)로 번지지 않는다.

**수명은 손으로, 감시는 자동으로** — 핸들은 정수라 JS GC가 네이티브 수명을 모른다.
`submit()` 반환의 `objects`(live 객체 수)와 네이티브 경고 로그(4096개 초과)로 destroy 누락을
잡고, 엔진이 `FinalizationRegistry`를 지원하면 GC 자동 해제 안전망이 켜진다.

## 실제 WebGPU 셰이더 호환성

공식 [webgpu-samples](https://github.com/webgpu/webgpu-samples)의 WGSL 69개를 **손대지 않고** 통과시켜
본 결과 (번역 + 실제 Metal 컴파일까지):

| 결과 | 수 |
|---|---|
| 그대로 통과 | **61 / 67 (91%)** — 그대로 60 + 같은 폴더 조각을 붙여 1 |
| 호스트가 `constants`만 주면 동작 | 4 (분모에 포함) |
| 실패 | 2 |

**남은 2건은 코퍼스 셰이더가 파일만으로 완성되지 않는 경우다** — `skinnedMesh/gltf.wgsl`은 호스트가 glTF
접근자를 보고 `VertexInput`을 런타임 생성해 붙이고, `wireframe/wireframeBufferView.wgsl`은 `requires buffer_view;`로
**WGSL 확장**을 요구한다 (코어 명세 밖). 런타임 크기 배열(`arrayLength`), 외부 텍스처(`textureSampleBaseClampToEdge`),
타입 없는 상수식(`vec3(1)`)은 모두 지원한다.

측정은 저장소 안에 하네스로 들어 있어 언제든 다시 잴 수 있다 ([docs/TESTING.md](docs/TESTING.md) §7).

## 검증

Swift 366개 + JS 133개 테스트가 몇 초 안에 돈다 — 시뮬레이터도 기기도 필요 없다.

- 트랜스파일러 테스트는 생성된 MSL을 **실제 Metal 컴파일러로** 통과시킨다.
- 렌더 테스트는 오프스크린 텍스처에 그린 뒤 **픽셀 값을 단언**한다 (삼각형, 유니폼, 인덱스 드로우,
  알파 블렌딩, 컴퓨트 + 리드백, 텍스처 샘플링, 가장자리 클램프 샘플링, 깊이 테스트).
  `rgba16float` 표면은 되읽었을 때 **1.0을 넘는 값과 음수가 살아 있는지**까지 본다 — HDR 결과를
  8비트로 떨어뜨리지 않는다는 근거다.
- 커맨드 스트림의 계약도 단언한다 — `writeTexture`가 같은 배치의 앞선 렌더 패스 **뒤에**
  실행되는지(큐 순서), 스테이징 풀이 프레임을 거듭해도 자라지 않는지, in-flight 카운터가
  커밋/완료로 오르내리는지.
- **"브라우저에서만 깨지는 코드"를 막는 검증도 테스트로 고정한다** — 렌더 번들의 상태 격리,
  매핑 중인 버퍼의 큐 작업 거부, read-only 어태치먼트 강제처럼 Metal이 그냥 통과시키는 것들이다.
  새 검증에는 **구현을 임시로 되돌려 실제로 실패하는지** 확인한 테스트만 넣는다.
- `arrayLength()`처럼 런타임이 셰이더에 몰래 넘기는 값도 GPU에서 되짚어 **숫자로 단언**한다.
- **적합성 스위트(`LynxWebGPUConformance`, 29검사)는 런타임을 갈아끼워도 그대로 돈다** —
  커맨드 스트림·`readCanvasPixels`·`readBuffer`·`adapterInfo`만 쓰므로, 저장소 밖 백엔드도
  같은 검사로 자신을 증명한다. 기본 Metal 런타임 29/29에 더해, 진짜 Dawn 위에 얹은 백엔드가
  같은 스위트를 28/29(+1 시뮬레이터 건너뜀)로 통과한다 ([docs/TESTING.md](docs/TESTING.md) §2-1).
  **두 백엔드 모두 iOS 시뮬레이터에서도 같은 스위트를 돌린다** (스킴 `WebGPUCheck` · `DawnCheck`) —
  `swift test`는 macOS라 드라이버·GPU 패밀리가 다르고, 실려 나가는 곳은 iOS이기 때문이다.
- JS shim은 node 내장 러너로 검증한다 — 바이너리 경로의 타입·바이트를 단언하고,
  프레임당 브리지 왕복이 1회인지 목으로 센다.

```zsh
swift test           # 엔진·트랜스파일러·GPU 렌더
cd JS && npm test    # JS shim (의존성 없음)
```

## 데모 앱

`Projects/WebGPUDemo`에 Tuist 데모 호스트 앱과 Lynx 번들 **24종**이 들어 있다. 앱을 켜면 씬 목록이 뜨고,
각 씬은 오프스크린 하네스가 자동 검증하는 기능과 1:1로 대응한다 — 회전 삼각형, 3D 큐브(깊이 테스트),
입자 4096개(컴퓨트 + 인스턴싱), 텍스처·샘플러, **동적 텍스처(CPU 플라스마를 매 프레임 `writeTexture`로)**,
알파 블렌딩, **스텐실 마스크(`stencil8` 단독 포맷)**, **GPU-driven 렌더링(컴퓨트가 정한 개수로 간접 드로우)**,
**렌더 번들(드로우 120개를 커맨드 6개로)**, **쿼리·오류 스코프(occlusion 샘플 수 · 타임스탬프 · `pushErrorScope`)**,
컴퓨트 리드백(`mapAsync`), 파이프라인 상수(`override`), MSL 탈출구,
홀로그래픽 카드(터치 → 3D 자세 → 포일), WGSL 호환성(`arrayLength` · 외부 텍스처 · 타입 없는 상수식),
바이너리 브리징, 브리지 비용 측정, **HDR 게인맵 재구성(`rgba16float` 중간 텍스처 → EDR 출력)**,
스크롤 통과(`<scroll-view>` 위 캔버스의 `passthrough-touches` 검증).
전부 60fps로 돌며 Lynx의 `<text>` HUD가 캔버스 위에 합성된다.

```zsh
mise exec -- tuist generate --no-open
# Xcode에서 WebGPUDemo 실행, 또는
xcrun simctl launch <device> org.lynxwebgpu.demo                  # 씬 목록
xcrun simctl launch <device> org.lynxwebgpu.demo -demo particles  # 바로 진입
```

## 요구 사항

Xcode 26.2 / Swift 6.2 툴체인 · iOS 17.0+ · macOS 14.0+ (개발 루프용)

매니페스트는 `swift-tools-version: 6.0`이고 컴파일은 **Swift 5 언어 모드**다 — GPU 객체 그래프를
컴파일러 격리 대신 명시적 락으로 지키기 때문이다 (`Package.swift` 상단 주석 · `docs/ARCHITECTURE.md` §7).
Swift 6 언어 모드를 쓰는 앱에서도 그대로 링크된다.

**Lynx SDK는 이 패키지가 가져오지 않는다** — 앱이 고른 것을 쓴다 (외부 의존성 0).
데모 앱은 [xenonClient/Lynx-XCFramework](https://github.com/xenonClient/Lynx-XCFramework)를
직접 물고 있다 (device + simulator 슬라이스 포함) — 이는 **데모의 선택**이지 라이브러리의
요구사항이 아니다. 다른 저장소·버전으로 바꾸려면 `Projects/WebGPUDemo/Project.swift`만 고치면 된다.

## 참고

- [Dawn](https://dawn.googlesource.com/dawn) (Google의 WebGPU 구현, BSD-3-Clause) — 명세 해석의
  기준으로 참고했고, `Projects/DawnCheck`가 프리빌트 바이너리를 링크해 **같은 적합성 스위트로
  교차 검증**한다. 라이브러리 product는 Dawn을 링크하지 않는다.
- [react-native-webgpu](https://github.com/wcandillon/react-native-webgpu) (MIT) — 레이어 분할을
  참고했다 (`docs/extra/RN-WEBGPU-LAYERING.md`). 코드는 이식하지 않았다.

데모·검증 앱을 배포할 때 실을 오픈소스 고지 원문은 [Projects/NOTICES.md](Projects/NOTICES.md)에
모아 두었다 (라이브러리 product만 쓰면 해당 없음 — 외부 의존성 0).
