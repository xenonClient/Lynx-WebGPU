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
.package(url: "https://github.com/xenonClient/Lynx-WebGPU", from: "0.3.1")
```

호스트 앱 연동은 세 단계다:

```swift
let host = try LynxWebGPUHost()
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
- [docs/TESTING.md](docs/TESTING.md) — 테스트 전략/하네스/컨벤션
- [docs/ROADMAP.md](docs/ROADMAP.md) — 다음 기능 계획 (압축 텍스처 → 스텐실 → 간접 드로우)
- [Examples/HelloTriangle.tsx](Examples/HelloTriangle.tsx) — ReactLynx 최소 예제

## 모듈

| 모듈 | 역할 |
|---|---|
| `LynxWebGPUCore` | WebGPU 열거형·디스크립터·오류·핸들 레지스트리 (Metal-free) |
| `LynxWebGPUShader` | WGSL 렉서/파서/리플렉션/MSL 방출기 (순수 Swift) |
| `LynxWebGPU` | Metal 백엔드 + 캔버스 표면 + 커맨드 스트림 해석기 |
| `LynxWebGPUBridge` | Lynx NativeModule + `<webgpu-canvas>` 엘리먼트 (iOS 전용) |

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

**프레임 경로에서 기다리지 않는다** — 캔버스 크기는 제출 응답으로 갱신되는 캐시를 읽어
프레임당 브리지 왕복이 정확히 1회다. `writeBuffer`/`writeTexture`는 스테이징(풀로 재사용) + blit으로
큐 순서를 타므로 GPU 완주를 기다리지 않는다 — **동적 텍스처를 매 프레임 올릴 수 있다.**
GPU가 밀리면 in-flight 3프레임 제한이 프레임 티커를 건너뛰어, `nextDrawable()` 블로킹이
JS 스레드(터치·타이머 포함)로 번지지 않는다.

**수명은 손으로, 감시는 자동으로** — 핸들은 정수라 JS GC가 네이티브 수명을 모른다.
`submit()` 반환의 `objects`(live 객체 수)와 네이티브 경고 로그(4096개 초과)로 destroy 누락을
잡고, 엔진이 `FinalizationRegistry`를 지원하면 GC 자동 해제 안전망이 켜진다.

## 실제 WebGPU 셰이더 호환성

공식 [webgpu-samples](https://github.com/webgpu/webgpu-samples)의 WGSL 68개를 **손대지 않고** 통과시켜
본 결과 (번역 + 실제 Metal 컴파일까지):

| 결과 | 수 |
|---|---|
| 그대로 통과 | **60 / 67 (89%)** |
| 호스트가 `constants`만 주면 동작 | 4 |
| 코퍼스 자체가 단독 파일이 아님 | 3 |

**남은 3건은 트랜스파일러의 빈틈이 아니다** — 선언이 다른 `.wgsl` 파일에 있어 샘플이 이어 붙여 쓰거나,
`texture_storage_2d<{OUTPUT_FORMAT}, write>`처럼 JS가 문자열을 치환할 자리를 담고 있는 조각이다.
런타임 크기 배열(`arrayLength`), 외부 텍스처(`textureSampleBaseClampToEdge`),
타입 없는 상수식(`vec3(1)`)은 모두 지원한다.

측정은 저장소 안에 하네스로 들어 있어 언제든 다시 잴 수 있다 ([docs/TESTING.md](docs/TESTING.md) §7).

## 검증

Swift 188개 + JS 43개 테스트가 몇 초 안에 돈다 — 시뮬레이터도 기기도 필요 없다.

- 트랜스파일러 테스트는 생성된 MSL을 **실제 Metal 컴파일러로** 통과시킨다.
- 렌더 테스트는 오프스크린 텍스처에 그린 뒤 **픽셀 값을 단언**한다 (삼각형, 유니폼, 인덱스 드로우,
  알파 블렌딩, 컴퓨트 + 리드백, 텍스처 샘플링, 가장자리 클램프 샘플링, 깊이 테스트).
  `rgba16float` 표면은 되읽었을 때 **1.0을 넘는 값과 음수가 살아 있는지**까지 본다 — HDR 결과를
  8비트로 떨어뜨리지 않는다는 근거다.
- 커맨드 스트림의 계약도 단언한다 — `writeTexture`가 같은 배치의 앞선 렌더 패스 **뒤에**
  실행되는지(큐 순서), 스테이징 풀이 프레임을 거듭해도 자라지 않는지, in-flight 카운터가
  커밋/완료로 오르내리는지.
- `arrayLength()`처럼 런타임이 셰이더에 몰래 넘기는 값도 GPU에서 되짚어 **숫자로 단언**한다.
- JS shim은 node 내장 러너로 검증한다 — 바이너리 경로의 타입·바이트를 단언하고,
  프레임당 브리지 왕복이 1회인지 목으로 센다.

```zsh
swift test           # 엔진·트랜스파일러·GPU 렌더
cd JS && npm test    # JS shim (의존성 없음)
```

## 데모 앱

`Projects/WebGPUDemo`에 Tuist 데모 호스트 앱과 Lynx 번들 **15종**이 들어 있다. 앱을 켜면 씬 목록이 뜨고,
각 씬은 오프스크린 하네스가 자동 검증하는 기능과 1:1로 대응한다 — 회전 삼각형, 3D 큐브(깊이 테스트),
입자 4096개(컴퓨트 + 인스턴싱), 텍스처·샘플러, **동적 텍스처(CPU 플라스마를 매 프레임 `writeTexture`로)**,
알파 블렌딩, 컴퓨트 리드백(`mapAsync`), 파이프라인 상수(`override`), MSL 탈출구,
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

Xcode 26.2 / Swift 6.2 · iOS 17.0+ · macOS 14.0+ (개발 루프용)
Lynx SDK는 [xenonClient/Lynx-XCFramework](https://github.com/xenonClient/Lynx-XCFramework) 4.0.0을
SPM `binaryTarget`으로 가져온다 (device + simulator 슬라이스 포함).
