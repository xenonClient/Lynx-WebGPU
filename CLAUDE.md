# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Lynx-WebGPU — [Lynx](https://lynxjs.org) 렌더 엔진 위에서, Lynx 번들의 JS가 **WebGPU 모양으로 GPU에 접근**하게 해 주는 SPM 라이브러리.
[W3C WebGPU 명세](https://www.w3.org/TR/webgpu/)의 객체 모델과 [WGSL](https://www.w3.org/TR/WGSL/)을 Metal로 옮긴다.
Swift 6.2 / iOS 17.0+ / macOS 14.0+. **이 패키지는 외부 의존성이 0이다** — Lynx SDK의 버전·배포처는
앱이 정한다 (`docs/LYNX-INTEGRATION.md` §1). 데모 앱은 [xenonClient/Lynx-XCFramework](https://github.com/xenonClient/Lynx-XCFramework)를 직접 물고 있다.

설계: `docs/ARCHITECTURE.md` · 지원 API: `docs/WEBGPU-API.md` · WGSL 서브셋: `docs/WGSL.md` ·
Lynx 연동: `docs/LYNX-INTEGRATION.md` · 커맨드 스트림 명세: `docs/COMMAND-STREAM.md` ·
번들(JS) 작성: `docs/JS-AUTHORING.md` ·
테스트: `docs/TESTING.md` · 로드맵: `docs/ROADMAP.md`

## Build & Test

```zsh
# macOS 개발 루프 — Lynx 없이 엔진/트랜스파일러만 빌드·테스트 (가장 빠르다)
swift build
swift test                                   # 364개 테스트, ~7초
swift test --filter LynxWebGPUShaderTests    # WGSL → MSL 트랜스파일러만
swift test --filter RenderPipelineTests      # GPU 오프스크린 렌더 검증
swift test --filter ConformanceTests         # 적합성 스위트(29검사) — 런타임 무관 계약

# 외부 백엔드 주입 검증 — Core의 public 표면을 바꿨으면 반드시 (회귀 훅에는 없다)
swift build --package-path Examples/ExternalRuntime
swift run --package-path Examples/ExternalRuntime external-runtime-check

# Dawn 연동 검증 — 진짜 Dawn 위의 WebGPURuntime을 같은 적합성 스위트에 건다 (docs/TESTING.md §2-1)
# 데모와 **같은 워크스페이스**다 — 위의 tuist generate가 스킴 셋(WebGPUDemo·DawnCheck·DawnDemo)을 다 만든다.
arch -arm64 xcodebuild -workspace LynxWebGPUDemo.xcworkspace -scheme DawnCheck \
  -destination 'platform=iOS Simulator,name=iPhone 17,OS=26.2' -derivedDataPath .derivedData-cli test
xcrun simctl launch <device> org.lynxwebgpu.dawndemo -demo triangle   # DawnDemo 스킴 빌드 후 화면 확인

# JS 클라이언트(shim) — 런타임 의존성 0. TypeScript는 **검사·선언 생성 전용**이다 (빌드 산출물 없음)
cd JS && npm test            # node 내장 러너, 133개
cd JS && npm run typecheck   # JSDoc 기준 타입 검사 (tsc --noEmit)
cd JS && npm run types       # webgpu.d.ts 를 JSDoc에서 다시 생성

# iOS 컴파일 확인 — **엔진만**이다 (브리지는 여기 안 들어온다, 아래 참고)
# --scratch-path 필수: 기본 .build를 같이 쓰면 이후 macOS `swift test`가 깨진다.
swift build --scratch-path .build-ios --sdk "$(xcrun --sdk iphonesimulator --show-sdk-path)" \
  --triple arm64-apple-ios17.0-simulator
swift build --scratch-path .build-ios-device --sdk "$(xcrun --sdk iphoneos --show-sdk-path)" \
  --triple arm64-apple-ios17.0   # 실기기

# **Lynx 브리지 컴파일 확인은 데모 앱 빌드가 유일하다.**
# 브리지는 SPM 타깃이 아니라 데모의 Tuist 타깃이 컴파일한다 (거기서만 Lynx가 보인다) —
# 브리지를 고쳤으면 반드시 아래 xcodebuild까지 돌릴 것.

# 데모 호스트 앱 (Tuist) — 브리지 검증 + 눈으로 확인
mise exec -- tuist generate --no-open
arch -arm64 xcodebuild -workspace LynxWebGPUDemo.xcworkspace -scheme WebGPUDemo \
  -configuration Debug -sdk iphonesimulator \
  -destination 'platform=iOS Simulator,name=iPhone 17,OS=26.2' -derivedDataPath .derivedData-cli build
xcrun simctl launch <device> org.lynxwebgpu.demo               # 씬 목록 화면
xcrun simctl launch <device> org.lynxwebgpu.demo -demo wgsl    # 바로 진입 (스크린샷 자동화용)
xcrun simctl launch <device> org.lynxwebgpu.demo -demo interactive -cardTilt 0.42  # 카드를 고정 각도로

# 데모 Lynx 번들 다시 만들기
cd Projects/WebGPUDemo/DemoSrc && mise exec -- npm run sync

# 외부 WGSL 코퍼스 호환성 리포트 (트랜스파일러를 크게 고친 뒤 반드시)
LYNXWEBGPU_WGSL_CORPUS=/path/to/webgpu-samples/sample swift test --filter SampleCorpus
```

## Architecture

의존성 그래프 (화살표 = "depends on"):

```
LynxWebGPUCore    ← (없음)                    WebGPU 열거형·디스크립터·**커맨드 디코딩**·오류·
                                              핸들 레지스트리 + **`WebGPURuntime` 프로토콜**.
                                              Metal을 import하지 않는다 (GPU 없이 테스트된다).
LynxWebGPUShader  ← Core                      WGSL 렉서/파서/리플렉션/MSL 방출기. 순수 Swift.
LynxWebGPU        ← Core, Shader              `WebGPURuntime`의 **기본 구현** — Metal 백엔드
                                              (리소스·파이프라인·인코더), 캔버스 표면, 해석기.
LynxWebGPUConformance ← Core                  런타임 무관 **적합성 스위트**(29검사). 테스트 타깃이
                                              아니라 **라이브러리**다 — 저장소 밖 백엔드가 같은
                                              계약을 지키는지 스스로 재려면 그래야 한다.
LynxWebGPUBridge  ← Core, Lynx                NativeModule(`NativeModules.WebGPU`) +
  (SPM 타깃 아님 — 소스)                        `<webgpu-canvas>` 엘리먼트. iOS 전용.
                                              **GPU 백엔드를 모른다** — 프로토콜만 본다.
```

핵심 원칙:
- **Core와 Shader는 Metal-free.** GPU 없이도 단위 테스트가 돌아야 한다. Metal 타입이 필요한 코드는 LynxWebGPU에만 둔다.
  (Core가 이름을 아는 유일한 그래픽 타입은 `CAMetalLayer`다 — `WebGPURuntime.attachCanvas`가 받는
  **불투명 핸들**이고, Dawn도 Apple 플랫폼에서 같은 레이어를 받는다. GPU 객체는 만들지 않는다.)
- **런타임(GPU 백엔드)은 앱이 주입한다.** 브리지는 `WebGPURuntime` 프로토콜만 보고,
  `LynxWebGPUHost(runtime:)`에 구현체가 들어온다 — Lynx SDK를 가져오지 않는 것과 같은 이유다.
  다른 백엔드(Dawn 등)를 붙일 때 **브리지도 JS 번들도 바뀌지 않는다** (`docs/extra/DAWN-BACKEND-REVIEW.md`).
- **커맨드 스트림의 디코딩·디스패치·검증·와이어 정책은 전부 Core에서 끝난다** (`WGPUDescriptors.swift` =
  명세 디스크립터, `WGPUCommands.swift` = op 인자, `WGPUCommand.swift` = op 이름 → 케이스 디스패치 표,
  `WGPUErrorScopeStack`/`WGPUBatchResult`/`WGPUDeferredErrorQueue` = 오류 스코프·응답 조립·지연 오류).
  디스패치(51케이스 exhaustive switch)와 명세 검증·프레임 수명·매핑 게이트·직렬화는
  **`WGPUBackendEngine`(Core) 한 곳**이 실행하고, 백엔드는 **`WGPUBackend` 동사 프로토콜**만
  구현한다 — op을 더하면 프로토콜 요구가 늘어 컴파일러가 모든 백엔드의 누락을 잡고,
  백엔드를 갈아끼울 때 다시 쓸 것이 "인코딩"으로만 좁혀진다 (Metal `WGPUMetalBackend`과
  Dawn `DawnBackend`가 같은 엔진 위에서 실제로 그렇게 돈다).
- **Lynx 심볼(`LynxModule`, `LynxUI` 등)은 LynxWebGPUBridge 안에서만** 참조하고 반드시 `#if canImport(Lynx)` 가드 안에 둔다.
  **이 패키지는 Lynx를 의존성으로 가져오지 않는다** — 버전·배포처를 앱이 정하게 하기 위해서다
  (`docs/LYNX-INTEGRATION.md` §1). 그래서 `Sources/LynxWebGPUBridge/`는 **SPM 타깃이 아니고**,
  Lynx가 보이는 앱 쪽 타깃(데모의 Tuist 타깃 · 앱 타깃 직접)에서 컴파일된다. `swift build`로는 가드가 꺼진 채 지나가므로 **브리지 수정은 데모 앱 빌드로 확인**할 것.
- **MSL 방출은 셰이더 모듈 생성 시점이 아니라 파이프라인 생성 시점에** 한다. `@group/@binding` → Metal 인덱스 배정이
  파이프라인 레이아웃에 달려 있기 때문이다 (Dawn이 셰이더를 레이아웃마다 다시 컴파일하는 것과 같은 이유).
- WebGPU 열거형의 raw value는 **명세 철자 그대로**다 (`"rgba8unorm"`, `"triangle-list"`). 바꾸면 JS 코드가 깨진다.

## Directory Structure

```
Sources/
├── LynxWebGPUCore/     — WGPUEnums / WGPUDescriptors / WGPUCommands(op 인자) /
│                         WGPUCommand(op → 케이스 디스패치 표) / WGPUValueReader
│                         WGPUHandle / WGPUError / WGPUAssetProvider
│                         WGPUErrorScopeStack · WGPUBatchResult · WGPUDeferredErrorQueue(와이어 정책)
│                         WebGPURuntime(런타임 프로토콜 — 앱이 구현체를 넣는다. processEvents 펌프 포함)
│                         WGPUBackend(백엔드 동사 프로토콜) · WGPUBackendEngine(오케스트레이션 —
│                         디스패치·검증·오류 스코프·프레임 수명·매핑 게이트·직렬화 락) ·
│                         WGPUEngineObjects(핸들 메타데이터 래퍼) · WGPUImageDecoder(ImageIO)
│                         WGPUFrameCoordinator(present 시점·in-flight 회계 — 백엔드 무관)
├── LynxWebGPUShader/   — WGSLLexer → WGSLParser → WGSLReflection → MSLEmitter
│                         WGSLLayout(vec3 배치 보정) · WGSLBindings(Metal 인덱스 배정)
│                         MSLPrelude(타입 추론을 C++ 템플릿에 위임하는 셰이더 프렐류드)
├── LynxWebGPU/         — WGPUMetalMapping / WGPUResources / WGPUPipeline / WGPUSurface
│                         WGPUMetalBackend(= WGPUBackend의 Metal 구현 — 인코딩 동사만)
│                         LynxWebGPUContext(= 엔진 + Metal 백엔드 조합 — WebGPURuntime 기본 구현)
├── LynxWebGPUConformance/ — WebGPUConformance(스위트 입구) · ConformanceHarness ·
│                         Checks(렌더·동치성·오류 19) · LifecycleChecks(경계 계약 10)
│                         **커맨드 스트림과 `WebGPURuntime`만 쓴다** — 백엔드 내부를 보는
│                         검사는 여기 들어올 수 없다 (다른 런타임에서 안 돈다)
├── LynxWebGPUBridge/   — LynxWebGPUHost / WebGPUNativeModule / WebGPUCanvasUI / WebGPUFrameTicker
└── (다른 백엔드)        — 이 저장소 밖. `WGPUBackend` 동사를 구현해 `WGPUBackendEngine`에
                          얹으면 오케스트레이션 없이 런타임이 된다 (Projects/DawnCheck가 실물)
Tests/
├── LynxWebGPUCoreTests/    — 디스크립터·커맨드 디코딩, 핸들 레지스트리, 와이어 정책, 프레임 회계
├── LynxWebGPUShaderTests/  — 트랜스파일 + **실제 Metal 컴파일러 통과 검증**(MetalCompilerHarness)
└── LynxWebGPUTests/        — 오프스크린 GPU 렌더 검증(RenderHarness) + 커맨드 해석기 계약
JS/                     — webgpu.js(클라이언트 shim) / webgpu.d.ts(**JSDoc에서 생성**) / elements.d.ts
                          lynx-env.d.ts(호스트 전역·네이티브 모듈 선언) · tsconfig.json(검사·선언 생성 전용)
                          tests/(node:test — 코덱·캔버스 크기·수명)
Examples/HelloTriangle.tsx  — ReactLynx 최소 예제
Examples/ExternalRuntime/   — 외부 백엔드 주입 검증 픽스처 (Core+Conformance만 링크하는 중첩 SPM)
Projects/WebGPUDemo/    — Tuist 데모 호스트 앱 (Sources/) + 데모 번들 rspeedy 소스 (DemoSrc/)
                          Tools/ — 빌드 시점 애셋 변환 (HDR HEIC → GPU가 바로 먹는 바이너리)
Projects/DawnCheck/     — 진짜 Dawn 위의 백엔드 (루트 워크스페이스의 별도 프로젝트, 시뮬레이터
                          전용). DawnBackend가 `WGPUBackend`를 구현하고 공유 엔진에 얹힌다.
                          스킴 둘: DawnCheck(오프스크린 적합성 29검사) ·
                          DawnDemo(화면 실증 앱 — 데모 씬·브리지·JS 무변경, 런타임만 Dawn)
Tuist.swift · Workspace.swift — 데모 앱 전용. 라이브러리 자체는 SPM만으로 완결된다
docs/                   — ARCHITECTURE / WEBGPU-API / WGSL / LYNX-INTEGRATION / JS-AUTHORING / TESTING
.claude/skills/         — webgpu-command / wgsl-feature / gpu-smoke
```

## Release

버전은 semver 태그로만 매긴다 (`0.1.0` — `v` 접두사 없이).

```zsh
git tag -a 0.2.0 -m "0.2.0 — 요약"
```

태그를 만들면 **PostToolUse 훅이 `README.md`와 `docs/LYNX-INTEGRATION.md`의 SPM 버전 표기를
최신 태그로 맞춘다** (`.claude/hooks/sync-readme-version.sh`). 손으로 돌려도 되고, 멱등이다.
훅은 `.claude/settings.json`에 `if: "Bash(git tag *)"` 필터로 걸려 있어 태그 명령에만 붙는다.

태그를 문서 갱신 커밋 **뒤에** 옮기는 것을 잊지 말 것 (`git tag -f <버전> HEAD`).

## 회귀 테스트 훅

작업이 끝날 때 `.claude/hooks/regression.sh`가 자동으로 돈다 (`.claude/settings.json`의 Stop 훅).
**변경이 없으면 건너뛴다.** 도는 것:

1. `npm run typecheck` — JSDoc이 빠지면 여기서 걸린다
2. `webgpu.d.ts` 드리프트 — 다시 뽑아 보고 달라지면 실패 (구현보다 뒤처졌다는 뜻)
3. 데모 사본(`DemoSrc/src/webgpu.js`) 드리프트
4. `npm test` · `swift test`

층이 넷이라(엔진 · shim · 트랜스파일러 · 데모 번들) 한 층만 고치고 넘어가면 조용히 어긋난다.
**데모 앱 빌드와 시뮬레이터 실행은 넣지 않았다** — 몇 분이 걸린다. 브리지나 씬을 고쳤으면
`docs/TESTING.md` §8의 명령을 직접 돌릴 것. 건너뛰려면 `LYNXWEBGPU_SKIP_REGRESSION=1`.

## Git Conventions

- Commit format: `type: Korean description` (예: `feat: 컴퓨트 파이프라인 지원`, `fix: vec3 유니폼 배치 오류 수정`)
- Types: `feat`, `fix`, `perf`, `refactor`, `chore`, `docs`, `test`
- Do NOT include `Co-Authored-By` in commit messages
- Do NOT include "Generated with Claude Code" in PR body

## Important Notes

- **시뮬레이터 타깃은 iPhone 17, iOS 26.2로 고정**한다. 기기/OS를 임의로 바꾸지 않는다.
- 이 환경의 셸에서는 `xcodebuild`가 x86_64로 뜨면서 CoreSimulator 로드에 실패한다. **`arch -arm64 xcodebuild`** 로 실행할 것.
- CLI 빌드는 전용 DerivedData(`-derivedDataPath .derivedData-cli`)를 쓴다. 사용자가 Xcode에서 동시에 빌드할 수 있다.
- **Swift 5 언어 모드**를 쓴다 (`Package.swift`의 `swiftSettings`). GPU 객체 그래프는 Sendable이 아닌 참조 타입을
  JS 스레드와 메인 스레드가 공유하므로, 컴파일러 격리 대신 `WGPUObjectRegistry`/컨텍스트의 명시적 락으로 동시성을 보장한다.
- **커맨드 실행은 JS 스레드에서 그대로 한다.** Metal 인코딩은 메인 스레드를 요구하지 않으므로 넘기면 UI와 경쟁만 한다.
  단 `CAMetalLayer` 프로퍼티 설정은 메인 스레드 전용이라 **비동기**로 넘긴다 (`main.sync`는 교착 위험 — `WGPUMetalLayerSurface` 참고).
- **`JS/webgpu.js`를 고치면 JSDoc을 반드시 함께 쓰거나 갱신한다.** 공개 선언 `JS/webgpu.d.ts`는
  손으로 쓰는 파일이 아니라 **JSDoc에서 뽑아내는 산출물**이다 (`cd JS && npm run types`).
  - **고쳤으면 반드시 이 두 개를 직접 돌린다** — 훅이 대신해 주지 않는다:
    ```zsh
    cd JS && npm run typecheck   # JSDoc이 빠졌으면 여기서 걸린다
    cd JS && npm run types       # webgpu.d.ts 재생성
    ```
    타입 없이 커밋하면 `npm run typecheck`가 CI/다음 작업에서 터지고, `npm run types`를
    빠뜨리면 공개 선언이 구현보다 뒤처진다 (지금까지 실제로 반복된 드리프트다).
  - **파일에 `// @ts-check`를 넣지 말 것.** 데모(`Projects/WebGPUDemo/DemoSrc/src/`)가 이 파일을
    복사해 쓰는데 거기엔 `lynx-env.d.ts`가 없어 빌드가 깨진다. 검사는 `JS/tsconfig.json`의
    `checkJs`가 켠다.
  - 호스트 전역(`NativeModules.WebGPU` 등)의 시그니처는 `JS/lynx-env.d.ts`에 있고,
    **`WebGPUNativeModule.swift`의 `methodLookup`과 짝이 맞아야 한다.** 한쪽만 고치면 런타임에 깨진다.
  - shim을 고쳤으면 데모 사본도 맞춘다:
    `cp JS/webgpu.js JS/webgpu.d.ts Projects/WebGPUDemo/DemoSrc/src/`
- **커맨드 스트림의 필드 이름은 타입 검사가 잡아 주지 않는다.** JS는 `Record<string, any>`로 싣고
  Swift는 문자열 키로 읽으므로, 이름이 어긋나도 양쪽 다 컴파일된다. op를 추가·수정할 때는
  `.claude/skills/webgpu-command/SKILL.md`의 순서를 그대로 따라 양쪽을 함께 고칠 것.
- **버퍼를 쓰는 새 op은 `WGPUBackendEngine.unmappedBuffer(_:path:)`를 거쳐야 한다.**
  레지스트리 lookup을 직접 부르면 매핑 검사를 건너뛰어, `mapAsync` 중인 버퍼에 GPU가 쓰는
  경쟁이 그 경로로 샌다 (명세의 "unavailable" 상태). 바인드 그룹 경로는 엔진의
  `applyDrawState()`가 그룹 메타데이터의 버퍼 목록을 훑어 따로 막는다.
- **드로우·디스패치 전 상태 확인은 엔진의 `applyDrawState()` 한 곳에 모여 있다.** 새 드로우 op을
  추가하면 이 함수를 부를 것 — 바인드 그룹·정점 버퍼 완전성과 번들 격리 계약이 여기 걸려 있다.
  이 검사들은 백엔드가 준 파이프라인 메타데이터(`WGPURenderPipelineInfo`)로 돈다 — nil을 준
  항목은 "백엔드가 네이티브로 검증한다"는 뜻이다 (Dawn이 그렇게 한다).
- 트랜스파일러를 고칠 때는 **반드시 `MetalCompilerHarness.assertCompiles`가 붙은 테스트**를 추가한다.
  문자열만 맞고 컴파일이 안 되는 MSL을 막기 위한 장치다.
- Metal 검증 레이어는 디스크립터 `label`에 nil을 넣으면 단언으로 죽는다. `if let label = …` 로 감쌀 것.
- 정점 버퍼는 Metal 버퍼 인자 테이블의 **위쪽(30번부터 역순)**, 바인드 그룹 버퍼는 **0번부터** 배정하고,
  **22번은 `arrayLength()`용 버퍼 크기 표로 예약**한다 (`WGSLMetalLimits`).
  이 규칙을 바꾸면 셰이더 방출과 인코딩을 **함께** 고쳐야 한다.
- 트랜스파일러를 고친 뒤에는 **외부 코퍼스 통과율을 다시 잰다** (`docs/TESTING.md` §7).
  로컬 테스트는 통과하는데 실제 셰이더 통과율이 내려가는 변경이 실제로 있었다.
- WGSL 식별자가 MSL 예약어(`texture` `sampler` `device` `char` …)와 충돌하면 방출기가 이름을 바꾼다.
  **선언과 사용처가 같은 `MSLTypeMapping.identifier(_:)`를 거쳐야** 한다 — 한쪽만 고치면 조용히 깨진다.
- 데모 앱에 **소스나 리소스 파일을 추가/삭제한 뒤에는 반드시 `mise exec -- tuist generate`** 로 재생성한다.
  소스를 빠뜨리면 "cannot find type in scope"로 깨지고, **번들(.lynx.bundle)을 빠뜨리면 앱이 조용히
  "번들이 없다"를 띄운다** — glob은 생성 시점에 펼쳐진다. 라이브러리 쪽은 SPM이라 재생성이 필요 없다.
- **iOS 시뮬레이터는 간접 드로우·디스패치를 지원하지 않는다** (Apple GPU family 2로 보고 —
  Metal은 family 3 이상을 요구). `gpudriven` 씬처럼 `drawIndirect` 계열을 쓰는 경로는
  시뮬레이터에서 `unsupported` 오류가 나고, **실기기(A12 이상)에서는 정상 동작**한다.
  막지 않으면 `MTLValidateFeatureSupport … failed assertion`으로 프로세스가 죽는다
  (`WGPUDeviceCapability.supportsIndirectArguments`가 걸러 준다).
- **시뮬레이터에는 터치 주입 수단이 없다** (`simctl`에 탭 API가 없고 `osascript` 클릭은 접근성 권한이 필요).
  터치로만 보이는 상태는 런치 인자로 고정해 캡처한다 (`-cardTilt` 참고 — initData로 JS까지 전달된다).
- **EDR(HDR 출력)은 시뮬레이터에서 확인할 수 없다.** 실기기 디스플레이 기능이라 스크린샷에도
  안 잡힌다. `hdr` 씬의 EDR 버튼은 실기기에서만 의미가 있다 (`docs/WEBGPU-API.md` §2).
  값이 실제로 1.0을 넘는지는 같은 씬의 **클리핑 표시**로 확인한다 — 그건 화면과 무관하다.
- **`<webgpu-canvas>`의 터치는 Lynx 표준 이벤트를 쓴다** (`bindtouchstart` 등). UIKit `touchesBegan`을
  가로채면 Lynx의 hitTest·pointer-events·버블링·제스처 아레나를 우회해 웹과 다르게 동작한다
  (`docs/LYNX-INTEGRATION.md` §6).
- **`LynxWebGPUCore`의 public 저장 프로퍼티를 바꾸면 `rm -rf .build` 후 다시 빌드한다.**
  SwiftPM 증분 빌드가 의존 모듈(Shader/LynxWebGPU)을 옛 메모리 레이아웃으로 남겨 두어,
  테스트가 **signal 11로 죽는다** — 코드 버그처럼 보이지만 아니다 (`WGPUError`에 필드를
  더했을 때 실제로 겪었다). 클린 빌드로 재현되지 않으면 그 경우다.
- 임시 산출물(빌드 로그, 렌더 덤프 PNG 등)은 `.tmp/` 아래에 둔다 (git-ignored).
