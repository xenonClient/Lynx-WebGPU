# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Lynx-WebGPU — [Lynx](https://lynxjs.org) 렌더 엔진 위에서, Lynx 번들의 JS가 **WebGPU 모양으로 GPU에 접근**하게 해 주는 SPM 라이브러리.
[W3C WebGPU 명세](https://www.w3.org/TR/webgpu/)의 객체 모델과 [WGSL](https://www.w3.org/TR/WGSL/)을 Metal로 옮긴다.
Swift 6.2 / iOS 17.0+ / macOS 14.0+, Lynx는 [xenonClient/Lynx-XCFramework](https://github.com/xenonClient/Lynx-XCFramework) 4.0.0을 SPM `binaryTarget`으로 연동한다.

설계: `docs/ARCHITECTURE.md` · 지원 API: `docs/WEBGPU-API.md` · WGSL 서브셋: `docs/WGSL.md` ·
Lynx 연동: `docs/LYNX-INTEGRATION.md` · 번들(JS) 작성: `docs/JS-AUTHORING.md` ·
테스트: `docs/TESTING.md` · 로드맵: `docs/ROADMAP.md`

## Build & Test

```zsh
# macOS 개발 루프 — Lynx 없이 엔진/트랜스파일러만 빌드·테스트 (가장 빠르다)
swift build
swift test                                   # 239개 테스트, ~4초
swift test --filter LynxWebGPUShaderTests    # WGSL → MSL 트랜스파일러만
swift test --filter RenderPipelineTests      # GPU 오프스크린 렌더 검증

# JS 클라이언트(shim) — 런타임 의존성 0. TypeScript는 **검사·선언 생성 전용**이다 (빌드 산출물 없음)
cd JS && npm test            # node 내장 러너, 84개
cd JS && npm run typecheck   # JSDoc 기준 타입 검사 (tsc --noEmit)
cd JS && npm run types       # webgpu.d.ts 를 JSDoc에서 다시 생성

# iOS 컴파일 확인 — Lynx 브리지 포함 (SPM 크로스 빌드)
# 루트에 Tuist 워크스페이스(LynxWebGPUDemo.xcworkspace)가 있으면 xcodebuild가 패키지 스킴
# (LynxWebGPUBridge)을 못 찾는다. swift build 크로스 빌드는 워크스페이스와 무관하게 동작한다.
# --scratch-path 필수: 기본 .build를 같이 쓰면 이후 macOS `swift test`가 깨진다.
swift build --scratch-path .build-ios --sdk "$(xcrun --sdk iphonesimulator --show-sdk-path)" \
  --triple arm64-apple-ios17.0-simulator
swift build --scratch-path .build-ios-device --sdk "$(xcrun --sdk iphoneos --show-sdk-path)" \
  --triple arm64-apple-ios17.0   # 실기기

# 데모 호스트 앱 (Tuist) — 눈으로 확인할 때만 필요하다
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
LynxWebGPUCore    ← (없음)                    WebGPU 열거형·디스크립터·오류·핸들 레지스트리.
                                              Metal/Lynx/UIKit 어느 것도 참조하지 않는다.
LynxWebGPUShader  ← Core                      WGSL 렉서/파서/리플렉션/MSL 방출기. 순수 Swift.
LynxWebGPU        ← Core, Shader              Metal 백엔드(디바이스·리소스·파이프라인·인코더),
                                              캔버스 표면, 커맨드 스트림 해석기.
LynxWebGPUBridge  ← LynxWebGPU, Lynx          NativeModule(`NativeModules.WebGPU`) +
                                              `<webgpu-canvas>` 엘리먼트. iOS 전용.
```

핵심 원칙:
- **Core와 Shader는 Metal-free.** GPU 없이도 단위 테스트가 돌아야 한다. Metal 타입이 필요한 코드는 LynxWebGPU에만 둔다.
- **Lynx 심볼(`LynxModule`, `LynxUI` 등)은 LynxWebGPUBridge 안에서만** 참조하고 반드시 `#if canImport(Lynx)` 가드 안에 둔다.
  Lynx 의존성은 `Package.swift`에서 iOS로 조건부 제한되므로, macOS 빌드에서는 이 가드가 꺼진 채 컴파일된다.
- **MSL 방출은 셰이더 모듈 생성 시점이 아니라 파이프라인 생성 시점에** 한다. `@group/@binding` → Metal 인덱스 배정이
  파이프라인 레이아웃에 달려 있기 때문이다 (Dawn이 셰이더를 레이아웃마다 다시 컴파일하는 것과 같은 이유).
- WebGPU 열거형의 raw value는 **명세 철자 그대로**다 (`"rgba8unorm"`, `"triangle-list"`). 바꾸면 JS 코드가 깨진다.

## Directory Structure

```
Sources/
├── LynxWebGPUCore/     — WGPUEnums / WGPUDescriptors / WGPUValueReader / WGPUHandle / WGPUError
├── LynxWebGPUShader/   — WGSLLexer → WGSLParser → WGSLReflection → MSLEmitter
│                         WGSLLayout(vec3 배치 보정) · WGSLBindings(Metal 인덱스 배정)
│                         MSLPrelude(타입 추론을 C++ 템플릿에 위임하는 셰이더 프렐류드)
├── LynxWebGPU/         — WGPUMetalMapping / WGPUResources / WGPUPipeline / WGPUSurface
│                         WGPUCommandInterpreter / LynxWebGPUContext
│                         WGPUAssetProvider(loadAsset 이름 해석 — 호스트가 갈아끼운다)
└── LynxWebGPUBridge/   — LynxWebGPUHost / WebGPUNativeModule / WebGPUCanvasUI / WebGPUFrameTicker
Tests/
├── LynxWebGPUCoreTests/    — 디스크립터 디코딩, 핸들 레지스트리
├── LynxWebGPUShaderTests/  — 트랜스파일 + **실제 Metal 컴파일러 통과 검증**(MetalCompilerHarness)
└── LynxWebGPUTests/        — 오프스크린 GPU 렌더 검증(RenderHarness) + 커맨드 해석기 계약
JS/                     — webgpu.js(클라이언트 shim) / webgpu.d.ts(**JSDoc에서 생성**) / elements.d.ts
                          lynx-env.d.ts(호스트 전역·네이티브 모듈 선언) · tsconfig.json(검사·선언 생성 전용)
                          tests/(node:test — 코덱·캔버스 크기·수명)
Examples/HelloTriangle.tsx  — ReactLynx 최소 예제
Projects/WebGPUDemo/    — Tuist 데모 호스트 앱 (Sources/) + 데모 번들 rspeedy 소스 (DemoSrc/)
                          Tools/ — 빌드 시점 애셋 변환 (HDR HEIC → GPU가 바로 먹는 바이너리)
Tuist.swift · Workspace.swift — 데모 앱 전용. 라이브러리 자체는 SPM만으로 완결된다
docs/                   — ARCHITECTURE / WEBGPU-API / WGSL / LYNX-INTEGRATION / JS-AUTHORING / TESTING
.claude/skills/         — webgpu-command / wgsl-feature / gpu-smoke
```

## Release

버전은 semver 태그로만 매긴다 (`0.1.0` — `v` 접두사 없이, Lynx-XCFramework와 같은 규칙).

```zsh
git tag -a 0.2.0 -m "0.2.0 — 요약"
```

태그를 만들면 **PostToolUse 훅이 `README.md`와 `docs/LYNX-INTEGRATION.md`의 SPM 버전 표기를
최신 태그로 맞춘다** (`.claude/hooks/sync-readme-version.sh`). 손으로 돌려도 되고, 멱등이다.
훅은 `.claude/settings.json`에 `if: "Bash(git tag *)"` 필터로 걸려 있어 태그 명령에만 붙는다.

태그를 문서 갱신 커밋 **뒤에** 옮기는 것을 잊지 말 것 (`git tag -f <버전> HEAD`).

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
- **버퍼를 쓰는 새 op은 `WGPUCommandInterpreter.unmappedBuffer(_:field:)`를 거쳐야 한다.**
  `registry.lookup(..., as: WGPUBufferObject.self, ...)`를 직접 부르면 매핑 검사를 건너뛰어,
  `mapAsync` 중인 버퍼에 GPU가 쓰는 경쟁이 그 경로로 샌다 (명세의 "unavailable" 상태).
  바인드 그룹 경로는 `applyDrawState()`가 `bufferObjects`를 훑어 따로 막는다.
- **드로우·디스패치 전 상태 확인은 `applyDrawState()` 한 곳에 모여 있다.** 새 드로우 op을
  추가하면 이 함수를 부를 것 — 바인드 그룹·정점 버퍼 완전성과 번들 격리 계약이 여기 걸려 있다.
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
- **시뮬레이터에는 터치 주입 수단이 없다** (`simctl`에 탭 API가 없고 `osascript` 클릭은 접근성 권한이 필요).
  터치로만 보이는 상태는 런치 인자로 고정해 캡처한다 (`-cardTilt` 참고 — initData로 JS까지 전달된다).
- **EDR(HDR 출력)은 시뮬레이터에서 확인할 수 없다.** 실기기 디스플레이 기능이라 스크린샷에도
  안 잡힌다. `hdr` 씬의 EDR 버튼은 실기기에서만 의미가 있다 (`docs/WEBGPU-API.md` §2).
  값이 실제로 1.0을 넘는지는 같은 씬의 **클리핑 표시**로 확인한다 — 그건 화면과 무관하다.
- **`<webgpu-canvas>`의 터치는 Lynx 표준 이벤트를 쓴다** (`bindtouchstart` 등). UIKit `touchesBegan`을
  가로채면 Lynx의 hitTest·pointer-events·버블링·제스처 아레나를 우회해 웹과 다르게 동작한다
  (`docs/LYNX-INTEGRATION.md` §5).
- 임시 산출물(빌드 로그, 렌더 덤프 PNG 등)은 `.tmp/` 아래에 둔다 (git-ignored).
