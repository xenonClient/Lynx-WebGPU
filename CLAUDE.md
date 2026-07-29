# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Lynx-WebGPU — [Lynx](https://lynxjs.org) 렌더 엔진 위에서, Lynx 번들의 JS가 **WebGPU 모양으로 GPU에 접근**하게 해 주는 SPM 라이브러리.
[W3C WebGPU 명세](https://www.w3.org/TR/webgpu/)의 객체 모델과 [WGSL](https://www.w3.org/TR/WGSL/)을 Metal로 옮긴다.
Swift 6.2 / iOS 17.0+ / macOS 14.0+, Lynx는 [xenonClient/Lynx-XCFramework](https://github.com/xenonClient/Lynx-XCFramework) 4.0.0을 SPM `binaryTarget`으로 연동한다.

설계: `docs/ARCHITECTURE.md` · 지원 API: `docs/WEBGPU-API.md` · WGSL 서브셋: `docs/WGSL.md` ·
Lynx 연동: `docs/LYNX-INTEGRATION.md` · 번들(JS) 작성: `docs/JS-AUTHORING.md` · 테스트: `docs/TESTING.md`

## Build & Test

```zsh
# macOS 개발 루프 — Lynx 없이 엔진/트랜스파일러만 빌드·테스트 (가장 빠르다)
swift build
swift test                                   # 60개 테스트, 3초
swift test --filter LynxWebGPUShaderTests    # WGSL → MSL 트랜스파일러만
swift test --filter RenderPipelineTests      # GPU 오프스크린 렌더 검증

# iOS 시뮬레이터 — Lynx 브리지까지 포함한 전체 빌드
arch -arm64 xcodebuild -scheme LynxWebGPUBridge \
  -destination 'platform=iOS Simulator,name=iPhone 17,OS=26.2' \
  -derivedDataPath .derivedData-cli build

# 실기기 빌드 확인
arch -arm64 xcodebuild -scheme LynxWebGPUBridge -destination 'generic/platform=iOS' \
  -derivedDataPath .derivedData-device CODE_SIGNING_ALLOWED=NO build
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
├── LynxWebGPU/         — WGPUMetalMapping / WGPUResources / WGPUPipeline / WGPUSurface
│                         WGPUCommandInterpreter / LynxWebGPUContext
└── LynxWebGPUBridge/   — LynxWebGPUHost / WebGPUNativeModule / WebGPUCanvasUI / WebGPUFrameTicker
Tests/
├── LynxWebGPUCoreTests/    — 디스크립터 디코딩, 핸들 레지스트리
├── LynxWebGPUShaderTests/  — 트랜스파일 + **실제 Metal 컴파일러 통과 검증**(MetalCompilerHarness)
└── LynxWebGPUTests/        — 오프스크린 GPU 렌더 검증(RenderHarness) + 커맨드 해석기 계약
JS/                     — webgpu.js(클라이언트 shim) / webgpu.d.ts / elements.d.ts
Examples/HelloTriangle.tsx  — ReactLynx 최소 예제
docs/                   — ARCHITECTURE / WEBGPU-API / WGSL / LYNX-INTEGRATION / JS-AUTHORING / TESTING
.claude/skills/         — webgpu-command / wgsl-feature / gpu-smoke
```

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
- 트랜스파일러를 고칠 때는 **반드시 `MetalCompilerHarness.assertCompiles`가 붙은 테스트**를 추가한다.
  문자열만 맞고 컴파일이 안 되는 MSL을 막기 위한 장치다.
- Metal 검증 레이어는 디스크립터 `label`에 nil을 넣으면 단언으로 죽는다. `if let label = …` 로 감쌀 것.
- 정점 버퍼는 Metal 버퍼 인자 테이블의 **위쪽(30번부터 역순)**, 바인드 그룹 버퍼는 **0번부터** 배정한다
  (`WGSLMetalLimits`). 이 규칙을 바꾸면 셰이더 방출과 인코딩을 **함께** 고쳐야 한다.
- 임시 산출물(빌드 로그, 렌더 덤프 PNG 등)은 `.tmp/` 아래에 둔다 (git-ignored).
