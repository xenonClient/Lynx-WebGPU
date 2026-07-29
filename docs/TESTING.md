# 테스트 가이드

## 1. 전략

GPU 코드는 "돌려 보고 눈으로 확인"에 기대기 쉽다. 이 저장소는 그 반대로 간다 —
**모든 계층을 시뮬레이터·기기 없이 검증**하고, 눈으로 볼 것은 디버깅 보조로만 남긴다.

| 계층 | 어떻게 검증하나 | 도구 |
|---|---|---|
| Core (디스크립터·핸들) | 순수 단위 테스트 | XCTest |
| Shader (WGSL → MSL) | 문자열 단언 **+ 실제 Metal 컴파일러 통과** | `MetalCompilerHarness` |
| Metal 백엔드 | 오프스크린 렌더 후 **픽셀 값 단언** | `RenderHarness` |
| 커맨드 해석기 | 오류 누적·경로·핸들 수명 계약 | `RenderHarness` |
| Lynx 브리지 | 컴파일 검증 + 호스트 앱 수동 확인 | `xcodebuild` |
| JS 클라이언트 | Swift 상수와의 값 일치 | `JSConstantParityTests` |

브리지는 Lynx 런타임이 있어야 의미가 있으므로 단위 테스트 대상이 아니다 —
로직이 생기면 `LynxWebGPU`로 내리는 것이 원칙이다.

## 2. 실행

```zsh
swift test                                      # 전체 (65개, 약 4초)
swift test --filter LynxWebGPUCoreTests         # 디스크립터/핸들
swift test --filter LynxWebGPUShaderTests       # 트랜스파일러 (+ Metal 컴파일 검증)
swift test --filter LynxWebGPUTests             # GPU 렌더 + 해석기
swift test --filter test_삼각형이_그려지고        # 개별 테스트
```

Lynx 브리지 컴파일 확인 (iOS 시뮬레이터 고정: iPhone 17 / iOS 26.2):

```zsh
arch -arm64 xcodebuild -scheme LynxWebGPUBridge \
  -destination 'platform=iOS Simulator,name=iPhone 17,OS=26.2' \
  -derivedDataPath .derivedData-cli build
```

## 3. Metal 컴파일러 하네스

트랜스파일러 테스트에서 **문자열 단언만으로는 부족하다.** 기대한 조각이 다 들어 있어도
컴파일되지 않는 MSL이 나올 수 있기 때문이다. `MetalCompilerHarness.assertCompiles(msl)`이
생성된 MSL을 `xcrun -sdk macosx metal -c`로 실제 컴파일하고, 실패하면 **진단 + 줄 번호가 붙은 MSL 전문**을 낸다.

```swift
private func translate(_ source: String, entryPoints: [String]) throws -> String {
    let module = try WGSLShaderModule(source: source)
    let bindings = try WGSLBindingAssigner.assign(groups: module.autoBindGroupLayouts(entryPoints: entryPoints))
    let msl = try module.translateToMSL(entryPoints: entryPoints, bindings: bindings)
    MetalCompilerHarness.assertCompiles(msl)     // ← 반드시 함께
    return msl
}
```

Metal 툴체인이 없는 환경에서는 조용히 건너뛴다 (문자열 단언은 그대로 돈다).

## 4. 오프스크린 렌더 하네스

`RenderHarness`는 `WGPUOffscreenSurface`(화면 없는 `MTLTexture`)를 캔버스로 등록해,
JS가 보낼 것과 **완전히 같은 커맨드 스트림**을 실행하고 결과 픽셀을 읽는다.

```swift
let harness = try XCTUnwrap(RenderHarness.make(width: 64, height: 64))

harness.executeExpectingSuccess([
    ["op": "configureCanvas", "canvas": "test", "format": "rgba8unorm"],
    ["op": "createShaderModule", "id": 1, "code": shader],
    // …
    ["op": "draw", "vertexCount": 3],
    ["op": "endPass"],
])

try harness.assertPixel(x: 32, y: 32, equals: (255, 0, 0, 255), "삼각형 내부")
try harness.assertPixel(x: 1, y: 1, equals: (0, 0, 255, 255), "삼각형 외부(클리어색)")
```

- `executeExpectingSuccess`는 실패 시 오류를 **경로와 함께** 그대로 보여 준다.
- `assertPixel`의 기본 허용 오차는 2 (래스터화·색공간 오차 흡수).
- 눈으로 봐야 할 때는 `harness.dumpPNG(named: "triangle")` → `.tmp/triangle.png`.
  의존성 없는 PNG 인코더가 들어 있어 별도 설치가 필요 없다.

테스트를 짤 때는 **두 점 이상**을 단언한다 — 클리어 색만 나와도 통과하는 테스트가 되기 쉽다.
"안쪽 한 점 + 바깥쪽 한 점"이 최소 조합이다.

## 5. 컨벤션

- 테스트 파일은 대상 타입/영역당 1개, 각 모듈 `Tests/` 아래.
- 메서드 명명: `test_<대상동작>_<조건>_<기대결과>` — 한국어.
  예: `test_vec3뒤_스칼라는_packed벡터와_패딩으로_WGSL배치를_맞춘다`
- GPU가 필요한 테스트는 `setUpWithError`에서 `XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, …)`.
- 비동기 경로(`readBuffer`)는 `XCTestExpectation`으로 검증한다.
- 테스트 더블은 손으로 만든다 (모킹 라이브러리 없음).

## 6. 커버리지 대상 (현재 65개)

| 영역 | 파일 | 주요 케이스 |
|---|---|---|
| 값 디코딩 | `WGPUValueReaderTests` | 기본값, NSNull, 열거형 후보 안내, 색/크기 두 표기, base64·바이트배열, 경로 누적 |
| 디스크립터 | `WGPUDescriptorTests` | 크기 유추, 범위 검증, 명세 기본값, auto/명시 레이아웃, 블렌드 기본값 |
| 핸들 레지스트리 | `WGPUObjectRegistryTests` | 등록/조회/해제, 타입 불일치 진단 |
| JS↔Swift 상수 | `JSConstantParityTests` | `JS/webgpu.js`의 사용 플래그·스테이지·컬러마스크가 Swift OptionSet과 같은 값인지 |
| WGSL 트랜스파일 | `WGSLTranspilerTests` | 삼각형(정점속성+유니폼+헬퍼), 리소스 스레딩, 리플렉션, vec3 배치, 컴퓨트/스토리지, 텍스처/샘플러/스토리지텍스처, 제어흐름, workgroup 변수, 오류 보고, 바인딩 배정, **MSL 예약어 맹글링**, **부동소수 `%`**, **벡터 성분 추론**, **파이프라인 상수**, **확장 선언** |
| 외부 코퍼스 | `SampleCorpusTests` | 공식 webgpu-samples 셰이더 통과율 리포트 (§7, 기본 스킵) |
| GPU 렌더 | `RenderPipelineTests` | 삼각형, 유니폼, 인덱스 드로우, 알파 블렌딩, 컴퓨트+readback, 텍스처 샘플링, 깊이 테스트 |
| 커맨드 해석기 | `CommandInterpreterTests` | 알 수 없는 명령, 없는 핸들, 오류 누적, 패스 상태, 캔버스 진단, 셰이더 실패 시 MSL 첨부, 드로어블 핸들 수명, 복사/읽기, 범위 검증, reset, 어댑터 정보 |

새 기능을 넣으면 위 표에 행을 추가하고 같은 컨벤션으로 테스트를 쓴다.

## 7. 외부 WGSL 코퍼스 리포트

직접 쓴 테스트만으로는 "우리가 생각한 문법"만 검증하게 된다. 실제 세상의 셰이더가 얼마나 그대로
통과하는지 재려면 **남의 코퍼스**가 필요하다.

```zsh
git clone --depth 1 https://github.com/webgpu/webgpu-samples.git /tmp/webgpu-samples
LYNXWEBGPU_WGSL_CORPUS=/tmp/webgpu-samples/sample swift test --filter SampleCorpus
```

디렉토리의 모든 `.wgsl`을 진입점별로 번역하고 **실제 Metal 컴파일러에 통과시킨 뒤**,
통과율과 실패 원인을 한 화면에 출력한다. 리포트가 목적이므로 실패해도 테스트를 깨지 않는다.

```
│ 그대로 통과: 54/67  (80%)
│ 호스트가 constants를 줘야 하는 것: 4건 — …
├─ 실패 9건 ───────────────────────────────
│ ✗ computeBoids/updateSprites.wgsl
│     번역[main]: WGSL: arrayLength() — …
```

번역 결과를 눈으로 보려면 덤프를 켠다:

```zsh
LYNXWEBGPU_WGSL_CORPUS=… LYNXWEBGPU_WGSL_DUMP=/tmp/msl swift test --filter SampleCorpus
```

**트랜스파일러를 크게 고친 뒤에는 이 수치를 다시 재고, 떨어졌으면 원인을 찾는다.**
로컬 테스트는 통과하는데 코퍼스 통과율이 내려가는 변경이 실제로 있었다
(WGSL의 AbstractInt 리터럴 규칙 — `docs/WGSL.md` §4).

## 8. 데모 호스트 앱

눈으로 확인해야 할 때는 Tuist로 만든 데모 앱을 쓴다 (`Projects/WebGPUDemo`).
앱을 켜면 **씬 목록**이 뜨고, 각 행이 오프스크린 하네스가 자동 검증하는 기능과 1:1로 대응한다:

| 씬 | 밟는 경로 |
|---|---|
| `triangle` | 정점 버퍼 + 유니폼 + 매 프레임 writeBuffer |
| `cube` | 인덱스 드로우 + 깊이 테스트 + 백페이스 컬링 + MVP |
| `particles` | 컴퓨트 + 스토리지 버퍼 + 인스턴싱 + 가산 블렌딩 |
| `texture` | createTexture + writeTexture + repeat 샘플러 + textureSample |
| `blending` | src-alpha 블렌딩 + discard + 유니폼 구조체 배열 |
| `readback` | 컴퓨트 결과를 `mapAsync`로 CPU가 읽어 화면에 표시 |
| `constants` | 같은 셰이더 모듈 + 다른 `override` 값 → 파이프라인 3개 |
| `msl` | `language: 'msl'` 직접 주입 + 명시적 파이프라인 레이아웃 |

목록 ↔ 씬을 오갈 때마다 LynxView와 WebGPU 런타임이 새로 만들어지고 해제되므로,
**생성/해제 경로까지 함께 확인된다.**

```zsh
mise exec -- tuist generate --no-open
arch -arm64 xcodebuild -workspace LynxWebGPUDemo.xcworkspace -scheme WebGPUDemo \
  -configuration Debug -sdk iphonesimulator \
  -destination 'platform=iOS Simulator,name=iPhone 17,OS=26.2' -derivedDataPath .derivedData-cli build

xcrun simctl install <device> .derivedData-cli/Build/Products/Debug-iphonesimulator/WebGPUDemo.app
xcrun simctl launch <device> org.lynxwebgpu.demo                # 씬 목록
xcrun simctl launch <device> org.lynxwebgpu.demo -demo cube      # 목록을 건너뛰고 바로 진입
xcrun simctl io <device> screenshot .tmp/demo-cube.png
```

번들 소스(JS/TSX)는 `Projects/WebGPUDemo/DemoSrc`이며, 고친 뒤에는 번들을 다시 만들어 넣는다:

```zsh
cd Projects/WebGPUDemo/DemoSrc && mise exec -- npm run sync   # build + Resources/ 로 복사
```
