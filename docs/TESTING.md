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
| JS 클라이언트 | Swift 상수와의 값 일치 + 동작 검증(바이너리 경로·캐시·수명) | `JSConstantParityTests`, `JS/tests` (node:test) |

브리지는 Lynx 런타임이 있어야 의미가 있으므로 단위 테스트 대상이 아니다 —
로직이 생기면 `LynxWebGPU`로 내리는 것이 원칙이다.

## 2. 실행

```zsh
swift test                                      # 전체 (227개, 약 4초)
swift test --filter LynxWebGPUCoreTests         # 디스크립터/핸들
swift test --filter LynxWebGPUShaderTests       # 트랜스파일러 (+ Metal 컴파일 검증)
swift test --filter LynxWebGPUTests             # GPU 렌더 + 해석기
swift test --filter test_삼각형이_그려지고        # 개별 테스트
```

JS 클라이언트(shim) 테스트 — 의존성 없이 node 내장 러너로 돈다.
`NativeModules.WebGPU`를 목으로 바꿔 커맨드 페이로드·왕복 횟수를 단언한다:

```zsh
cd JS && npm test            # NODE_OPTIONS=--expose-gc node --test 'tests/*.test.mjs' — 62개
cd JS && npm run typecheck   # JSDoc 기준 타입 검사
```

`--expose-gc`는 **`NODE_OPTIONS`로 넘겨야 한다.** `node --expose-gc --test`로 주면 러너가
테스트 파일을 도는 자식 프로세스에 그 플래그를 물려주지 않아 `globalThis.gc`가 없고,
GC 수명 테스트 3개가 **조용히 스킵된다** (실패가 아니라 스킵이라 통과처럼 보인다).

바이너리 경로는 커맨드에 실린 `data`가 **진짜 `ArrayBuffer`인지**까지 단언한다. 뷰(TypedArray)가
새어 나가면 Lynx가 `{"0":1,…}` 객체로 바꿔 **오류 없이 조용히** 깨지므로, 바이트 비교만으로는
부족하고 `instanceof ArrayBuffer`를 함께 봐야 한다.

타입 검사는 `JS/tsconfig.json`(`checkJs`)이 JSDoc을 읽어 돌린다. 브라우저 전역을 실수로 쓰는 것을
막으려고 **DOM lib을 켜지 않고**, 호스트가 실제로 주는 것만 `JS/lynx-env.d.ts`에 적어 둔다.

Lynx 브리지 컴파일 확인 — SPM 크로스 빌드를 쓴다 (루트의 Tuist 워크스페이스가
`xcodebuild`의 패키지 스킴 탐색을 가리므로). `--scratch-path`를 꼭 준다 —
기본 `.build`에 iOS 산출물이 섞이면 이후 macOS `swift test`가 깨진다:

```zsh
swift build --scratch-path .build-ios --sdk "$(xcrun --sdk iphonesimulator --show-sdk-path)" \
  --triple arm64-apple-ios17.0-simulator
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
  의존성 없는 PNG 인코더가 들어 있어 별도 설치가 필요 없다. float 표면은 0~1로 잘라서 굽는다.

테스트를 짤 때는 **두 점 이상**을 단언한다 — 클리어 색만 나와도 통과하는 테스트가 되기 쉽다.
"안쪽 한 점 + 바깥쪽 한 점"이 최소 조합이다.

### 4-1. 8비트가 아닌 표면 되읽기

`WGPUOffscreenSurface.readPixels(queue:)`는 `Data`가 아니라 **`WGPUPixelReadback`**을 돌려준다.
바이트와 함께 `format` · `width` · `height` · `bytesPerRow`를 들고 있어, 호출 측이 픽셀 크기를
가정할 필요가 없다. 행 간격은 표면 포맷의 `bytesPerPixel`에서 나온다.

```swift
harness.executeExpectingSuccess([
    ["op": "configureCanvas", "canvas": "test", "format": "rgba16float"],
    // …
])

let readback = try harness.readback()
XCTAssertEqual(readback.format, .rgba16float)
XCTAssertEqual(readback.bytesPerRow, 64 * 8)          // 픽셀당 8바이트

// 1.0 초과·음수가 그대로 살아 있는지 — half-float을 쓰는 이유가 이것이다.
try harness.assertPixelFloat(x: 32, y: 32, equals: SIMD4<Float>(2.5, 0.5, -0.25, 1))
```

- `harness.pixel(x:y:)`(0~255 정수)는 **8비트 표면용**이다. float 표면에는
  `pixelFloat(x:y:)` / `assertPixelFloat(...)`을 쓴다 (기본 허용 오차 0.01).
- `readback.rgba(x:y:)`는 채널이 균일하게 늘어선 포맷만 편다 — unorm8/snorm8 계열,
  f16 계열, f32 계열. `bgra8unorm`은 RGBA 순서로 바꿔 준다. **색공간 변환은 하지 않는다**
  (`-srgb` 포맷도 저장된 값을 그대로 정규화할 뿐이다).
- `rgb10a2unorm`·`rg11b10ufloat`처럼 비트가 채널 경계를 넘어 팩된 포맷과 정수 포맷은
  **던진다** — `data`를 직접 해석해야 한다.
- **depth/stencil 표면은 `readPixels` 자체가 던진다.** Metal blit이 aspect 지정 없이
  한 덩어리로 복사할 수 없고, `depth32float-stencil8`은 픽셀당 바이트가 연속된 블록으로
  존재하지도 않는다.

이 계약은 `OffscreenReadbackTests`(표면 쪽)와 `WGPUPixelReadbackTests`(해석 쪽, GPU 불필요)가
못 박는다. 예전 구현은 픽셀당 4바이트를 가정해서 `rgba16float` 표면에서 **오류 없이** 길이도
해석도 틀린 바이트를 돌려줬다 — 그 회귀를 막는 것이 이 두 파일의 목적이다.

`Data`를 돌려주던 시절의 코드를 옮기는 법은
[`docs/extra/260801-readpixel-migration-guide.md`](extra/260801-readpixel-migration-guide.md)에 있다.

### 4-2. 동치성 단언 — "같은 결과를 내야 하는 두 경로"

점 단언으로는 약한 기능이 있다. 간접 드로우·렌더 번들처럼 **계약 자체가 "직접 경로와 결과가
같다"**인 것은 고른 두 점만 우연히 맞아도 통과하기 때문이다. 그럴 때는 프레임 전체를 비교한다:

```swift
harness.executeExpectingSuccess(직접경로)
let reference = try harness.frameBytes()

harness.executeExpectingSuccess(간접경로)
try harness.assertFrameEquals(reference, "간접 드로우는 직접 드로우와 같아야 한다")
```

실패하면 **처음 어긋난 픽셀의 좌표와 두 값**이 나온다 ("N바이트 다름"만으로는 못 고친다).
이 단언이 다름을 실제로 잡는지는 `RenderHarnessTests`가 못 박는다 — 토대가 되는 단언이라
"항상 통과"가 되면 그 위의 모든 동치성 테스트가 조용히 무의미해진다.

버퍼 리드백은 동기 헬퍼를 쓴다 (`XCTestExpectation` 보일러플레이트가 리드백 테스트마다
반복되던 것을 없앤다):

```swift
let values = try harness.readBufferSync(handle: 3, as: Float.self, size: 32)
```

기기마다 갈리는 기능은 `supports(_:)`로 나눠 건너뛴다 — GPU 유무만 보던 조건의 확장이다:

```swift
try XCTSkipUnless(harness.supports(.timestampQuery), "타임스탬프 카운터를 지원하지 않는 기기")
```

## 5. 컨벤션

- 테스트 파일은 대상 타입/영역당 1개, 각 모듈 `Tests/` 아래.
- 메서드 명명: `test_<대상동작>_<조건>_<기대결과>` — 한국어.
  예: `test_vec3뒤_스칼라는_packed벡터와_패딩으로_WGSL배치를_맞춘다`
- GPU가 필요한 테스트는 `setUpWithError`에서 `XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, …)`.
- 비동기 경로(`readBuffer`)는 `XCTestExpectation`으로 검증한다.
- 테스트 더블은 손으로 만든다 (모킹 라이브러리 없음).

## 6. 커버리지 대상 (Swift 227개 + JS 62개)

| 영역 | 파일 | 주요 케이스 |
|---|---|---|
| 값 디코딩 | `WGPUValueReaderTests` | 기본값, NSNull, 열거형 후보 안내, 색/크기 두 표기, **바이너리 세 표현(Data·base64·바이트배열)**, 경로 누적 |
| 디스크립터 | `WGPUDescriptorTests` | 크기 유추, 범위 검증, 명세 기본값, auto/명시 레이아웃, 블렌드 기본값 |
| 핸들 레지스트리 | `WGPUObjectRegistryTests` | 등록/조회/해제, 타입 불일치 진단, **증식 경고 임계(4096 → 두 배씩)** |
| 픽셀 되읽기 해석 | `WGPUPixelReadbackTests` | half→float(서브노멀·Inf·NaN), **1.0 초과·음수 보존**, BGRA 순서, 없는 채널 기본값, 행 패딩, 팩된/정수 포맷 거부 |
| JS↔Swift 상수 | `JSConstantParityTests` | `JS/webgpu.js`의 사용 플래그·스테이지·컬러마스크가 Swift OptionSet과 같은 값인지 |
| in-flight 프레임 | `SurfaceInFlightTests` | 카운터 계약(3에서 포화·완료로 해제), 컨텍스트 집계, 해석기 커밋/완료 통지(표면당 1회), CAMetalLayer 헤드리스 왕복 |
| JS 클라이언트 | `JS/tests` (node:test) | **바이너리 경로(ArrayBuffer 타입·뷰 오프셋 반영·불필요한 복사 없음·양방향)**, 캔버스 크기 캐시(프레임당 왕복 1회·리사이즈 반영), GC 자동 해제(중복 방지·프레임 스코프 제외), objects 전달, **짝 없는 pop의 `OperationError` reject**, **알 수 없는 filter의 동기 `TypeError`**, **브리지 호출 실패 시 대기 스코프 정리**, **번들 `finish()` 재호출 거부·리소스 retain**, **`device.features`(요청한 것만·미지원 요구 거부)·영원히 pending인 `device.lost`**, **flush의 `present` 표시(submit=프레임 / popErrorScope·mapAsync=내부)**, **기록 시점 스냅샷(호출 뒤 copySize·origin·clearValue 재사용/리셋이 스트림에 안 샘)**, **비동기 파이프라인 생성(스코프 두 겹을 한 배치에·`GPUPipelineError` reason 매핑·실패 핸들 회수)** |
| WGSL 트랜스파일 | `WGSLTranspilerTests` | 삼각형(정점속성+유니폼+헬퍼), 리소스 스레딩, 리플렉션, vec3 배치, 컴퓨트/스토리지, 텍스처/샘플러/스토리지텍스처, 제어흐름, workgroup 변수, 오류 보고, 바인딩 배정, **MSL 예약어 맹글링**, **부동소수 `%`**, **벡터 성분 추론**, **추상 정수 벡터(문맥으로 굳는 `vec3(1)`)**, **파이프라인 상수**, **확장 선언**, **`arrayLength()` 크기 표**, **외부 텍스처**, **함수 지역 `const` 배열 크기**, **전역 섀도잉**(가려진 전역은 주입 안 함 — 매개변수·지역·전이 전달, 주입과 겹친 지역 선언 리네임, 중첩 블록 복원) |
| 외부 코퍼스 | `SampleCorpusTests` | 공식 webgpu-samples 셰이더 통과율 리포트 (§7, 기본 스킵) |
| GPU 렌더 | `RenderPipelineTests` | 삼각형, 유니폼, 인덱스 드로우, 알파 블렌딩, 컴퓨트+readback, 텍스처 샘플링, **`arrayLength()`가 바인딩된 크기를 돌려주는지**, **가장자리 클램프 샘플링**, 깊이 테스트, **rgba16float 표면이 SDR 범위 밖 값을 보존하는지**, **전역 섀도잉의 스코프 해석이 런타임 값까지 옳은지** |
| 오프스크린 되읽기 | `OffscreenReadbackTests` | 포맷별 행 간격·길이(1~16B/픽셀), **depth/stencil 거부**, configure 전 거부 |
| 커맨드 해석기 | `CommandInterpreterTests` | 알 수 없는 명령, 없는 핸들, 오류 누적, 패스 상태, 캔버스 진단, 셰이더 실패 시 MSL 첨부, 드로어블 핸들 수명, **프레임 경계가 배치가 아니라 present인지**, 복사/읽기, 범위 검증, reset, 어댑터 정보, **writeTexture 큐 순서**, **배열 레이어 업로드**, **버퍼 매핑 상태(매핑 중 큐 작업 거부·중복 매핑 거부·unmap 후 복귀)**, **`MAP_READ`/`MAP_WRITE` usage 조합**, **파이프라인 없는 간접 드로우/디스패치의 op별 메시지**, **`present: false` 내부 제출은 커맨드 버퍼가 있어도 드로어블·프레임 핸들을 유지** |
| 스테이징 풀 | `StagingPoolTests` | 크기 클래스 반올림, 같은 인스턴스 재사용, 최적합 선택, 총량 상한, 프레임 반복 시 풀 크기 불변 |
| 하네스 자신 | `RenderHarnessTests` | **동치성 단언이 다름을 실제로 잡는지**(§4-2), 동기 리드백의 실패 보고 |
| Metal 매핑 | `MetalMappingTests` | 스텐실 연산·비교 함수 **전수**(CaseIterable), 네 연산이 제 슬롯에 들어가는지, 마스크. GPU 불필요 |
| 스텐실 | `StencilTests` | 마스킹(안/밖) + **같은 영역의 시저와 프레임 전체 비교**, `setStencilReference`가 쓰기와 비교 양쪽에, read/writeMask, `depthFailOp`(섀도 볼륨 경로), `stencil8` 단독 포맷 회귀, **스텐실 성분 없는 포맷 + 비기본 상태 거부**, **`depthReadOnly`/`stencilReadOnly` 강제**(읽기만 하는 파이프라인은 통과), **음수 참조값이 프로세스를 죽이지 않는지** |
| 간접 드로우 | `IndirectDrawTests` | **직접 호출과 프레임 전체 동치성**(인자 칸 순서), `firstVertex`, 인덱스 바인딩 오프셋 + `firstIndex` 이중 적용 회귀, 간접 디스패치, **컴퓨트가 인자를 쓰는 GPU-driven 경로** |
| 오류 스코프 | `ErrorScopeTests` | 가로채기(전역으로 안 샘), 필터 매칭, 중첩에서 안쪽 우선 + 안 맞으면 바깥으로, **배치를 넘는 수명**, 처음 잡힌 하나만, **짝 없는 pop은 오류 대신 reject 상태**(인덱스도 안 민다), **필터를 못 읽어도 스택 깊이 유지**, reset, **두 겹(validation+internal)이 파이프라인의 두 실패를 모두 가져가는지** — 비동기 생성이 기대는 계약 |
| 렌더 번들 | `RenderBundleTests` | **직접 인코딩과 프레임 전체 동치성**, 재사용(두 프레임 연속), 실행 순서, **상태 격리 양방향**(파이프라인·바인드 그룹·정점 버퍼 셋 다), **실행 중 오류가 나도 격리가 성립하는지**, 포맷·어태치먼트 수 불일치, **후행 `null` 무시**, **어태치먼트 최소 하나**, **깊이 전용 MSAA 패스의 `sampleCount`**, **`depthReadOnly` 패스에는 readOnly 번들만**, 번들에 금지된 명령, **하나만 비호환이어도 앞의 호환 번들까지 미실행** |
| 쿼리셋 | `QuerySetTests` | occlusion은 **값 단언**(전체 통과 = 64×64, 완전히 잘린 드로우 = 정확히 0), 구간 resolve, 타임스탬프는 **구조만**(길이·단조·초기값 아님, 절대 시간 임계 금지), 기기 지원과 `adapter.features` 일치, 중첩·범위·usage·256 정렬 계약, **개수 상한(4096)**, **`timestampWrites` 인덱스 최소 하나·중복 금지**, **occlusion 미종료·인덱스 재사용 거부** |

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
│ 그대로 통과: 60/67  (89%)
│ 호스트가 constants를 줘야 하는 것: 4건 — …
├─ 실패 3건 ───────────────────────────────
│ ✗ cornell/rasterizer.wgsl
│     MSL[vs_main]: error: use of undeclared identifier 'common_uniforms'
```

남은 3건은 **코퍼스 쪽 사정**이다 (다른 `.wgsl`과 이어 붙여 쓰거나, 호스트가 문자열을
치환해 쓰는 조각 — `docs/WGSL.md` §4-1).

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
| `dynamic` | CPU 플라스마를 **매 프레임 writeTexture**로 — 큐 순서 업로드 + 스테이징 풀 + 업로드 처리량. HUD의 live 객체 수(`objects`)가 일정해야 정상 |
| `blending` | 미리 곱해진 알파 합성 + discard + 유니폼 구조체 배열 |
| `stencil` | **스텐실 마스크** — `stencil8` 단독 포맷(깊이 없음)에 같은 풀스크린 삼각형을 3번 그린다. 갈리는 이유가 스텐실뿐이라, 안 먹으면 화면이 한 색으로 덮인다. 마킹 패스는 `writeMask: 0`, 채우기 두 번은 `equal`/`not-equal`. 버튼이 `setStencilReference`로 마스크를 뒤집는다 |
| `gpudriven` | **간접 드로우/디스패치** — 컴퓨트가 이번 프레임의 개수를 정해 인자 버퍼에 쓰고, `dispatchWorkgroupsIndirect` + `drawIndexedIndirect`가 그 버퍼를 읽는다. HUD는 인자 버퍼를 되짚어 GPU가 정한 수를 띄운다. 직접 모드는 개수를 모르니 늘 최대치를 그린다 |
| `bundle` | **렌더 번들** — 드로우 120개를 한 번만 기록해 매 프레임 되돌린다. HUD의 커맨드 수는 `submit()` 반환의 `commandCount`라 추정이 아니다 (직접 128개 → 번들 6개) |
| `query` | **occlusion 쿼리 · 타임스탬프 · 오류 스코프** — 막대가 원을 가릴수록 살아남은 샘플 수가 줄고 완전히 가려지면 0이 된다. 타임스탬프는 `adapter.features.has('timestamp-query')`인 기기에서만 (시뮬레이터는 미지원으로 표시). 버튼 둘이 같은 잘못된 호출을 스코프 **안/밖**에서 실행해, 노란 줄과 빨간 줄로 갈리는 것을 보여 준다 |
| `readback` | 컴퓨트 결과를 `mapAsync`로 CPU가 읽어 화면에 표시 |
| `constants` | 같은 셰이더 모듈 + 다른 `override` 값 → 파이프라인 3개 |
| `msl` | `language: 'msl'` 직접 주입 + 명시적 파이프라인 레이아웃 |
| `interactive` | 홀로그래픽 카드 — Lynx 표준 터치 → 3D 자세 → 포일/정반사/반짝임. **위아래로 겹친 Lynx 컴포넌트의 입력 라우팅**도 함께 확인 |
| `wgsl` | `arrayLength()`로 센 칸 수 + 외부 텍스처(왼쪽만 가장자리 클램프) + 타입 없는 상수식. 셰이더가 센 길이를 CPU가 리드백으로 되짚어 HUD에 ✓/✗로 띄운다 |
| `bench` | **브리지 비용 측정** — 같은 커맨드의 `data`만 base64 문자열/`ArrayBuffer`로 바꿔 인코딩·제출 비용을 잰다. 네이티브가 두 표현을 다 받으므로 그 아래(스테이징·blit)는 완전히 같다. 캔버스를 쓰지 않는다 |
| `arraybuffer` | **Lynx 값 변환기 스모크** — 바이트열이 `ArrayBuffer`로 **양방향** 오가는지 본다. 올릴 때는 커맨드의 중첩 위치(`commands[i].data`), 내릴 때는 `mapAsync`. 페이로드 타입까지 단언한다. 화면이 초록이면 통과, 빨강이면 실패 |
| `hdr` | **HDR 게인맵 재구성** — `loadAsset`으로 받은 애셋을 컴퓨트로 `rgba16float`에 되살리고, 좌우로 갈라 8비트 원본과 같은 조건으로 비교한다. 드래그로 경계 이동. 버튼 셋: 노출 ±, **클리핑**(원본 선형값이 1.0을 넘는 픽셀만 표시 — 오른쪽에만 떠야 정상), **EDR**(캔버스를 `rgba16float` + `toneMapping: extended`로 재configure). **EDR은 실기기에서만 확인된다** |
| `scrollpass` | **스크롤 통과** — `<scroll-view>` 리스트 **위에** 캔버스 밴드가 형제로 겹친다. 밴드를 세로로 드래그: `passthrough-touches` ON이면 리스트가 스크롤되고 캔버스는 `touchcancel`을 받는다, OFF면 웹 기본처럼 스크롤이 막히고 `touchmove`가 계속 온다. HUD의 스크롤 오프셋·터치 로그로 판정한다. **터치 주입 수단이 없어 손으로 만져야 한다** |
| `three` | **three.js WebGPURenderer 기능 체크리스트** — 렌더러가 자기 부트스트랩(navigator.gpu → adapter.features → requiredFeatures → device.lost → rAF 루프)을 그대로 밟은 뒤, 렌더 타깃에 그려 `readRenderTargetPixelsAsync`로 **픽셀 값을 단언**한다: shim 직접 프로브 2종(버퍼/텍스처 왕복 — three 실패 시 층 가르기용) → 클리어색 → 단색 쿼드(노드 셰이더→WGSL) → DataTexture 샘플링 → Standard+Directional 조명. HUD에 ✓/✗와 실제 (r,g,b), 스트림 통계(P/I 배치·오류 수)가 뜨고 체커 큐브가 회전한다. **기록 시점 스냅샷 버그를 잡아낸 화면**이다 (copySize가 flush 전에 reset되어 폭 0 복사가 나가던 것) |

`interactive` · `hdr` · `scrollpass`는 **모달 전체 화면**으로 올라온다 (닫기 버튼은 화면 왼쪽 위).
밀어서 뒤로 가기 제스처가 캔버스를 끄는 드래그를 가로채기 때문이다 — `DemoScene.coversFullScreen`.

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

# 시뮬레이터에는 터치를 주입할 방법이 없다. 손으로 눌러야 보이는 상태는 런치 인자로 고정한다:
xcrun simctl launch <device> org.lynxwebgpu.demo -demo interactive -cardTilt 0.42
# -altMode 1 은 토글이 있는 씬(stencil·gpudriven·bundle)을 **기본이 아닌 쪽**으로 시작시킨다.
xcrun simctl launch <device> org.lynxwebgpu.demo -demo bundle -altMode 1
xcrun simctl io <device> screenshot .tmp/demo-cube.png
```

번들 소스(JS/TSX)는 `Projects/WebGPUDemo/DemoSrc`이며, 고친 뒤에는 번들을 다시 만들어 넣는다:

```zsh
cd Projects/WebGPUDemo/DemoSrc && mise exec -- npm run sync   # build + Resources/ 로 복사
```
