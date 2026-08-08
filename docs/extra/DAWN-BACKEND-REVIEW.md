# Dawn 백엔드 — 대체 가능 영역 검토

> 목적: Lynx 브리지와 JS shim은 그대로 두고, **엔진(WGSL 트랜스파일러 + Metal 백엔드)을
> [google/dawn](https://github.com/google/dawn)으로 갈아끼울 수 있는지**와, 그러려면 지금
> 무엇을 먼저 잘라 두어야 하는지를 본다. 구현이 아니라 절단면 검토다.
>
> 전제: 고르는 주체는 **호스트 앱**(SPM 소비자)이다. JS 번들은 어느 쪽이든 손대지 않는다 —
> 이 성질을 지키는 것이 아래 설계의 제1 제약이다.

## 0. 한 줄 요약

커맨드 스트림이 **이미 백엔드 중립 프로토콜**이라, 절단면은 `execute(payload)` 하나로 충분하다.
Dawn의 사정권에 드는 것은 Swift 10,387줄 중 **약 5,000줄(≈50%)**이고, 그중 **3,498줄이
트랜스파일러 하나**다. 나머지 절반(브리지·shim·디코딩·애셋·프레임 정책)은 Dawn에 대응물이 없다.

## 1. 모듈별 대조

| 이 저장소 | 줄 수 | Dawn 대응 | 판정 |
|---|---:|---|---|
| **LynxWebGPUShader** 전체 | 3,498 | **Tint** (`src/tint/lang/wgsl/reader` → `lang/msl/writer`) | **전량 대체** |
| `WGPUResources` · `WGPUPipeline` · `WGPUMetalMapping` | 1,309 | `dawn::native` Metal 백엔드 | **전량 대체** |
| `WGPUStagingPool` | 94 | `wgpuQueueWriteBuffer` 내부 업로드 링 | **삭제** |
| `WGPUCommandInterpreter` | 1,861 | 인코딩 절반만 (`wgpuRenderPassEncoder*` 등) | **절반 대체** |
| `WGPUSurface` | 274 | `WGPUSurfaceSourceMetalLayer` + configure/getCurrentTexture/present | **절반 대체** |
| `LynxWebGPUContext` | 421 | `adapterInfo()`의 limits 손계산 → `wgpuAdapterGetLimits` | **일부 대체** |
| `WGPUEnums` · `WGPUDescriptors` | 1,276 | 구조체 **모양**은 있으나 문자열 raw value 디코딩은 없다 | **거의 유지** |
| `WGPUValueReader` · `WGPUHandle` | 497 | 개념은 `dawn::wire` client/server와 같으나 Lynx 딕셔너리 전용 | **유지** |
| `WGPUError` | 90 | `WGPUErrorType`·에러 스코프는 있음. `commands[3].vertex…` 경로는 없음 | **유지** |
| `WGPUAssetProvider` · `WGPUImageDecoder` · `WGPUPixelReadback` | 468 | 없음 (브라우저 측 관심사) | **유지** |
| **LynxWebGPUBridge** 전체 | 579 | 없음 | **유지** |
| **JS/webgpu.js** | 2,578 | `dawn::wire` client가 개념적 대응물이나 JS에서 못 쓴다 | **유지** |

### 1-1. Dawn이 주는 것 중 지금 없는 것

| 얻는 것 | 지금 상태 |
|---|---|
| 완전한 WGSL | `docs/WGSL.md` §4의 거부 목록(`atomicCompareExchangeWeak` · `modf`/`frexp` · `workgroupUniformLoad` · `break if`)이 사라진다. 코퍼스 92% → 사실상 100% |
| 셰이더 로버스트니스 | **자체 구현으로 해소됐다** — `docs/WGSL.md` §3-2 "안전 변환"(인덱싱 클램프·정수 나눗셈·시프트·포화 변환·workgroup 0초기화 등 7종). Dawn 도입 논거에서 빠진다 |
| 명세 검증 전부 | `docs/ARCHITECTURE.md` §8이 "의도적으로 안 한다"고 적어 둔 사용 플래그·포맷 호환성·동기화 스코프 검증. **브라우저와 같은 곳에서 같이 거부된다** = 이식성 |
| 진짜 렌더 번들 | 지금은 명령 목록 재생(`WGPURenderBundleObject`) |
| Vulkan / D3D12 / GL 백엔드 | **Android·데스크톱 경로.** Lynx가 크로스 플랫폼인데 이 패키지는 Metal 전용이다 — 구조적으로 가장 큰 논거다 |
| CTS 적합성 | 상류가 검증한다 |

### 1-2. Tint가 우리 손계산과 정확히 겹치는 지점

`src/tint/lang/msl/writer/common/options.h`:

| 우리 것 | Tint Options |
|---|---|
| `WGSLBindingAssigner` (`@group/@binding` → Metal 인덱스) | `bindings` (binding remapper) |
| `WGSLMetalLimits.bufferSizesIndex` (22번 `arrayLength` 크기 표) | `array_length_from_constants` |
| `WGSLMetalLimits.vertexBufferIndex` (30번부터 역순) | `vertex_pulling_config` / 같은 remapper |
| — | `disable_robustness` · `fixed_sample_mask` · `strip_all_names` |

즉 **배정표를 그대로 넘겨 주면 된다.** 지금 방출기와 인코딩이 같은 표를 보는 계약이 그대로 산다.

## 2. 절단면 — `execute(payload)` 하나

세 후보를 놓고 봤다.

| 안 | 절단 위치 | 평가 |
|---|---|---|
| **A** | `LynxWebGPUContext` 전체를 프로토콜로 (런타임 = 커맨드 스트림 실행기) | **채택.** 추상화가 새지 않고 JS·브리지가 무변경 |
| B | 디코딩(공유) / 인코딩(백엔드) 사이에 백엔드 프로토콜 | 공유 코드는 늘지만 **의미가 갈리는 자리에서 정확히 샌다** — 에러 스코프·번들·present 시점·매핑을 Dawn은 스스로 한다. 우리 것과 이중 관리가 된다 |
| C | 셰이더만 (MSLEmitter ↔ Tint) | **하지 않기로 정했다** (아래) |

### 왜 Tint 단독 투입을 하지 않나

기술적으로는 가장 작은 위험으로 가장 큰 이득을 얻는 경로다(§1-2). 그런데 **Dawn을 도입한다면
전면 도입**이라는 것이 이 프로젝트의 결정이므로, Tint 단독은 목적지가 아니라 **중간 정거장**이 된다.
그 중간 정거장에는 값보다 비용이 크다:

- Tint를 붙이는 순간 **CMake → XCFramework 빌드 파이프라인이 이미 필요하다.** 전면 Dawn과
  같은 비용을 먼저 치르면서 얻는 것은 셰이더 층뿐이다.
- 그 상태는 **트랜스파일러 두 벌(우리 것 + Tint)을 동시에 유지**하는 기간을 만든다. 전면 Dawn으로
  가면 어차피 둘 다 버려지므로 그 유지 비용이 회수되지 않는다.
- 바이너리 크기라는 결정적 관문(§4)을 **Tint 크기로 재게 되어** 판단이 낙관 쪽으로 치우친다.
  재야 할 것은 처음부터 `libwebgpu_dawn` 전체다.

그래서 선택지는 **"지금 구조를 유지"** 또는 **"전면 Dawn 런타임"** 둘뿐이고, 절단면 A가
그 둘을 잇는다.

A를 고르는 이유는 하나다: **커맨드 스트림이 이미 직렬화된 백엔드 중립 프로토콜**이다
(`{op, …}` 배열 → `{ok, errors, canvases, errorScopes}`). 두 런타임이 지켜야 할 계약이 이미
데이터로 존재하므로, 추가 추상화를 발명할 필요가 없다.

> 2026-08-08 보충: 절단면은 A 그대로다. 다만 Dawn이 대신해 주지 **않는** 와이어 정책 —
> 디스패치 표(`WGPUCommand`), 오류 스코프 스택(`WGPUErrorScopeStack`), 응답 모양
> (`WGPUBatchResult`), 지연 오류 전달(`WGPUDeferredErrorQueue`) — 은 Core의 공용 타입이
> 되어, B가 우려한 이중 관리 없이 두 런타임이 같은 코드를 쓴다. Dawn 구현이 다시 쓰는 것은
> 정말로 "인코딩"뿐이다.

```
                        JS/webgpu.js  (무변경)
                              │  {commands, present}
                    LynxWebGPUBridge  (무변경 — 프로토콜만 봄)
                              │
                    ┌─────────┴─────────┐
        LynxWebGPUContext        DawnWebGPUContext
        (Metal 직접)              (dawn::native + Tint)
              │                          │
      LynxWebGPUShader              libwebgpu_dawn.xcframework
              └──────── LynxWebGPUCore ─┘   (디코딩·에러 모양 공유)
```

## 3. 선행작업 — 모듈 단위로 자른 순서

각 항목은 **Dawn이 끝내 안 들어와도 그 자체로 값이 있다.** 그게 선행작업의 조건이다.

> **진행 상태 (2026-08-08):** 1~5에 이어 **인터페이스 완결(5+)**까지 끝났다. 남은 것은
> (6) 빌드 경로뿐이고, 거기서 재는 바이너리 크기가 도입 여부를 가른다.
>
> | # | 항목 | 상태 |
> |---|---|---|
> | 1 | `WebGPURuntime` 프로토콜 | **완료** — `Sources/LynxWebGPUCore/WebGPURuntime.swift`. 브리지가 `LynxWebGPUCore`만 의존한다 |
> | 2 | 디코딩 완결 | **완료** — `Sources/LynxWebGPUCore/WGPUCommands.swift`. 해석기의 인라인 필드 읽기 81곳 → 0곳 |
> | 3 | 커맨드 스트림 문서 | **완료** — `docs/COMMAND-STREAM.md` (51개 op 전수 + §5-1 스레딩·수명 규약 + §5-2 adapterInfo 등급) |
> | 4 | 적합성 스위트 분리 | **완료** — `Sources/LynxWebGPUConformance`(라이브러리 product). **28개 검사** — 프레임 수명(present:false 생존·만료)·readBuffer·resize·컴파일 진단·msl-optional·decodeImage 포함 |
> | 5 | 프레임 정책 분리 | **완료** — `WGPUFrameCoordinator` · `WGPUFrameBoundary`(Core). GPU 없이 검증된다 |
> | 5+ | 인터페이스 완결 | **완료 (2026-08-08)** — ① 디스패치 표 `WGPUCommand`(51케이스 열거형 — 백엔드는 default 없는 switch만 쓰고, op 추가 누락은 컴파일이 잡는다) ② 와이어 정책의 Core 이관 (`WGPUErrorScopeStack` · `WGPUBatchResult` · `WGPUDeferredErrorQueue`) ③ 비동기 펌프 훅 `processEvents()` ④ `RenderHarness` 런타임 매개변수화 ⑤ 외부 주입 픽스처 `Examples/ExternalRuntime` (Core+Conformance만 링크한 스텁이 28검사를 전부 판정) |
> | 6 | 빌드·배포 경로 | **시제품 완료 (2026-08-08)** — 프리빌트는 [Dawn-xcFramework](https://github.com/xenonClient/Dawn-xcFramework)(별도 저장소, SPM binaryTarget)가 제공하고, 그 위의 `WebGPURuntime` 구현 시제품이 `Projects/DawnCheck`에 있다. **시뮬레이터 적합성 27/28 통과 · 1 건너뜀(간접 드로우 — 시뮬레이터 제약으로 미광고) · 0 실패.** 화면 표면(`WGPUSurfaceSourceMetalLayer`)까지 배선되어 **DawnDemo 앱이 데모 씬(triangle·cube)을 브리지·JS 무변경으로 그린다** (wgsl 씬은 §4 동작 발산 사례 — uniformity 위반으로 거부되고 오류 오버레이가 뜬다) — in-flight 페이싱은 `WGPUFrameCoordinator`의 `noteCommitted`/`noteCompleted` 그대로다. **24씬 전체 스위프 결과는 `docs/TESTING.md` §2-1** — 15씬 무오류(three.js 16/16 포함), 발산 4건(전부 씬의 명세 위반), 이 빌드 제약 1건(비교 샘플러 비활성). 남은 것: 시제품을 실제 배포 저장소(Lynx-WebGPU-Dawn)로 옮기고 실기기 확인 |
>
> Dawn 런타임이 채워야 할 자리는 이제 **`WebGPURuntime`의 14개 멤버**(기본 no-op인
> `processEvents` 포함)로 닫혀 있고, 지켜야 할 계약은 `docs/COMMAND-STREAM.md`, 증명 수단은
> `WebGPUConformance.run(on:)`(28검사), 출발점은 `Examples/ExternalRuntime`의 `StubRuntime`
> (컴파일 가능성)과 `Projects/DawnCheck`의 `DawnWebGPURuntime`(실물 — 27/28)이다.

### 1) `WebGPURuntime` 프로토콜 추출 · 소 · 위험 낮음

브리지가 `LynxWebGPUContext` 구체 타입을 보던 것을 프로토콜로 바꿨다. **실제 시그니처**
(정본은 `Sources/LynxWebGPUCore/WebGPURuntime.swift` — 제안 당시 모양에서 표면 API가
attach/detach/resize/오프스크린/픽셀 읽기 5멤버로 자라고, 적합성 스위트의 픽셀 통로가
계약에 들어갔다):

```swift
public protocol WebGPURuntime: AnyObject {
    func execute(_ payload: [String: Any]) -> [String: Any]      // 조립은 WGPUBatchResult로
    func adapterInfo() -> [String: Any]
    func shaderCompilationInfo(handle: Int) -> [String: Any]
    func canvasInfo(identifier: String) -> [String: Any]
    func readBuffer(handle: Int, offset: Int, size: Int?, completion: @escaping ([String: Any]) -> Void)
    func decodeImage(handle:data:name:options:provider:completion:)
    func attachCanvas(identifier: String, layer: CAMetalLayer)   // 레이어 초기 속성은 런타임 몫
    func attachOffscreenCanvas(identifier: String, size: CGSize) throws   // 적합성 픽셀 통로
    func resizeCanvas(identifier: String, drawableSize: CGSize)
    func detachCanvas(identifier: String)                        // 임의 스레드에서 온다
    func readCanvasPixels(identifier: String) throws -> WGPUPixelReadback // 적합성 픽셀 통로
    var isReadyForNextFrame: Bool { get }
    func processEvents()   // 기본 no-op — Dawn의 wgpuInstanceProcessEvents 자리
    func reset()
}
```

핵심은 **`WGPUMetalLayerSurface` 생성을 브리지에서 걷어내는 것**이다. `<webgpu-canvas>`가
`CAMetalLayer`를 backing layer로 쓰는 사실은 양쪽 공통이므로 (Dawn도 `WGPUSurfaceSourceMetalLayer`가
같은 레이어를 받는다) 엘리먼트 코드는 정말로 무변경이 된다 — 레이어의 초기 속성
(`pixelFormat` 등)까지 런타임이 정하므로 엘리먼트는 레이어를 넘기기만 한다.

> Dawn 없이도 이득: 브리지를 가짜 런타임으로 단위 테스트할 수 있게 된다 (지금은 못 한다 —
> 브리지에 테스트가 하나도 없는 이유이기도 하다).

### 2) 디코딩 완결 — 순수 리팩터 · 중 · 위험 낮음

지금 **51개 op 중 14개만** Core의 타입 있는 디스크립터를 지난다. 나머지는 해석기 안에서
리더를 직접 읽는다 (`command.int(…)` 계열 **81곳**):

| 지나는 것 (14) | 안 지나는 것 |
|---|---|
| `createBuffer` `createTexture` `createTextureView` `createSampler` `createShaderModule` `createBindGroupLayout` `createPipelineLayout` `createBindGroup` `createQuerySet` `createRenderBundle` `createRenderPipeline` `createComputePipeline` `beginRenderPass` `beginComputePass` | `writeBuffer` `writeTexture` `copy*` 5종 `getCurrentTexture` `configureCanvas` `setPipeline` `setBindGroup` `setVertexBuffer` `setIndexBuffer` `setViewport` `setScissorRect` `setBlendConstant` `draw*` 4종 `dispatch*` 2종 `executeBundles` `*OcclusionQuery` `resolveQuerySet` `*DebugGroup` `clearBuffer` `unmapBuffer` |

전부 Core의 struct로 올린다. 그러면 **디코딩 단계가 100% 공유**되고, Dawn 구현자는 인코딩
절반만 쓰면 된다. 동작 변경이 0이므로 기존 테스트가 그대로 회귀 그물이 된다.

> Dawn 없이도 이득: `CLAUDE.md`가 경고하는 "JS와 Swift의 필드 이름이 어긋나도 양쪽 다
> 컴파일된다" 문제의 표면이 절반으로 준다 — 이름이 한 곳(Core)에만 남는다.

### 3) `docs/COMMAND-STREAM.md` — 소 · 위험 없음

op × 필드 × 의미를 명세로 적는다. 지금은 `.claude/skills/webgpu-command/SKILL.md`와 코드에
흩어져 있다. 이게 곧 **Dawn 구현자의 계약서**이자, 두 런타임의 동치를 판정하는 기준이 된다.

### 4) 적합성 테스트 분리 — 중 · 위험 낮음

지금 `Tests/LynxWebGPUTests`가 두 종류를 섞고 있다:

| 종류 | 파일 | Dawn 경로에서 |
|---|---|---|
| 커맨드 스트림 계약 (런타임 무관) | `CommandInterpreterTests` `RenderPipelineTests` `ErrorScopeTests` `RenderBundleTests` `QuerySetTests` `StencilTests` `IndirectDrawTests` `CompressedTextureTests` `ExternalImageTests` `OffscreenReadbackTests` | **그대로 재사용 — 이게 동치 증명이다** |
| Metal 내부 | `MetalMappingTests` `StagingPoolTests` `SurfaceInFlightTests` `RenderHarnessTests` | Metal 경로 전용으로 남는다 |

`RenderHarness`를 런타임으로 매개변수화한다 (**완료** — `make(runtime:width:height:)`가
주입점이고, Metal 내부 관찰만 `context` 탈출구로 남았다). 그러면 같은 스위트를 두 백엔드에
돌려 **"픽셀까지 같은가"**를 기계로 확인할 수 있다. 이게 없으면 Dawn 전환은 신뢰 근거가 없다.

### 5) 프레임 정책을 런타임 밖으로 — 소~중 · 위험 중

두 백엔드가 공통으로 필요로 하는데 **Dawn이 대신해 주지 않는** 것들이다:

- **present 지연** (`execute({present:false})` — 프레임 중간 내부 제출). 명세의 "프레임 끝에
  만료"를 지키는 정책이지 Metal 사정이 아니다.
- **in-flight 백프레셔** (`maxFramesInFlight = 3`, 포화 시 틱 스킵). Dawn에는 "지금 present하면
  블록되는가"를 묻는 공개 API가 없다 — `wgpuQueueOnSubmittedWorkDone`으로 **다시 세워야 한다.**
- 드로어블 텍스처/뷰의 프레임 스코프 핸들 만료.

`WGPUFrameCoordinator`로 뽑고, 런타임은 "드로어블 획득 / present / 완료 통지"만 노출한다.

> 이건 순수 리팩터가 아니다. 실기기 프레임 페이싱에 직결되므로 `SurfaceInFlightTests`를
> 먼저 두껍게 만든 뒤 손대는 게 맞다.

### 6) 빌드·배포 경로 — 대 · 위험 높음

Dawn은 CMake다. **SPM 타깃이 될 수 없다.** 이 저장소가 Lynx에 대해 이미 내린 결론
(`Package.swift`의 긴 주석 — "버전·배포처를 앱이 정하게 한다")을 그대로 적용하는 것이 맞다:

```
Lynx-WebGPU            (이 저장소 — 외부 의존성 0 유지)
  └ LynxWebGPUCore / Shader / LynxWebGPU / Bridge(소스)

Lynx-WebGPU-Dawn       (별도 저장소)
  └ DawnWebGPURuntime  → WebGPURuntime 구현
  └ libwebgpu_dawn.xcframework  (CMake로 만든 산출물, ios-arm64 + ios-arm64-simulator)
```

이렇게 하면 **이 패키지의 "의존성 0" 성질이 그대로 유지**되고, Dawn을 쓰는 앱만 사이드
패키지를 추가한다. `Package.swift`에 트레이트를 거는 것보다 이쪽이 이 저장소의 기존 판단과 일관된다.

Swift ↔ Dawn은 **C API(`webgpu.h`)로 충분하다** — C++ interop 불필요, 모듈 맵 하나면 된다.

## 4. 비용과 위험 — 정직하게

| 항목 | 내용 |
|---|---|
| **바이너리 크기** | **실측 완료 (2026-08-08, Dawn-xcFramework 릴리스 빌드)** — 정적 라이브러리 19MB(기기 슬라이스), 런타임+적합성 스위트까지 **링크한 최종 바이너리 11MB.** "900MB대" 보고는 디버그 미스트립 수치였다. 단독 기각 사유가 아니다 — 관문 통과 |
| **빌드 파이프라인** | CMake → 기기/시뮬레이터 슬라이스 → XCFramework → CI. Tuist 데모까지 엮으면 작지 않다 |
| **비동기 펌프** | **해소** — `WebGPURuntime.processEvents()`(기본 no-op)가 그 자리다. 프레임 티커가 준비 게이트 **앞**에서 틱마다 부르고 (완료 통지가 펌프에서 나오면 게이트 뒤에선 포화가 안 풀린다), 적합성 하네스도 콜백 대기 중에 부른다. 티커 없는 구성은 런타임이 자체 대기 수단을 갖출 것 (프로토콜 문서) |
| **동작 발산** | Dawn은 엄격히 검증한다 — **지금 도는 코드 중 일부가 거부된다.** 그게 목적이긴 하지만, 문서화된 선택이어야 한다. **실사례 (2026-08-08):** wgsl 데모 씬의 varying 분기 안 `textureSample`이 uniformity 분석으로 거부됐다 — 자체 트랜스파일러는 관대하게 통과시키던 명세 위반이고, 브라우저에서도 거부됐을 코드다. 샘플을 분기 앞으로 올리면 해소됨을 확인했지만 데모 씬은 무변경으로 두었다 — 수정 여부는 별도 결정 (`docs/TESTING.md` §2-1) |
| **진단 품질 하락** | 우리 오류에는 `commands[3].vertex.buffers[0].format` 경로와 한국어 메시지가 붙는다. Dawn은 영어 문자열 하나다. `kind` 매핑은 문제없다 (`WGPUErrorType_Validation`/`OutOfMemory`/`Internal` ↔ 우리 4종) |
| **시뮬레이터 제약** | 간접 드로우 미지원 같은 기기 갈림은 Dawn을 써도 그대로다 (Metal 자체 제약) |

## 5. 무엇을 해도 우리 것으로 남는가

- Lynx 브리지 (NativeModule · `<webgpu-canvas>` · CADisplayLink 티커) — 579줄
- JS shim과 커맨드 스트림 포맷 — 2,578줄
- JS 딕셔너리 → 디스크립터 디코딩, 문자열 raw value — 1,276줄
- 애셋 공급자 · 이미지 디코딩 — 308줄
- 캔버스 식별·등록 모델, 프레임 정책, 오류 경로 진단

**합쳐서 Swift의 절반 + JS 전부.** Dawn은 "엔진을 빌려주는 것"이지 이 라이브러리를 대체하지 않는다.

## 6. 권하는 순서

1. **(2) 디코딩 완결** → 순수 리팩터, 지금 당장 값이 있다
2. **(1) `WebGPURuntime` 프로토콜** + **(3) 커맨드 스트림 문서**
3. **(4) 적합성 스위트 분리** — 여기까지가 "선택 가능한 구조"의 완성이다
4. **(5) 프레임 정책 분리** — 두 백엔드가 공통으로 필요로 하는데 Dawn이 대신해 주지 않는 것
5. **(6) 빌드 경로로 `libwebgpu_dawn` 크기를 실측** — 이것이 도입 여부를 가르는 관문이다
6. 크기가 받아들일 만하면 **전면 Dawn 런타임**을 `WebGPURuntime` 구현으로 쓴다

3번까지는 Dawn 도입 여부와 무관하게 코드가 좋아진다. **5번이 실질적 분기점**이다 —
중간 단계(Tint 단독)를 두지 않기로 했으므로, 판단 근거는 처음부터 전체 크기여야 한다.

---

참고: [Dawn](https://dawn.googlesource.com/dawn/+/refs/heads/main/README.md) (BSD-3-Clause) ·
[webgpu-headers `Surfaces`](https://webgpu-native.github.io/webgpu-headers/Surfaces.html) ·
[비동기 연산](https://webgpu-native.github.io/webgpu-headers/Asynchronous-Operations.html) ·
[MapLibre Native의 iOS Dawn 빌드](https://maplibre.org/maplibre-native/docs/book/platforms/ios/ios-webgpu.html)
