# 로드맵

무엇을 먼저 할지는 **이 라이브러리가 실제로 쓰이는 곳**을 기준으로 정한다. 주 사용처는
Lynx 앱 안의 **이미지 에디터·필터**지 3D 렌더러가 아니다. 그래서 같은 미지원 목록
(`docs/WEBGPU-API.md` §8)이라도 3D 기준으로 매긴 우선순위와는 순서가 다르다.

| 순서 | 기능 | 규모 | 효과 |
|---|---|---|---|
| 1 | 웹 라이브러리 이식 갭 (§1) | 중 | **표면은 다 메웠다** — 남은 것은 상위 기능 검증(섀도맵·포스트프로세싱·TSL)과 실기기 확인 |
| 2 | 이미지 처리 경로 다듬기 (§2) | 소~중 | limits 노출 · 색공간 · 큰 이미지 업로드 |
| 3 | 트랜스파일러 코퍼스 (§3) | — | **숙제 없음** — 92%, 남은 1건은 호스트 생성 코드가 필요한 셰이더다 |
| — | 압축 텍스처 (ASTC · ETC2 · BC) | 대 | **보류** — 편집 파이프라인에 끼울 수 없다 (아래) |

### 이미 된 것

| 기능 | 비고 |
|---|---|
| 캔버스 EDR 출력 (`colorSpace` · `toneMapping`) | `hdr` 데모 씬에서 확인. 실기기 전용 |
| 애셋 로딩 (`loadAsset` + `WGPUAssetProvider`) | 브라우저 `fetch()` 자리 |
| 스텐실 상태 (`stencilFront`/`stencilBack`/마스크) | `StencilTests`. `stencil8` 단독 포맷도 |
| 간접 드로우/디스패치 (`drawIndirect` 계열 3종) | `IndirectDrawTests`. 컴퓨트가 인자를 쓰는 GPU-driven 경로까지 |
| 오류 스코프 (`pushErrorScope` / `popErrorScope`) | `ErrorScopeTests`. GPU 불필요 |
| 렌더 번들 (`GPURenderBundle` / `executeBundles`) | `RenderBundleTests`. 명령 목록을 되풀이하는 방식 |
| 쿼리셋 (occlusion · 타임스탬프 · `resolveQuerySet`) | `QuerySetTests`. 타임스탬프는 `adapter.features`로 기기 확인 |
| read-only 깊이/스텐실 어태치먼트 | `StencilTests`. 선언만 받는 것이 아니라 **강제**한다 |
| 버퍼 매핑 상태 (`mapAsync` ~ `unmap`) | `CommandInterpreterTests`. 매핑 중 큐 작업 거부 — 리드백 경쟁을 없앤다 |

공통 절차는 `.claude/skills/webgpu-command/SKILL.md`(새 op 추가)와
`.claude/skills/wgsl-feature/SKILL.md`(셰이더 문법 — §3만 해당)를 따른다.
§1·§2는 **트랜스파일러를 건드리지 않으므로** 외부 코퍼스 재측정(`docs/TESTING.md` §7)이 불필요하다.

---

## 1. 웹 라이브러리 이식 갭

Three.js `WebGPURenderer`를 실제로 올려 보며(데모 `three` 씬) 드러난 것들이다. 검증 방식은
그 씬의 체크리스트를 따른다 — **렌더 타깃에 그리고 픽셀 값을 단언**하므로 "도는 것 같다"가
아니라 값으로 확인된다.

| 항목 | 규모 | 왜 필요한가 |
|---|---|---|
| 상위 기능 검증 — 남은 것 (섀도맵 · 포스트프로세싱 · TSL 컴퓨트 · glTF 로딩) | 중 | 체크리스트에 행을 늘려 가며 하나씩. 각각이 다른 경로(비교 샘플러 · 다중 패스 · 자체 컴퓨트 파이프라인 · `fetch`→`loadAsset`)를 밟는다 |
| 실기기 확인 | — | 지금까지 전부 시뮬레이터다. `three` 체크리스트와 `hdr` EDR(시뮬레이터로는 원리적으로 불가)은 기기에서 한 번 봐야 한다 |

### 이미 메운 것 (참고)

| 갭 | 어떻게 |
|---|---|
| `device.features` / `device.lost` | 명세대로 노출. `features`는 요청한 것만, `lost`는 영원히 pending |
| `self` · `performance` · `navigator` · `GPU*` 전역 | shim이 `lynx*` 전역을 얹고 번들러 `define`이 bare 식별자를 잇는다 (`docs/JS-AUTHORING.md` §10) |
| 프레임 중간 제출이 드로어블을 present하던 문제 | `execute({present})` — `popErrorScope`·`mapAsync`는 내부 제출로 표시 |
| 커맨드 인자가 기록 뒤 변이되던 문제 | `Recorder.push`가 기록 시점 값으로 고정 (three가 디스크립터를 `reset()`한다) |
| WGSL 전역 섀도잉 오역 | 사용 분석이 스코프를 따른다 (기계 생성 셰이더의 일상 패턴) |
| `createRenderPipelineAsync` / `createComputePipelineAsync` | 오류 스코프 두 겹으로 감싸 즉시 제출, `GPUPipelineError`로 거부 |
| `device.onuncapturederror` | 명세 통로 + `addEventListener`. `GPUValidationError` 계열로 갈린다 |
| `requestAnimationFrame` | shim의 `installAnimationFrame()` — `startFrameLoop` 위에 얹힌다 |
| `entryPoint` 생략 | 명세의 "get the entry point" — 그 스테이지의 유일한 진입점을 쓴다 (three 밉맵이 이걸 밟는다) |
| `adapter.limits` | 명세 `GPUSupportedLimits` 전 항목을 명세 철자로 |
| 명세 읽기 전용 속성 | `GPUTexture` 7종 · `GPUBuffer.mapState` |
| `clearBuffer` · `copyBufferToBuffer` 짧은 형태 | 명세 오버로드까지 |
| 상위 기능 검증 (깊이 · 인스턴싱 · 밉맵 · 블렌딩 · 비동기 컴파일) | `three` 씬 체크리스트 14종이 픽셀 값으로 확인한다 |

---

## 3. 트랜스파일러 코퍼스 — 남은 1건

`webgpu-samples` 완성 모듈 66개 중 **61개 통과(92%)**. 이전에 "실패 3건"으로 세던 것은
**하나가 진짜 버그, 둘은 측정 방식 문제**였다:

| 파일 | 실제 원인 | 결과 |
|---|---|---|
| `cornell/rasterizer.wgsl` | **버그였다** — 진입점이 부르지도 않는 함수까지 방출해, `layout: "auto"`에 없는 리소스를 그 함수가 찾다 실패했다 (`common.wgsl`을 여럿이 나눠 쓰는 구성) | 도달 가능한 함수만 방출하도록 고침 → 통과 |
| `cornell/tonemapper.wgsl` | `{OUTPUT_FORMAT}`을 호스트가 치환해 쓰는 **템플릿**이다 — 그대로는 WGSL이 아니다 | 하네스가 템플릿으로 분류 (분모에서 제외, 이름은 찍는다) |
| `skinnedMesh/gltf.wgsl` | 호스트가 glTF 접근자를 보고 `VertexInput`을 **런타임 생성**해 붙인다 | **남은 1건.** 파일만으로는 완성될 수 없다 — 지원 대상이 아니라고 보는 것이 맞다 |

즉 **트랜스파일러 쪽 숙제는 없다.** 통과율을 더 올리려면 코퍼스를 늘리는 편이 낫다
(Three.js 노드 시스템이 생성하는 셰이더가 좋은 후보다 — 사람이 안 쓰는 패턴을 기계가 만든다).

트랜스파일러를 고친 뒤에는 `docs/TESTING.md` §7의 통과율을 **반드시 다시 잰다** (로컬만
통과하고 코퍼스가 내려간 변경이 실제로 있었다).

---

## (보류) 압축 텍스처 (ASTC · ETC2 · BC)

> **왜 보류인가.** 블록 압축은 손실 압축인 데다 **렌더 타깃도 스토리지 텍스처도 될 수 없다** —
> GPU에 디코더는 있어도 인코더가 없기 때문이다. 이미지 에디터의 핵심 루프는 "텍스처를 읽어
> 필터를 걸고 다시 텍스처에 쓰기"인데 그 출력이 압축 포맷일 수 없고, 편집을 왕복할 때마다
> 세대 손실이 쌓인다. 품질이 훨씬 나은 BC7·ASTC라도 이 제약은 같으므로 **화질 문제가 아니라
> 구조 문제**다.
>
> 읽기 전용 에셋(스티커·브러시 소재)이 수백 장 들어가게 되면 그때 다시 본다. 그건 에셋 로딩
> 문제지 편집 파이프라인 문제가 아니다. 아래 설계는 그때를 위해 남겨 둔다.

### 목표

`createTexture` + `queue.writeTexture` + `copyBufferToTexture`로 블록 압축 텍스처를 만들고
채우고 샘플링할 수 있다. WebGPU 명세의 세 feature에 대응한다:

| WebGPU feature | 포맷 수 | Apple GPU 지원 |
|---|---|---|
| `texture-compression-astc` | 28 (`astc-4x4-unorm` … `astc-12x12-unorm-srgb`) | 전 iOS 기기 + Apple Silicon Mac |
| `texture-compression-etc2` | 10 (`etc2-rgb8unorm` … `eac-rg11snorm`) | 전 iOS 기기 + Apple Silicon Mac |
| `texture-compression-bc` | 14 (`bc1-rgba-unorm` … `bc7-rgba-unorm-srgb`) | `device.supportsBCTextureCompression`인 기기만 |

ASTC를 1순위로 한다 — iOS에서 보편이고 압축률 선택 폭이 가장 넓다. ETC2는 매핑만 추가하면
같은 경로를 타므로 동반 구현하고, BC는 지원 기기에서만 켠다 (Intel Mac 개발 루프는 BC만 가능).

### 바꿀 곳 (계층 순)

**LynxWebGPUCore — 포맷 메타데이터가 핵심 작업이다**

- `WGPUEnums.swift`의 `WGPUTextureFormat`에 케이스 52종 추가. **raw value는 명세 철자 그대로**
  (`"astc-4x4-unorm"` 등 — 바꾸면 JS가 깨진다).
- `bytesPerPixel`(주석부터 "블록 압축 포맷은 지원하지 않으므로"라고 못 박고 있다)을
  **블록 메타데이터로 일반화**한다:
  ```swift
  var blockSize: (width: Int, height: Int)   // 비압축은 (1, 1)
  var bytesPerBlock: Int                     // 비압축은 bytesPerPixel과 같다
  ```
  `bytesPerPixel`을 그대로 두고 압축 포맷에서 호출하면 잘못된 값이 조용히 퍼지므로,
  **비압축 전제인 호출처를 전수 수정**한다. 현재 호출처는 세 곳:
  `WGPUCommandInterpreter.writeTexture` / `copyTextureToBuffer` / `copyBufferToTexture`의
  기본 `bytesPerRow` 계산.

**LynxWebGPU**

- `WGPUMetalMapping.pixelFormat` — `MTLPixelFormat.astc_4x4_ldr`/`_srgb`, `.etc2_rgb8`,
  `.eac_r11Unorm`, `.bc1_rgba` … 매핑 추가. sRGB 변형 주의.
- `WGPUCommandInterpreter.writeTexture` — 기본/검증 로직을 블록 단위로:
  - 기본 `bytesPerRow` = `ceil(width / blockWidth) × bytesPerBlock`
  - `rowsPerImage`·데이터 부족 검증도 **블록 행** 기준 (`ceil(height / blockHeight)`)
  - `WGPUTextureObject.encodeWrite`는 바이트 단위 파라미터를 그대로 받으므로 수정 불필요 —
    Metal blit의 `sourceBytesPerRow`는 원래 블록 행의 바이트 수다.
- `copyBufferToTexture` / `copyTextureToBuffer` — 같은 블록 단위 계산.
- 압축 포맷 + `RENDER_ATTACHMENT`/`STORAGE_BINDING` usage는 Metal이 거부한다 —
  프로젝트 방침(검증은 Metal에 맡긴다)대로 두되, `createTexture`에서 디바이스가 포맷을
  지원하지 않으면 **명확한 `unsupported` 오류**를 낸다 (BC 미지원 기기, Intel Mac의 ASTC).
- `LynxWebGPUContext.adapterInfo()` — `features: ["texture-compression-astc", …]` 배열을
  실제 디바이스 지원(`supportsFamily(.apple2)`, `supportsBCTextureCompression`)으로 채운다.

**JS shim**

- `GPUAdapter`에 `features`(Set 흉내 — `has(name)` 메서드면 충분)를 추가해 웹과 같은
  분기(`adapter.features.has('texture-compression-astc')`)가 동작하게 한다.
- 포맷은 문자열 그대로 지나가므로 그 외 변경 없음. `webgpu.d.ts`에 feature 타입 추가.

### 테스트 (필수)

- **ASTC void-extent 블록을 손으로 만들어 픽셀로 단언한다.** void-extent는 16바이트로
  단색을 인코딩하는 ASTC 블록이라 외부 인코더 없이 결정적 테스트가 된다:
  `writeTexture(astc-4x4-unorm, 단색 블록)` → 풀스크린 샘플 → `assertPixel`.
- 포맷 메타데이터 표 단위 테스트 — 블록 크기·bytesPerBlock을 명세 표와 대조 (Core, GPU 불필요).
- `copyBufferToTexture`의 블록 정렬 `bytesPerRow` 기본값 테스트.
- 8×8 ASTC(2×2 블록) 업로드로 **여러 블록 행**의 stride 계산 검증.
- 지원 없는 디바이스에서는 `XCTSkipIf` — `RenderHarness`에 `supportsASTC` 헬퍼를 추가한다.
- `JSConstantParityTests` 성격의 스펙 철자 검증: 추가한 raw value 전수를 명세 문자열과 대조.

### 데모 씬 (필수 — 눈 검증)

`compressed` 씬: 같은 이미지를 `rgba8unorm` vs `astc-8x8-unorm`으로 나란히 그리고 HUD에
메모리 크기를 표기한다 (예: 256×256 기준 256KB vs 16KB). 에셋은 사전 인코딩한 base64를
번들에 넣는다 (`xcrun` 계열 인코더 또는 오프라인 생성 — 빌드 파이프라인에 넣지 않는다).

### 완료 기준

`swift test` 전부 통과(비지원 환경 skip 포함) · 데모 씬 60fps · `WEBGPU-API.md` §3 포맷 표와
§8 미지원 목록 갱신 · `adapter.features` 문서화 (`JS-AUTHORING.md` §3 표).

---

## 2. 이미지 처리 경로 다듬기

3D 기준 미지원 목록에는 없지만, 이미지 에디터에서는 이쪽이 먼저 아프다.

| 항목 | 지금 | 필요한 것 |
|---|---|---|
| ~~`maxTextureDimension2D`~~ | **끝났다** — 명세 `GPUSupportedLimits` 전 항목을 명세 철자로 싣는다 (`docs/WEBGPU-API.md` §1) | — |
| 색공간 | `hdr` 애셋을 빌드 시점에 sRGB로 변환해 둔다 | Display P3 원본을 그대로 다루려면 파이프라인 어디서 변환할지 정해야 한다. 감마를 두 번 먹거나 색역을 잘라먹기 쉬운 지점이다 |
| 큰 이미지 업로드 | 스테이징 풀은 있다 (`WGPUStagingPool`) | 수천만 픽셀을 올릴 때의 버퍼 재사용·분할 업로드 정책 |
| 필터 체인 중간 텍스처 | 씬이 직접 만든다 | 패스마다 만들고 버리지 않도록 풀링. 8비트로 왕복하면 밴딩이 쌓이므로 `rgba16float`가 기본이어야 한다 |

### 완료 기준

`maxTextureDimension2D`를 포함한 limits가 `adapter.limits`로 나가고,
`JS-AUTHORING.md`에 "큰 이미지를 다룰 때" 항목이 생긴다.

---

## 공통 완료 기준

1. `swift test` 전부 통과 — GPU 테스트는 `MetalCompilerHarness`가 아니라 `RenderHarness`
   픽셀 단언 (셰이더 문법 변경이 없으므로).
2. `cd JS && npm test` 통과 (shim을 바꾼 경우).
3. iOS 크로스 빌드 (`swift build --scratch-path .build-ios --sdk … --triple …`) 통과.
4. 데모 번들 재생성(`npm run sync`) + **씬을 추가했다면 `tuist generate`** (glob은 생성 시점에 펼쳐진다).
5. `WEBGPU-API.md` §8 미지원 표에서 해당 행 제거, `TESTING.md` 커버리지 표에 행 추가.
6. 커밋은 기능당 1개 이상, `feat:` 타입 한국어 요약 (`CLAUDE.md` Git Conventions).
