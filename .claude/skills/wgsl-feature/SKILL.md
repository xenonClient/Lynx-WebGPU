---
name: wgsl-feature
description: WGSL → MSL 트랜스파일러(LynxWebGPUShader)에 문법·타입·내장 함수를 추가하거나 번역 버그를 고친다. 렉서/파서/리플렉션/방출기 중 어디를 고쳐야 하는지, 배치 규칙과 필수 테스트 형태를 담는다. "WGSL의 X를 지원해줘", "셰이더가 컴파일 안 돼", "MSL 번역이 이상해" 같은 요청에 쓴다.
---

# WGSL 트랜스파일러 확장

## 0. 먼저 재현

증상을 **최소 WGSL**로 줄이고 실제 MSL을 눈으로 본다.

```swift
// Tests/LynxWebGPUShaderTests/ 에 임시 테스트를 넣고 돌리는 것이 가장 빠르다
let module = try WGSLShaderModule(source: source)
let bindings = try WGSLBindingAssigner.assign(groups: module.autoBindGroupLayouts(entryPoints: ["main"]))
print(try module.translateToMSL(entryPoints: ["main"], bindings: bindings))
```

런타임 경로에서도 볼 수 있다 — 파이프라인 생성이 실패하면 오류 메시지에 **줄 번호가 붙은 MSL 전문**이 들어 있다.

실패는 세 종류이고 고칠 곳이 다르다:

| 증상 | 어디 |
|---|---|
| `WGSL 파싱 실패 (line N)` | 렉서/파서 (§1, §2) |
| `MSL 컴파일 실패` + MSL 전문 | 방출기 (§3) |
| 컴파일은 되는데 **결과가 틀림** | 배치(§4) 또는 바인딩 배정(§5) |

## 1. 렉서 (`WGSLLexer.swift`)

새 토큰·연산자·리터럴 접미사가 필요할 때만. `operators` 배열은 **긴 것부터** 정렬돼 있어야 한다
(`<<=`가 `<`+`<=`로 잘리지 않게).

## 2. 파서 (`WGSLParser.swift`)

- **새 문장**: `parseStatement()`의 `switch current.text`에 케이스 추가 + `WGSLStatement`에 case 추가.
- **새 타입**: `parseType()`. 제네릭을 받는 이름이면 `templateNames`에도 넣는다.
- **`<` 모호성** — WGSL은 `<`가 제네릭인지 비교인지 문맥으로 갈린다. 제네릭 인자 안에서
  표현식을 파싱할 때는 `templateDepth`를 올려 `>`가 이항 연산자로 먹히지 않게 한다
  (`array<f32, 64>`의 길이 식이 이 경우다).
- 중첩 제네릭은 렉서가 `>>`로 붙여 놓으므로 `expectGenericClose()`로 한 겹씩 떼어낸다.

AST를 늘렸으면 **`WGSLReflectionBuilder.collect`의 식별자 수집에도 반영**한다 —
빠뜨리면 그 문장 안에서 쓰인 유니폼이 진입점 인자로 전달되지 않아 "정의되지 않은 식별자"가 난다.

## 3. 방출기 (`MSLEmitter.swift` / `MSLTypeMapping.swift`)

| 무엇 | 어디 |
|---|---|
| 이름만 다른 내장 함수 | `MSLTypeMapping.renamedBuiltins` 한 줄 |
| 메서드 호출로 바뀌는 텍스처 함수 | `MSLEmitter.textureCall` |
| 아토믹 | `MSLEmitter.atomicCall` / `atomicFetchOperations` |
| 타입 매핑 | `MSLTypeMapping.type` / `textureType` |
| `@builtin` | `MSLTypeMapping.builtin` |
| 문장 번역 | `MSLEmitter.emitStatement` / `simpleStatement` |
| 옮길 수 없는 것 | `MSLTypeMapping.unsupportedBuiltins`에 **이유와 대안**을 적어 거부 |

원칙: **모르는 이름은 그대로 통과시킨다.** WGSL 내장 함수 대부분이 MSL과 이름이 같고,
정말 없는 이름이면 Metal 컴파일러가 정확한 진단을 준다. 억지로 비슷한 것에 매핑하지 않는다.

구조적 변환 두 가지를 건드릴 일이 생기면 `docs/ARCHITECTURE.md` §4를 먼저 읽을 것:
- **리소스 스레딩** — MSL에 가변 전역이 없어 전역을 함수 인자로 내려보낸다.
  `threadedGlobals(for:)`의 **정렬 순서가 선언과 호출에서 같아야** 한다.
- **진입점 래핑** — 스테이지 I/O 속성은 `wgpu_<entry>_in/out` 전용 구조체에만 붙인다.
  일반 구조체에 `[[attribute]]`를 붙이면 유니폼 버퍼 레이아웃이 깨진다.

## 4. 구조체 배치 (`WGSLLayout.swift`)

결과가 미묘하게 틀리면 대개 여기다. WGSL `vec3<f32>`는 크기 12, MSL `float3`는 16이다.

- 오프셋 계산은 WGSL 명세(`AlignOf`/`SizeOf`/`RoundUp`)를 따른다. `@align`/`@size`를 존중한다.
- uniform 주소 공간이면 구조체/배열 정렬이 16으로 올라간다 (`uniformStructNames`가 판별).
- vec3 뒤 12~15바이트를 다른 멤버가 쓰면 `packed_float3`, 아니면 `float3` + `char` 패딩.

배치를 고쳤으면 **오프셋을 직접 단언**하는 테스트를 넣는다:

```swift
let placement = WGSLLayout.layout(of: structure, module: module, uniform: true)
XCTAssertEqual(placement.members.map(\.offset), [0, 12, 16])
XCTAssertEqual(placement.size, 80)
```

## 5. 바인딩 배정 (`WGSLBindings.swift`)

`@group/@binding` → Metal 인덱스. **방출기와 인코더가 같은 표를 봐야 한다** —
한쪽만 고치면 셰이더는 컴파일되는데 엉뚱한 버퍼를 읽는다.

- 바인드 그룹: 그룹 → 바인딩 오름차순으로 종류별 카운터 0부터.
- 정점 버퍼: `30 - slot` (테이블 위쪽부터 역순).

이 규칙을 바꾸면 `WGPUMetalBackend.applyVertexBuffer` / `applyBindGroup(_:at:dynamicOffsets:)`와
`WGPURenderPipelineObject.vertexDescriptor`도 **함께** 고친다.

## 6. 테스트 (필수)

`Tests/LynxWebGPUShaderTests/WGSLTranspilerTests.swift`에 케이스를 추가한다.
**반드시 `translate` 헬퍼를 거칠 것** — 그 안에 `MetalCompilerHarness.assertCompiles`가 들어 있다.

```swift
func test_theNewFeatureIsTranslatedToMSL() throws {
    let msl = try translate(source, entryPoints: ["main"])   // ← Metal 컴파일까지 검증된다
    XCTAssertTrue(msl.contains("the expected fragment"))
}
```

문자열 단언만 있는 테스트는 받지 않는다 — "그럴듯하지만 컴파일 안 되는 MSL"이 통과해 버린다.

번역 결과가 **런타임에 옳은지**까지 봐야 하면 `Tests/LynxWebGPUTests/RenderPipelineTests.swift`에
오프스크린 렌더 테스트를 추가한다 (`gpu-smoke` 스킬 참고).

## 7. 문서

`docs/WGSL.md`의 지원 표(§1)나 미지원 표(§4)를 갱신한다. 거부하기로 했다면 **이유와 대안**을 적는다.

## 8. 검증

```zsh
swift test --filter LynxWebGPUShaderTests     # 트랜스파일러 (Metal 컴파일 포함)
swift test                                    # 전체 — 렌더 결과 회귀까지
```

**크게 고쳤다면 외부 코퍼스 통과율을 반드시 다시 잰다:**

```zsh
git clone --depth 1 https://github.com/webgpu/webgpu-samples.git /tmp/webgpu-samples
LYNXWEBGPU_WGSL_CORPUS=/tmp/webgpu-samples/sample swift test --filter SampleCorpus
LYNXWEBGPU_WGSL_DUMP=/tmp/msl …               # 생성된 MSL을 눈으로 볼 때
```

로컬 테스트는 전부 통과하는데 코퍼스 통과율만 내려가는 변경이 실제로 있었다 —
WGSL의 AbstractInt 리터럴은 문맥 타입을 따르므로, 타입 추론 없이 "정확해 보이는" 규칙을 넣으면
오히려 더 자주 틀린다. **수치로 확인하고 들어갈 것.**

## 9. 프렐류드 (`MSLPrelude`)

타입 추론이 필요한 자리는 C++ 템플릿의 `decltype`에 넘긴다 (`wgpu_mod` `wgpu_vec2` `wgpu_max` …).
새 헬퍼를 더할 때는:

1. `MSLPrelude.source`에 오버로드를 넣고 **`xcrun -sdk macosx metal -c`로 먼저 컴파일해 본다**
   (템플릿 오버로드 해소는 눈으로 맞히기 어렵다).
2. `redirectedBuiltins`에 WGSL 이름 → 헬퍼 이름을 등록한다.
3. `inferredType(of:)`이 같은 규칙을 쓰는지 확인한다 — **방출기와 추론기가 어긋나면**
   구조체 타입과 초기값 타입이 달라져 조용히 깨진다 (실제로 겪은 회귀다).
