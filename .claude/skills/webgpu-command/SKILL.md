---
name: webgpu-command
description: Lynx-WebGPU에 새 WebGPU API(커맨드 스트림 op)를 끝에서 끝까지 추가한다. 디스크립터 → Metal 백엔드 → 해석기 → JS shim → 테스트 → 문서 순서와 각 지점의 규칙을 담는다. "drawIndirect 지원해줘", "쿼리셋 추가", "새 op 만들어줘" 같은 요청에 쓴다.
---

# WebGPU 커맨드 추가

WebGPU API 하나를 늘리는 일은 **여섯 지점**을 순서대로 건드리는 일이다. 하나라도 빠지면
"JS에서 부르면 아무 일도 안 일어남" 또는 "알 수 없는 명령" 오류가 난다.

## 0. 먼저 확인

- 명세 확인: https://www.w3.org/TR/webgpu/ 의 해당 인터페이스 — **필드 이름과 기본값을 명세 그대로** 쓴다.
- 이미 미지원 목록에 있는지: `docs/WEBGPU-API.md` §8
- Metal 대응이 있는지. 없으면 `WGPUError.unsupported`로 명시적으로 거부하는 편이 낫다 —
  비슷한 것으로 조용히 바꾸면 원인을 못 찾는 버그가 된다.

## 1. 디스크립터 (`Sources/LynxWebGPUCore/`)

새 인자 묶음이 있으면 `WGPUDescriptors.swift`에 구조체를 더한다.

```swift
public struct WGPUFooDescriptor {
    public var bar: Int
    public var mode: WGPUFooMode

    public init(from reader: WGPUValueReader) throws {
        bar = try reader.requiredInt("bar")
        mode = try reader.enumValue("mode", default: WGPUFooMode.someDefault)   // 명세 기본값
    }
}
```

규칙:
- 새 열거형은 `WGPUEnums.swift`에 두고 **raw value는 명세 철자 그대로** (`"one-minus-src-alpha"`).
  `CaseIterable`을 붙여야 오류 메시지에 후보 목록이 나온다.
- 명세에 기본값이 있는 필드는 `enumValue(_:default:)` / `int(_:default:)`를 쓴다. required는 명세가 required인 것만.
- 검증(범위·조합)은 `init`에서 던진다. 여기서 막아야 Metal 검증 레이어의 단언(=크래시)까지 안 간다.

## 2. Metal 백엔드 (`Sources/LynxWebGPU/`)

- 열거형 → Metal 변환은 `WGPUMetalMapping.swift`. **대응이 없으면 던진다.**
- 새 GPU 객체 타입은 `WGPUResources.swift`(리소스) 또는 `WGPUPipeline.swift`(파이프라인/바인딩)에
  `final class`로. 생성자에서 Metal 객체를 만들고 실패하면 `.outOfMemory` / `.backend`.
- **디스크립터 `label`은 `if let`으로 감쌀 것** — Metal 검증 레이어는 nil label에 단언으로 죽는다.

## 3. 해석기 (`WGPUCommandInterpreter.swift`)

```swift
private func perform(_ command: WGPUValueReader, at index: Int) throws {
    let op = try command.requiredString("op")
    switch op {
    // …
    case "fooBar": try fooBar(command)      // ← 여기에 한 줄
```

```swift
private func fooBar(_ command: WGPUValueReader) throws {
    let encoder = try requireRenderEncoder()          // 패스가 필요한 명령이면
    let target = try registry.lookup(
        try command.requiredHandle("target"), as: WGPUFooObject.self, kind: "GPUFoo"
    )
    try applyBindGroups()                             // draw/dispatch 계열이면 반드시
    encoder.doSomething(target.metalThing)
}
```

지켜야 할 것:
- **던지기만 한다.** 오류 수집·경로 부착은 `execute`가 한다.
- 리소스 생성 명령이면 `id`를 `requiredHandle("id")`로 받아 `registry.insert(_:at:)`.
- blit(복사·업로드)이 필요하면 `activeBlitEncoder()` — 렌더/컴퓨트 패스 안에서는 알아서 거부된다.
- 프레임 안에서만 유효한 핸들이면 `frameScopedHandles`에 넣는다.

## 4. JS shim (`JS/webgpu.js` + `webgpu.d.ts`)

브라우저 WebGPU와 **같은 이름·같은 인자 순서**로 만든다. 웹 코드를 그대로 옮겨올 수 있어야 한다.

```js
class GPURenderPassEncoder extends GPUPassEncoderBase {
  fooBar(target, mode) {
    this._commands.push({ op: 'fooBar', target: target.id, mode })
  }
}
```

- 패스 명령은 `this._commands`(인코더 로컬), 리소스 생성/큐 작업은 `this._recorder.push`.
- **네이티브를 동기로 부르지 않는다** — 프레임당 왕복 1회 계약이 깨진다.
  `mapAsync`/`canvasInfo`처럼 원래 왕복이 필요한 것만 예외다.
- 핸들은 `this._recorder.allocate()`로 JS가 발급한다.
- `.d.ts`에 시그니처를 더한다.

## 5. 테스트 (`Tests/LynxWebGPUTests/`)

**두 가지를 함께** 쓴다:

```swift
// (1) 동작 — 렌더 결과를 픽셀로 단언 (RenderPipelineTests)
harness.executeExpectingSuccess([ …, ["op": "fooBar", "target": 3], … ])
try harness.assertPixel(x: 32, y: 32, equals: (255, 0, 0, 255), "안쪽")
try harness.assertPixel(x: 1, y: 1, equals: (0, 0, 255, 255), "바깥쪽")

// (2) 계약 — 잘못 쓰면 크래시가 아니라 오류가 나는가 (CommandInterpreterTests)
let result = harness.execute([["op": "fooBar", "target": 999]])
XCTAssertEqual(errors(result).first?["kind"] as? String, "validation")
```

픽셀 단언은 **안쪽 한 점 + 바깥쪽 한 점**이 최소 조합이다 (클리어 색만 나와도 통과하는 테스트를 막는다).

## 6. 문서

- `docs/WEBGPU-API.md` — 해당 절에 사용 예를 넣고, §8 미지원 목록에서 지운다.
- 새 WGSL 문법/내장 함수를 함께 쓴다면 `docs/WGSL.md`도.

## 7. 검증

```zsh
swift test                                      # 전체 (약 3초)
arch -arm64 xcodebuild -scheme LynxWebGPUBridge \
  -destination 'platform=iOS Simulator,name=iPhone 17,OS=26.2' \
  -derivedDataPath .derivedData-cli build       # Lynx 브리지까지 컴파일되는지
```

## 흔한 실수

| 증상 | 원인 |
|---|---|
| "알 수 없는 명령 'fooBar'" | 3번(해석기 switch) 누락 |
| JS에서 불러도 아무 일 없음 | 4번에서 `_commands` 대신 다른 배열에 push |
| 바인딩이 안 꽂힌 채 draw | `applyBindGroups()` 호출 누락 |
| Metal 단언으로 크래시 | label nil, 또는 검증을 1번에서 안 했다 |
| 프레임이 갑자기 느려짐 | shim이 프레임 안에서 네이티브를 동기 호출한다 |
