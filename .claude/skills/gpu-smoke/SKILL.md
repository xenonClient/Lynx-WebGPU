---
name: gpu-smoke
description: Lynx-WebGPU의 렌더 결과를 오프스크린으로 실행해 픽셀로 검증한다. 화면 없이 GPU 파이프라인을 확인하거나, 검게 나오는 원인을 좁히거나, PNG로 떠서 눈으로 볼 때 쓴다. "렌더가 안 나와", "검은 화면", "이 셰이더 결과 확인해줘", "렌더 테스트 추가" 같은 요청에 쓴다.
---

# GPU 렌더 검증

시뮬레이터도 호스트 앱도 없이, JS가 보낼 것과 **완전히 같은 커맨드 스트림**을 실행하고
결과 픽셀을 읽는다. 결정적이고 3초 안에 끝난다.

## 1. 빠른 실행

```zsh
swift test --filter RenderPipelineTests      # 렌더 파이프라인 전체
swift test --filter CommandInterpreterTests  # 해석기 계약(오류·핸들 수명)
swift test --filter test_삼각형이_그려지고    # 개별
```

## 2. 새 렌더 검증 쓰기

`Tests/LynxWebGPUTests/RenderPipelineTests.swift`에 추가한다.

```swift
func test_무엇이_어떻게_보인다() throws {
    harness.executeExpectingSuccess([
        ["op": "configureCanvas", "canvas": "test", "format": "rgba8unorm"],
        ["op": "createShaderModule", "id": 1, "code": shader],
        ["op": "createBuffer", "id": 2, "usage": TestUsage.vertex, "data": vertices.base64],
        ["op": "createRenderPipeline", "id": 3, "layout": "auto",
         "vertex": ["module": 1, "entryPoint": "vs_main", "buffers": [ … ]],
         "fragment": ["module": 1, "entryPoint": "fs_main", "targets": [["format": "rgba8unorm"]]]],
        ["op": "getCurrentTexture", "id": 10, "canvas": "test"],
        ["op": "createTextureView", "id": 11, "texture": 10],
        ["op": "beginRenderPass", "colorAttachments": [[
            "view": 11, "loadOp": "clear", "storeOp": "store",
            "clearValue": ["r": 0, "g": 0, "b": 1, "a": 1],
        ]]],
        ["op": "setPipeline", "pipeline": 3],
        ["op": "setVertexBuffer", "slot": 0, "buffer": 2],
        ["op": "draw", "vertexCount": 3],
        ["op": "endPass"],
    ])

    try harness.assertPixel(x: 32, y: 32, equals: (255, 0, 0, 255), "삼각형 내부")
    try harness.assertPixel(x: 1, y: 1, equals: (0, 0, 255, 255), "삼각형 외부(클리어색)")
}
```

규칙:
- **안쪽 한 점 + 바깥쪽 한 점**을 최소로 단언한다. 한 점만 보면 "클리어 색만 나와도 통과"하는 테스트가 된다.
- 기본 캔버스는 64×64, id는 `"test"`. `RenderHarness.make(width:height:)`로 바꿀 수 있다.
- 버퍼 데이터는 `[Float].base64` / `Data(...).base64EncodedString()`.
- 사용 플래그는 `TestUsage` (JS 상수와 같은 값).
- 컴퓨트 결과는 `harness.context.readBuffer(handle:)` + `XCTestExpectation`으로 읽는다.

정점 버퍼 없이 화면을 채우고 싶으면 `@builtin(vertex_index)`로 큰 삼각형 하나를 그리는 관용구를 쓴다:

```wgsl
var positions = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
return vec4f(positions[index], 0.0, 1.0);
```

## 3. 눈으로 보기

```swift
harness.dumpPNG(named: "my-case")     // → .tmp/my-case.png
```

의존성 없는 PNG 인코더가 하네스에 들어 있다. `.tmp/`는 git-ignored다.

## 4. "검은 화면" 좁히기

위에서부터 순서대로 확인한다. 대부분 1~3에서 끝난다.

1. **오류가 이미 나고 있는가** — `execute()` 반환의 `errors`. `executeExpectingSuccess`는 경로까지 찍어 준다.
2. **셰이더가 컴파일됐는가** — `kind: 'backend'` 오류라면 메시지에 **생성된 MSL 전문**이 있다.
3. **클리어 색은 나오는가** — draw를 지우고 `beginRenderPass` + `endPass`만 남겨 본다.
   클리어 색도 안 나오면 어태치먼트/표면 문제, 나오면 파이프라인 문제다.
4. **지오메트리가 화면 안인가** — WebGPU NDC는 x/y가 -1~1, z가 0~1이다 (Metal과 같다).
   `arrayStride`와 attribute `offset`이 정점 데이터와 맞는지 본다.
5. **컬링** — `primitive.cullMode`가 `back`인데 정점 순서가 `frontFace`와 반대면 전부 사라진다.
   `cullMode: 'none'`으로 바꿔 확인.
6. **깊이** — `depthCompare`가 `less`인데 깊이 버퍼를 클리어하지 않았거나, 파이프라인에
   `depthStencil`이 있는데 패스에 `depthStencilAttachment`가 없는 경우.
7. **바인딩** — 유니폼이 0으로 읽히면 바인드 그룹이 안 꽂힌 것이다. `layout: "auto"`를 쓸 때
   `getBindGroupLayout`으로 받은 레이아웃으로 바인드 그룹을 만들었는지 확인한다.

## 5. 실제 화면(호스트 앱)에서만 나는 문제

오프스크린이 통과하는데 앱에서만 안 되면 원인은 표면 쪽이다:

- `<webgpu-canvas canvas-id>`가 JS의 `getCanvasContext(id)`와 같은가
- `host.attach(to: lynxView)`를 불렀는가 (안 부르면 캔버스가 등록되지 않는다)
- 캔버스 크기가 0인가 — 레이아웃 전이면 `getCurrentTexture`가 오류를 낸다. `bindcanvasresize`로 확인.
- Xcode 콘솔의 `org.lynxwebgpu` / `Canvas` 카테고리 로그에 등록 메시지가 있는가

자세한 절차는 `docs/LYNX-INTEGRATION.md` §6.
