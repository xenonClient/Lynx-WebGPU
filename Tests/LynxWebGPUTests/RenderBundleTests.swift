import XCTest
import Metal
import LynxWebGPUCore
@testable import LynxWebGPU

/// 렌더 번들 — 계약이 **"직접 인코딩과 같은 결과"**이므로 동치성이 곧 검증이다.
///
/// 이 구현에는 Metal 대응 객체가 없어 명령 목록을 저장했다가 되풀이한다. 그래서 "번들이
/// 뭔가 다르게 동작할" 여지가 오히려 좁고, 대신 **상태 격리와 재사용**이 틀리기 쉬운 지점이다.
final class RenderBundleTests: XCTestCase {
    private var harness: RenderHarness!

    override func setUpWithError() throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "Metal 디바이스 없음")
        harness = try XCTUnwrap(RenderHarness.make())
    }

    override func tearDown() {
        harness = nil
        super.tearDown()
    }

    /// 화면 전체를 유니폼 색으로 덮는다.
    private static let shader = """
    struct Tint { color: vec4f };
    @group(0) @binding(0) var<uniform> tint: Tint;

    @vertex
    fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
        var corners = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
        return vec4f(corners[index], 0.0, 1.0);
    }

    @fragment
    fn fs_main() -> @location(0) vec4f {
        return tint.color;
    }
    """

    private let red = (r: 255, g: 0, b: 0, a: 255)
    private let green = (r: 0, g: 255, b: 0, a: 255)

    private func errors(_ result: [String: Any]) -> [[String: Any]] {
        result["errors"] as? [[String: Any]] ?? []
    }

    /// 파이프라인 1개 + 빨강/초록 바인드 그룹 2개 (핸들 6, 7).
    private func setUpResources() -> [[String: Any]] {
        [
            ["op": "configureCanvas", "canvas": "test", "format": "rgba8unorm"],
            ["op": "createShaderModule", "id": 1, "code": Self.shader],
            ["op": "createRenderPipeline", "id": 2, "layout": "auto",
             "vertex": ["module": 1, "entryPoint": "vs_main"],
             "fragment": ["module": 1, "entryPoint": "fs_main",
                          "targets": [["format": "rgba8unorm"]]]],
            ["op": "createBuffer", "id": 3, "size": 16, "usage": TestUsage.uniform,
             "data": [Float]([1, 0, 0, 1]).base64],
            ["op": "createBuffer", "id": 4, "size": 16, "usage": TestUsage.uniform,
             "data": [Float]([0, 1, 0, 1]).base64],
            ["op": "getBindGroupLayout", "id": 5, "pipeline": 2, "index": 0],
            ["op": "createBindGroup", "id": 6, "layout": 5,
             "entries": [["binding": 0, "resource": ["buffer": 3]]]],
            ["op": "createBindGroup", "id": 7, "layout": 5,
             "entries": [["binding": 0, "resource": ["buffer": 4]]]],
        ]
    }

    private let acquireDrawable: [[String: Any]] = [
        ["op": "getCurrentTexture", "id": 20, "canvas": "test"],
        ["op": "createTextureView", "id": 21, "texture": 20],
    ]

    private let beginPass: [String: Any] = [
        "op": "beginRenderPass",
        "colorAttachments": [[
            "view": 21, "loadOp": "clear", "storeOp": "store",
            "clearValue": ["r": 0, "g": 0, "b": 1, "a": 1],
        ]],
    ]

    /// 화면을 `bindGroup` 색으로 덮는 드로우 세 줄 — 직접 인코딩과 번들이 공유하는 몸통이다.
    private func fullScreenDraw(bindGroup: Int) -> [[String: Any]] {
        [
            ["op": "setPipeline", "pipeline": 2],
            ["op": "setBindGroup", "index": 0, "bindGroup": bindGroup],
            ["op": "draw", "vertexCount": 3],
        ]
    }

    private func createBundle(id: Int, commands: [[String: Any]]) -> [String: Any] {
        ["op": "createRenderBundle", "id": id, "colorFormats": ["rgba8unorm"], "commands": commands]
    }

    // MARK: - 동치성

    func test_번들이_직접_인코딩과_같은_프레임을_낸다() throws {
        harness.executeExpectingSuccess(setUpResources() + [
            createBundle(id: 10, commands: fullScreenDraw(bindGroup: 6)),
        ])

        harness.executeExpectingSuccess(acquireDrawable + [beginPass]
            + fullScreenDraw(bindGroup: 6) + [["op": "endPass"]])
        try harness.assertPixel(x: 32, y: 32, equals: red, "직접 인코딩이 실제로 그렸는지")
        let direct = try harness.frameBytes()

        harness.executeExpectingSuccess(acquireDrawable + [
            beginPass,
            ["op": "executeBundles", "bundles": [10]],
            ["op": "endPass"],
        ])

        try harness.assertFrameEquals(direct, "번들의 계약은 직접 인코딩과 같은 결과다")
    }

    /// 번들의 존재 이유는 재사용이다 — 한 번 실행하면 상하는 자료 구조가 아니어야 한다.
    func test_같은_번들을_두_프레임_연속_실행해도_같은_결과다() throws {
        harness.executeExpectingSuccess(setUpResources() + [
            createBundle(id: 10, commands: fullScreenDraw(bindGroup: 6)),
        ])

        let frame: [[String: Any]] = acquireDrawable + [
            beginPass,
            ["op": "executeBundles", "bundles": [10]],
            ["op": "endPass"],
        ]

        harness.executeExpectingSuccess(frame)
        try harness.assertPixel(x: 32, y: 32, equals: red)
        let first = try harness.frameBytes()

        harness.executeExpectingSuccess(frame)
        try harness.assertFrameEquals(first, "두 번째 실행도 같아야 한다")
    }

    func test_여러_번들이_넘긴_순서대로_실행된다() throws {
        harness.executeExpectingSuccess(setUpResources() + [
            createBundle(id: 10, commands: fullScreenDraw(bindGroup: 6)),   // 빨강
            createBundle(id: 11, commands: fullScreenDraw(bindGroup: 7)),   // 초록
        ])

        harness.executeExpectingSuccess(acquireDrawable + [
            beginPass,
            ["op": "executeBundles", "bundles": [10, 11]],
            ["op": "endPass"],
        ])
        try harness.assertPixel(x: 32, y: 32, equals: green, "나중에 온 번들이 위에 그려진다")

        harness.executeExpectingSuccess(acquireDrawable + [
            beginPass,
            ["op": "executeBundles", "bundles": [11, 10]],
            ["op": "endPass"],
        ])
        try harness.assertPixel(x: 32, y: 32, equals: red, "순서를 뒤집으면 결과도 뒤집힌다")
    }

    /// 호환성 검증은 실행보다 먼저, **전체 목록에 대해** 끝나야 한다 — 하나만 비호환이어도
    /// 앞의 호환 번들까지 실행되지 않는다. 절반만 그려진 프레임을 남기지 않기 위한 계약인데,
    /// 검증 루프와 실행 루프를 합치면 조용히 깨지므로 픽셀로 못 박는다.
    func test_비호환_번들이_섞이면_앞의_호환_번들도_실행되지_않는다() throws {
        let setUp = setUpResources() + [
            createBundle(id: 10, commands: fullScreenDraw(bindGroup: 6)),   // 호환 (빨강)
            ["op": "createRenderBundle", "id": 12, "colorFormats": ["bgra8unorm"],
             "commands": fullScreenDraw(bindGroup: 7)],                     // 패스와 포맷 불일치
        ]
        let result = harness.execute(setUp + acquireDrawable + [
            beginPass,
            ["op": "executeBundles", "bundles": [10, 12]],
            ["op": "endPass"],
        ])

        XCTAssertEqual(errors(result).first?["kind"] as? String, "validation")
        // 호환인 10번(빨강)도 그려지지 않아야 한다 — 중앙이 클리어색 그대로여야 한다.
        try harness.assertPixel(x: 32, y: 32, equals: (0, 0, 255, 255), "앞의 호환 번들이 실행됐다")
    }

    /// 명세의 `GPURenderBundleEncoder`는 `GPUDebugCommandsMixin`을 포함한다 — 마커를 담을 수 있다.
    ///
    /// 허용 목록에서 빠뜨리면 **마커 하나 때문에 번들 전체가 거부되고**, 사용자는 마커가
    /// 원인이라고 생각하기 어렵다 (드로우를 의심하며 엉뚱한 곳을 고친다).
    func test_번들에_디버그_마커를_담을_수_있다() throws {
        harness.executeExpectingSuccess(setUpResources() + [
            createBundle(id: 10, commands:
                [["op": "pushDebugGroup", "groupLabel": "번들 구간"]]
                + fullScreenDraw(bindGroup: 6)
                + [["op": "insertDebugMarker", "markerLabel": "드로우 뒤"],
                   ["op": "popDebugGroup"]]
            ),
        ])

        harness.executeExpectingSuccess(acquireDrawable + [
            beginPass,
            ["op": "executeBundles", "bundles": [10]],
            ["op": "endPass"],
        ])
        // 마커가 그림을 바꾸지 않는다 — 번들은 평소대로 그려야 한다.
        try harness.assertPixel(x: 32, y: 32, equals: red, "마커가 섞여도 그려져야 한다")
    }

    // MARK: - 상태 격리

    /// 명세는 번들 실행이 패스 상태를 **복원**하는 것이 아니라 **무효화**한다고 정한다.
    /// 그래서 이어서 그리려면 `setPipeline`부터 다시 해야 한다 — 그러지 않으면 번들이 남긴
    /// 바인딩으로 그려져, 브라우저에서는 오류인 코드가 여기서만 돌아간다.
    func test_번들_실행_뒤에는_파이프라인을_다시_지정해야_한다() {
        let result = harness.execute(setUpResources() + [
            createBundle(id: 10, commands: fullScreenDraw(bindGroup: 6)),
        ] + acquireDrawable + [
            beginPass,
            ["op": "executeBundles", "bundles": [10]],
            ["op": "draw", "vertexCount": 3],   // setPipeline 없이
            ["op": "endPass"],
        ])

        XCTAssertTrue(
            ((errors(result).first?["message"] as? String) ?? "").contains("setPipeline"),
            "번들이 남긴 상태를 물려받으면 안 된다: \(harness.describeErrors(result))"
        )
    }

    /// 반대 방향 — 번들은 패스가 이미 지정해 둔 파이프라인을 물려받지 않는다.
    func test_번들은_패스의_파이프라인을_물려받지_않는다() {
        let result = harness.execute(setUpResources() + [
            // setPipeline 없이 draw만 담은 번들.
            createBundle(id: 10, commands: [
                ["op": "setBindGroup", "index": 0, "bindGroup": 6],
                ["op": "draw", "vertexCount": 3],
            ]),
        ] + acquireDrawable + [
            beginPass,
            ["op": "setPipeline", "pipeline": 2],   // 패스 쪽에서 미리 지정해 둔다
            ["op": "executeBundles", "bundles": [10]],
            ["op": "endPass"],
        ])

        XCTAssertTrue(
            ((errors(result).first?["message"] as? String) ?? "").contains("setPipeline"),
            "번들 안에서 다시 지정해야 한다: \(harness.describeErrors(result))"
        )
    }

    /// 반대 방향 — 번들은 패스가 이미 올려 둔 바인드 그룹도 물려받지 않는다.
    ///
    /// 파이프라인과 달리 바인드 그룹은 **Metal 인코더에 그대로 남아 있어서** 그림자 상태만
    /// 비워서는 격리되지 않는다. 레이아웃이 요구하는 그룹이 다 바인드되었는지 봐야 잡힌다.
    func test_번들은_패스의_바인드_그룹을_물려받지_않는다() {
        let result = harness.execute(setUpResources() + [
            createBundle(id: 10, commands: [
                ["op": "setPipeline", "pipeline": 2],   // setBindGroup 없이
                ["op": "draw", "vertexCount": 3],
            ]),
        ] + acquireDrawable + [
            beginPass,
            ["op": "setPipeline", "pipeline": 2],
            ["op": "setBindGroup", "index": 0, "bindGroup": 6],   // 패스 쪽에서 미리 올려 둔다
            ["op": "executeBundles", "bundles": [10]],
            ["op": "endPass"],
        ])

        XCTAssertTrue(
            ((errors(result).first?["message"] as? String) ?? "").contains("@group(0)"),
            "번들 안에서 다시 바인드해야 한다: \(harness.describeErrors(result))"
        )
    }

    /// 번들 실행 뒤 `setPipeline`만 다시 하고 그리면, 바인드 그룹은 번들이 남긴 것이 쓰인다.
    func test_번들_실행_뒤에는_바인드_그룹도_다시_지정해야_한다() {
        let result = harness.execute(setUpResources() + [
            createBundle(id: 10, commands: fullScreenDraw(bindGroup: 6)),
        ] + acquireDrawable + [
            beginPass,
            ["op": "executeBundles", "bundles": [10]],
            ["op": "setPipeline", "pipeline": 2],   // setBindGroup 없이
            ["op": "draw", "vertexCount": 3],
            ["op": "endPass"],
        ])

        XCTAssertTrue(
            ((errors(result).first?["message"] as? String) ?? "").contains("@group(0)"),
            harness.describeErrors(result)
        )
    }

    // MARK: - 상태 격리 (정점 버퍼)

    /// 정점 버퍼를 실제로 읽는 파이프라인 — 격리를 보려면 슬롯을 요구하는 쪽이 있어야 한다.
    private static let vertexPullingShader = """
    @vertex
    fn vs_main(@location(0) position: vec2f) -> @builtin(position) vec4f {
        return vec4f(position, 0.0, 1.0);
    }

    @fragment
    fn fs_main() -> @location(0) vec4f {
        return vec4f(1.0, 0.0, 0.0, 1.0);
    }
    """

    /// 셰이더(40) · 파이프라인(41) · 정점 버퍼(42).
    private func setUpVertexPulling() -> [[String: Any]] {
        [
            ["op": "configureCanvas", "canvas": "test", "format": "rgba8unorm"],
            ["op": "createShaderModule", "id": 40, "code": Self.vertexPullingShader],
            ["op": "createRenderPipeline", "id": 41, "layout": "auto",
             "vertex": ["module": 40, "entryPoint": "vs_main",
                        "buffers": [[
                            "arrayStride": 8,
                            "attributes": [["shaderLocation": 0, "offset": 0, "format": "float32x2"]],
                        ]]],
             "fragment": ["module": 40, "entryPoint": "fs_main",
                          "targets": [["format": "rgba8unorm"]]]],
            ["op": "createBuffer", "id": 42, "size": 24, "usage": TestUsage.vertex,
             "data": [Float]([-1, -1, 3, -1, -1, 3]).base64],
        ]
    }

    private func vertexBundle(id: Int, commands: [[String: Any]]) -> [String: Any] {
        ["op": "createRenderBundle", "id": id, "colorFormats": ["rgba8unorm"], "commands": commands]
    }

    /// 기준 — 번들이 자기 정점 버퍼를 담으면 정상으로 그려진다.
    func test_정점_버퍼를_담은_번들은_정상으로_그린다() throws {
        harness.executeExpectingSuccess(setUpVertexPulling() + [
            vertexBundle(id: 43, commands: [
                ["op": "setPipeline", "pipeline": 41],
                ["op": "setVertexBuffer", "slot": 0, "buffer": 42],
                ["op": "draw", "vertexCount": 3],
            ]),
        ] + acquireDrawable + [
            beginPass,
            ["op": "executeBundles", "bundles": [43]],
            ["op": "endPass"],
        ])

        try harness.assertPixel(x: 32, y: 32, equals: red)
    }

    /// 패스 쪽에서만 정점 버퍼를 올린 경우 — 번들은 물려받지 않으므로 거부되어야 한다.
    func test_번들은_패스의_정점_버퍼를_물려받지_않는다() {
        let result = harness.execute(setUpVertexPulling() + [
            vertexBundle(id: 43, commands: [
                ["op": "setPipeline", "pipeline": 41],   // setVertexBuffer 없이
                ["op": "draw", "vertexCount": 3],
            ]),
        ] + acquireDrawable + [
            beginPass,
            ["op": "setVertexBuffer", "slot": 0, "buffer": 42],   // 패스 쪽에서 미리 올려 둔다
            ["op": "executeBundles", "bundles": [43]],
            ["op": "endPass"],
        ])

        XCTAssertTrue(
            ((errors(result).first?["message"] as? String) ?? "").contains("vertex buffer slot 0"),
            harness.describeErrors(result)
        )
    }

    /// 반대 방향 — 번들이 올린 정점 버퍼는 실행이 끝나면 무효화된다.
    func test_번들_실행_뒤에는_정점_버퍼도_다시_지정해야_한다() {
        let result = harness.execute(setUpVertexPulling() + [
            vertexBundle(id: 43, commands: [
                ["op": "setPipeline", "pipeline": 41],
                ["op": "setVertexBuffer", "slot": 0, "buffer": 42],
                ["op": "draw", "vertexCount": 3],
            ]),
        ] + acquireDrawable + [
            beginPass,
            ["op": "executeBundles", "bundles": [43]],
            ["op": "setPipeline", "pipeline": 41],   // setVertexBuffer 없이
            ["op": "draw", "vertexCount": 3],
            ["op": "endPass"],
        ])

        XCTAssertTrue(
            ((errors(result).first?["message"] as? String) ?? "").contains("vertex buffer slot 0"),
            harness.describeErrors(result)
        )
    }

    /// 인덱스 버퍼도 번들 경계에서 무효화된다 — 문서가 "파이프라인·바인드 그룹·정점 버퍼·
    /// **인덱스 버퍼** 네 가지"라고 적어 둔 자리다. 남아 있으면 이어지는 `drawIndexed`가
    /// 번들이 남긴 인덱스로 **실제로 그려져** 브라우저와 다르게 동작한다.
    func test_번들_실행_뒤에는_인덱스_버퍼도_다시_지정해야_한다() {
        let result = harness.execute(setUpVertexPulling() + [
            ["op": "createBuffer", "id": 44, "size": 6, "usage": TestUsage.index,
             "data": Data([0, 0, 1, 0, 2, 0]).base64EncodedString()],
            vertexBundle(id: 43, commands: [
                ["op": "setPipeline", "pipeline": 41],
                ["op": "setVertexBuffer", "slot": 0, "buffer": 42],
                ["op": "setIndexBuffer", "buffer": 44, "format": "uint16"],
                ["op": "drawIndexed", "indexCount": 3],
            ]),
        ] + acquireDrawable + [
            beginPass,
            ["op": "executeBundles", "bundles": [43]],
            ["op": "setPipeline", "pipeline": 41],
            ["op": "setVertexBuffer", "slot": 0, "buffer": 42],
            ["op": "drawIndexed", "indexCount": 3],   // setIndexBuffer 없이
            ["op": "endPass"],
        ])

        XCTAssertTrue(
            ((errors(result).first?["message"] as? String) ?? "").contains("setIndexBuffer is required"),
            harness.describeErrors(result)
        )
    }

    /// 반대 방향 — 패스에서 묶은 인덱스 버퍼를 번들이 물려받지 않는다.
    func test_번들은_패스의_인덱스_버퍼를_물려받지_않는다() {
        let result = harness.execute(setUpVertexPulling() + [
            ["op": "createBuffer", "id": 44, "size": 6, "usage": TestUsage.index,
             "data": Data([0, 0, 1, 0, 2, 0]).base64EncodedString()],
            vertexBundle(id: 43, commands: [
                ["op": "setPipeline", "pipeline": 41],
                ["op": "setVertexBuffer", "slot": 0, "buffer": 42],
                ["op": "drawIndexed", "indexCount": 3],   // 번들 안에서는 묶은 적이 없다
            ]),
        ] + acquireDrawable + [
            beginPass,
            ["op": "setIndexBuffer", "buffer": 44, "format": "uint16"],
            ["op": "executeBundles", "bundles": [43]],
            ["op": "endPass"],
        ])

        XCTAssertTrue(
            ((errors(result).first?["message"] as? String) ?? "").contains("setIndexBuffer is required"),
            harness.describeErrors(result)
        )
    }

    /// 회귀 — 번들 명령 하나가 실패해도 패스 바인딩 초기화는 건너뛰면 안 된다.
    ///
    /// 해석기의 계약은 "오류가 프레임을 죽이지 않고 누적된다"이므로 실행은 다음 명령부터 계속된다.
    /// 그때 초기화를 빠뜨리면 이어지는 draw가 번들이 남긴 파이프라인으로 **실제로 그려진다** —
    /// 브라우저에서는 거부되는 코드가 여기서만 엉뚱한 픽셀을 낸다.
    func test_번들_실행_중_오류가_나도_패스_바인딩이_초기화된다() {
        let result = harness.execute(setUpResources() + [
            // 두 번째 명령의 바인드 그룹 핸들이 없다 — 실행 도중에 실패한다.
            createBundle(id: 10, commands: [
                ["op": "setPipeline", "pipeline": 2],
                ["op": "setBindGroup", "index": 0, "bindGroup": 999],
            ]),
        ] + acquireDrawable + [
            beginPass,
            ["op": "executeBundles", "bundles": [10]],
            ["op": "draw", "vertexCount": 3],   // setPipeline 없이
            ["op": "endPass"],
        ])

        let messages = errors(result).compactMap { $0["message"] as? String }
        XCTAssertEqual(
            messages.count, 2,
            "번들 실패 1건 + draw 거부 1건이어야 한다: \(harness.describeErrors(result))"
        )
        XCTAssertTrue(
            (messages.last ?? "").contains("setPipeline"),
            "번들이 남긴 파이프라인을 물고 그리면 안 된다: \(harness.describeErrors(result))"
        )
    }

    /// 회귀 — 패스의 `sampleCount`를 컬러 어태치먼트에서만 유도하면, 컬러 없이 깊이만 있는
    /// MSAA 패스(그림자 맵·깊이 프리패스)에서 패스가 1로 잡힌다. 그러면 명세대로 정확히
    /// `sampleCount: 4`를 선언한 번들이 오탐으로 거부된다.
    func test_컬러_없는_MSAA_깊이_패스에서_sampleCount가_깊이_뷰로_결정된다() throws {
        try XCTSkipUnless(harness.context!.device.supportsTextureSampleCount(4), "4x MSAA 미지원")

        let result = harness.execute([
            ["op": "createTexture", "id": 2, "size": ["width": 64, "height": 64],
             "format": "depth32float", "sampleCount": 4, "usage": TestUsage.renderAttachment],
            ["op": "createTextureView", "id": 3, "texture": 2],
            ["op": "createRenderBundle", "id": 10, "colorFormats": [String?](),
             "depthStencilFormat": "depth32float", "sampleCount": 4, "commands": [[String: Any]]()],
            ["op": "beginRenderPass",
             "colorAttachments": [[String: Any]](),
             "depthStencilAttachment": [
                "view": 3, "depthClearValue": 1.0, "depthLoadOp": "clear", "depthStoreOp": "store",
             ]],
            ["op": "executeBundles", "bundles": [10]],
            ["op": "endPass"],
        ])

        XCTAssertEqual(result["ok"] as? Bool, true, harness.describeErrors(result))
    }

    // MARK: - 계약

    func test_번들의_컬러_포맷이_패스와_다르면_거부한다() {
        let result = harness.execute(setUpResources() + [
            ["op": "createRenderBundle", "id": 10, "colorFormats": ["bgra8unorm"],
             "commands": fullScreenDraw(bindGroup: 6)],
        ] + acquireDrawable + [
            beginPass,   // 패스는 rgba8unorm이다
            ["op": "executeBundles", "bundles": [10]],
            ["op": "endPass"],
        ])

        XCTAssertEqual(errors(result).first?["kind"] as? String, "validation")
        let message = (errors(result).first?["message"] as? String) ?? ""
        XCTAssertTrue(message.contains("colorFormats"), message)
        XCTAssertTrue(message.contains("bgra8unorm") && message.contains("rgba8unorm"), message)
    }

    func test_번들의_어태치먼트_수가_패스와_다르면_거부한다() {
        let result = harness.execute(setUpResources() + [
            ["op": "createRenderBundle", "id": 10, "colorFormats": ["rgba8unorm", "rgba8unorm"],
             "commands": fullScreenDraw(bindGroup: 6)],
        ] + acquireDrawable + [
            beginPass,
            ["op": "executeBundles", "bundles": [10]],
            ["op": "endPass"],
        ])

        XCTAssertTrue(
            ((errors(result).first?["message"] as? String) ?? "").contains("attachment count"),
            harness.describeErrors(result)
        )
    }

    /// 명세의 레이아웃 동치 비교는 **후행 null을 무시한다.** 자르지 않으면 유효한 조합이 거부된다.
    func test_번들_colorFormats의_후행_null은_무시한다() {
        let result = harness.execute(setUpResources() + [
            ["op": "createRenderBundle", "id": 10,
             // JS가 `null`을 실어 보내면 브리지에서 NSNull로 도착한다.
             "colorFormats": ["rgba8unorm", NSNull()] as [Any],
             "commands": fullScreenDraw(bindGroup: 6)],
        ] + acquireDrawable + [
            beginPass,   // 컬러 어태치먼트 1개
            ["op": "executeBundles", "bundles": [10]],
            ["op": "endPass"],
        ])

        XCTAssertEqual(result["ok"] as? Bool, true, harness.describeErrors(result))
    }

    /// 어태치먼트가 하나도 없는 번들은 **만들 때** 거부한다.
    /// 지금도 결국 `makeRenderCommandEncoder`가 실패하지만, 그러면 오류가 엉뚱한 자리에서 난다.
    func test_어태치먼트가_없는_번들은_만들_때_거부한다() {
        let result = harness.execute([
            ["op": "createRenderBundle", "id": 10, "colorFormats": [String?](),
             "commands": [[String: Any]]()],
        ])

        XCTAssertEqual(errors(result).first?["kind"] as? String, "validation")
        XCTAssertTrue(
            ((errors(result).first?["message"] as? String) ?? "").contains("attachment"),
            harness.describeErrors(result)
        )
        XCTAssertEqual(harness.liveObjects, 0, "거부한 번들이 등록되면 안 된다")
    }

    func test_번들의_깊이_포맷이_패스와_다르면_거부한다() {
        let result = harness.execute(setUpResources() + [
            ["op": "createRenderBundle", "id": 10, "colorFormats": ["rgba8unorm"],
             "depthStencilFormat": "depth32float", "commands": fullScreenDraw(bindGroup: 6)],
        ] + acquireDrawable + [
            beginPass,   // 깊이 어태치먼트가 없다
            ["op": "executeBundles", "bundles": [10]],
            ["op": "endPass"],
        ])

        XCTAssertTrue(
            ((errors(result).first?["message"] as? String) ?? "").contains("depthStencilFormat"),
            harness.describeErrors(result)
        )
    }

    /// 명세가 번들에 담지 못하게 한 명령들. **번들을 만들 때** 거부해야 한다 —
    /// 실행 시점까지 미루면 몇 프레임 뒤에야 드러난다.
    func test_번들_안에_금지된_명령이_있으면_만들_때_거부한다() {
        for forbidden in [
            ["op": "setViewport", "x": 0, "y": 0, "width": 8, "height": 8],
            ["op": "setScissorRect", "x": 0, "y": 0, "width": 8, "height": 8],
            ["op": "setStencilReference", "reference": 1],
            ["op": "endPass"],
            ["op": "executeBundles", "bundles": [99]],
            ["op": "copyBufferToBuffer", "source": 1, "destination": 2, "size": 4],
            // occlusion 쿼리는 렌더 **패스**의 것이다 — 명세의 번들 인코더에는 아예 없다.
            ["op": "beginOcclusionQuery", "queryIndex": 0],
            ["op": "endOcclusionQuery"],
            // 컴퓨트·쓰기 계열도 마찬가지다.
            ["op": "dispatchWorkgroups", "x": 1],
            ["op": "writeBuffer", "buffer": 1, "data": [0, 0, 0, 0]],
            ["op": "clearBuffer", "buffer": 1],
        ] as [[String: Any]] {
            let result = harness.execute([createBundle(id: 10, commands: [forbidden])])
            XCTAssertEqual(
                errors(result).first?["kind"] as? String, "validation",
                "'\(forbidden["op"] ?? "?")'은 번들에 담을 수 없어야 한다"
            )
            XCTAssertEqual(harness.liveObjects, 0, "거부한 번들이 등록되면 안 된다")
        }
    }

    /// read-only로 연 패스에는 "나도 안 쓴다"고 선언한 번들만 넣을 수 있다.
    /// 반대(쓰기 가능 패스에 read-only 번들)는 문제가 없으므로 한 방향만 본다.
    func test_depthReadOnly_패스에는_readOnly_번들만_실행할_수_있다() {
        let depthTarget: [[String: Any]] = [
            ["op": "createTexture", "id": 50, "size": ["width": 64, "height": 64],
             "format": "depth32float", "usage": TestUsage.renderAttachment],
            ["op": "createTextureView", "id": 51, "texture": 50],
        ]
        let readOnlyPass: [String: Any] = [
            "op": "beginRenderPass",
            "colorAttachments": [[
                "view": 21, "loadOp": "clear", "storeOp": "store",
                "clearValue": ["r": 0, "g": 0, "b": 1, "a": 1],
            ]],
            "depthStencilAttachment": ["view": 51, "depthReadOnly": true],
        ]
        func bundle(id: Int, readOnly: Bool) -> [String: Any] {
            var command: [String: Any] = [
                "op": "createRenderBundle", "id": id, "colorFormats": ["rgba8unorm"],
                "depthStencilFormat": "depth32float", "commands": [[String: Any]](),
            ]
            if readOnly { command["depthReadOnly"] = true }
            return command
        }

        let rejected = harness.execute(setUpResources() + depthTarget + [
            bundle(id: 52, readOnly: false),
        ] + acquireDrawable + [
            readOnlyPass,
            ["op": "executeBundles", "bundles": [52]],
            ["op": "endPass"],
        ])
        XCTAssertTrue(
            ((errors(rejected).first?["message"] as? String) ?? "").contains("depthReadOnly"),
            harness.describeErrors(rejected)
        )

        harness.executeExpectingSuccess(setUpResources() + depthTarget + [
            bundle(id: 53, readOnly: true),
        ] + acquireDrawable + [
            readOnlyPass,
            ["op": "executeBundles", "bundles": [53]],
            ["op": "endPass"],
        ])
    }

    func test_패스없이_executeBundles하면_오류다() {
        let result = harness.execute(setUpResources() + [
            createBundle(id: 10, commands: fullScreenDraw(bindGroup: 6)),
            ["op": "executeBundles", "bundles": [10]],
        ])

        XCTAssertTrue(
            ((errors(result).first?["message"] as? String) ?? "").contains("beginRenderPass"),
            harness.describeErrors(result)
        )
    }

    func test_없는_번들_핸들은_validation_오류다() {
        let result = harness.execute(setUpResources() + acquireDrawable + [
            beginPass,
            ["op": "executeBundles", "bundles": [999]],
            ["op": "endPass"],
        ])

        XCTAssertEqual(errors(result).first?["kind"] as? String, "validation")
        XCTAssertTrue(((errors(result).first?["message"] as? String) ?? "").contains("GPURenderBundle"))
    }
}
