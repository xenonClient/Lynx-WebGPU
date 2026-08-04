import XCTest
import Metal
import LynxWebGPUCore
@testable import LynxWebGPU

/// 스텐실 테스트 — **컬러로 간접 단언**한다.
///
/// `readPixels`는 depth/stencil 표면을 읽을 수 없다 (`docs/TESTING.md` §4-1). 하지만 스텐실이
/// 정하는 것은 "어느 프래그먼트가 살아남는가"이므로, 살아남은 프래그먼트의 **색이 곧 스텐실
/// 동작의 증거**다. 스텐실 버퍼를 직접 못 읽는 것이 검증의 제약이 되지 않는다.
final class StencilTests: XCTestCase {
    private var harness: RenderHarness!

    override func setUpWithError() throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "Metal 디바이스 없음")
        harness = try XCTUnwrap(RenderHarness.make())
    }

    override func tearDown() {
        harness = nil
        super.tearDown()
    }

    /// 화면을 덮는 삼각형 하나. 색과 깊이를 유니폼으로 받아 파이프라인을 여러 개 만들 필요를 줄인다.
    private static let shader = """
    struct Fill {
        color: vec4f,
        depth: f32,
    };
    @group(0) @binding(0) var<uniform> fill: Fill;

    @vertex
    fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
        var corners = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
        return vec4f(corners[index], fill.depth, 1.0);
    }

    @fragment
    fn fs_main() -> @location(0) vec4f {
        return fill.color;
    }
    """

    /// `Fill` 유니폼 한 벌 — vec4f(16B) + f32, 정렬 규칙에 따라 32B다.
    private func fill(red: Float, green: Float, blue: Float, depth: Float = 0) -> String {
        [red, green, blue, 1, depth, 0, 0, 0].base64
    }

    private let red = (r: 255, g: 0, b: 0, a: 255)
    private let clearBlue = (r: 0, g: 0, b: 255, a: 255)

    /// 색 유니폼 두 개(빨강/초록)와 셰이더를 올린다. 파이프라인은 테스트마다 다르므로 여기 없다.
    private func makeCommonResources() -> [[String: Any]] {
        [
            ["op": "configureCanvas", "canvas": "test", "format": "rgba8unorm"],
            ["op": "createShaderModule", "id": 1, "code": Self.shader],
            ["op": "createBuffer", "id": 4, "size": 32, "usage": TestUsage.uniform,
             "data": fill(red: 1, green: 0, blue: 0)],
            ["op": "createBuffer", "id": 5, "size": 32, "usage": TestUsage.uniform,
             "data": fill(red: 0, green: 1, blue: 0)],
        ]
    }

    /// 스텐실 상태만 다른 파이프라인 하나. `colorWriteMask: 0`이면 스텐실만 쓰고 색은 남기지 않는다.
    private func pipeline(
        id: Int,
        format: String = "stencil8",
        stencil: [String: Any],
        writesColor: Bool = true,
        depthCompare: String = "always",
        depthWrite: Bool = false
    ) -> [String: Any] {
        var target: [String: Any] = ["format": "rgba8unorm"]
        if !writesColor { target["writeMask"] = 0 }
        var depthStencil: [String: Any] = [
            "format": format, "depthCompare": depthCompare, "depthWriteEnabled": depthWrite,
        ]
        depthStencil.merge(stencil) { _, new in new }
        return [
            "op": "createRenderPipeline", "id": id, "layout": "auto",
            "vertex": ["module": 1, "entryPoint": "vs_main"],
            "fragment": ["module": 1, "entryPoint": "fs_main", "targets": [target]],
            "depthStencil": depthStencil,
        ]
    }

    /// 앞/뒤 면에 같은 상태를 넣는다 (테스트 도형은 한 방향이라 면을 나눌 이유가 없다).
    private func bothFaces(_ state: [String: Any]) -> [String: Any] {
        ["stencilFront": state, "stencilBack": state]
    }

    private func beginPass(stencilView: Int) -> [String: Any] {
        [
            "op": "beginRenderPass",
            "colorAttachments": [[
                "view": 21, "loadOp": "clear", "storeOp": "store",
                "clearValue": ["r": 0, "g": 0, "b": 1, "a": 1],
            ]],
            "depthStencilAttachment": [
                "view": stencilView,
                "depthClearValue": 1.0, "depthLoadOp": "clear", "depthStoreOp": "store",
                "stencilClearValue": 0, "stencilLoadOp": "clear", "stencilStoreOp": "store",
            ],
        ]
    }

    private let acquireDrawable: [[String: Any]] = [
        ["op": "getCurrentTexture", "id": 20, "canvas": "test"],
        ["op": "createTextureView", "id": 21, "texture": 20],
    ]

    // MARK: - 마스킹

    func test_스텐실_마스크가_뒤이은_드로우를_잘라낸다() throws {
        harness.executeExpectingSuccess(makeCommonResources() + [
            ["op": "createTexture", "id": 2, "size": ["width": 64, "height": 64],
             "format": "stencil8", "usage": TestUsage.renderAttachment],
            ["op": "createTextureView", "id": 3, "texture": 2],
            // 마킹 — 그린 곳의 스텐실을 ref로 바꾸고, 색은 남기지 않는다.
            pipeline(id: 6, stencil: bothFaces(["compare": "always", "passOp": "replace"]),
                     writesColor: false),
            // 칠하기 — 스텐실이 ref와 같은 곳만.
            pipeline(id: 7, stencil: bothFaces(["compare": "equal"])),
            ["op": "getBindGroupLayout", "id": 8, "pipeline": 6, "index": 0],
            ["op": "createBindGroup", "id": 9, "layout": 8,
             "entries": [["binding": 0, "resource": ["buffer": 4]]]],
            ["op": "createBindGroup", "id": 10, "layout": 8,
             "entries": [["binding": 0, "resource": ["buffer": 5]]]],
        ] + acquireDrawable + [
            beginPass(stencilView: 3),
            ["op": "setStencilReference", "reference": 1],
            ["op": "setPipeline", "pipeline": 6],
            ["op": "setBindGroup", "index": 0, "bindGroup": 10],   // 초록 — writeMask 0이라 보이면 안 된다
            ["op": "setScissorRect", "x": 0, "y": 0, "width": 32, "height": 64],
            ["op": "draw", "vertexCount": 3],
            ["op": "setScissorRect", "x": 0, "y": 0, "width": 64, "height": 64],
            ["op": "setPipeline", "pipeline": 7],
            ["op": "setBindGroup", "index": 0, "bindGroup": 9],    // 빨강
            ["op": "draw", "vertexCount": 3],
            ["op": "endPass"],
        ])

        try harness.assertPixel(x: 16, y: 32, equals: red, "스텐실 통과 영역")
        try harness.assertPixel(x: 48, y: 32, equals: clearBlue, "스텐실 실패 → 클리어색")
    }

    /// 같은 그림을 시저로 만들어 놓고 프레임 전체를 비교한다.
    ///
    /// 두 점만 보면 "스텐실이 아니라 다른 이유로 잘린" 경우를 못 거른다. 경계 픽셀까지
    /// 똑같아야 스텐실이 정확히 그 영역을 잘랐다고 말할 수 있다.
    func test_스텐실_마스크_결과가_같은_영역의_시저와_일치한다() throws {
        harness.executeExpectingSuccess(makeCommonResources() + [
            ["op": "createTexture", "id": 2, "size": ["width": 64, "height": 64],
             "format": "stencil8", "usage": TestUsage.renderAttachment],
            ["op": "createTextureView", "id": 3, "texture": 2],
            pipeline(id: 6, stencil: bothFaces(["compare": "always", "passOp": "replace"]),
                     writesColor: false),
            pipeline(id: 7, stencil: bothFaces(["compare": "equal"])),
            // 대조군 — 스텐실 없이 시저만 쓰는 파이프라인.
            ["op": "createRenderPipeline", "id": 30, "layout": "auto",
             "vertex": ["module": 1, "entryPoint": "vs_main"],
             "fragment": ["module": 1, "entryPoint": "fs_main",
                          "targets": [["format": "rgba8unorm"]]]],
            ["op": "getBindGroupLayout", "id": 8, "pipeline": 6, "index": 0],
            ["op": "createBindGroup", "id": 9, "layout": 8,
             "entries": [["binding": 0, "resource": ["buffer": 4]]]],
        ])

        // 기준 — 시저로 왼쪽 절반만 빨강.
        harness.executeExpectingSuccess(acquireDrawable + [
            ["op": "beginRenderPass", "colorAttachments": [[
                "view": 21, "loadOp": "clear", "storeOp": "store",
                "clearValue": ["r": 0, "g": 0, "b": 1, "a": 1],
            ]]],
            ["op": "setPipeline", "pipeline": 30],
            ["op": "setBindGroup", "index": 0, "bindGroup": 9],
            ["op": "setScissorRect", "x": 0, "y": 0, "width": 32, "height": 64],
            ["op": "draw", "vertexCount": 3],
            ["op": "endPass"],
        ])
        let scissored = try harness.frameBytes()

        // 같은 그림을 스텐실 마스크로.
        harness.executeExpectingSuccess(acquireDrawable + [
            beginPass(stencilView: 3),
            ["op": "setStencilReference", "reference": 1],
            ["op": "setPipeline", "pipeline": 6],
            ["op": "setBindGroup", "index": 0, "bindGroup": 9],
            ["op": "setScissorRect", "x": 0, "y": 0, "width": 32, "height": 64],
            ["op": "draw", "vertexCount": 3],
            ["op": "setScissorRect", "x": 0, "y": 0, "width": 64, "height": 64],
            ["op": "setPipeline", "pipeline": 7],
            ["op": "draw", "vertexCount": 3],
            ["op": "endPass"],
        ])

        try harness.assertFrameEquals(scissored, "스텐실이 시저와 정확히 같은 영역을 남겨야 한다")
    }

    // MARK: - 참조값 · 마스크

    func test_setStencilReference가_쓰기와_비교에_모두_반영된다() throws {
        harness.executeExpectingSuccess(makeCommonResources() + [
            ["op": "createTexture", "id": 2, "size": ["width": 64, "height": 64],
             "format": "stencil8", "usage": TestUsage.renderAttachment],
            ["op": "createTextureView", "id": 3, "texture": 2],
            pipeline(id: 6, stencil: bothFaces(["compare": "always", "passOp": "replace"]),
                     writesColor: false),
            pipeline(id: 7, stencil: bothFaces(["compare": "equal"])),
            ["op": "getBindGroupLayout", "id": 8, "pipeline": 6, "index": 0],
            ["op": "createBindGroup", "id": 9, "layout": 8,
             "entries": [["binding": 0, "resource": ["buffer": 4]]]],
        ] + acquireDrawable + [
            beginPass(stencilView: 3),
            ["op": "setPipeline", "pipeline": 6],
            ["op": "setBindGroup", "index": 0, "bindGroup": 9],
            // 왼쪽에는 1, 오른쪽에는 2를 쓴다 — replace가 쓰는 값이 곧 현재 reference다.
            ["op": "setStencilReference", "reference": 1],
            ["op": "setScissorRect", "x": 0, "y": 0, "width": 32, "height": 64],
            ["op": "draw", "vertexCount": 3],
            ["op": "setStencilReference", "reference": 2],
            ["op": "setScissorRect", "x": 32, "y": 0, "width": 32, "height": 64],
            ["op": "draw", "vertexCount": 3],
            // reference 2로 비교 → 오른쪽만 통과한다.
            ["op": "setScissorRect", "x": 0, "y": 0, "width": 64, "height": 64],
            ["op": "setPipeline", "pipeline": 7],
            ["op": "draw", "vertexCount": 3],
            ["op": "endPass"],
        ])

        try harness.assertPixel(x: 48, y: 32, equals: red, "reference 2를 쓴 영역")
        try harness.assertPixel(x: 16, y: 32, equals: clearBlue, "reference 1을 쓴 영역 — 2와 다르다")
    }

    func test_stencilWriteMask가_0이면_스텐실이_갱신되지_않는다() throws {
        harness.executeExpectingSuccess(makeCommonResources() + [
            ["op": "createTexture", "id": 2, "size": ["width": 64, "height": 64],
             "format": "stencil8", "usage": TestUsage.renderAttachment],
            ["op": "createTextureView", "id": 3, "texture": 2],
            // 왼쪽 마킹 — writeMask 0이라 replace가 아무 비트도 못 바꾼다.
            pipeline(id: 6,
                     stencil: bothFaces(["compare": "always", "passOp": "replace"])
                        .merging(["stencilWriteMask": 0]) { _, new in new },
                     writesColor: false),
            // 오른쪽 마킹 — 같은 상태에 마스크만 기본값.
            pipeline(id: 7, stencil: bothFaces(["compare": "always", "passOp": "replace"]),
                     writesColor: false),
            pipeline(id: 8, stencil: bothFaces(["compare": "equal"])),
            ["op": "getBindGroupLayout", "id": 9, "pipeline": 6, "index": 0],
            ["op": "createBindGroup", "id": 10, "layout": 9,
             "entries": [["binding": 0, "resource": ["buffer": 4]]]],
        ] + acquireDrawable + [
            beginPass(stencilView: 3),
            ["op": "setStencilReference", "reference": 1],
            ["op": "setBindGroup", "index": 0, "bindGroup": 10],
            ["op": "setPipeline", "pipeline": 6],
            ["op": "setScissorRect", "x": 0, "y": 0, "width": 32, "height": 64],
            ["op": "draw", "vertexCount": 3],
            ["op": "setPipeline", "pipeline": 7],
            ["op": "setScissorRect", "x": 32, "y": 0, "width": 32, "height": 64],
            ["op": "draw", "vertexCount": 3],
            ["op": "setScissorRect", "x": 0, "y": 0, "width": 64, "height": 64],
            ["op": "setPipeline", "pipeline": 8],
            ["op": "draw", "vertexCount": 3],
            ["op": "endPass"],
        ])

        try harness.assertPixel(x: 16, y: 32, equals: clearBlue, "writeMask 0 — 스텐실이 0으로 남는다")
        try harness.assertPixel(x: 48, y: 32, equals: red, "기본 마스크 — 스텐실이 1로 바뀐다")
    }

    func test_stencilReadMask가_비교_전에_값을_가린다() throws {
        harness.executeExpectingSuccess(makeCommonResources() + [
            ["op": "createTexture", "id": 2, "size": ["width": 64, "height": 64],
             "format": "stencil8", "usage": TestUsage.renderAttachment],
            ["op": "createTextureView", "id": 3, "texture": 2],
            pipeline(id: 6, stencil: bothFaces(["compare": "always", "passOp": "replace"]),
                     writesColor: false),
            // 마스크 없이 비교 — 저장된 3과 reference 1은 다르다.
            pipeline(id: 7, stencil: bothFaces(["compare": "equal"])),
            // 하위 1비트만 보고 비교 — (1 & 1) == (3 & 1) 이라 통과한다.
            pipeline(id: 8,
                     stencil: bothFaces(["compare": "equal"])
                        .merging(["stencilReadMask": 0x1]) { _, new in new }),
            ["op": "getBindGroupLayout", "id": 9, "pipeline": 6, "index": 0],
            ["op": "createBindGroup", "id": 10, "layout": 9,
             "entries": [["binding": 0, "resource": ["buffer": 4]]]],
        ] + acquireDrawable + [
            beginPass(stencilView: 3),
            ["op": "setBindGroup", "index": 0, "bindGroup": 10],
            // 화면 전체에 스텐실 3.
            ["op": "setStencilReference", "reference": 3],
            ["op": "setPipeline", "pipeline": 6],
            ["op": "draw", "vertexCount": 3],
            ["op": "setStencilReference", "reference": 1],
            ["op": "setPipeline", "pipeline": 7],
            ["op": "setScissorRect", "x": 0, "y": 0, "width": 32, "height": 64],
            ["op": "draw", "vertexCount": 3],
            ["op": "setPipeline", "pipeline": 8],
            ["op": "setScissorRect", "x": 32, "y": 0, "width": 32, "height": 64],
            ["op": "draw", "vertexCount": 3],
            ["op": "endPass"],
        ])

        try harness.assertPixel(x: 16, y: 32, equals: clearBlue, "마스크 없이 1 ≠ 3 → 실패")
        try harness.assertPixel(x: 48, y: 32, equals: red, "readMask 0x1이면 1 == 3 → 통과")
    }

    // MARK: - 깊이와의 조합

    /// 섀도 볼륨이 쓰는 경로 — "깊이 테스트에 **진** 프래그먼트"만 스텐실에 표시한다.
    /// `passOp`(둘 다 통과)와 `depthFailOp`(스텐실만 통과)가 뒤바뀌면 여기서 드러난다.
    func test_depthFailOp가_깊이에_진_영역만_스텐실에_표시한다() throws {
        harness.executeExpectingSuccess(makeCommonResources() + [
            ["op": "createTexture", "id": 2, "size": ["width": 64, "height": 64],
             "format": "depth24plus-stencil8", "usage": TestUsage.renderAttachment],
            ["op": "createTextureView", "id": 3, "texture": 2],
            // 가까운 면(0.2)을 깊이 버퍼에 남긴다.
            ["op": "createBuffer", "id": 11, "size": 32, "usage": TestUsage.uniform,
             "data": fill(red: 0, green: 1, blue: 0, depth: 0.2)],
            // 먼 면(0.8) — 깊이 테스트에서 진다.
            ["op": "createBuffer", "id": 12, "size": 32, "usage": TestUsage.uniform,
             "data": fill(red: 0, green: 1, blue: 0, depth: 0.8)],
            pipeline(id: 6, format: "depth24plus-stencil8", stencil: [:],
                     writesColor: false, depthCompare: "less", depthWrite: true),
            pipeline(id: 7, format: "depth24plus-stencil8",
                     stencil: bothFaces(["compare": "always", "depthFailOp": "replace"]),
                     writesColor: false, depthCompare: "less"),
            pipeline(id: 8, format: "depth24plus-stencil8",
                     stencil: bothFaces(["compare": "equal"])),
            ["op": "getBindGroupLayout", "id": 9, "pipeline": 6, "index": 0],
            ["op": "createBindGroup", "id": 13, "layout": 9,
             "entries": [["binding": 0, "resource": ["buffer": 11]]]],
            ["op": "createBindGroup", "id": 14, "layout": 9,
             "entries": [["binding": 0, "resource": ["buffer": 12]]]],
            ["op": "createBindGroup", "id": 15, "layout": 9,
             "entries": [["binding": 0, "resource": ["buffer": 4]]]],
        ] + acquireDrawable + [
            beginPass(stencilView: 3),
            ["op": "setStencilReference", "reference": 1],
            // 1) 깊이 0.2를 깔아 둔다.
            ["op": "setPipeline", "pipeline": 6],
            ["op": "setBindGroup", "index": 0, "bindGroup": 13],
            ["op": "draw", "vertexCount": 3],
            // 2) 깊이 0.8 — 스텐실 테스트는 통과하고 깊이 테스트에 져서 depthFailOp가 돈다.
            //    왼쪽 절반에만 그려 "표시된 곳/아닌 곳"을 한 화면에 만든다.
            ["op": "setPipeline", "pipeline": 7],
            ["op": "setBindGroup", "index": 0, "bindGroup": 14],
            ["op": "setScissorRect", "x": 0, "y": 0, "width": 32, "height": 64],
            ["op": "draw", "vertexCount": 3],
            // 3) 표시된 곳만 빨강.
            ["op": "setScissorRect", "x": 0, "y": 0, "width": 64, "height": 64],
            ["op": "setPipeline", "pipeline": 8],
            ["op": "setBindGroup", "index": 0, "bindGroup": 15],
            ["op": "draw", "vertexCount": 3],
            ["op": "endPass"],
        ])

        try harness.assertPixel(x: 16, y: 32, equals: red, "깊이에 진 영역 — depthFailOp가 표시했다")
        try harness.assertPixel(x: 48, y: 32, equals: clearBlue, "깊이 테스트를 치르지 않은 영역")
    }

    // MARK: - 회귀

    /// 회귀 — 예전에는 `depthAttachmentPixelFormat`을 무조건 세팅해서, 깊이가 없는 `stencil8`
    /// 파이프라인이 "있지도 않은 깊이 어태치먼트"를 요구하며 생성 자체에 실패했다.
    func test_stencil8_단독_포맷_파이프라인이_만들어진다() {
        harness.executeExpectingSuccess(makeCommonResources() + [
            pipeline(id: 6, stencil: bothFaces(["compare": "equal", "passOp": "replace"])),
        ])
    }

    // MARK: - 계약

    /// 스텐실 성분이 없는 포맷에 스텐실 상태를 붙이면 Metal은 **조용히 무시**한다.
    /// 그러면 "스텐실 마스킹이 왜 안 먹지"를 오류 하나 없이 디버깅하게 되므로 여기서 막는다.
    func test_스텐실_없는_깊이_포맷에_스텐실_상태를_주면_거부한다() {
        for format in ["depth32float", "depth24plus", "depth16unorm"] {
            let result = harness.execute(makeCommonResources() + [
                pipeline(id: 6, format: format,
                         stencil: bothFaces(["compare": "equal", "passOp": "replace"])),
            ])
            let message = (result["errors"] as? [[String: Any]])?.first?["message"] as? String ?? ""
            XCTAssertTrue(
                message.contains("스텐실 성분"),
                "\(format) + 비기본 stencilFront는 거부되어야 한다 — 받은 것: \(message)"
            )
        }
    }

    /// `GPUStencilValue`는 `u32`이고 WebIDL 변환은 modulo다 — 음수는 wrap될 뿐 트랩하지 않는다.
    /// 비-truncating 이니셜라이저를 쓰면 이 한 줄로 프로세스가 죽는다.
    func test_음수_스텐실_참조값이_프로세스를_죽이지_않는다() {
        harness.executeExpectingSuccess(makeCommonResources() + [
            ["op": "createTexture", "id": 2, "size": ["width": 64, "height": 64],
             "format": "stencil8", "usage": TestUsage.renderAttachment],
            ["op": "createTextureView", "id": 3, "texture": 2],
            pipeline(id: 6, stencil: bothFaces(["compare": "always", "passOp": "replace"])),
        ] + acquireDrawable + [
            [
                "op": "beginRenderPass",
                "colorAttachments": [[
                    "view": 21, "loadOp": "clear", "storeOp": "store",
                    "clearValue": ["r": 0, "g": 0, "b": 1, "a": 1],
                ]],
                "depthStencilAttachment": [
                    "view": 3,
                    "stencilClearValue": -1, "stencilLoadOp": "clear", "stencilStoreOp": "store",
                ],
            ],
            ["op": "setStencilReference", "reference": -1],
            ["op": "endPass"],
        ])
    }

    /// 반대로 스텐실 상태를 주지 않으면 깊이 전용 포맷도 그대로 통과해야 한다.
    func test_스텐실_상태가_기본값이면_깊이_전용_포맷이_통과한다() {
        harness.executeExpectingSuccess(makeCommonResources() + [
            pipeline(id: 6, format: "depth32float", stencil: [:], depthCompare: "less", depthWrite: true),
        ])
    }
}
