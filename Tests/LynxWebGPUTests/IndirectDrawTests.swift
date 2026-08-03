import XCTest
import Metal
import LynxWebGPUCore
@testable import LynxWebGPU

/// 간접 드로우/디스패치 — **동치성으로** 검증한다.
///
/// 드로우 인자가 GPU 버퍼 안에 있어 커맨드 스트림만 봐서는 무엇을 그릴지 알 수 없다. 그래서
/// 같은 뜻의 직접 호출과 **프레임 전체를 비교**한다. 인자 구조체의 칸 순서를 잘못 이해하는 것이
/// 이 기능의 최대 버그원인데, 칸이 하나만 밀려도 그림이 달라지므로 그대로 드러난다.
///
/// 기준 프레임에는 픽셀 단언을 함께 건다 — 두 경로가 **똑같이 아무것도 안 그려도** 동치성은
/// 통과하기 때문이다.
final class IndirectDrawTests: XCTestCase {
    private var harness: RenderHarness!

    override func setUpWithError() throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "Metal 디바이스 없음")
        harness = try XCTUnwrap(RenderHarness.make())
    }

    override func tearDown() {
        harness = nil
        super.tearDown()
    }

    /// 정점 6개 = 화면을 반씩 덮는 삼각형 두 장. 인스턴스가 뒤쪽 삼각형을 고르고 색도 바꾼다.
    ///
    /// - `vertexCount`가 밀리면 삼각형이 깨지고,
    /// - `firstVertex`가 밀리면 **다른 절반**이 칠해지고,
    /// - `instanceCount`가 밀리면 색이 하나만 나온다.
    ///
    /// 세 칸이 서로 다른 방식으로 그림을 바꾸므로, 인자 구조체의 칸 순서가 어긋나면 반드시 보인다.
    private static let shader = """
    struct Out {
        @builtin(position) position: vec4f,
        @location(0) color: vec3f,
    };

    @vertex
    fn vs_main(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> Out {
        var corners = array<vec2f, 6>(
            vec2f(-1.0, -1.0), vec2f( 1.0, -1.0), vec2f(-1.0,  1.0),   // 왼쪽 아래 절반
            vec2f( 1.0,  1.0), vec2f(-1.0,  1.0), vec2f( 1.0, -1.0),   // 오른쪽 위 절반
        );
        var out: Out;
        out.position = vec4f(corners[(vertex + instance * 3u) % 6u], 0.0, 1.0);
        if (instance == 0u) {
            out.color = vec3f(1.0, 0.0, 0.0);
        } else {
            out.color = vec3f(0.0, 1.0, 0.0);
        }
        return out;
    }

    @fragment
    fn fs_main(in: Out) -> @location(0) vec4f {
        return vec4f(in.color, 1.0);
    }
    """

    private let red = (r: 255, g: 0, b: 0, a: 255)
    private let green = (r: 0, g: 255, b: 0, a: 255)
    private let clearBlue = (r: 0, g: 0, b: 255, a: 255)

    /// 왼쪽 아래 절반 한 점 · 오른쪽 위 절반 한 점 (반대각선으로 갈린다).
    private let lowerLeft = (x: 16, y: 48)
    private let upperRight = (x: 48, y: 16)

    private func setUpPipeline() -> [[String: Any]] {
        [
            ["op": "configureCanvas", "canvas": "test", "format": "rgba8unorm"],
            ["op": "createShaderModule", "id": 1, "code": Self.shader],
            ["op": "createRenderPipeline", "id": 2, "layout": "auto",
             "vertex": ["module": 1, "entryPoint": "vs_main"],
             "fragment": ["module": 1, "entryPoint": "fs_main",
                          "targets": [["format": "rgba8unorm"]]]],
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

    /// `u32` 배열을 커맨드에 실을 base64로.
    private func arguments(_ values: [UInt32]) -> String {
        values.withUnsafeBufferPointer { Data(buffer: $0).base64EncodedString() }
    }

    // MARK: - drawIndirect

    func test_drawIndirect가_직접_드로우와_같은_프레임을_낸다() throws {
        harness.executeExpectingSuccess(setUpPipeline() + [
            // vertexCount 3, instanceCount 2, firstVertex 0, firstInstance 0
            ["op": "createBuffer", "id": 3, "size": 16,
             "usage": TestUsage.indirect | TestUsage.copyDst,
             "data": arguments([3, 2, 0, 0])],
        ])

        harness.executeExpectingSuccess(acquireDrawable + [
            beginPass,
            ["op": "setPipeline", "pipeline": 2],
            ["op": "draw", "vertexCount": 3, "instanceCount": 2],
            ["op": "endPass"],
        ])
        // 기준 프레임이 실제로 뭔가를 그렸는지 먼저 못 박는다 — 빈 프레임끼리는 늘 같다.
        try harness.assertPixel(x: lowerLeft.x, y: lowerLeft.y, equals: red, "인스턴스 0")
        try harness.assertPixel(x: upperRight.x, y: upperRight.y, equals: green, "인스턴스 1")
        let direct = try harness.frameBytes()

        harness.executeExpectingSuccess(acquireDrawable + [
            beginPass,
            ["op": "setPipeline", "pipeline": 2],
            ["op": "drawIndirect", "indirectBuffer": 3],
            ["op": "endPass"],
        ])

        try harness.assertFrameEquals(direct, "간접 인자 4칸이 draw의 4개 인자와 같은 뜻이어야 한다")
    }

    func test_drawIndirect의_firstVertex가_정점_시작_위치를_민다() throws {
        harness.executeExpectingSuccess(setUpPipeline() + [
            // firstVertex 3 — 뒤쪽 삼각형(오른쪽 위 절반)만 그린다.
            ["op": "createBuffer", "id": 3, "size": 16,
             "usage": TestUsage.indirect | TestUsage.copyDst,
             "data": arguments([3, 1, 3, 0])],
        ])

        harness.executeExpectingSuccess(acquireDrawable + [
            beginPass,
            ["op": "setPipeline", "pipeline": 2],
            ["op": "draw", "vertexCount": 3, "instanceCount": 1, "firstVertex": 3],
            ["op": "endPass"],
        ])
        try harness.assertPixel(x: upperRight.x, y: upperRight.y, equals: red, "뒤쪽 삼각형")
        try harness.assertPixel(x: lowerLeft.x, y: lowerLeft.y, equals: clearBlue, "앞쪽 삼각형은 안 그린다")
        let direct = try harness.frameBytes()

        harness.executeExpectingSuccess(acquireDrawable + [
            beginPass,
            ["op": "setPipeline", "pipeline": 2],
            ["op": "drawIndirect", "indirectBuffer": 3],
            ["op": "endPass"],
        ])

        try harness.assertFrameEquals(direct, "인자 3번째 칸이 firstVertex여야 한다")
    }

    // MARK: - drawIndexedIndirect

    /// 회귀 — 직접 경로(`drawIndexed`)는 `firstIndex × stride`를 인덱스 버퍼 오프셋에 **더하지만**,
    /// 간접 경로에서는 `firstIndex`가 인자 버퍼 안에 있어 GPU가 따로 적용한다. 섞으면 두 번
    /// 세어 조용히 다른 삼각형이 나온다. `setIndexBuffer(offset:)`도 함께 반영돼야 한다.
    func test_drawIndexedIndirect가_인덱스버퍼_오프셋과_firstIndex를_함께_반영한다() throws {
        // 앞 2칸은 미끼 — setIndexBuffer(offset: 4)가 실제로 건너뛰는지 보려고 넣는다.
        let indices: [UInt16] = [9, 9, 0, 1, 2, 3, 4, 5]
        let indexData = indices.withUnsafeBufferPointer { Data(buffer: $0).base64EncodedString() }

        harness.executeExpectingSuccess(setUpPipeline() + [
            ["op": "createBuffer", "id": 3, "usage": TestUsage.index | TestUsage.copyDst,
             "data": indexData],
            // indexCount 3, instanceCount 1, firstIndex 3, baseVertex 0, firstInstance 0
            ["op": "createBuffer", "id": 4, "size": 20,
             "usage": TestUsage.indirect | TestUsage.copyDst,
             "data": arguments([3, 1, 3, 0, 0])],
        ])

        harness.executeExpectingSuccess(acquireDrawable + [
            beginPass,
            ["op": "setPipeline", "pipeline": 2],
            ["op": "setIndexBuffer", "buffer": 3, "format": "uint16", "offset": 4],
            ["op": "drawIndexed", "indexCount": 3, "firstIndex": 3],
            ["op": "endPass"],
        ])
        try harness.assertPixel(x: upperRight.x, y: upperRight.y, equals: red, "인덱스 3·4·5")
        try harness.assertPixel(x: lowerLeft.x, y: lowerLeft.y, equals: clearBlue, "인덱스 0·1·2는 안 쓴다")
        let direct = try harness.frameBytes()

        harness.executeExpectingSuccess(acquireDrawable + [
            beginPass,
            ["op": "setPipeline", "pipeline": 2],
            ["op": "setIndexBuffer", "buffer": 3, "format": "uint16", "offset": 4],
            ["op": "drawIndexedIndirect", "indirectBuffer": 4],
            ["op": "endPass"],
        ])

        try harness.assertFrameEquals(direct, "바인딩 오프셋 + 인자 firstIndex가 한 번씩만 적용돼야 한다")
    }

    // MARK: - dispatchWorkgroupsIndirect

    private static let computeShader = """
    @group(0) @binding(0) var<storage, read_write> out: array<u32>;

    @compute @workgroup_size(2)
    fn mark(@builtin(global_invocation_id) id: vec3u) {
        out[id.x] = id.x + 1u;
    }
    """

    func test_dispatchWorkgroupsIndirect가_직접_디스패치와_같은_결과를_쓴다() throws {
        // 워크그룹 3개 × 크기 2 = 6칸이 채워지고 나머지는 0으로 남는다.
        let expected: [UInt32] = [1, 2, 3, 4, 5, 6, 0, 0]

        harness.executeExpectingSuccess([
            ["op": "createShaderModule", "id": 1, "code": Self.computeShader],
            ["op": "createComputePipeline", "id": 2, "layout": "auto",
             "compute": ["module": 1, "entryPoint": "mark"]],
            ["op": "getBindGroupLayout", "id": 3, "pipeline": 2, "index": 0],
            ["op": "createBuffer", "id": 4, "size": 32,
             "usage": TestUsage.storage | TestUsage.copySrc | TestUsage.mapRead],
            ["op": "createBuffer", "id": 5, "size": 32,
             "usage": TestUsage.storage | TestUsage.copySrc | TestUsage.mapRead],
            ["op": "createBindGroup", "id": 6, "layout": 3,
             "entries": [["binding": 0, "resource": ["buffer": 4]]]],
            ["op": "createBindGroup", "id": 7, "layout": 3,
             "entries": [["binding": 0, "resource": ["buffer": 5]]]],
            ["op": "createBuffer", "id": 8, "size": 12,
             "usage": TestUsage.indirect | TestUsage.copyDst, "data": arguments([3, 1, 1])],
            ["op": "beginComputePass"],
            ["op": "setPipeline", "pipeline": 2],
            ["op": "setBindGroup", "index": 0, "bindGroup": 6],
            ["op": "dispatchWorkgroups", "x": 3],
            ["op": "setBindGroup", "index": 0, "bindGroup": 7],
            ["op": "dispatchWorkgroupsIndirect", "indirectBuffer": 8],
            ["op": "endPass"],
        ])

        let direct = try harness.readBufferSync(handle: 4, as: UInt32.self, size: 32)
        let indirect = try harness.readBufferSync(handle: 5, as: UInt32.self, size: 32)
        XCTAssertEqual(direct, expected, "직접 디스패치가 6칸을 채워야 한다")
        XCTAssertEqual(indirect, direct, "간접 디스패치는 같은 워크그룹 수를 돌려야 한다")
    }

    // MARK: - GPU-driven

    /// 이 기능의 존재 이유 — **컴퓨트가 만든 인자로 같은 배치에서 그린다.**
    ///
    /// 커맨드 스트림 순서(컴퓨트 → 드로우)가 실제 실행 순서라는 것도 함께 확인한다.
    /// 순서가 뒤집히면 드로우가 아직 0인 인자를 읽어 아무것도 안 그린다.
    func test_컴퓨트가_쓴_인자로_같은_배치에서_드로우한다() throws {
        let argumentWriter = """
        @group(0) @binding(0) var<storage, read_write> args: array<u32>;

        @compute @workgroup_size(1)
        fn fill() {
            args[0] = 3u;   // vertexCount
            args[1] = 2u;   // instanceCount
            args[2] = 0u;   // firstVertex
            args[3] = 0u;   // firstInstance
        }
        """

        harness.executeExpectingSuccess(setUpPipeline() + [
            ["op": "createShaderModule", "id": 10, "code": argumentWriter],
            ["op": "createComputePipeline", "id": 11, "layout": "auto",
             "compute": ["module": 10, "entryPoint": "fill"]],
            ["op": "getBindGroupLayout", "id": 12, "pipeline": 11, "index": 0],
            // 컴퓨트가 쓰고(STORAGE) 커맨드 프로세서가 읽는(INDIRECT) 버퍼. 처음엔 전부 0이다.
            ["op": "createBuffer", "id": 13, "size": 16,
             "usage": TestUsage.storage | TestUsage.indirect | TestUsage.copySrc | TestUsage.mapRead],
            ["op": "createBindGroup", "id": 14, "layout": 12,
             "entries": [["binding": 0, "resource": ["buffer": 13]]]],
        ])

        harness.executeExpectingSuccess(acquireDrawable + [
            ["op": "beginComputePass"],
            ["op": "setPipeline", "pipeline": 11],
            ["op": "setBindGroup", "index": 0, "bindGroup": 14],
            ["op": "dispatchWorkgroups", "x": 1],
            ["op": "endPass"],
            beginPass,
            ["op": "setPipeline", "pipeline": 2],
            ["op": "drawIndirect", "indirectBuffer": 13],
            ["op": "endPass"],
        ])

        try harness.assertPixel(x: lowerLeft.x, y: lowerLeft.y, equals: red, "컴퓨트가 정한 인스턴스 0")
        try harness.assertPixel(x: upperRight.x, y: upperRight.y, equals: green, "컴퓨트가 정한 인스턴스 1")
        // 인자 버퍼를 되짚어 "그림이 우연히 맞은" 경우를 배제한다.
        XCTAssertEqual(
            try harness.readBufferSync(handle: 13, as: UInt32.self, size: 16), [3, 2, 0, 0],
            "컴퓨트가 인자를 실제로 썼는지"
        )
    }
}
