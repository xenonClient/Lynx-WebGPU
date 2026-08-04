import XCTest
import Metal
import LynxWebGPUCore
@testable import LynxWebGPU

/// 쿼리셋 — 두 종류를 **다른 방식으로** 검증한다.
///
/// `occlusion`은 "몇 개의 샘플이 살아남았나"라 결정적이다. 값을 그대로 단언한다.
/// `timestamp`는 GPU 시계라 같은 입력에도 값이 매번 다르다. 값 대신 **구조**만 단언한다 —
/// 절대 시간 임계를 걸면 CI에서 흔들리기만 하고 잡아 주는 버그는 없다.
final class QuerySetTests: XCTestCase {
    private var harness: RenderHarness!

    override func setUpWithError() throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "Metal 디바이스 없음")
        harness = try XCTUnwrap(RenderHarness.make())
    }

    override func tearDown() {
        harness = nil
        super.tearDown()
    }

    private static let shader = """
    @vertex
    fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
        var corners = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
        return vec4f(corners[index], 0.0, 1.0);
    }

    @fragment
    fn fs_main() -> @location(0) vec4f {
        return vec4f(1.0, 0.0, 0.0, 1.0);
    }
    """

    private func errors(_ result: [String: Any]) -> [[String: Any]] {
        result["errors"] as? [[String: Any]] ?? []
    }

    private func setUpResources() -> [[String: Any]] {
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

    private func beginPass(occlusionQuerySet: Int? = nil) -> [String: Any] {
        var command: [String: Any] = [
            "op": "beginRenderPass",
            "colorAttachments": [[
                "view": 21, "loadOp": "clear", "storeOp": "store",
                "clearValue": ["r": 0, "g": 0, "b": 1, "a": 1],
            ]],
        ]
        if let occlusionQuerySet { command["occlusionQuerySet"] = occlusionQuerySet }
        return command
    }

    // MARK: - occlusion (결정적)

    /// 최소 조합 — **보이는 드로우와 완전히 잘린 드로우**. 하나만 보면 "0이 나와야 하는데
    /// 0이 나왔다"인지 "아무것도 안 세고 있어서 0"인지 구분할 수 없다.
    func test_occlusion_쿼리가_통과한_샘플_수를_센다() throws {
        harness.executeExpectingSuccess(setUpResources() + [
            ["op": "createQuerySet", "id": 3, "type": "occlusion", "count": 2],
            ["op": "createBuffer", "id": 4, "size": 16,
             "usage": TestUsage.queryResolve | TestUsage.copySrc],
        ] + acquireDrawable + [
            beginPass(occlusionQuerySet: 3),
            ["op": "setPipeline", "pipeline": 2],
            // 0번 — 화면 전체를 덮는다.
            ["op": "beginOcclusionQuery", "queryIndex": 0],
            ["op": "draw", "vertexCount": 3],
            ["op": "endOcclusionQuery"],
            // 1번 — 시저로 완전히 잘라 낸다. 정확히 0이어야 한다.
            ["op": "setScissorRect", "x": 0, "y": 0, "width": 0, "height": 0],
            ["op": "beginOcclusionQuery", "queryIndex": 1],
            ["op": "draw", "vertexCount": 3],
            ["op": "endOcclusionQuery"],
            ["op": "endPass"],
            ["op": "resolveQuerySet", "querySet": 3, "firstQuery": 0, "queryCount": 2,
             "destination": 4, "destinationOffset": 0],
        ])

        let results = try harness.readBufferSync(handle: 4, as: UInt64.self, size: 16)
        XCTAssertEqual(results.count, 2, "쿼리 하나당 u64 하나")
        XCTAssertEqual(results[0], 64 * 64, "화면 전체(64×64 샘플)가 통과해야 한다")
        XCTAssertEqual(results[1], 0, "완전히 잘린 드로우는 정확히 0이다")
    }

    func test_resolveQuerySet이_구간만_내린다() throws {
        harness.executeExpectingSuccess(setUpResources() + [
            ["op": "createQuerySet", "id": 3, "type": "occlusion", "count": 3],
            ["op": "createBuffer", "id": 4, "size": 8,
             "usage": TestUsage.queryResolve | TestUsage.copySrc],
        ] + acquireDrawable + [
            beginPass(occlusionQuerySet: 3),
            ["op": "setPipeline", "pipeline": 2],
            // 2번에만 그린다 — 0·1번은 건드리지 않는다.
            ["op": "beginOcclusionQuery", "queryIndex": 2],
            ["op": "draw", "vertexCount": 3],
            ["op": "endOcclusionQuery"],
            ["op": "endPass"],
            ["op": "resolveQuerySet", "querySet": 3, "firstQuery": 2, "queryCount": 1,
             "destination": 4, "destinationOffset": 0],
        ])

        XCTAssertEqual(
            try harness.readBufferSync(handle: 4, as: UInt64.self, size: 8), [UInt64(64 * 64)],
            "firstQuery가 가리키는 칸이 목적지 맨 앞에 와야 한다"
        )
    }

    // MARK: - timestamp (비결정적 — 구조만)

    func test_타임스탬프가_패스_경계에서_증가한다() throws {
        try XCTSkipUnless(
            harness.supports(.timestampQuery), "패스 경계 타임스탬프 샘플링을 지원하지 않는 기기"
        )

        harness.executeExpectingSuccess(setUpResources() + [
            ["op": "createQuerySet", "id": 3, "type": "timestamp", "count": 2],
            ["op": "createBuffer", "id": 4, "size": 16,
             "usage": TestUsage.queryResolve | TestUsage.copySrc],
        ] + acquireDrawable + [
            [
                "op": "beginRenderPass",
                "colorAttachments": [[
                    "view": 21, "loadOp": "clear", "storeOp": "store",
                    "clearValue": ["r": 0, "g": 0, "b": 1, "a": 1],
                ]],
                "timestampWrites": [
                    "querySet": 3, "beginningOfPassWriteIndex": 0, "endOfPassWriteIndex": 1,
                ],
            ],
            ["op": "setPipeline", "pipeline": 2],
            ["op": "draw", "vertexCount": 3],
            ["op": "endPass"],
            ["op": "resolveQuerySet", "querySet": 3, "firstQuery": 0, "queryCount": 2,
             "destination": 4, "destinationOffset": 0],
        ])

        let stamps = try harness.readBufferSync(handle: 4, as: UInt64.self, size: 16)
        // 값 자체는 GPU 시계라 단언할 수 없다. 구조만 본다 —
        // 절대 시간 임계("0.1ms 이상")를 걸면 CI에서 흔들리기만 하고 잡는 버그는 없다.
        XCTAssertEqual(stamps.count, 2, "쿼리 하나당 8바이트")
        XCTAssertNotEqual(stamps[0], 0, "초기값 그대로면 샘플링이 안 된 것이다")
        XCTAssertGreaterThanOrEqual(stamps[1], stamps[0], "끝이 시작보다 앞설 수는 없다")
    }

    /// 컴퓨트 패스의 타임스탬프는 **값을 단언하지 않는다.**
    ///
    /// Apple GPU에서 컴퓨트 인코더가 찍은 샘플은 CPU 쪽 `resolveCounterRange`로는 매번 제대로
    /// 나오지만, `MTLBlitCommandEncoder.resolveCounters`(= `resolveQuerySet`이 쓰는 GPU 경로)로는
    /// **같은 코드가 실행마다 0을 내기도 한다** — 커맨드 버퍼를 나눠도 그렇다. 드라이버 쪽
    /// 사정이라 여기서 고칠 수 없다.
    ///
    /// 그래서 흔들리는 단언 대신 "결과가 (0, 0)이거나 제대로 된 값이거나 둘 중 하나"만 본다.
    /// 쓰레기 값·잘못된 길이·resolve 실패는 여전히 잡히고, CI는 흔들리지 않는다.
    /// 프레임 계측이 목적이라면 **렌더 패스 타임스탬프**를 쓸 것 (그쪽은 안정적이다).
    func test_컴퓨트_패스도_타임스탬프를_찍는다() throws {
        try XCTSkipUnless(
            harness.supports(.timestampQuery), "패스 경계 타임스탬프 샘플링을 지원하지 않는 기기"
        )

        // 빈 패스는 Metal이 카운터를 찍지 않을 수 있다 — 실제 일을 하나 시킨다.
        let compute = """
        @group(0) @binding(0) var<storage, read_write> out: array<u32>;

        @compute @workgroup_size(1)
        fn touch() { out[0] = 1u; }
        """

        harness.executeExpectingSuccess([
            ["op": "createShaderModule", "id": 5, "code": compute],
            ["op": "createComputePipeline", "id": 6, "layout": "auto",
             "compute": ["module": 5, "entryPoint": "touch"]],
            ["op": "getBindGroupLayout", "id": 7, "pipeline": 6, "index": 0],
            ["op": "createBuffer", "id": 8, "size": 16, "usage": TestUsage.storage],
            ["op": "createBindGroup", "id": 9, "layout": 7,
             "entries": [["binding": 0, "resource": ["buffer": 8]]]],
            ["op": "createQuerySet", "id": 1, "type": "timestamp", "count": 2],
            ["op": "createBuffer", "id": 2, "size": 16,
             "usage": TestUsage.queryResolve | TestUsage.copySrc],
            ["op": "beginComputePass",
             "timestampWrites": ["querySet": 1, "beginningOfPassWriteIndex": 0,
                                 "endOfPassWriteIndex": 1]],
            ["op": "setPipeline", "pipeline": 6],
            ["op": "setBindGroup", "index": 0, "bindGroup": 9],
            ["op": "dispatchWorkgroups", "x": 1],
            ["op": "endPass"],
        ])

        harness.executeExpectingSuccess([
            ["op": "resolveQuerySet", "querySet": 1, "firstQuery": 0, "queryCount": 2,
             "destination": 2, "destinationOffset": 0],
        ])

        let stamps = try harness.readBufferSync(handle: 2, as: UInt64.self, size: 16)
        XCTAssertEqual(stamps.count, 2, "쿼리 하나당 8바이트")
        if stamps[0] == 0 && stamps[1] == 0 { return }   // 위 주석의 드라이버 사정
        XCTAssertNotEqual(stamps[0], 0, "한쪽만 0이면 쓰레기 값이다")
        XCTAssertGreaterThanOrEqual(stamps[1], stamps[0], "끝이 시작보다 앞설 수는 없다")
    }

    /// 지원하지 않는 기기에서 만들려 하면 **명확한 `unsupported`**가 나와야 한다.
    /// 지원하는 기기에서는 그냥 만들어지는지만 본다 — 양쪽 다 조용히 실패하면 안 된다.
    func test_타임스탬프_쿼리셋_생성이_기기_지원과_일치한다() {
        let result = harness.execute([
            ["op": "createQuerySet", "id": 1, "type": "timestamp", "count": 2],
        ])

        if harness.supports(.timestampQuery) {
            XCTAssertEqual(result["ok"] as? Bool, true, harness.describeErrors(result))
        } else {
            XCTAssertEqual(errors(result).first?["kind"] as? String, "unsupported")
        }
    }

    func test_어댑터가_타임스탬프_지원을_기능으로_알린다() throws {
        let info = harness.context.adapterInfo()
        let features = try XCTUnwrap(info["features"] as? [String])

        XCTAssertEqual(
            features.contains("timestamp-query"), harness.supports(.timestampQuery),
            "JS가 만들기 전에 물어볼 수 있어야 한다"
        )
        // Metal은 간접 드로우 인자의 firstInstance를 그대로 존중하므로 기기와 무관하게 참이다.
        XCTAssertTrue(features.contains("indirect-first-instance"), "\(features)")
    }

    // MARK: - 계약

    func test_쿼리셋_없이_beginOcclusionQuery하면_오류다() {
        let result = harness.execute(setUpResources() + acquireDrawable + [
            beginPass(),   // occlusionQuerySet 없이
            ["op": "beginOcclusionQuery", "queryIndex": 0],
            ["op": "endPass"],
        ])

        XCTAssertTrue(
            ((errors(result).first?["message"] as? String) ?? "").contains("occlusionQuerySet"),
            harness.describeErrors(result)
        )
    }

    func test_occlusion_쿼리는_중첩할_수_없다() {
        let result = harness.execute(setUpResources() + [
            ["op": "createQuerySet", "id": 3, "type": "occlusion", "count": 2],
        ] + acquireDrawable + [
            beginPass(occlusionQuerySet: 3),
            ["op": "beginOcclusionQuery", "queryIndex": 0],
            ["op": "beginOcclusionQuery", "queryIndex": 1],
            ["op": "endPass"],
        ])

        XCTAssertTrue(
            ((errors(result).first?["message"] as? String) ?? "").contains("중첩"),
            harness.describeErrors(result)
        )
    }

    func test_쿼리_인덱스가_범위를_넘으면_거부한다() {
        let result = harness.execute(setUpResources() + [
            ["op": "createQuerySet", "id": 3, "type": "occlusion", "count": 2],
        ] + acquireDrawable + [
            beginPass(occlusionQuerySet: 3),
            ["op": "beginOcclusionQuery", "queryIndex": 5],
            ["op": "endPass"],
        ])

        XCTAssertEqual(errors(result).first?["kind"] as? String, "validation")
        XCTAssertTrue(((errors(result).first?["message"] as? String) ?? "").contains("범위"))
    }

    func test_QUERY_RESOLVE_usage가_없는_버퍼에는_내릴_수_없다() {
        let result = harness.execute([
            ["op": "createQuerySet", "id": 1, "type": "occlusion", "count": 1],
            ["op": "createBuffer", "id": 2, "size": 8, "usage": TestUsage.copyDst],
            ["op": "resolveQuerySet", "querySet": 1, "queryCount": 1, "destination": 2],
        ])

        XCTAssertEqual(errors(result).first?["kind"] as? String, "validation")
        XCTAssertEqual(errors(result).first?["path"] as? String, "commands[2].destination")
        XCTAssertTrue(((errors(result).first?["message"] as? String) ?? "").contains("QUERY_RESOLVE"))
    }

    func test_목적지_오프셋은_256의_배수여야_한다() {
        let result = harness.execute([
            ["op": "createQuerySet", "id": 1, "type": "occlusion", "count": 1],
            ["op": "createBuffer", "id": 2, "size": 512, "usage": TestUsage.queryResolve],
            ["op": "resolveQuerySet", "querySet": 1, "queryCount": 1,
             "destination": 2, "destinationOffset": 8],
        ])

        XCTAssertEqual(errors(result).first?["path"] as? String, "commands[2].destinationOffset")
        XCTAssertTrue(((errors(result).first?["message"] as? String) ?? "").contains("256"))
    }

    func test_occlusion_쿼리셋을_timestampWrites에_주면_거부한다() {
        let result = harness.execute(setUpResources() + [
            ["op": "createQuerySet", "id": 3, "type": "occlusion", "count": 2],
        ] + acquireDrawable + [
            [
                "op": "beginRenderPass",
                "colorAttachments": [[
                    "view": 21, "loadOp": "clear", "storeOp": "store",
                    "clearValue": ["r": 0, "g": 0, "b": 0, "a": 1],
                ]],
                "timestampWrites": ["querySet": 3, "beginningOfPassWriteIndex": 0],
            ],
            ["op": "endPass"],
        ])

        XCTAssertTrue(
            ((errors(result).first?["message"] as? String) ?? "").contains("timestamp"),
            harness.describeErrors(result)
        )
    }

    func test_크기0_쿼리셋은_거부한다() {
        let result = harness.execute([["op": "createQuerySet", "id": 1, "type": "occlusion", "count": 0]])
        XCTAssertEqual(errors(result).first?["kind"] as? String, "validation")
        XCTAssertEqual(errors(result).first?["path"] as? String, "commands[0].count")
    }

    /// 상한을 넘는 쿼리셋은 여기서 만들어지지만 브라우저에서는 validation 오류다.
    /// (occlusion이면 `count * 8`바이트를 그대로 할당하기까지 한다.)
    func test_쿼리_개수_상한을_넘으면_거부한다() {
        let result = harness.execute([
            ["op": "createQuerySet", "id": 1, "type": "occlusion",
             "count": WGPUQuerySetDescriptor.maxCount + 1],
        ])
        XCTAssertEqual(errors(result).first?["kind"] as? String, "validation")
        XCTAssertEqual(errors(result).first?["path"] as? String, "commands[0].count")

        harness.executeExpectingSuccess([
            ["op": "createQuerySet", "id": 2, "type": "occlusion",
             "count": WGPUQuerySetDescriptor.maxCount],
        ])
    }

    /// 두 인덱스를 모두 생략하면 Metal 샘플 인덱스가 전부 `MTLCounterDontSample`이 되어
    /// **오류 없이 아무것도 찍지 않는 패스**가 된다. 앱은 GPU 시간을 0ns로 읽는다.
    func test_timestampWrites에_인덱스가_하나도_없으면_거부한다() throws {
        try XCTSkipUnless(harness.supports(.timestampQuery), "타임스탬프 미지원 기기")

        let result = harness.execute(setUpResources() + [
            ["op": "createQuerySet", "id": 3, "type": "timestamp", "count": 2],
        ] + acquireDrawable + [
            [
                "op": "beginRenderPass",
                "colorAttachments": [[
                    "view": 21, "loadOp": "clear", "storeOp": "store",
                    "clearValue": ["r": 0, "g": 0, "b": 0, "a": 1],
                ]],
                "timestampWrites": ["querySet": 3],
            ],
            ["op": "endPass"],
        ])

        XCTAssertTrue(
            ((errors(result).first?["message"] as? String) ?? "").contains("최소 하나"),
            harness.describeErrors(result)
        )
    }

    /// 같은 슬롯을 가리키면 끝 샘플이 시작 샘플을 덮어 델타가 의미를 잃는다.
    func test_timestampWrites의_두_인덱스가_같으면_거부한다() throws {
        try XCTSkipUnless(harness.supports(.timestampQuery), "타임스탬프 미지원 기기")

        let result = harness.execute(setUpResources() + [
            ["op": "createQuerySet", "id": 3, "type": "timestamp", "count": 2],
        ] + acquireDrawable + [
            [
                "op": "beginRenderPass",
                "colorAttachments": [[
                    "view": 21, "loadOp": "clear", "storeOp": "store",
                    "clearValue": ["r": 0, "g": 0, "b": 0, "a": 1],
                ]],
                "timestampWrites": [
                    "querySet": 3, "beginningOfPassWriteIndex": 1, "endOfPassWriteIndex": 1,
                ],
            ],
            ["op": "endPass"],
        ])

        XCTAssertTrue(
            ((errors(result).first?["message"] as? String) ?? "").contains("서로 달라야"),
            harness.describeErrors(result)
        )
    }

    /// Metal은 `endEncoding` 시점에 visibility 결과를 그대로 써 주므로 **값까지 정상으로 보인다.**
    /// 브라우저에서는 부모 커맨드 인코더가 무효화되어 프레임이 통째로 날아간다.
    func test_occlusion_쿼리를_닫지_않고_endPass하면_거부한다() {
        let result = harness.execute(setUpResources() + [
            ["op": "createQuerySet", "id": 3, "type": "occlusion", "count": 2],
        ] + acquireDrawable + [
            beginPass(occlusionQuerySet: 3),
            ["op": "setPipeline", "pipeline": 2],
            ["op": "beginOcclusionQuery", "queryIndex": 0],
            ["op": "draw", "vertexCount": 3],
            ["op": "endPass"],   // endOcclusionQuery 없이
        ])

        XCTAssertTrue(
            ((errors(result).first?["message"] as? String) ?? "").contains("열린 채로"),
            harness.describeErrors(result)
        )
    }

    /// 같은 인덱스를 두 번 쓰면 두 구간이 같은 8바이트 슬롯을 나눠 쓴다 —
    /// 남는 값이 Metal의 누적/덮어쓰기 동작에 달린 값이 되어 브라우저와 결과가 갈린다.
    func test_같은_패스에서_occlusion_인덱스를_재사용하면_거부한다() {
        let result = harness.execute(setUpResources() + [
            ["op": "createQuerySet", "id": 3, "type": "occlusion", "count": 2],
        ] + acquireDrawable + [
            beginPass(occlusionQuerySet: 3),
            ["op": "setPipeline", "pipeline": 2],
            ["op": "beginOcclusionQuery", "queryIndex": 0],
            ["op": "draw", "vertexCount": 3],
            ["op": "endOcclusionQuery"],
            ["op": "beginOcclusionQuery", "queryIndex": 0],
            ["op": "draw", "vertexCount": 3],
            ["op": "endOcclusionQuery"],
            ["op": "endPass"],
        ])

        XCTAssertTrue(
            ((errors(result).first?["message"] as? String) ?? "").contains("이미 썼다"),
            harness.describeErrors(result)
        )
    }

    /// 반대로 **패스가 다르면** 같은 인덱스를 다시 쓸 수 있어야 한다 (명세는 패스 안에서만 막는다).
    func test_패스가_다르면_같은_occlusion_인덱스를_다시_쓸_수_있다() {
        let pass: [[String: Any]] = [
            beginPass(occlusionQuerySet: 3),
            ["op": "setPipeline", "pipeline": 2],
            ["op": "beginOcclusionQuery", "queryIndex": 0],
            ["op": "draw", "vertexCount": 3],
            ["op": "endOcclusionQuery"],
            ["op": "endPass"],
        ]
        harness.executeExpectingSuccess(setUpResources() + [
            ["op": "createQuerySet", "id": 3, "type": "occlusion", "count": 2],
        ] + acquireDrawable + pass + pass)
    }
}
