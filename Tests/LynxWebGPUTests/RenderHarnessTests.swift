import XCTest
import Metal
import LynxWebGPUCore
@testable import LynxWebGPU

/// 하네스 자신의 계약.
///
/// 동치성 단언(`assertFrameEquals`)은 그 위에 얹는 모든 테스트의 토대라, **다른 것을 다르다고
/// 하는지**를 여기서 못 박아 둔다. 이게 없으면 항상 통과하는 단언이 되어도 아무도 모른다.
final class RenderHarnessTests: XCTestCase {
    private var harness: RenderHarness!

    override func setUpWithError() throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "Metal 디바이스 없음")
        harness = try XCTUnwrap(RenderHarness.make(width: 16, height: 16))
    }

    override func tearDown() {
        harness = nil
        super.tearDown()
    }

    /// 클리어 색만 다른 한 프레임 — 셰이더 없이 표면 전체를 칠한다.
    private func clearFrame(red: Double) -> [[String: Any]] {
        [
            ["op": "getCurrentTexture", "id": 10, "canvas": "test"],
            ["op": "createTextureView", "id": 11, "texture": 10],
            ["op": "beginRenderPass", "colorAttachments": [[
                "view": 11, "loadOp": "clear", "storeOp": "store",
                "clearValue": ["r": red, "g": 0, "b": 0, "a": 1],
            ]]],
            ["op": "endPass"],
        ]
    }

    func test_runningTheSameCommandsTwiceGivesByteIdenticalFrames() throws {
        harness.executeExpectingSuccess([
            ["op": "configureCanvas", "canvas": "test", "format": "rgba8unorm"],
        ])

        harness.executeExpectingSuccess(clearFrame(red: 1))
        let reference = try harness.frameBytes()

        harness.executeExpectingSuccess(clearFrame(red: 1))
        try harness.assertFrameEquals(reference, "같은 입력은 같은 프레임이어야 한다")
    }

    func test_theEquivalenceAssertionCatchesADifferingFrame() throws {
        harness.executeExpectingSuccess([
            ["op": "configureCanvas", "canvas": "test", "format": "rgba8unorm"],
        ])

        harness.executeExpectingSuccess(clearFrame(red: 1))
        let reference = try harness.frameBytes()

        harness.executeExpectingSuccess(clearFrame(red: 0))
        // assertFrameEquals를 그대로 부르면 이 테스트가 실패하므로, 같은 비교를 뒤집어 확인한다.
        XCTAssertNotEqual(try harness.frameBytes(), reference, "다른 입력은 다른 프레임이어야 한다")
    }

    func test_synchronousReadbackReportsFailureAsAnError() {
        // 없는 핸들 — 콜백이 오지 않아 타임아웃으로 매달리면 안 되고, 오류로 즉시 돌아와야 한다.
        XCTAssertThrowsError(try harness.readBufferSync(handle: 999)) { error in
            XCTAssertTrue(
                "\(error)".contains("GPUBuffer"),
                "무엇이 없는지 알려 줘야 한다: \(error)"
            )
        }
    }
}
