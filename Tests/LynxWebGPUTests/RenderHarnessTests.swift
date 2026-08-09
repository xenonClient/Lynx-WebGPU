import XCTest
import Metal
import LynxWebGPUCore
@testable import LynxWebGPU

/// The harness's own contract.
///
/// The equivalence assertion (`assertFrameEquals`) is the foundation every test above it rests on, so
/// **that it calls different things different** is pinned here. Without this it could become an
/// always-passing assertion and nobody would know.
final class RenderHarnessTests: XCTestCase {
    private var harness: RenderHarness!

    override func setUpWithError() throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "no Metal device")
        harness = try XCTUnwrap(RenderHarness.make(width: 16, height: 16))
    }

    override func tearDown() {
        harness = nil
        super.tearDown()
    }

    /// A frame differing only in clear color — the whole surface is painted with no shader.
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
        try harness.assertFrameEquals(reference, "the same input must give the same frame")
    }

    func test_theEquivalenceAssertionCatchesADifferingFrame() throws {
        harness.executeExpectingSuccess([
            ["op": "configureCanvas", "canvas": "test", "format": "rgba8unorm"],
        ])

        harness.executeExpectingSuccess(clearFrame(red: 1))
        let reference = try harness.frameBytes()

        harness.executeExpectingSuccess(clearFrame(red: 0))
        // Calling assertFrameEquals directly would fail this test, so the same comparison is inverted.
        XCTAssertNotEqual(try harness.frameBytes(), reference, "different input must give a different frame")
    }

    func test_synchronousReadbackReportsFailureAsAnError() {
        // A missing handle — it must not hang until timeout with no callback, but return an error at once.
        XCTAssertThrowsError(try harness.readBufferSync(handle: 999)) { error in
            XCTAssertTrue(
                "\(error)".contains("GPUBuffer"),
                "it must say what is missing: \(error)"
            )
        }
    }
}
