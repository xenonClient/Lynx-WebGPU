import XCTest
import Metal
import LynxWebGPUCore
@testable import LynxWebGPU

/// Runs a frame from a payload shaped exactly the way the host (Lynx) hands one over.
///
/// Every other test builds its command stream from Swift literals, so none of them crosses the
/// Objective-C bridge at all — and the bridge is where the crash happened: an `NSDictionary` received
/// as `[String: Any]` has only its top level in native storage, and the nested containers stay windows
/// onto host objects. Here the whole payload is built as a mutable Objective-C tree so that path runs.
final class HostPayloadTests: XCTestCase {
    private var harness: RenderHarness!

    override func setUpWithError() throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "no Metal device")
        harness = try XCTUnwrap(RenderHarness.make())
    }

    override func tearDown() {
        harness = nil
        super.tearDown()
    }

    /// A fullscreen triangle without a vertex buffer — leaving only nested descriptors on the path.
    private static let shader = """
    @vertex
    fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
        var positions = array<vec2f, 3>(
            vec2f(-1.0, -1.0),
            vec2f( 3.0, -1.0),
            vec2f(-1.0,  3.0),
        );
        return vec4f(positions[index], 0.0, 1.0);
    }

    @fragment
    fn fs_main() -> @location(0) vec4f {
        return vec4f(0.0, 1.0, 0.0, 1.0);
    }
    """

    /// Turns a Swift literal tree into the **mutable** Objective-C tree Lynx would hand over.
    private func cocoaTree(_ value: Any) -> Any {
        if let dictionary = value as? [String: Any] {
            let result = NSMutableDictionary()
            for (key, element) in dictionary { result[key] = cocoaTree(element) }
            return result
        }
        if let array = value as? [Any] {
            return NSMutableArray(array: array.map(cocoaTree))
        }
        return value
    }

    func test_aFrameRendersFromAnObjectiveCTreePayload() throws {
        let commands: [Any] = [
            ["op": "configureCanvas", "canvas": harness.canvasId, "format": "rgba8unorm"],
            ["op": "createShaderModule", "id": 1, "code": Self.shader],
            ["op": "createRenderPipeline", "id": 2, "layout": "auto",
             "vertex": ["module": 1, "entryPoint": "vs_main"],
             "fragment": ["module": 1, "entryPoint": "fs_main",
                          "targets": [["format": "rgba8unorm"]]]],
            ["op": "getCurrentTexture", "id": 10, "canvas": harness.canvasId],
            ["op": "createTextureView", "id": 11, "texture": 10],
            ["op": "beginRenderPass", "colorAttachments": [[
                "view": 11, "loadOp": "clear", "storeOp": "store",
                "clearValue": ["r": 1.0, "g": 0.0, "b": 0.0, "a": 1.0],
            ]]],
            ["op": "setPipeline", "pipeline": 2],
            ["op": "draw", "vertexCount": 3],
            ["op": "endPass"],
        ]
        let payload = try XCTUnwrap(cocoaTree(["commands": commands, "present": true]) as? [String: Any])

        let result = harness.runtime.execute(payload)
        XCTAssertEqual(result["ok"] as? Bool, true, harness.describeErrors(result))

        // The clear color is red and the shader writes green — green means the whole pipeline ran.
        try harness.assertPixel(x: 32, y: 32, equals: (0, 255, 0, 255), "Objective-C tree payload render")
    }
}
