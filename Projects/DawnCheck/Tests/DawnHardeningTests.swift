import XCTest
import CoreGraphics
import LynxWebGPUCore

/// Crash hardening — whether "bad arguments never kill the process" (`WGPUError`) holds on the Dawn
/// runtime too. JS can send any integer, so a negative, an enormous value or a NaN must become
/// **a validation error rather than a trap (process death)** when converted to a GPU argument width.
final class DawnHardeningTests: XCTestCase {

    func test_hostileArgumentsAreRejectedAsValidationWithoutCrashing() throws {
        let runtime = try DawnWebGPURuntime()
        defer { runtime.reset() }
        try runtime.attachOffscreenCanvas(identifier: "h", size: CGSize(width: 8, height: 8))

        // Negative sizes, offsets and indices; u32/u16 overflow; missing handles — every family in one batch.
        let hostile: [[String: Any]] = [
            ["op": "createBuffer", "id": 1, "size": -16, "usage": 0x20],
            ["op": "createBuffer", "id": 2, "size": 16, "usage": 0x0040 | 0x0008],
            ["op": "writeBuffer", "buffer": 2, "data": [1, 2, 3, 4], "bufferOffset": -8],
            ["op": "createTexture", "id": 3, "size": ["width": -4, "height": 4],
             "format": "rgba8unorm", "usage": 0x10],
            ["op": "createTexture", "id": 4, "size": ["width": 4, "height": 4],
             "format": "rgba8unorm", "usage": 0x10, "mipLevelCount": -1],
            ["op": "createSampler", "id": 5, "maxAnisotropy": 99999],
            ["op": "configureCanvas", "canvas": "h", "format": "rgba8unorm"],
            ["op": "getCurrentTexture", "id": 10, "canvas": "h"],
            ["op": "createTextureView", "id": 11, "texture": 10, "baseMipLevel": -3],
            ["op": "beginRenderPass",
             "colorAttachments": [["view": 99, "loadOp": "clear", "storeOp": "store"]]],
            ["op": "draw", "vertexCount": -3],
            ["op": "setScissorRect", "x": -1, "y": 0, "width": 4, "height": 4],
            ["op": "setBindGroup", "index": -1, "bindGroup": 999],
            ["op": "dispatchWorkgroups", "x": 4_294_967_296_000],
            ["op": "copyBufferToBuffer", "source": 2, "sourceOffset": -4, "destination": 2],
            ["op": "clearBuffer", "buffer": 2, "offset": -4],
            ["op": "resolveQuerySet", "querySet": 999, "firstQuery": -1, "destination": 2],
        ]
        let result = runtime.execute(commands: hostile, present: true)
        XCTAssertEqual(result["ok"] as? Bool, false, "hostile input was reported as success")
        let errors = result["errors"] as? [[String: Any]] ?? []
        XCTAssertFalse(errors.isEmpty, "hostile input passed with no error")
        // Every one must classify as one of the four spec error kinds (not a trap or crash).
        for error in errors {
            XCTAssertNotNil(error["kind"], "an error with no kind: \(error)")
        }

        // Both the process and the device are alive — a following normal batch works unchanged.
        let sane = runtime.execute(commands: [
            ["op": "createBuffer", "id": 20, "size": 16, "usage": 0x0040],
        ], present: false)
        XCTAssertEqual(sane["ok"] as? Bool, true, "a normal batch failed after the hostile one")

        // Size attacks — a NaN or negative resize is ignored, and reading a bad canvas throws (no trap).
        runtime.resizeCanvas(identifier: "h", drawableSize: CGSize(width: CGFloat.nan, height: -5))
        XCTAssertThrowsError(try runtime.readCanvasPixels(identifier: "no-such-canvas"))
        let info = runtime.canvasInfo(identifier: "h")
        XCTAssertEqual(info["ok"] as? Bool, true, "the NaN resize corrupted the canvas state")
    }
}
