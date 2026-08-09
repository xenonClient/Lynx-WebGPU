import XCTest
@testable import LynxWebGPUCore

/// Completeness of the `WGPUCommand` decoder table — runs with no GPU.
///
/// `opName` is an exhaustive switch so a missing case is a compile error, but **a gap in the decoder
/// table (string → case) surfaces only at runtime** (leaking into `default`'s unsupported). The
/// exhaustive round trip of these fixtures closes that hole — add an op, add a fixture line
/// (`docs/COMMAND-STREAM.md` §7).
final class WGPUCommandDecodeTests: XCTestCase {

    /// op name → the smallest payload that decodes (the test fills in the op field).
    private static let fixtures: [String: [String: Any]] = [
        // Resources
        "createBuffer": ["id": 1, "size": 16, "usage": 0x0020],
        "writeBuffer": ["buffer": 1, "data": [0, 1, 2, 3]],
        "unmapBuffer": ["buffer": 1],
        "createTexture": [
            "id": 1, "size": ["width": 4, "height": 4], "format": "rgba8unorm", "usage": 0x10,
        ],
        "writeTexture": ["texture": 1, "data": [0, 0, 0, 0], "size": ["width": 1, "height": 1]],
        "copyExternalImageToTexture": [
            "source": ["source": 1], "destination": ["texture": 2],
        ],
        "createTextureView": ["id": 2, "texture": 1],
        "createSampler": ["id": 1],
        "createShaderModule": ["id": 1, "code": "@vertex fn v() {}"],
        "createBindGroupLayout": ["id": 1, "entries": [[String: Any]]()],
        "createPipelineLayout": ["id": 1, "bindGroupLayouts": [Int]()],
        "createBindGroup": ["id": 2, "layout": 1, "entries": [[String: Any]]()],
        "createQuerySet": ["id": 1, "type": "occlusion", "count": 4],
        "createRenderBundle": [
            "id": 1, "commands": [[String: Any]](), "colorFormats": ["bgra8unorm"],
        ],
        "createRenderPipeline": ["id": 2, "vertex": ["module": 1]],
        "createComputePipeline": ["id": 2, "compute": ["module": 1]],
        "getBindGroupLayout": ["id": 3, "pipeline": 2, "index": 0],
        "destroy": ["id": 1],
        // Error scopes
        "pushErrorScope": ["filter": "validation"],
        "popErrorScope": [:],
        // Canvas
        "configureCanvas": ["canvas": "main"],
        "getCurrentTexture": ["id": 1, "canvas": "main"],
        // Render pass
        "beginRenderPass": [
            "colorAttachments": [["view": 1, "loadOp": "clear", "storeOp": "store"]],
        ],
        "setPipeline": ["pipeline": 1],
        "setBindGroup": ["index": 0, "bindGroup": 1],
        "setVertexBuffer": ["slot": 0, "buffer": 1],
        "setIndexBuffer": ["buffer": 1, "format": "uint16"],
        "setViewport": ["width": 1, "height": 1],
        "setScissorRect": ["width": 1, "height": 1],
        "setBlendConstant": [:],
        "setStencilReference": [:],
        "draw": ["vertexCount": 3],
        "drawIndexed": ["indexCount": 3],
        "drawIndirect": ["indirectBuffer": 1],
        "drawIndexedIndirect": ["indirectBuffer": 1],
        "executeBundles": ["bundles": [Int]()],
        "beginOcclusionQuery": ["queryIndex": 0],
        "endOcclusionQuery": [:],
        // Compute pass
        "beginComputePass": [:],
        "dispatchWorkgroups": [:],
        "dispatchWorkgroupsIndirect": ["indirectBuffer": 1],
        "endPass": [:],
        // Copies
        "copyBufferToBuffer": ["source": 1, "destination": 2],
        "clearBuffer": ["buffer": 1],
        "copyTextureToBuffer": [
            "source": ["texture": 1], "destination": ["buffer": 2],
            "copySize": ["width": 1, "height": 1],
        ],
        "copyBufferToTexture": [
            "source": ["buffer": 1], "destination": ["texture": 2],
            "copySize": ["width": 1, "height": 1],
        ],
        "copyTextureToTexture": [
            "source": ["texture": 1], "destination": ["texture": 2],
            "copySize": ["width": 1, "height": 1],
        ],
        // Queries
        "resolveQuerySet": ["querySet": 1, "destination": 2],
        // Debug markers
        "pushDebugGroup": ["groupLabel": "section"],
        "popDebugGroup": [:],
        "insertDebugMarker": ["markerLabel": "marker"],
    ]

    func test_everyOpDecodesAndOpNameRoundTrips() throws {
        // Catches a gap in the fixtures themselves — bump this count whenever a case is added.
        XCTAssertEqual(Self.fixtures.count, 51, "the op fixture count differs from the case count")

        for (op, fields) in Self.fixtures {
            var payload = fields
            payload["op"] = op
            let decoded: WGPUCommand
            do {
                decoded = try WGPUCommand(from: WGPUValueReader(payload))
            } catch {
                XCTFail("'\(op)' failed to decode: \(error)")
                continue
            }
            XCTAssertEqual(decoded.opName, op, "'\(op)' decoded into a different case")
        }
    }

    func test_anUnknownOpIsUnsupportedWithAPath() throws {
        // Build the reader path exactly as in production — the shape execute splits with requiredObjects("commands").
        let commands = try WGPUValueReader(["commands": [["op": "teleport"]]])
            .requiredObjects("commands")
        XCTAssertThrowsError(try WGPUCommand(from: commands[0])) { error in
            guard let error = error as? WGPUError else { return XCTFail("not a WGPUError") }
            XCTAssertEqual(error.kind, .unsupported)
            XCTAssertEqual(error.path, "commands[0].op")
        }
    }

    func test_pushErrorScopeStillBuildsACaseWithABrokenFilter() throws {
        let decoded = try WGPUCommand(from: WGPUValueReader([
            "op": "pushErrorScope", "filter": "warp-core",
        ]))
        guard case .pushErrorScope(let filter, let decodeFailure) = decoded else {
            return XCTFail("not the pushErrorScope case: \(decoded.opName)")
        }
        // Placeholder plus the accompanying failure — a backend pushes first, then throws (the depth-keeping contract).
        XCTAssertNil(filter)
        XCTAssertEqual(decodeFailure?.kind, .validation)
    }

    func test_createRenderBundleCarriesTheCommandReadersUnchanged() throws {
        let decoded = try WGPUCommand(from: WGPUValueReader([
            "op": "createRenderBundle", "id": 1,
            "commands": [["op": "draw", "vertexCount": 3]],
            "colorFormats": ["bgra8unorm"],
        ]))
        guard case .createRenderBundle(let command) = decoded else {
            return XCTFail("not the createRenderBundle case")
        }
        // The replay-time decoding contract — feeding a stored reader back into this initializer yields the command.
        XCTAssertEqual(command.commands.count, 1)
        XCTAssertEqual(try WGPUCommand(from: command.commands[0]).opName, "draw")
    }
}
