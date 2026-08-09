import XCTest
import Metal
import LynxWebGPUCore
@testable import LynxWebGPU

/// Whether the safety transformations work **on a real GPU** — checked by value, not by the emitted MSL string.
///
/// String assertions (`ShaderSafetyTests`) only reach "it is emitted that way". Here we read buffer
/// values to see whether an out-of-range index really is clamped and a discarded fragment's write really is blocked.
final class ShaderSafetyRenderTests: XCTestCase {
    private var harness: RenderHarness!

    override func setUpWithError() throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "no Metal device")
        harness = try XCTUnwrap(RenderHarness.make())
    }

    override func tearDown() {
        harness = nil
        super.tearDown()
    }

    /// A **read** far out of range is clamped to the last element — it does not read adjacent memory.
    func test_anOutOfRangeReadIsClampedToTheLastElement() throws {
        let source: [Float] = [10, 20, 30, 40]
        harness.executeExpectingSuccess([
            ["op": "createShaderModule", "id": 1, "code": """
            @group(0) @binding(0) var<storage, read> data: array<f32, 4>;
            @group(0) @binding(1) var<storage, read_write> out: array<f32>;
            @group(0) @binding(2) var<uniform> idx: u32;
            @compute @workgroup_size(1) fn cs() { out[0] = data[idx]; }
            """],
            ["op": "createBuffer", "id": 2, "usage": TestUsage.storage, "data": source.base64],
            ["op": "createBuffer", "id": 3, "size": 4,
             "usage": TestUsage.storage | TestUsage.copySrc],
            // Index 100 into a 4-element array.
            ["op": "createBuffer", "id": 4, "usage": TestUsage.uniform,
             "data": [UInt32(100)].withUnsafeBufferPointer { Data(buffer: $0).base64EncodedString() }],
            ["op": "createBuffer", "id": 5, "size": 4,
             "usage": TestUsage.copyDst | TestUsage.mapRead],
            ["op": "createComputePipeline", "id": 6, "layout": "auto",
             "compute": ["module": 1, "entryPoint": "cs"]],
            ["op": "getBindGroupLayout", "id": 7, "pipeline": 6, "index": 0],
            ["op": "createBindGroup", "id": 8, "layout": 7, "entries": [
                ["binding": 0, "resource": ["buffer": 2]],
                ["binding": 1, "resource": ["buffer": 3]],
                ["binding": 2, "resource": ["buffer": 4]],
            ]],
            ["op": "beginComputePass"],
            ["op": "setPipeline", "pipeline": 6],
            ["op": "setBindGroup", "index": 0, "bindGroup": 8],
            ["op": "dispatchWorkgroups", "x": 1],
            ["op": "endPass"],
            ["op": "copyBufferToBuffer", "source": 3, "destination": 5, "size": 4],
        ])
        let result = try harness.readBufferSync(handle: 5, as: Float.self)
        XCTAssertEqual(result.first, 40, "an out-of-range index must clamp to the last element")
    }

    /// A **write** out of range is clamped inside the buffer — unclamped it would overwrite someone else's memory.
    func test_anOutOfRangeWriteIsClampedInsideTheBuffer() throws {
        harness.executeExpectingSuccess([
            ["op": "createShaderModule", "id": 1, "code": """
            @group(0) @binding(0) var<storage, read_write> out: array<f32>;
            @group(0) @binding(1) var<uniform> idx: u32;
            @compute @workgroup_size(1) fn cs() { out[idx] = 9.0; }
            """],
            ["op": "createBuffer", "id": 2, "size": 16,
             "usage": TestUsage.storage | TestUsage.copySrc],
            ["op": "createBuffer", "id": 3, "usage": TestUsage.uniform,
             "data": [UInt32(1_000_000)].withUnsafeBufferPointer { Data(buffer: $0).base64EncodedString() }],
            ["op": "createBuffer", "id": 4, "size": 16,
             "usage": TestUsage.copyDst | TestUsage.mapRead],
            ["op": "createComputePipeline", "id": 5, "layout": "auto",
             "compute": ["module": 1, "entryPoint": "cs"]],
            ["op": "getBindGroupLayout", "id": 6, "pipeline": 5, "index": 0],
            ["op": "createBindGroup", "id": 7, "layout": 6, "entries": [
                ["binding": 0, "resource": ["buffer": 2]],
                ["binding": 1, "resource": ["buffer": 3]],
            ]],
            ["op": "beginComputePass"],
            ["op": "setPipeline", "pipeline": 5],
            ["op": "setBindGroup", "index": 0, "bindGroup": 7],
            ["op": "dispatchWorkgroups", "x": 1],
            ["op": "endPass"],
            ["op": "copyBufferToBuffer", "source": 2, "destination": 4, "size": 16],
        ])
        let result = try harness.readBufferSync(handle: 4, as: Float.self)
        XCTAssertEqual(result, [0, 0, 0, 9], "the millionth write must clamp to the buffer's last slot")
    }

    /// `var<workgroup>` starts at zero — reading without writing must give 0.
    func test_workgroupMemoryStartsAtZero() throws {
        harness.executeExpectingSuccess([
            ["op": "createShaderModule", "id": 1, "code": """
            var<workgroup> tile: array<f32, 4>;
            @group(0) @binding(0) var<storage, read_write> out: array<f32>;
            @compute @workgroup_size(4)
            fn cs(@builtin(local_invocation_id) lid: vec3u) {
                out[lid.x] = tile[lid.x];
            }
            """],
            ["op": "createBuffer", "id": 2, "size": 16,
             "usage": TestUsage.storage | TestUsage.copySrc],
            ["op": "createBuffer", "id": 3, "size": 16,
             "usage": TestUsage.copyDst | TestUsage.mapRead],
            ["op": "createComputePipeline", "id": 4, "layout": "auto",
             "compute": ["module": 1, "entryPoint": "cs"]],
            ["op": "getBindGroupLayout", "id": 5, "pipeline": 4, "index": 0],
            ["op": "createBindGroup", "id": 6, "layout": 5,
             "entries": [["binding": 0, "resource": ["buffer": 2]]]],
            ["op": "beginComputePass"],
            ["op": "setPipeline", "pipeline": 4],
            ["op": "setBindGroup", "index": 0, "bindGroup": 6],
            ["op": "dispatchWorkgroups", "x": 1],
            ["op": "endPass"],
            ["op": "copyBufferToBuffer", "source": 2, "destination": 3, "size": 16],
        ])
        let result = try harness.readBufferSync(handle: 3, as: Float.self)
        XCTAssertEqual(result, [0, 0, 0, 0], "read uninitialized threadgroup memory")
    }

    /// **A discarded fragment does not corrupt storage.**
    ///
    /// MSL's `discard_fragment()` is not an immediate return and the code after it keeps running — unblocked,
    /// this test's buffer would fill with 1.0.
    func test_aDiscardedFragmentDoesNotWriteToStorage() throws {
        harness.executeExpectingSuccess([
            ["op": "configureCanvas", "canvas": "test", "format": "rgba8unorm"],
            ["op": "createShaderModule", "id": 1, "code": """
            @group(0) @binding(0) var<storage, read_write> out: array<f32>;
            @vertex fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
                var positions = array<vec2f, 3>(
                    vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0),
                );
                return vec4f(positions[index], 0.0, 1.0);
            }
            @fragment fn fs() -> @location(0) vec4f {
                discard;
                out[0] = 1.0;
                return vec4f(1.0, 0.0, 0.0, 1.0);
            }
            """],
            ["op": "createBuffer", "id": 2, "size": 4,
             "usage": TestUsage.storage | TestUsage.copySrc],
            ["op": "createBuffer", "id": 3, "size": 4,
             "usage": TestUsage.copyDst | TestUsage.mapRead],
            ["op": "createRenderPipeline", "id": 4, "layout": "auto",
             "vertex": ["module": 1, "entryPoint": "vs"],
             "fragment": ["module": 1, "entryPoint": "fs",
                          "targets": [["format": "rgba8unorm"]]]],
            ["op": "getBindGroupLayout", "id": 5, "pipeline": 4, "index": 0],
            ["op": "createBindGroup", "id": 6, "layout": 5,
             "entries": [["binding": 0, "resource": ["buffer": 2]]]],
            ["op": "getCurrentTexture", "id": 10, "canvas": "test"],
            ["op": "createTextureView", "id": 11, "texture": 10],
            ["op": "beginRenderPass", "colorAttachments": [[
                "view": 11, "loadOp": "clear", "storeOp": "store",
                "clearValue": ["r": 0, "g": 0, "b": 1, "a": 1],
            ]]],
            ["op": "setPipeline", "pipeline": 4],
            ["op": "setBindGroup", "index": 0, "bindGroup": 6],
            ["op": "draw", "vertexCount": 3],
            ["op": "endPass"],
            ["op": "copyBufferToBuffer", "source": 2, "destination": 3, "size": 4],
        ])
        let result = try harness.readBufferSync(handle: 3, as: Float.self)
        XCTAssertEqual(result.first, 0, "a discarded fragment wrote into the storage buffer")
        // The discard itself must still happen — only the clear color remains on screen.
        try harness.assertPixel(x: 32, y: 32, equals: (0, 0, 255, 255), "a discarded fragment was drawn")
    }

    /// Division by zero gives the value WGSL specifies **without killing the process** (`x / 0 == x`, `x % 0 == 0`).
    func test_divisionByZeroBehavesAsSpecified() throws {
        harness.executeExpectingSuccess([
            ["op": "createShaderModule", "id": 1, "code": """
            @group(0) @binding(0) var<storage, read_write> out: array<i32>;
            @group(0) @binding(1) var<uniform> zero: i32;
            @compute @workgroup_size(1) fn cs() {
                out[0] = 100 / zero;
                out[1] = 100 % zero;
            }
            """],
            ["op": "createBuffer", "id": 2, "size": 8,
             "usage": TestUsage.storage | TestUsage.copySrc],
            ["op": "createBuffer", "id": 3, "usage": TestUsage.uniform,
             "data": [Int32(0)].withUnsafeBufferPointer { Data(buffer: $0).base64EncodedString() }],
            ["op": "createBuffer", "id": 4, "size": 8,
             "usage": TestUsage.copyDst | TestUsage.mapRead],
            ["op": "createComputePipeline", "id": 5, "layout": "auto",
             "compute": ["module": 1, "entryPoint": "cs"]],
            ["op": "getBindGroupLayout", "id": 6, "pipeline": 5, "index": 0],
            ["op": "createBindGroup", "id": 7, "layout": 6, "entries": [
                ["binding": 0, "resource": ["buffer": 2]],
                ["binding": 1, "resource": ["buffer": 3]],
            ]],
            ["op": "beginComputePass"],
            ["op": "setPipeline", "pipeline": 5],
            ["op": "setBindGroup", "index": 0, "bindGroup": 7],
            ["op": "dispatchWorkgroups", "x": 1],
            ["op": "endPass"],
            ["op": "copyBufferToBuffer", "source": 2, "destination": 4, "size": 8],
        ])
        let result = try harness.readBufferSync(handle: 4, as: Int32.self)
        XCTAssertEqual(result, [100, 0], "WGSL defines x / 0 == x and x % 0 == 0")
    }
}
