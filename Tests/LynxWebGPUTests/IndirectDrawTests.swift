import XCTest
import Metal
import LynxWebGPUCore
@testable import LynxWebGPU

/// Indirect draw and dispatch — verified **by equivalence.**
///
/// The draw arguments live in a GPU buffer, so the command stream alone cannot tell what will be
/// drawn. So we **compare whole frames** against the direct call that means the same thing.
/// Misreading the slot order of the argument struct is the biggest source of bugs here, and a single
/// slot out of place changes the picture, so it shows up immediately.
///
/// The baseline frame also carries pixel assertions — two paths that **both draw nothing** would pass
/// equivalence otherwise.
final class IndirectDrawTests: XCTestCase {
    private var harness: RenderHarness!

    override func setUpWithError() throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "no Metal device")
        harness = try XCTUnwrap(RenderHarness.make())
        // Indirect arguments need Apple family 3 or above — the iOS simulator is family 2 and drops out.
        try XCTSkipUnless(harness.supports(.indirectArguments), "device without indirect argument support")
    }

    override func tearDown() {
        harness = nil
        super.tearDown()
    }

    /// 6 vertices = two triangles each covering half the screen. The instance picks the back triangle and changes the color.
    ///
    /// - a shifted `vertexCount` breaks the triangle,
    /// - a shifted `firstVertex` paints **the other half**, and
    /// - a shifted `instanceCount` yields only one color.
    ///
    /// The three slots change the picture in different ways, so a wrong slot order is always visible.
    private static let shader = """
    struct Out {
        @builtin(position) position: vec4f,
        @location(0) color: vec3f,
    };

    @vertex
    fn vs_main(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> Out {
        var corners = array<vec2f, 6>(
            vec2f(-1.0, -1.0), vec2f( 1.0, -1.0), vec2f(-1.0,  1.0),   // the lower-left half
            vec2f( 1.0,  1.0), vec2f(-1.0,  1.0), vec2f( 1.0, -1.0),   // the upper-right half
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

    /// One point in the lower-left half and one in the upper-right (split along the anti-diagonal).
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

    /// A `u32` array as the base64 carried in a command.
    private func arguments(_ values: [UInt32]) -> String {
        values.withUnsafeBufferPointer { Data(buffer: $0).base64EncodedString() }
    }

    // MARK: - drawIndirect

    func test_drawIndirectProducesTheSameFrameAsADirectDraw() throws {
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
        // First pin down that the baseline frame really drew something — two empty frames always match.
        try harness.assertPixel(x: lowerLeft.x, y: lowerLeft.y, equals: red, "instance 0")
        try harness.assertPixel(x: upperRight.x, y: upperRight.y, equals: green, "instance 1")
        let direct = try harness.frameBytes()

        harness.executeExpectingSuccess(acquireDrawable + [
            beginPass,
            ["op": "setPipeline", "pipeline": 2],
            ["op": "drawIndirect", "indirectBuffer": 3],
            ["op": "endPass"],
        ])

        try harness.assertFrameEquals(direct, "the four indirect slots must mean the same as draw's four arguments")
    }

    func test_drawIndirectFirstVertexShiftsTheVertexStart() throws {
        harness.executeExpectingSuccess(setUpPipeline() + [
            // firstVertex 3 — draws only the back triangle (the upper-right half).
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
        try harness.assertPixel(x: upperRight.x, y: upperRight.y, equals: red, "the back triangle")
        try harness.assertPixel(x: lowerLeft.x, y: lowerLeft.y, equals: clearBlue, "the front triangle is not drawn")
        let direct = try harness.frameBytes()

        harness.executeExpectingSuccess(acquireDrawable + [
            beginPass,
            ["op": "setPipeline", "pipeline": 2],
            ["op": "drawIndirect", "indirectBuffer": 3],
            ["op": "endPass"],
        ])

        try harness.assertFrameEquals(direct, "the third slot must be firstVertex")
    }

    // MARK: - drawIndexedIndirect

    /// Regression — the direct path (`drawIndexed`) **adds** `firstIndex × stride` to the index buffer
    /// offset, while on the indirect path `firstIndex` lives in the argument buffer and the GPU applies
    /// it separately. Mixing the two counts twice and quietly yields a different triangle.
    /// `setIndexBuffer(offset:)` must be reflected too.
    func test_drawIndexedIndirectAppliesBothTheIndexBufferOffsetAndFirstIndex() throws {
        // The first 2 slots are bait — they exist to check that setIndexBuffer(offset: 4) really skips them.
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
        try harness.assertPixel(x: upperRight.x, y: upperRight.y, equals: red, "indices 3, 4, 5")
        try harness.assertPixel(x: lowerLeft.x, y: lowerLeft.y, equals: clearBlue, "indices 0, 1, 2 are unused")
        let direct = try harness.frameBytes()

        harness.executeExpectingSuccess(acquireDrawable + [
            beginPass,
            ["op": "setPipeline", "pipeline": 2],
            ["op": "setIndexBuffer", "buffer": 3, "format": "uint16", "offset": 4],
            ["op": "drawIndexedIndirect", "indirectBuffer": 4],
            ["op": "endPass"],
        ])

        try harness.assertFrameEquals(direct, "the binding offset and the argument firstIndex must each apply exactly once")
    }

    // MARK: - dispatchWorkgroupsIndirect

    private static let computeShader = """
    @group(0) @binding(0) var<storage, read_write> out: array<u32>;

    @compute @workgroup_size(2)
    fn mark(@builtin(global_invocation_id) id: vec3u) {
        out[id.x] = id.x + 1u;
    }
    """

    func test_dispatchWorkgroupsIndirectWritesTheSameResultAsADirectDispatch() throws {
        // 3 workgroups × size 2 = 6 slots filled, the rest left at 0.
        let expected: [UInt32] = [1, 2, 3, 4, 5, 6, 0, 0]

        harness.executeExpectingSuccess([
            ["op": "createShaderModule", "id": 1, "code": Self.computeShader],
            ["op": "createComputePipeline", "id": 2, "layout": "auto",
             "compute": ["module": 1, "entryPoint": "mark"]],
            ["op": "getBindGroupLayout", "id": 3, "pipeline": 2, "index": 0],
            ["op": "createBuffer", "id": 4, "size": 32,
             "usage": TestUsage.storage | TestUsage.copySrc],
            ["op": "createBuffer", "id": 5, "size": 32,
             "usage": TestUsage.storage | TestUsage.copySrc],
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
        XCTAssertEqual(direct, expected, "the direct dispatch must fill 6 slots")
        XCTAssertEqual(indirect, direct, "the indirect dispatch must run the same number of workgroups")
    }

    // MARK: - GPU-driven

    /// The reason this feature exists — **drawing in the same batch with arguments compute produced.**
    ///
    /// It also confirms that command stream order (compute → draw) is the real execution order.
    /// Reversed, the draw would read arguments still at 0 and draw nothing.
    func test_drawsInTheSameBatchWithArgumentsComputeWrote() throws {
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
            // A buffer compute writes (STORAGE) and the command processor reads (INDIRECT). It starts all zero.
            ["op": "createBuffer", "id": 13, "size": 16,
             "usage": TestUsage.storage | TestUsage.indirect | TestUsage.copySrc],
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

        try harness.assertPixel(x: lowerLeft.x, y: lowerLeft.y, equals: red, "instance 0 as compute decided")
        try harness.assertPixel(x: upperRight.x, y: upperRight.y, equals: green, "instance 1 as compute decided")
        // Read the argument buffer back to rule out the picture matching by chance.
        XCTAssertEqual(
            try harness.readBufferSync(handle: 13, as: UInt32.self, size: 16), [3, 2, 0, 0],
            "whether compute really wrote the arguments"
        )
    }
}
