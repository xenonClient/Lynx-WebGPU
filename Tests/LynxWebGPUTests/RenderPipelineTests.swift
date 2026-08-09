import XCTest
import Metal
import LynxWebGPUCore
@testable import LynxWebGPU

/// Runs the whole WGSL → MSL → Metal pipeline on the GPU and verifies it **by pixel**.
final class RenderPipelineTests: XCTestCase {
    private var harness: RenderHarness!

    override func setUpWithError() throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "no Metal device")
        harness = try XCTUnwrap(RenderHarness.make())
    }

    override func tearDown() {
        harness = nil
        super.tearDown()
    }

    // MARK: - Triangle

    private static let triangleShader = """
    struct VertexOutput {
        @builtin(position) position: vec4f,
        @location(0) color: vec3f,
    };

    @vertex
    fn vs_main(@location(0) position: vec2f, @location(1) color: vec3f) -> VertexOutput {
        var out: VertexOutput;
        out.position = vec4f(position, 0.0, 1.0);
        out.color = color;
        return out;
    }

    @fragment
    fn fs_main(in: VertexOutput) -> @location(0) vec4f {
        return vec4f(in.color, 1.0);
    }
    """

    func test_theTriangleIsDrawnAndDistinctFromTheClearColor() throws {
        // Interleaved position (x, y) + color (r, g, b) — stride 20B
        let vertices: [Float] = [
            -0.5, -0.5, 1, 0, 0,
             0.5, -0.5, 1, 0, 0,
             0.0,  0.5, 1, 0, 0,
        ]

        harness.executeExpectingSuccess([
            ["op": "configureCanvas", "canvas": "test", "format": "rgba8unorm"],
            ["op": "createShaderModule", "id": 1, "code": Self.triangleShader],
            ["op": "createBuffer", "id": 2, "usage": TestUsage.vertex | TestUsage.copyDst,
             "data": vertices.base64],
            ["op": "createRenderPipeline", "id": 3, "layout": "auto",
             "vertex": [
                "module": 1, "entryPoint": "vs_main",
                "buffers": [[
                    "arrayStride": 20,
                    "attributes": [
                        ["format": "float32x2", "offset": 0, "shaderLocation": 0],
                        ["format": "float32x3", "offset": 8, "shaderLocation": 1],
                    ],
                ]],
             ],
             "fragment": [
                "module": 1, "entryPoint": "fs_main",
                "targets": [["format": "rgba8unorm"]],
             ]],
            ["op": "getCurrentTexture", "id": 10, "canvas": "test"],
            ["op": "createTextureView", "id": 11, "texture": 10],
            ["op": "beginRenderPass", "colorAttachments": [[
                "view": 11, "loadOp": "clear", "storeOp": "store",
                "clearValue": ["r": 0.0, "g": 0.0, "b": 1.0, "a": 1.0],
            ]]],
            ["op": "setPipeline", "pipeline": 3],
            ["op": "setVertexBuffer", "slot": 0, "buffer": 2],
            ["op": "draw", "vertexCount": 3],
            ["op": "endPass"],
        ])

        // The center is inside the triangle → red. The top-left corner is outside → the clear color (blue).
        try harness.assertPixel(x: 32, y: 32, equals: (255, 0, 0, 255), "inside the triangle")
        try harness.assertPixel(x: 1, y: 1, equals: (0, 0, 255, 255), "outside the triangle (the clear color)")
    }

    func test_aUniformBufferReachesTheFragmentOutput() throws {
        let shader = """
        struct Tint { color: vec4f };
        @group(0) @binding(0) var<uniform> tint: Tint;

        @vertex
        fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
            // A triangle covering the whole screen (no vertex buffer, driven by vertex_index alone)
            var positions = array<vec2f, 3>(
                vec2f(-1.0, -1.0),
                vec2f( 3.0, -1.0),
                vec2f(-1.0,  3.0),
            );
            return vec4f(positions[index], 0.0, 1.0);
        }

        @fragment
        fn fs_main() -> @location(0) vec4f {
            return tint.color;
        }
        """
        let tint: [Float] = [0.0, 1.0, 0.0, 1.0]   // green

        harness.executeExpectingSuccess([
            ["op": "configureCanvas", "canvas": "test", "format": "rgba8unorm"],
            ["op": "createShaderModule", "id": 1, "code": shader],
            ["op": "createBuffer", "id": 2, "size": 16,
             "usage": TestUsage.uniform | TestUsage.copyDst, "data": tint.base64],
            ["op": "createRenderPipeline", "id": 3, "layout": "auto",
             "vertex": ["module": 1, "entryPoint": "vs_main"],
             "fragment": ["module": 1, "entryPoint": "fs_main", "targets": [["format": "rgba8unorm"]]]],
            ["op": "getBindGroupLayout", "id": 4, "pipeline": 3, "index": 0],
            ["op": "createBindGroup", "id": 5, "layout": 4,
             "entries": [["binding": 0, "resource": ["buffer": 2]]]],
            ["op": "getCurrentTexture", "id": 10, "canvas": "test"],
            ["op": "createTextureView", "id": 11, "texture": 10],
            ["op": "beginRenderPass", "colorAttachments": [[
                "view": 11, "loadOp": "clear", "storeOp": "store",
                "clearValue": ["r": 0.0, "g": 0.0, "b": 0.0, "a": 1.0],
            ]]],
            ["op": "setPipeline", "pipeline": 3],
            ["op": "setBindGroup", "index": 0, "bindGroup": 5],
            ["op": "draw", "vertexCount": 3],
            ["op": "endPass"],
        ])

        try harness.assertPixel(x: 32, y: 32, equals: (0, 255, 0, 255), "the uniform color")
        try harness.assertPixel(x: 5, y: 60, equals: (0, 255, 0, 255), "it must cover the whole screen")
    }

    /// Checks the layout of an integer `vec3` uniform **by the value the GPU read**.
    ///
    /// WGSL `vec3<i32>` is 12 bytes; MSL `int3` is 16. Without the emitter using `packed_int3`, the
    /// following fields shift by 4 bytes and **a different value is read with no error** — transpiler
    /// tests only reach "the string matches and it compiles", so whether it really points at the same
    /// place is checked here.
    ///
    /// (`packed_int3`/`packed_uint3` exist in MSL and are 12 bytes. If the name matched but the size
    /// were 16, this test would break.)
    func test_anIntegerVec3UniformReadsAtTheWGSLOffsets() throws {
        let shader = """
        struct Counts {
            offsets: vec3<i32>,   // offset 0  (12B)
            total: i32,           // offset 12
            sizes: vec3<u32>,     // offset 16 (12B)
            stride: u32,          // offset 28
        };
        @group(0) @binding(0) var<uniform> counts: Counts;

        @vertex
        fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
            var positions = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
            return vec4f(positions[index], 0.0, 1.0);
        }

        @fragment
        fn fs_main() -> @location(0) vec4f {
            let a = counts.offsets.x + counts.offsets.y + counts.offsets.z + counts.total;
            let b = counts.sizes.x + counts.sizes.y + counts.sizes.z + counts.stride;
            let c = counts.offsets.z * 10 + i32(counts.sizes.y);
            return vec4f(f32(a) / 255.0, f32(b) / 255.0, f32(c) / 255.0, 1.0);
        }
        """
        // offsets(1,2,3) total 4 · sizes(5,6,7) stride 8 — filled at the WGSL offsets exactly.
        let values: [Int32] = [1, 2, 3, 4, 5, 6, 7, 8]
        let data = values.withUnsafeBufferPointer { Data(buffer: $0).base64EncodedString() }

        harness.executeExpectingSuccess([
            ["op": "configureCanvas", "canvas": "test", "format": "rgba8unorm"],
            ["op": "createShaderModule", "id": 1, "code": shader],
            ["op": "createBuffer", "id": 2, "size": 32,
             "usage": TestUsage.uniform | TestUsage.copyDst, "data": data],
            ["op": "createRenderPipeline", "id": 3, "layout": "auto",
             "vertex": ["module": 1, "entryPoint": "vs_main"],
             "fragment": ["module": 1, "entryPoint": "fs_main", "targets": [["format": "rgba8unorm"]]]],
            ["op": "getBindGroupLayout", "id": 4, "pipeline": 3, "index": 0],
            ["op": "createBindGroup", "id": 5, "layout": 4,
             "entries": [["binding": 0, "resource": ["buffer": 2]]]],
            ["op": "getCurrentTexture", "id": 10, "canvas": "test"],
            ["op": "createTextureView", "id": 11, "texture": 10],
            ["op": "beginRenderPass", "colorAttachments": [[
                "view": 11, "loadOp": "clear", "storeOp": "store",
                "clearValue": ["r": 0, "g": 0, "b": 0, "a": 1],
            ]]],
            ["op": "setPipeline", "pipeline": 3],
            ["op": "setBindGroup", "index": 0, "bindGroup": 5],
            ["op": "draw", "vertexCount": 3],
            ["op": "endPass"],
        ])

        // r = 1+2+3+4 = 10 · g = 5+6+7+8 = 26 · b = 3*10 + 6 = 36.
        // A shifted layout changes **all three** of these.
        try harness.assertPixel(x: 32, y: 32, equals: (10, 26, 36, 255), "integer vec3 layout")
    }

    func test_anIndexedDrawFillsTheQuad() throws {
        let vertices: [Float] = [
            -1, -1, 1, 1, 0,
             1, -1, 1, 1, 0,
             1,  1, 1, 1, 0,
            -1,  1, 1, 1, 0,
        ]
        let indices: [UInt16] = [0, 1, 2, 0, 2, 3]
        let indexData = indices.withUnsafeBufferPointer { Data(buffer: $0).base64EncodedString() }

        harness.executeExpectingSuccess([
            ["op": "configureCanvas", "canvas": "test", "format": "rgba8unorm"],
            ["op": "createShaderModule", "id": 1, "code": Self.triangleShader],
            ["op": "createBuffer", "id": 2, "usage": TestUsage.vertex, "data": vertices.base64],
            ["op": "createBuffer", "id": 3, "usage": TestUsage.index, "data": indexData],
            ["op": "createRenderPipeline", "id": 4, "layout": "auto",
             "vertex": [
                "module": 1, "entryPoint": "vs_main",
                "buffers": [[
                    "arrayStride": 20,
                    "attributes": [
                        ["format": "float32x2", "offset": 0, "shaderLocation": 0],
                        ["format": "float32x3", "offset": 8, "shaderLocation": 1],
                    ],
                ]],
             ],
             "fragment": ["module": 1, "entryPoint": "fs_main", "targets": [["format": "rgba8unorm"]]]],
            ["op": "getCurrentTexture", "id": 10, "canvas": "test"],
            ["op": "createTextureView", "id": 11, "texture": 10],
            ["op": "beginRenderPass", "colorAttachments": [[
                "view": 11, "loadOp": "clear", "storeOp": "store",
                "clearValue": ["r": 0, "g": 0, "b": 0, "a": 1],
            ]]],
            ["op": "setPipeline", "pipeline": 4],
            ["op": "setVertexBuffer", "slot": 0, "buffer": 2],
            ["op": "setIndexBuffer", "buffer": 3, "format": "uint16"],
            ["op": "drawIndexed", "indexCount": 6],
            ["op": "endPass"],
        ])

        // The whole screen must fill with yellow (1,1,0).
        try harness.assertPixel(x: 32, y: 32, equals: (255, 255, 0, 255))
        try harness.assertPixel(x: 2, y: 61, equals: (255, 255, 0, 255))
    }

    func test_alphaBlendingIsApplied() throws {
        let shader = """
        @vertex
        fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
            var positions = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
            return vec4f(positions[index], 0.0, 1.0);
        }
        @fragment
        fn fs_main() -> @location(0) vec4f {
            return vec4f(1.0, 0.0, 0.0, 0.5);
        }
        """
        harness.executeExpectingSuccess([
            ["op": "configureCanvas", "canvas": "test", "format": "rgba8unorm"],
            ["op": "createShaderModule", "id": 1, "code": shader],
            ["op": "createRenderPipeline", "id": 2, "layout": "auto",
             "vertex": ["module": 1, "entryPoint": "vs_main"],
             "fragment": ["module": 1, "entryPoint": "fs_main", "targets": [[
                "format": "rgba8unorm",
                "blend": [
                    "color": ["srcFactor": "src-alpha", "dstFactor": "one-minus-src-alpha", "operation": "add"],
                    "alpha": ["srcFactor": "one", "dstFactor": "one-minus-src-alpha", "operation": "add"],
                ],
             ]]]],
            ["op": "getCurrentTexture", "id": 10, "canvas": "test"],
            ["op": "createTextureView", "id": 11, "texture": 10],
            ["op": "beginRenderPass", "colorAttachments": [[
                "view": 11, "loadOp": "clear", "storeOp": "store",
                "clearValue": ["r": 0, "g": 0, "b": 1, "a": 1],
            ]]],
            ["op": "setPipeline", "pipeline": 2],
            ["op": "draw", "vertexCount": 3],
            ["op": "endPass"],
        ])

        // Red at 50% over blue → (0.5, 0, 0.5)
        try harness.assertPixel(x: 32, y: 32, equals: (128, 0, 128, 255), tolerance: 3)
    }

    /// Stacks two layers with **the same blend settings and colors as the demo** (the `blending` scene)
    /// and checks the pixels match what the premultiplied-alpha src-over formula produces.
    ///
    ///   result = src·a + dst·(1 − a)
    ///
    /// Judging "that blend looks off" by eye is easy to get wrong, so it is pinned numerically.
    func test_premultipliedAlphaCompositingMatchesTheFormula() throws {
        let shader = """
        struct Layer { color: vec4f };
        @group(0) @binding(0) var<uniform> layer: Layer;

        @vertex
        fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
            var corners = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
            return vec4f(corners[index], 0.0, 1.0);
        }

        @fragment
        fn fs_main() -> @location(0) vec4f {
            // As in the demo, alpha is premultiplied into RGB on output.
            return vec4f(layer.color.rgb * layer.color.a, layer.color.a);
        }
        """
        // The demo's colors and background exactly.
        let background: [Double] = [0.043, 0.055, 0.08]
        let first: [Float] = [1.0, 0.25, 0.3, 0.55]
        let second: [Float] = [0.25, 0.9, 0.45, 0.55]

        harness.executeExpectingSuccess([
            ["op": "configureCanvas", "canvas": "test", "format": "rgba8unorm"],
            ["op": "createShaderModule", "id": 1, "code": shader],
            ["op": "createBuffer", "id": 2, "size": 16,
             "usage": TestUsage.uniform | TestUsage.copyDst, "data": first.base64],
            ["op": "createBuffer", "id": 3, "size": 16,
             "usage": TestUsage.uniform | TestUsage.copyDst, "data": second.base64],
            ["op": "createRenderPipeline", "id": 4, "layout": "auto",
             "vertex": ["module": 1, "entryPoint": "vs_main"],
             "fragment": ["module": 1, "entryPoint": "fs_main", "targets": [[
                "format": "rgba8unorm",
                "blend": [
                    "color": ["srcFactor": "one", "dstFactor": "one-minus-src-alpha", "operation": "add"],
                    "alpha": ["srcFactor": "one", "dstFactor": "one-minus-src-alpha", "operation": "add"],
                ],
             ]]]],
            ["op": "getBindGroupLayout", "id": 5, "pipeline": 4, "index": 0],
            ["op": "createBindGroup", "id": 6, "layout": 5,
             "entries": [["binding": 0, "resource": ["buffer": 2]]]],
            ["op": "createBindGroup", "id": 7, "layout": 5,
             "entries": [["binding": 0, "resource": ["buffer": 3]]]],
            ["op": "getCurrentTexture", "id": 10, "canvas": "test"],
            ["op": "createTextureView", "id": 11, "texture": 10],
            ["op": "beginRenderPass", "colorAttachments": [[
                "view": 11, "loadOp": "clear", "storeOp": "store",
                "clearValue": ["r": background[0], "g": background[1], "b": background[2], "a": 1.0],
            ]]],
            ["op": "setPipeline", "pipeline": 4],
            ["op": "setBindGroup", "index": 0, "bindGroup": 6],
            ["op": "draw", "vertexCount": 3],
            // The second layer goes only over the right half — one pass shows 1-layer and 2-layer together.
            ["op": "setScissorRect", "x": 32, "y": 0, "width": 32, "height": 64],
            ["op": "setBindGroup", "index": 0, "bindGroup": 7],
            ["op": "draw", "vertexCount": 3],
            ["op": "endPass"],
        ])

        /// Computes src-over (premultiplied alpha) on the CPU.
        func over(_ source: [Float], on destination: [Double]) -> [Double] {
            let alpha = Double(source[3])
            return (0..<3).map { Double(source[$0]) * alpha + destination[$0] * (1 - alpha) }
        }
        func bytes(_ color: [Double]) -> (r: Int, g: Int, b: Int, a: Int) {
            (Int((color[0] * 255).rounded()), Int((color[1] * 255).rounded()),
             Int((color[2] * 255).rounded()), 255)
        }

        let oneLayer = over(first, on: background)
        let twoLayers = over(second, on: oneLayer)

        // One layer: red at 55% over the background → (145, 41, 51)
        try harness.assertPixel(x: 16, y: 32, equals: bytes(oneLayer), "one layer")
        // Two layers: green at 55% over that result → (100, 145, 86)
        try harness.assertPixel(x: 48, y: 32, equals: bytes(twoLayers), "two layers")

        // Drawn over an opaque background, so alpha must stay 1 (the canvas must not show through).
        XCTAssertEqual(try harness.pixel(x: 48, y: 32).a, 255)
    }

    // MARK: - Compute

    func test_aComputeShaderComputesIntoAStorageBuffer() throws {
        let shader = """
        @group(0) @binding(0) var<storage, read> input: array<f32>;
        @group(0) @binding(1) var<storage, read_write> output: array<f32>;

        @compute @workgroup_size(4)
        fn double_values(@builtin(global_invocation_id) id: vec3u) {
            output[id.x] = input[id.x] * 2.0;
        }
        """
        let input: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]

        harness.executeExpectingSuccess([
            ["op": "createShaderModule", "id": 1, "code": shader],
            ["op": "createBuffer", "id": 2, "usage": TestUsage.storage | TestUsage.copyDst,
             "data": input.base64],
            ["op": "createBuffer", "id": 3, "size": 32,
             "usage": TestUsage.storage | TestUsage.copySrc],
            ["op": "createComputePipeline", "id": 4, "layout": "auto",
             "compute": ["module": 1, "entryPoint": "double_values"]],
            ["op": "getBindGroupLayout", "id": 5, "pipeline": 4, "index": 0],
            ["op": "createBindGroup", "id": 6, "layout": 5, "entries": [
                ["binding": 0, "resource": ["buffer": 2]],
                ["binding": 1, "resource": ["buffer": 3]],
            ]],
            ["op": "beginComputePass"],
            ["op": "setPipeline", "pipeline": 4],
            ["op": "setBindGroup", "index": 0, "bindGroup": 6],
            ["op": "dispatchWorkgroups", "x": 2],
            ["op": "endPass"],
        ])

        let output = try harness.readBufferSync(handle: 3, as: Float.self, size: 32)
        XCTAssertEqual(output, [2, 4, 6, 8, 10, 12, 14, 16])
    }

    func test_arrayLengthReturnsTheBoundSize() throws {
        // A shader cannot know a buffer's size, so the runtime plugs a size table into the reserved index.
        // What is checked here: (1) a whole binding, (2) a partial binding, (3) an array at the end of a struct.
        let shader = """
        struct Particles {
            count: u32,
            items: array<vec4f>,
        };

        @group(0) @binding(0) var<storage, read> whole: array<f32>;
        @group(0) @binding(1) var<storage, read> part: array<vec4f>;
        @group(0) @binding(2) var<storage, read> particles: Particles;
        @group(0) @binding(3) var<storage, read_write> out: array<u32>;

        @compute @workgroup_size(1)
        fn probe() {
            out[0] = arrayLength(&whole);
            out[1] = arrayLength(&part);
            out[2] = arrayLength(&particles.items);
        }
        """
        harness.executeExpectingSuccess([
            ["op": "createShaderModule", "id": 1, "code": shader],
            // 10 f32 = 40 bytes → length 10
            ["op": "createBuffer", "id": 2, "size": 40, "usage": TestUsage.storage],
            // A 96-byte buffer with only 48 bytes bound → 3 vec4f
            ["op": "createBuffer", "id": 3, "size": 96, "usage": TestUsage.storage],
            // count(4) + padding(12) + 2 vec4f(32) = 48 bytes → items length 2
            ["op": "createBuffer", "id": 4, "size": 48, "usage": TestUsage.storage],
            ["op": "createBuffer", "id": 5, "size": 16,
             "usage": TestUsage.storage | TestUsage.copySrc],
            ["op": "createComputePipeline", "id": 6, "layout": "auto",
             "compute": ["module": 1, "entryPoint": "probe"]],
            ["op": "getBindGroupLayout", "id": 7, "pipeline": 6, "index": 0],
            ["op": "createBindGroup", "id": 8, "layout": 7, "entries": [
                ["binding": 0, "resource": ["buffer": 2]],
                ["binding": 1, "resource": ["buffer": 3, "size": 48]],
                ["binding": 2, "resource": ["buffer": 4]],
                ["binding": 3, "resource": ["buffer": 5]],
            ]],
            ["op": "beginComputePass"],
            ["op": "setPipeline", "pipeline": 6],
            ["op": "setBindGroup", "index": 0, "bindGroup": 8],
            ["op": "dispatchWorkgroups", "x": 1],
            ["op": "endPass"],
        ])

        let lengths = try harness.readBufferSync(handle: 5, as: UInt32.self, size: 16)
        XCTAssertEqual(Array(lengths.prefix(3)), [10, 3, 2])
    }

    // MARK: - Textures

    func test_textureSamplingWorks() throws {
        let shader = """
        @group(0) @binding(0) var tex: texture_2d<f32>;
        @group(0) @binding(1) var samp: sampler;

        struct Out {
            @builtin(position) position: vec4f,
            @location(0) uv: vec2f,
        };

        @vertex
        fn vs_main(@builtin(vertex_index) index: u32) -> Out {
            var positions = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
            var out: Out;
            out.position = vec4f(positions[index], 0.0, 1.0);
            out.uv = positions[index] * 0.5 + vec2f(0.5, 0.5);
            return out;
        }

        @fragment
        fn fs_main(in: Out) -> @location(0) vec4f {
            return textureSample(tex, samp, in.uv);
        }
        """
        // Fill the whole 2x2 texture with magenta (so the result is the same wherever it samples).
        let texels = [UInt8](repeating: 0, count: 16).enumerated().map { index, _ -> UInt8 in
            switch index % 4 {
            case 0: return 255   // R
            case 1: return 0     // G
            case 2: return 255   // B
            default: return 255  // A
            }
        }

        harness.executeExpectingSuccess([
            ["op": "configureCanvas", "canvas": "test", "format": "rgba8unorm"],
            ["op": "createShaderModule", "id": 1, "code": shader],
            ["op": "createTexture", "id": 2, "size": ["width": 2, "height": 2],
             "format": "rgba8unorm", "usage": TestUsage.textureBinding | TestUsage.copyDst],
            ["op": "writeTexture", "texture": 2, "data": Data(texels).base64EncodedString(),
             "size": ["width": 2, "height": 2], "bytesPerRow": 8],
            ["op": "createTextureView", "id": 3, "texture": 2],
            ["op": "createSampler", "id": 4, "magFilter": "linear", "minFilter": "linear"],
            ["op": "createRenderPipeline", "id": 5, "layout": "auto",
             "vertex": ["module": 1, "entryPoint": "vs_main"],
             "fragment": ["module": 1, "entryPoint": "fs_main", "targets": [["format": "rgba8unorm"]]]],
            ["op": "getBindGroupLayout", "id": 6, "pipeline": 5, "index": 0],
            ["op": "createBindGroup", "id": 7, "layout": 6, "entries": [
                ["binding": 0, "resource": ["textureView": 3]],
                ["binding": 1, "resource": ["sampler": 4]],
            ]],
            ["op": "getCurrentTexture", "id": 10, "canvas": "test"],
            ["op": "createTextureView", "id": 11, "texture": 10],
            ["op": "beginRenderPass", "colorAttachments": [[
                "view": 11, "loadOp": "clear", "storeOp": "store",
                "clearValue": ["r": 0, "g": 0, "b": 0, "a": 1],
            ]]],
            ["op": "setPipeline", "pipeline": 5],
            ["op": "setBindGroup", "index": 0, "bindGroup": 7],
            ["op": "draw", "vertexCount": 3],
            ["op": "endPass"],
        ])

        try harness.assertPixel(x: 32, y: 32, equals: (255, 0, 255, 255), "the sampled texel color")
    }

    func test_samplesAnExternalTextureWithEdgeClamping() throws {
        // `textureSampleBaseClampToEdge` pulls the coordinate half a texel inward. So even at uv 0 or 1
        // **the opposite texel does not blend in** — the mechanism that stops video frame edges bleeding.
        // Here a repeat sampler is used deliberately, creating the situation where the opposite side blends without the clamp.
        let shader = """
        @group(0) @binding(0) var frame: texture_external;
        @group(0) @binding(1) var samp: sampler;

        struct Out {
            @builtin(position) position: vec4f,
            @location(0) uv: vec2f,
        };

        @vertex
        fn vs_main(@builtin(vertex_index) index: u32) -> Out {
            var positions = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
            var out: Out;
            out.position = vec4f(positions[index], 0.0, 1.0);
            // Fill the whole screen with uv 1.0 (the bottom-right corner).
            out.uv = vec2f(1.0, 1.0);
            return out;
        }

        @fragment
        fn fs_main(in: Out) -> @location(0) vec4f {
            return textureSampleBaseClampToEdge(frame, samp, in.uv);
        }
        """
        // 2x2: (0,0) red, the rest blue. With the clamp working at uv=1.0 pure blue comes out;
        // without it, repeat wraps around and a quarter of red mixes in.
        let texels: [UInt8] = [
            255, 0, 0, 255,   0, 0, 255, 255,
            0, 0, 255, 255,   0, 0, 255, 255,
        ]

        harness.executeExpectingSuccess([
            ["op": "configureCanvas", "canvas": "test", "format": "rgba8unorm"],
            ["op": "createShaderModule", "id": 1, "code": shader],
            ["op": "createTexture", "id": 2, "size": ["width": 2, "height": 2],
             "format": "rgba8unorm", "usage": TestUsage.textureBinding | TestUsage.copyDst],
            ["op": "writeTexture", "texture": 2, "data": Data(texels).base64EncodedString(),
             "size": ["width": 2, "height": 2], "bytesPerRow": 8],
            ["op": "createTextureView", "id": 3, "texture": 2],
            ["op": "createSampler", "id": 4, "magFilter": "linear", "minFilter": "linear",
             "addressModeU": "repeat", "addressModeV": "repeat"],
            ["op": "createRenderPipeline", "id": 5, "layout": "auto",
             "vertex": ["module": 1, "entryPoint": "vs_main"],
             "fragment": ["module": 1, "entryPoint": "fs_main", "targets": [["format": "rgba8unorm"]]]],
            ["op": "getBindGroupLayout", "id": 6, "pipeline": 5, "index": 0],
            ["op": "createBindGroup", "id": 7, "layout": 6, "entries": [
                ["binding": 0, "resource": ["textureView": 3]],
                ["binding": 1, "resource": ["sampler": 4]],
            ]],
            ["op": "getCurrentTexture", "id": 10, "canvas": "test"],
            ["op": "createTextureView", "id": 11, "texture": 10],
            ["op": "beginRenderPass", "colorAttachments": [[
                "view": 11, "loadOp": "clear", "storeOp": "store",
                "clearValue": ["r": 0, "g": 0, "b": 0, "a": 1],
            ]]],
            ["op": "setPipeline", "pipeline": 5],
            ["op": "setBindGroup", "index": 0, "bindGroup": 7],
            ["op": "draw", "vertexCount": 3],
            ["op": "endPass"],
        ])

        try harness.assertPixel(x: 32, y: 32, equals: (0, 0, 255, 255), "only the edge texel must come out")
    }

    // MARK: - Depth buffer

    func test_theDepthTestHidesTheTriangleBehind() throws {
        let shader = """
        struct Uniforms { depth: f32, r: f32, g: f32, b: f32 };
        @group(0) @binding(0) var<uniform> u: Uniforms;

        @vertex
        fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
            var positions = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
            return vec4f(positions[index], u.depth, 1.0);
        }
        @fragment
        fn fs_main() -> @location(0) vec4f {
            return vec4f(u.r, u.g, u.b, 1.0);
        }
        """
        // Draw front (depth 0.2, red) then back (depth 0.8, green). The depth test must discard the back one.
        let near: [Float] = [0.2, 1, 0, 0]
        let far: [Float] = [0.8, 0, 1, 0]

        harness.executeExpectingSuccess([
            ["op": "configureCanvas", "canvas": "test", "format": "rgba8unorm"],
            ["op": "createShaderModule", "id": 1, "code": shader],
            ["op": "createTexture", "id": 2, "size": ["width": 64, "height": 64],
             "format": "depth32float", "usage": TestUsage.renderAttachment],
            ["op": "createTextureView", "id": 3, "texture": 2],
            ["op": "createBuffer", "id": 4, "size": 16, "usage": TestUsage.uniform, "data": near.base64],
            ["op": "createBuffer", "id": 5, "size": 16, "usage": TestUsage.uniform, "data": far.base64],
            ["op": "createRenderPipeline", "id": 6, "layout": "auto",
             "vertex": ["module": 1, "entryPoint": "vs_main"],
             "fragment": ["module": 1, "entryPoint": "fs_main", "targets": [["format": "rgba8unorm"]]],
             "depthStencil": ["format": "depth32float", "depthWriteEnabled": true, "depthCompare": "less"]],
            ["op": "getBindGroupLayout", "id": 7, "pipeline": 6, "index": 0],
            ["op": "createBindGroup", "id": 8, "layout": 7,
             "entries": [["binding": 0, "resource": ["buffer": 4]]]],
            ["op": "createBindGroup", "id": 9, "layout": 7,
             "entries": [["binding": 0, "resource": ["buffer": 5]]]],
            ["op": "getCurrentTexture", "id": 10, "canvas": "test"],
            ["op": "createTextureView", "id": 11, "texture": 10],
            ["op": "beginRenderPass",
             "colorAttachments": [[
                "view": 11, "loadOp": "clear", "storeOp": "store",
                "clearValue": ["r": 0, "g": 0, "b": 0, "a": 1],
             ]],
             "depthStencilAttachment": [
                "view": 3, "depthClearValue": 1.0, "depthLoadOp": "clear", "depthStoreOp": "store",
             ]],
            ["op": "setPipeline", "pipeline": 6],
            ["op": "setBindGroup", "index": 0, "bindGroup": 8],
            ["op": "draw", "vertexCount": 3],
            ["op": "setBindGroup", "index": 0, "bindGroup": 9],
            ["op": "draw", "vertexCount": 3],
            ["op": "endPass"],
        ])

        try harness.assertPixel(x: 32, y: 32, equals: (255, 0, 0, 255), "the front one (red) must remain")
    }

    // MARK: - HDR readback

    func test_anRGBA16FloatSurfaceDoesNotLoseValuesOutsideSDR() throws {
        // Inside the triangle is what the fragment wrote, outside is the clear value — both go past 1.0.
        // On an 8-bit surface everything here would clip to 1.0.
        let shader = """
        @vertex
        fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
            var positions = array<vec2f, 3>(vec2f(-0.5, -0.5), vec2f(0.5, -0.5), vec2f(0.0, 0.5));
            return vec4f(positions[index], 0.0, 1.0);
        }
        @fragment
        fn fs_main() -> @location(0) vec4f {
            return vec4f(2.5, 0.5, -0.25, 1.0);
        }
        """

        harness.executeExpectingSuccess([
            ["op": "configureCanvas", "canvas": "test", "format": "rgba16float"],
            ["op": "createShaderModule", "id": 1, "code": shader],
            ["op": "createRenderPipeline", "id": 2, "layout": "auto",
             "vertex": ["module": 1, "entryPoint": "vs_main"],
             "fragment": ["module": 1, "entryPoint": "fs_main", "targets": [["format": "rgba16float"]]]],
            ["op": "getCurrentTexture", "id": 10, "canvas": "test"],
            ["op": "createTextureView", "id": 11, "texture": 10],
            ["op": "beginRenderPass", "colorAttachments": [[
                "view": 11, "loadOp": "clear", "storeOp": "store",
                "clearValue": ["r": 4.0, "g": 0.0, "b": 0.0, "a": 1.0],
            ]]],
            ["op": "setPipeline", "pipeline": 2],
            ["op": "draw", "vertexCount": 3],
            ["op": "endPass"],
        ])

        let readback = try harness.readback()
        XCTAssertEqual(readback.format, .rgba16float)
        XCTAssertEqual(readback.bytesPerRow, 64 * 8, "it must be 8 bytes per pixel")
        XCTAssertEqual(readback.data.count, 64 * 64 * 8)

        try harness.assertPixelFloat(
            x: 32, y: 32, equals: SIMD4<Float>(2.5, 0.5, -0.25, 1),
            "inside the triangle — values above 1.0 and negatives must survive"
        )
        try harness.assertPixelFloat(
            x: 1, y: 1, equals: SIMD4<Float>(4, 0, 0, 1), "the clear value must not clip either"
        )
    }

    // MARK: - Global shadowing (whether scope resolution is right down to the value)

    func test_theScopeOfALocalShadowingAGlobalIsCorrectAtRuntime() throws {
        // base before the declaration is the global (green); after it, the local (red). A mistake in scope
        // resolution or rename substitution yields red — something a successful compile cannot catch.
        let shader = """
        var<private> base : vec4f;

        fn shade() -> vec4f {
            let inherited = base;
            var base : vec4f = vec4f(1.0, 0.0, 0.0, 1.0);
            return inherited + base * 0.0;
        }

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
            base = vec4f(0.0, 1.0, 0.0, 1.0);
            return shade();
        }
        """

        harness.executeExpectingSuccess([
            ["op": "configureCanvas", "canvas": "test", "format": "rgba8unorm"],
            ["op": "createShaderModule", "id": 1, "code": shader],
            ["op": "createRenderPipeline", "id": 2, "layout": "auto",
             "vertex": ["module": 1, "entryPoint": "vs_main"],
             "fragment": ["module": 1, "entryPoint": "fs_main", "targets": [["format": "rgba8unorm"]]]],
            ["op": "getCurrentTexture", "id": 10, "canvas": "test"],
            ["op": "createTextureView", "id": 11, "texture": 10],
            ["op": "beginRenderPass", "colorAttachments": [[
                "view": 11, "loadOp": "clear", "storeOp": "store",
                "clearValue": ["r": 0.0, "g": 0.0, "b": 0.0, "a": 1.0],
            ]]],
            ["op": "setPipeline", "pipeline": 2],
            ["op": "draw", "vertexCount": 3],
            ["op": "endPass"],
        ])

        try harness.assertPixel(x: 32, y: 32, equals: (0, 255, 0, 255), "a reference before the declaration must see the global")
    }
}
