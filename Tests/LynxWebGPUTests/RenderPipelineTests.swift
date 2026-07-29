import XCTest
import Metal
import LynxWebGPUCore
@testable import LynxWebGPU

/// WGSL → MSL → Metal 파이프라인 전체를 GPU에서 돌려 **픽셀로** 검증한다.
final class RenderPipelineTests: XCTestCase {
    private var harness: RenderHarness!

    override func setUpWithError() throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "Metal 디바이스 없음")
        harness = try XCTUnwrap(RenderHarness.make())
    }

    override func tearDown() {
        harness = nil
        super.tearDown()
    }

    // MARK: - 삼각형

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

    func test_삼각형이_그려지고_클리어색과_구분된다() throws {
        // 위치(x, y) + 색(r, g, b) 인터리브 — stride 20B
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

        // 중앙은 삼각형 안 → 빨강. 좌상단 구석은 밖 → 클리어색(파랑).
        try harness.assertPixel(x: 32, y: 32, equals: (255, 0, 0, 255), "삼각형 내부")
        try harness.assertPixel(x: 1, y: 1, equals: (0, 0, 255, 255), "삼각형 외부(클리어색)")
    }

    func test_유니폼버퍼가_프래그먼트_출력에_반영된다() throws {
        let shader = """
        struct Tint { color: vec4f };
        @group(0) @binding(0) var<uniform> tint: Tint;

        @vertex
        fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
            // 화면 전체를 덮는 삼각형 (정점 버퍼 없이 vertex_index만으로)
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
        let tint: [Float] = [0.0, 1.0, 0.0, 1.0]   // 초록

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

        try harness.assertPixel(x: 32, y: 32, equals: (0, 255, 0, 255), "유니폼 색")
        try harness.assertPixel(x: 5, y: 60, equals: (0, 255, 0, 255), "전체 화면을 덮어야 한다")
    }

    func test_인덱스드로우가_사각형을_채운다() throws {
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

        // 노랑(1,1,0)으로 화면 전체가 채워져야 한다.
        try harness.assertPixel(x: 32, y: 32, equals: (255, 255, 0, 255))
        try harness.assertPixel(x: 2, y: 61, equals: (255, 255, 0, 255))
    }

    func test_알파블렌딩이_적용된다() throws {
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

        // 빨강 50% over 파랑 → (0.5, 0, 0.5)
        try harness.assertPixel(x: 32, y: 32, equals: (128, 0, 128, 255), tolerance: 3)
    }

    // MARK: - 컴퓨트

    func test_컴퓨트셰이더가_스토리지버퍼를_계산한다() throws {
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
             "usage": TestUsage.storage | TestUsage.copySrc | TestUsage.mapRead],
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

        let expectation = expectation(description: "readBuffer")
        var output: [Float] = []
        harness.context.readBuffer(handle: 3, offset: 0, size: 32) { result in
            if let base64 = result["data"] as? String, let data = Data(base64Encoded: base64) {
                output = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            }
            expectation.fulfill()
        }
        wait(for: [expectation], timeout: 10)

        XCTAssertEqual(output, [2, 4, 6, 8, 10, 12, 14, 16])
    }

    // MARK: - 텍스처

    func test_텍스처_샘플링이_동작한다() throws {
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
        // 2x2 텍스처를 전부 자홍색으로 채운다 (샘플 위치와 무관하게 결과가 같도록).
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

        try harness.assertPixel(x: 32, y: 32, equals: (255, 0, 255, 255), "샘플링한 텍셀 색")
    }

    // MARK: - 깊이 버퍼

    func test_깊이테스트가_뒤쪽_삼각형을_가린다() throws {
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
        // 앞(깊이 0.2, 빨강) → 뒤(깊이 0.8, 초록) 순서로 그린다. 깊이 테스트가 뒤를 버려야 한다.
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

        try harness.assertPixel(x: 32, y: 32, equals: (255, 0, 0, 255), "앞쪽(빨강)이 남아야 한다")
    }
}
