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

    /// 정수 `vec3` 유니폼의 배치를 **GPU가 읽은 값으로** 확인한다.
    ///
    /// WGSL `vec3<i32>`는 12바이트, MSL `int3`는 16바이트다. 방출기가 `packed_int3`를 쓰지
    /// 않으면 뒤 필드가 4바이트씩 밀려 **오류 없이 다른 값**이 읽힌다 — 트랜스파일러 테스트는
    /// "문자열이 맞고 컴파일된다"까지만 보므로, 실제로 같은 자리를 가리키는지는 여기서 본다.
    ///
    /// (`packed_int3`/`packed_uint3`는 MSL에 있는 타입이고 12바이트다. 이름만 맞고 크기가
    /// 16이면 이 테스트가 깨진다.)
    func test_정수_vec3_유니폼이_WGSL_오프셋대로_읽힌다() throws {
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
        // offsets(1,2,3) total 4 · sizes(5,6,7) stride 8 — WGSL 오프셋 그대로 채운다.
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
        // 배치가 밀리면 이 셋이 **전부** 달라진다.
        try harness.assertPixel(x: 32, y: 32, equals: (10, 26, 36, 255), "정수 vec3 배치")
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

    /// 데모(`blending` 씬)와 **같은 블렌드 설정·같은 색**으로 두 겹을 쌓고,
    /// 미리 곱해진 알파 src-over 공식이 내는 값과 픽셀이 일치하는지 본다.
    ///
    ///   result = src·a + dst·(1 − a)
    ///
    /// 겹친 색이 "이상해 보인다"는 판단은 눈으로 하면 틀리기 쉬우므로 수치로 고정한다.
    func test_미리곱해진알파_합성이_공식과_일치한다() throws {
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
            // 데모와 같이 RGB에 알파를 미리 곱해 내보낸다.
            return vec4f(layer.color.rgb * layer.color.a, layer.color.a);
        }
        """
        // 데모의 색·배경 그대로.
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
            // 오른쪽 절반에만 둘째 겹을 올린다 — 한 패스에서 1겹/2겹을 같이 본다.
            ["op": "setScissorRect", "x": 32, "y": 0, "width": 32, "height": 64],
            ["op": "setBindGroup", "index": 0, "bindGroup": 7],
            ["op": "draw", "vertexCount": 3],
            ["op": "endPass"],
        ])

        /// src-over(미리 곱해진 알파)를 CPU에서 계산한다.
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

        // 한 겹: 빨강 55% over 배경 → (145, 41, 51)
        try harness.assertPixel(x: 16, y: 32, equals: bytes(oneLayer), "한 겹")
        // 두 겹: 초록 55% over 그 결과 → (100, 145, 86)
        try harness.assertPixel(x: 48, y: 32, equals: bytes(twoLayers), "두 겹")

        // 불투명 배경 위에 그렸으므로 알파는 1로 남아야 한다 (캔버스가 비쳐 보이면 안 된다).
        XCTAssertEqual(try harness.pixel(x: 48, y: 32).a, 255)
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

    func test_arrayLength가_바인딩된_크기를_돌려준다() throws {
        // 셰이더는 버퍼 크기를 알 수 없으므로 런타임이 예약 인덱스에 크기 표를 꽂아 준다.
        // 여기서 보는 것: (1) 전체 바인딩, (2) 일부만 바인딩, (3) 구조체 말미 배열.
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
            // 10개 f32 = 40바이트 → 길이 10
            ["op": "createBuffer", "id": 2, "size": 40, "usage": TestUsage.storage],
            // 96바이트 버퍼지만 48바이트만 바인딩 → vec4f 3개
            ["op": "createBuffer", "id": 3, "size": 96, "usage": TestUsage.storage],
            // count(4) + 패딩(12) + vec4f 2개(32) = 48바이트 → items 길이 2
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

    func test_외부텍스처를_가장자리_클램프로_샘플링한다() throws {
        // `textureSampleBaseClampToEdge`는 좌표를 텍셀 절반만큼 안쪽으로 물린다. 그래서 uv가
        // 0이나 1로 가도 **반대쪽 텍셀이 섞이지 않는다** — 비디오 프레임 경계가 번지는 것을 막는 장치다.
        // 여기서는 일부러 repeat 샘플러를 써서, 클램프가 없으면 반대쪽이 섞이는 상황을 만든다.
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
            // 화면 전체를 uv 1.0(오른쪽 아래 끝)으로 채운다.
            out.uv = vec2f(1.0, 1.0);
            return out;
        }

        @fragment
        fn fs_main(in: Out) -> @location(0) vec4f {
            return textureSampleBaseClampToEdge(frame, samp, in.uv);
        }
        """
        // 2x2: (0,0) 빨강, 나머지는 파랑. uv=1.0에서 클램프가 동작하면 순수 파랑이 나오고,
        // 없으면 repeat로 감싸며 빨강이 1/4 섞인다.
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

        try harness.assertPixel(x: 32, y: 32, equals: (0, 0, 255, 255), "가장자리 텍셀만 나와야 한다")
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

    // MARK: - HDR 되읽기

    func test_rgba16float_표면은_SDR범위_밖의_값을_잃지_않는다() throws {
        // 삼각형 안쪽은 프래그먼트가 쓴 값, 바깥쪽은 클리어 값 — 둘 다 1.0을 넘겨서 확인한다.
        // 8비트 표면이라면 여기서 전부 1.0으로 잘려 나간다.
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
        XCTAssertEqual(readback.bytesPerRow, 64 * 8, "픽셀당 8바이트여야 한다")
        XCTAssertEqual(readback.data.count, 64 * 64 * 8)

        try harness.assertPixelFloat(
            x: 32, y: 32, equals: SIMD4<Float>(2.5, 0.5, -0.25, 1),
            "삼각형 내부 — 1.0 초과와 음수가 그대로 살아야 한다"
        )
        try harness.assertPixelFloat(
            x: 1, y: 1, equals: SIMD4<Float>(4, 0, 0, 1), "클리어 값도 잘리지 않아야 한다"
        )
    }

    // MARK: - 전역 섀도잉 (스코프 해석이 값까지 옳은지)

    func test_전역을_가린_지역선언의_스코프가_런타임_값으로_옳다() throws {
        // 선언 앞의 base는 전역(초록), 뒤의 base는 지역(빨강)이다. 스코프 해석이나
        // 리네임 참조 치환이 틀리면 빨강이 나온다 — 컴파일 성공만으로는 못 잡는 부분이다.
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

        try harness.assertPixel(x: 32, y: 32, equals: (0, 255, 0, 255), "선언 앞의 참조는 전역을 봐야 한다")
    }
}
