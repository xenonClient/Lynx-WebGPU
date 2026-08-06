import XCTest
import Metal
import LynxWebGPUCore
@testable import LynxWebGPU

/// 안전 변환이 **실제 GPU에서** 동작하는지 — 방출된 MSL 문자열이 아니라 값으로 확인한다.
///
/// 문자열 단언(`ShaderSafetyTests`)은 "그렇게 방출된다"까지만 본다. 여기서는 범위를 벗어난
/// 인덱스가 정말로 잘리는지, 버려진 프래그먼트의 쓰기가 정말로 막히는지를 버퍼 값으로 읽는다.
final class ShaderSafetyRenderTests: XCTestCase {
    private var harness: RenderHarness!

    override func setUpWithError() throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "Metal 디바이스 없음")
        harness = try XCTUnwrap(RenderHarness.make())
    }

    override func tearDown() {
        harness = nil
        super.tearDown()
    }

    /// 범위를 크게 벗어난 **읽기**가 잘려 마지막 원소를 준다 — 인접 메모리를 읽지 않는다.
    func test_범위를_벗어난_읽기가_마지막_원소로_잘린다() throws {
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
            // 4개짜리 배열에 100번을 넣는다.
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
        XCTAssertEqual(result.first, 40, "범위를 벗어난 인덱스가 마지막 원소로 잘려야 한다")
    }

    /// 범위를 벗어난 **쓰기**가 잘려 버퍼 안에 떨어진다 — 잘리지 않으면 남의 메모리를 덮어쓴다.
    func test_범위를_벗어난_쓰기가_버퍼_안으로_잘린다() throws {
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
        XCTAssertEqual(result, [0, 0, 0, 9], "100만 번째 쓰기가 버퍼의 마지막 자리로 잘려야 한다")
    }

    /// `var<workgroup>`은 0에서 시작한다 — 쓰지 않고 읽으면 0이어야 한다.
    func test_workgroup_메모리가_0에서_시작한다() throws {
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
        XCTAssertEqual(result, [0, 0, 0, 0], "초기화되지 않은 threadgroup 메모리를 읽었다")
    }

    /// **버려진 프래그먼트는 스토리지를 오염시키지 않는다.**
    ///
    /// MSL의 `discard_fragment()`는 즉시 종료가 아니라 뒤의 코드가 계속 돈다 — 막지 않으면
    /// 이 테스트의 버퍼가 1.0으로 채워진다.
    func test_버려진_프래그먼트는_스토리지에_쓰지_않는다() throws {
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
        XCTAssertEqual(result.first, 0, "버려진 프래그먼트가 스토리지 버퍼에 썼다")
        // 폐기 자체는 그대로 일어나야 한다 — 화면에는 클리어색만 남는다.
        try harness.assertPixel(x: 32, y: 32, equals: (0, 0, 255, 255), "폐기된 프래그먼트가 그려졌다")
    }

    /// 0으로 나누기가 **프로세스를 죽이지 않고** WGSL이 정한 값을 준다 (`x / 0 == x`, `x % 0 == 0`).
    func test_0으로_나누기가_명세대로_동작한다() throws {
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
        XCTAssertEqual(result, [100, 0], "WGSL은 x / 0 == x, x % 0 == 0으로 정의한다")
    }
}
