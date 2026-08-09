import XCTest
import Metal
import LynxWebGPUCore
@testable import LynxWebGPU

/// 블록 압축 텍스처를 **실제로 GPU에 올려 샘플링**한다.
///
/// 산수만 맞고 Metal 포맷 대응이 틀리면 화면에는 쓰레기가 나오는데 오류는 없다. 그래서
/// 손으로 인코딩한 블록 하나를 올려 색을 되읽는다 — 블록 레이아웃·포맷 대응·복사 스트라이드가
/// 한 줄에 다 걸린다.
final class CompressedTextureTests: XCTestCase {
    var harness: RenderHarness!

    override func setUpWithError() throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "Metal 디바이스 없음")
        harness = try XCTUnwrap(RenderHarness.make())
    }

    override func tearDown() {
        harness = nil
        super.tearDown()
    }

    /// 상수 색 블록을 그대로 화면에 펼치는 최소 셰이더.
    private static let shader = """
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

    /// BC1 블록 하나 (8B). `color0 > color1`이면 4색 모드이고 인덱스 0은 `color0`이다.
    /// 인덱스를 전부 0으로 두면 4×4가 통째로 `color0` 한 색이 된다.
    private static func bc1Block(rgb565: UInt16) -> [UInt8] {
        [UInt8(rgb565 & 0xFF), UInt8(rgb565 >> 8), 0, 0, 0, 0, 0, 0]
    }

    /// ASTC "void extent" 블록 (16B) — 블록 전체가 한 색이라고 선언하는 형태다.
    /// 앞 9비트가 서명(`0b111111100`), 이어지는 범위 비트는 전부 1(=무시), 뒤 8바이트가
    /// UNORM16 RGBA다. 블록 크기와 무관하게 같은 인코딩이라 6x5에도 그대로 쓴다.
    private static func astcVoidExtent(r: UInt16, g: UInt16, b: UInt16, a: UInt16) -> [UInt8] {
        var block: [UInt8] = [0xFC, 0xFD, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]
        for value in [r, g, b, a] {
            block.append(UInt8(value & 0xFF))
            block.append(UInt8(value >> 8))
        }
        return block
    }

    /// 압축 텍스처를 올려 화면 전체에 펼치고 가운데 픽셀 색을 확인한다.
    private func renderCompressed(
        format: String, width: Int, height: Int, block: [UInt8], bytesPerRow: Int
    ) throws -> (r: Int, g: Int, b: Int, a: Int) {
        harness.executeExpectingSuccess([
            ["op": "configureCanvas", "canvas": "test", "format": "rgba8unorm"],
            ["op": "createShaderModule", "id": 1, "code": Self.shader],
            ["op": "createTexture", "id": 2, "size": ["width": width, "height": height],
             "format": format, "usage": TestUsage.textureBinding | TestUsage.textureCopyDst],
            ["op": "writeTexture", "texture": 2, "data": Data(block).base64EncodedString(),
             "size": ["width": width, "height": height], "bytesPerRow": bytesPerRow],
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
                "clearValue": ["r": 0, "g": 1, "b": 0, "a": 1],
            ]]],
            ["op": "setPipeline", "pipeline": 5],
            ["op": "setBindGroup", "index": 0, "bindGroup": 7],
            ["op": "draw", "vertexCount": 3],
            ["op": "endPass"],
        ])
        return try harness.pixel(x: 32, y: 32)
    }

    func test_BC1_텍스처를_샘플링한다() throws {
        try XCTSkipUnless(
            WGPUDeviceCapability.supportsCompression(.bc1RGBAUnorm, on: harness.context!.device),
            "이 기기는 BC를 지원하지 않는다"
        )
        // RGB565의 순수 빨강 = R5 최대(31) → 8비트로 펼치면 255다.
        let color = try renderCompressed(
            format: "bc1-rgba-unorm", width: 4, height: 4,
            block: Self.bc1Block(rgb565: 0xF800), bytesPerRow: 8
        )
        XCTAssertEqual(color.r, 255, "빨강 채널")
        XCTAssertEqual(color.g, 0, "초록이 새면 블록 해석이 틀린 것이다")
        XCTAssertEqual(color.b, 0)
    }

    /// 정사각이 아닌 블록(6×5)까지 확인한다 — 행 수를 높이로 세면 여기서 깨진다.
    func test_ASTC_6x5_텍스처를_샘플링한다() throws {
        try XCTSkipUnless(
            WGPUDeviceCapability.supportsCompression(.astc6x5Unorm, on: harness.context!.device),
            "이 기기는 ASTC를 지원하지 않는다"
        )
        let color = try renderCompressed(
            format: "astc-6x5-unorm", width: 6, height: 5,
            block: Self.astcVoidExtent(r: 0, g: 0, b: 0xFFFF, a: 0xFFFF), bytesPerRow: 16
        )
        XCTAssertEqual(color.b, 255, "파랑 채널")
        XCTAssertEqual(color.r, 0)
        XCTAssertEqual(color.g, 0)
    }

    /// `bytesPerRow`를 생략하면 **블록 단위로 올림**한 기본값이 나와야 한다.
    /// 픽셀로 계산하던 예전 식이면 데이터가 모자라 "부족하다"로 거부된다.
    func test_bytesPerRow를_생략해도_블록으로_계산한다() throws {
        try XCTSkipUnless(
            WGPUDeviceCapability.supportsCompression(.astc4x4Unorm, on: harness.context!.device),
            "이 기기는 ASTC를 지원하지 않는다"
        )
        // 8x8 = 4x4 블록이 2x2개 = 64바이트. bytesPerRow는 32여야 한다.
        let blocks = (0..<4).flatMap { _ in Self.astcVoidExtent(r: 0xFFFF, g: 0xFFFF, b: 0, a: 0xFFFF) }
        harness.executeExpectingSuccess([
            ["op": "createTexture", "id": 1, "size": ["width": 8, "height": 8],
             "format": "astc-4x4-unorm", "usage": TestUsage.textureBinding | TestUsage.textureCopyDst],
            ["op": "writeTexture", "texture": 1, "data": Data(blocks).base64EncodedString(),
             "size": ["width": 8, "height": 8]],
        ])
    }

    /// 압축 텍스처는 렌더 타깃이 될 수 없다. **Metal은 이것을 단언으로 죽인다** —
    /// 검증 오류로 돌려주는지가 요점이다.
    func test_rejectsACompressedTextureAsARenderTarget() throws {
        let result = harness.execute([
            ["op": "createTexture", "id": 1, "size": ["width": 4, "height": 4],
             "format": "astc-4x4-unorm", "usage": TestUsage.renderAttachment],
        ])
        XCTAssertEqual(result["ok"] as? Bool, false)
        XCTAssertTrue(harness.describeErrors(result).contains("render target"), harness.describeErrors(result))
    }

    func test_rejectsAMisalignedOriginOnACompressedTexture() throws {
        try XCTSkipUnless(
            WGPUDeviceCapability.supportsCompression(.astc4x4Unorm, on: harness.context!.device),
            "이 기기는 ASTC를 지원하지 않는다"
        )
        let block = Self.astcVoidExtent(r: 0xFFFF, g: 0, b: 0, a: 0xFFFF)
        let result = harness.execute([
            ["op": "createTexture", "id": 1, "size": ["width": 8, "height": 8],
             "format": "astc-4x4-unorm", "usage": TestUsage.textureBinding | TestUsage.textureCopyDst],
            // origin.x = 2는 4x4 블록 경계가 아니다.
            ["op": "writeTexture", "texture": 1, "data": Data(block).base64EncodedString(),
             "origin": ["x": 2, "y": 0], "size": ["width": 4, "height": 4], "bytesPerRow": 16],
        ])
        XCTAssertEqual(result["ok"] as? Bool, false)
        XCTAssertTrue(harness.describeErrors(result).contains("block boundary"), harness.describeErrors(result))
    }

    /// 가장자리 블록은 잘려 있으므로, **밉 레벨 끝에 닿는** 크기는 블록 배수가 아니어도 된다.
    func test_reachingTheMipLevelEndNeedNotBeAMultipleOfTheBlock() throws {
        try XCTSkipUnless(
            WGPUDeviceCapability.supportsCompression(.astc4x4Unorm, on: harness.context!.device),
            "이 기기는 ASTC를 지원하지 않는다"
        )
        let block = Self.astcVoidExtent(r: 0xFFFF, g: 0, b: 0, a: 0xFFFF)
        harness.executeExpectingSuccess([
            ["op": "createTexture", "id": 1, "size": ["width": 6, "height": 6],
             "format": "astc-4x4-unorm", "usage": TestUsage.textureBinding | TestUsage.textureCopyDst],
            // 6x6은 4의 배수가 아니지만 텍스처 끝에 닿는다 — 블록은 2x2개 필요하다.
            ["op": "writeTexture", "texture": 1,
             "data": Data(block + block + block + block).base64EncodedString(),
             "size": ["width": 6, "height": 6], "bytesPerRow": 32],
        ])
    }

    /// `adapter.features`가 실제 기기 능력과 일치하는지 — 있다고 알리고 못 만들면
    /// 확인하고 쓴 앱이 오히려 배신당한다.
    func test_featureAdvertisementMatchesActualSupport() throws {
        let info = harness.runtime.adapterInfo()
        let features = Set(info["features"] as? [String] ?? [])
        let device = harness.context!.device
        XCTAssertEqual(
            features.contains("texture-compression-astc"),
            WGPUDeviceCapability.supportsCompression(.astc4x4Unorm, on: device)
        )
        XCTAssertEqual(
            features.contains("texture-compression-bc"),
            WGPUDeviceCapability.supportsCompression(.bc1RGBAUnorm, on: device)
        )
        XCTAssertEqual(
            features.contains("texture-compression-etc2"),
            WGPUDeviceCapability.supportsCompression(.etc2RGB8Unorm, on: device)
        )
    }
}
