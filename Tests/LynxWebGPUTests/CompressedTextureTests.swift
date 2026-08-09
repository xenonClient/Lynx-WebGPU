import XCTest
import Metal
import LynxWebGPUCore
@testable import LynxWebGPU

/// **Actually uploads block-compressed textures to the GPU and samples them.**
///
/// If the arithmetic is right but the Metal format correspondence is wrong, the screen shows garbage
/// with no error. So we upload one hand-encoded block and read the color back — block layout, format
/// correspondence and copy stride all hang on that one line.
final class CompressedTextureTests: XCTestCase {
    var harness: RenderHarness!

    override func setUpWithError() throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "no Metal device")
        harness = try XCTUnwrap(RenderHarness.make())
    }

    override func tearDown() {
        harness = nil
        super.tearDown()
    }

    /// The smallest shader that spreads a constant-color block across the screen.
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

    /// One BC1 block (8B). With `color0 > color1` it is 4-color mode and index 0 is `color0`.
    /// Setting every index to 0 makes the whole 4×4 a single `color0`.
    private static func bc1Block(rgb565: UInt16) -> [UInt8] {
        [UInt8(rgb565 & 0xFF), UInt8(rgb565 >> 8), 0, 0, 0, 0, 0, 0]
    }

    /// An ASTC "void extent" block (16B) — the form declaring the whole block is one color.
    /// The first 9 bits are the signature (`0b111111100`), the range bits that follow are all 1
    /// (= ignored), and the last 8 bytes are UNORM16 RGBA. The encoding is the same regardless of
    /// block size, so it is used for 6x5 unchanged.
    private static func astcVoidExtent(r: UInt16, g: UInt16, b: UInt16, a: UInt16) -> [UInt8] {
        var block: [UInt8] = [0xFC, 0xFD, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]
        for value in [r, g, b, a] {
            block.append(UInt8(value & 0xFF))
            block.append(UInt8(value >> 8))
        }
        return block
    }

    /// Uploads a compressed texture, spreads it across the screen and checks the center pixel's color.
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

    func test_samplesABC1Texture() throws {
        try XCTSkipUnless(
            WGPUDeviceCapability.supportsCompression(.bc1RGBAUnorm, on: harness.context!.device),
            "this device does not support BC"
        )
        // Pure red in RGB565 = R5 at maximum (31) → expanded to 8 bits it is 255.
        let color = try renderCompressed(
            format: "bc1-rgba-unorm", width: 4, height: 4,
            block: Self.bc1Block(rgb565: 0xF800), bytesPerRow: 8
        )
        XCTAssertEqual(color.r, 255, "red channel")
        XCTAssertEqual(color.g, 0, "green leaking means the block interpretation is wrong")
        XCTAssertEqual(color.b, 0)
    }

    /// Checks a non-square block (6×5) too — counting rows by height breaks right here.
    func test_samplesAnASTC6x5Texture() throws {
        try XCTSkipUnless(
            WGPUDeviceCapability.supportsCompression(.astc6x5Unorm, on: harness.context!.device),
            "this device does not support ASTC"
        )
        let color = try renderCompressed(
            format: "astc-6x5-unorm", width: 6, height: 5,
            block: Self.astcVoidExtent(r: 0, g: 0, b: 0xFFFF, a: 0xFFFF), bytesPerRow: 16
        )
        XCTAssertEqual(color.b, 255, "blue channel")
        XCTAssertEqual(color.r, 0)
        XCTAssertEqual(color.g, 0)
    }

    /// Omitting `bytesPerRow` must produce a default **rounded up in blocks**.
    /// With the old pixel-based formula the data falls short and it is rejected as "not enough".
    func test_omittingBytesPerRowStillComputesInBlocks() throws {
        try XCTSkipUnless(
            WGPUDeviceCapability.supportsCompression(.astc4x4Unorm, on: harness.context!.device),
            "this device does not support ASTC"
        )
        // 8x8 = 2x2 blocks of 4x4 = 64 bytes. bytesPerRow must be 32.
        let blocks = (0..<4).flatMap { _ in Self.astcVoidExtent(r: 0xFFFF, g: 0xFFFF, b: 0, a: 0xFFFF) }
        harness.executeExpectingSuccess([
            ["op": "createTexture", "id": 1, "size": ["width": 8, "height": 8],
             "format": "astc-4x4-unorm", "usage": TestUsage.textureBinding | TestUsage.textureCopyDst],
            ["op": "writeTexture", "texture": 1, "data": Data(blocks).base64EncodedString(),
             "size": ["width": 8, "height": 8]],
        ])
    }

    /// A compressed texture cannot be a render target. **Metal kills the process over this** —
    /// the point is whether it comes back as a validation error.
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
            "this device does not support ASTC"
        )
        let block = Self.astcVoidExtent(r: 0xFFFF, g: 0, b: 0, a: 0xFFFF)
        let result = harness.execute([
            ["op": "createTexture", "id": 1, "size": ["width": 8, "height": 8],
             "format": "astc-4x4-unorm", "usage": TestUsage.textureBinding | TestUsage.textureCopyDst],
            // origin.x = 2 is not on a 4x4 block boundary.
            ["op": "writeTexture", "texture": 1, "data": Data(block).base64EncodedString(),
             "origin": ["x": 2, "y": 0], "size": ["width": 4, "height": 4], "bytesPerRow": 16],
        ])
        XCTAssertEqual(result["ok"] as? Bool, false)
        XCTAssertTrue(harness.describeErrors(result).contains("block boundary"), harness.describeErrors(result))
    }

    /// Edge blocks are cut off, so a size **reaching the end of the mip level** need not be a multiple of the block.
    func test_reachingTheMipLevelEndNeedNotBeAMultipleOfTheBlock() throws {
        try XCTSkipUnless(
            WGPUDeviceCapability.supportsCompression(.astc4x4Unorm, on: harness.context!.device),
            "this device does not support ASTC"
        )
        let block = Self.astcVoidExtent(r: 0xFFFF, g: 0, b: 0, a: 0xFFFF)
        harness.executeExpectingSuccess([
            ["op": "createTexture", "id": 1, "size": ["width": 6, "height": 6],
             "format": "astc-4x4-unorm", "usage": TestUsage.textureBinding | TestUsage.textureCopyDst],
            // 6x6 is not a multiple of 4 but reaches the texture's end — 2x2 blocks are needed.
            ["op": "writeTexture", "texture": 1,
             "data": Data(block + block + block + block).base64EncodedString(),
             "size": ["width": 6, "height": 6], "bytesPerRow": 32],
        ])
    }

    /// Whether `adapter.features` matches actual device capability — advertising support and then
    /// failing to create betrays the app that checked before using it.
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
