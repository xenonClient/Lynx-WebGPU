import XCTest
import Metal
import ImageIO
import CoreGraphics
import UniformTypeIdentifiers
import LynxWebGPUCore
@testable import LynxWebGPU

/// `importSharedTexture` end to end on the Metal runtime — two contexts on one device stand in for two
/// LynxViews, each with its own registry and handles, sharing one cache.
final class SharedTextureTests: XCTestCase {
    var device: MTLDevice!
    var cache: WGPUSharedTextureCache!
    var provider: WGPUFileAssetProvider!
    /// Two "views" — separate contexts, separate canvases, the same device and cache.
    var a: RenderHarness!
    var b: RenderHarness!

    override func setUpWithError() throws {
        device = try XCTUnwrap(MTLCreateSystemDefaultDevice(), "no Metal device")
        cache = WGPUSharedTextureCache()
        provider = WGPUFileAssetProvider()
        a = try XCTUnwrap(RenderHarness.make(runtime: makeContext(), width: 64, height: 64))
        b = try XCTUnwrap(RenderHarness.make(runtime: makeContext(), width: 64, height: 64))
        // Top half red, bottom half blue — asymmetric, so orientation is visible.
        provider.register(try Self.png(width: 8, height: 8, top: (255, 0, 0, 255), bottom: (0, 0, 255, 255)),
                          for: "halves.png")
    }

    override func tearDown() {
        a = nil
        b = nil
        super.tearDown()
    }

    private func makeContext() throws -> LynxWebGPUContext {
        try LynxWebGPUContext(device: device, frameCoordinator: WGPUFrameCoordinator(), sharedTextures: cache)
    }

    // MARK: - Helpers

    static func png(
        width: Int, height: Int,
        top: (UInt8, UInt8, UInt8, UInt8), bottom: (UInt8, UInt8, UInt8, UInt8)
    ) throws -> Data {
        var pixels = [UInt8]()
        for row in 0..<height {
            let color = row < height / 2 ? top : bottom
            for _ in 0..<width { pixels.append(contentsOf: [color.0, color.1, color.2, color.3]) }
        }
        let provider = try XCTUnwrap(CGDataProvider(data: Data(pixels) as CFData))
        let image = try XCTUnwrap(CGImage(
            width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
            bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue | CGBitmapInfo.byteOrder32Big.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent
        ))
        let output = NSMutableData()
        let destination = try XCTUnwrap(CGImageDestinationCreateWithData(output, UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        XCTAssertTrue(CGImageDestinationFinalize(destination))
        return output as Data
    }

    /// Imports through the protocol and waits — the callback must come exactly once.
    @discardableResult
    private func importShared(
        into harness: RenderHarness, handle: Int = 50, key: String = "halves",
        name: String = "halves.png", resizeWidth: Int? = nil, resizeHeight: Int? = nil
    ) -> [String: Any] {
        let done = expectation(description: "importSharedTexture")
        var result: [String: Any] = [:]
        var calls = 0
        harness.runtime.importSharedTexture(
            handle: handle,
            request: WGPUSharedTextureRequest(key: key, name: name, resizeWidth: resizeWidth, resizeHeight: resizeHeight),
            provider: provider
        ) {
            result = $0
            calls += 1
            done.fulfill()
        }
        wait(for: [done], timeout: 10)
        XCTAssertEqual(calls, 1)
        return result
    }

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
        out.uv = vec2f(positions[index].x * 0.5 + 0.5, 0.5 - positions[index].y * 0.5);
        return out;
    }

    @fragment
    fn fs_main(in: Out) -> @location(0) vec4f {
        return textureSample(tex, samp, in.uv);
    }
    """

    /// Samples the texture at `handle` across the whole canvas (ids 1…11 are this helper's).
    private func draw(_ harness: RenderHarness, texture handle: Int = 50) {
        harness.executeExpectingSuccess([
            ["op": "configureCanvas", "canvas": "test", "format": "rgba8unorm"],
            ["op": "createShaderModule", "id": 1, "code": Self.shader],
            ["op": "createTextureView", "id": 3, "texture": handle],
            ["op": "createSampler", "id": 4],
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
    }

    /// The image's top half red, bottom half blue — and not the green clear color.
    private func assertHalves(_ harness: RenderHarness, _ message: String, line: UInt = #line) throws {
        try harness.assertPixel(x: 32, y: 8, equals: (255, 0, 0, 255), "\(message): top", line: line)
        try harness.assertPixel(x: 32, y: 56, equals: (0, 0, 255, 255), "\(message): bottom", line: line)
    }

    // MARK: - Sharing (ensure 1–3)

    /// Two views asking for the same key — one decode, one MTLTexture, and both draw it.
    func test_twoContextsShareOneDecodeAndOneTexture() throws {
        let first = importShared(into: a)
        let second = importShared(into: b, handle: 77)   // handles are per view — the key is what matters
        XCTAssertEqual(first["ok"] as? Bool, true, "\(first)")
        XCTAssertEqual(second["ok"] as? Bool, true, "\(second)")
        XCTAssertEqual(first["width"] as? Int, 8)
        XCTAssertEqual(first["format"] as? String, "rgba8unorm")

        XCTAssertEqual(cache.makeCount, 1, "decoded once")
        XCTAssertEqual(cache.liveEntryCount, 1, "one texture for the key")

        draw(a)
        try assertHalves(a, "view A")
        draw(b, texture: 77)
        try assertHalves(b, "view B")
    }

    /// `device.destroy()` in one view (the runtime's reset) drops only that view's reference.
    func test_resettingOneContextLeavesTheOtherSampling() throws {
        importShared(into: a)
        importShared(into: b)
        draw(a)

        a.runtime.reset()
        XCTAssertEqual(cache.liveEntryCount, 1, "view B still holds it")

        draw(b)
        try assertHalves(b, "view B after view A reset")
    }

    /// When every view has let go — `destroy` in one, `reset` in the other — the cache entry is gone.
    func test_theCacheEntryGoesWhenEveryContextLetsGo() throws {
        importShared(into: a)
        importShared(into: b)
        draw(b)   // views and bind groups of the texture exist in B, too

        a.executeExpectingSuccess([["op": "destroy", "id": 50]])
        XCTAssertEqual(cache.liveEntryCount, 1)

        b.runtime.reset()
        XCTAssertEqual(cache.liveEntryCount, 0)
        XCTAssertNil(cache.entry(forKey: "halves", device: device))

        importShared(into: a)
        XCTAssertEqual(cache.makeCount, 2, "the next import decodes again")
    }

    /// A view outlives its texture's `destroy` — and keeps the cache entry with it, so a re-import
    /// does not put a second copy in memory.
    func test_aRemainingViewKeepsTheEntry() throws {
        importShared(into: a)
        a.executeExpectingSuccess([
            ["op": "createTextureView", "id": 60, "texture": 50],
            ["op": "destroy", "id": 50],
        ])
        XCTAssertEqual(cache.liveEntryCount, 1)
        importShared(into: b)
        XCTAssertEqual(cache.makeCount, 1)
    }

    // MARK: - Read-only

    /// Every way of writing to a shared texture is a validation error — the pixels are another view's too.
    func test_writesToASharedTextureAreRejected() throws {
        importShared(into: a)
        let writes: [(String, [[String: Any]])] = [
            ("writeTexture", [
                ["op": "writeTexture", "texture": 50, "data": [UInt8](repeating: 0, count: 16),
                 "size": ["width": 2, "height": 2], "bytesPerRow": 8],
            ]),
            ("copyBufferToTexture", [
                ["op": "createBuffer", "id": 20, "size": 1024, "usage": TestUsage.copySrc],
                ["op": "copyBufferToTexture", "source": ["buffer": 20, "bytesPerRow": 256],
                 "destination": ["texture": 50], "copySize": ["width": 2, "height": 2]],
            ]),
            ("copyTextureToTexture", [
                ["op": "createTexture", "id": 21, "size": ["width": 8, "height": 8], "format": "rgba8unorm",
                 "usage": TestUsage.textureCopySrc | TestUsage.textureBinding],
                ["op": "copyTextureToTexture", "source": ["texture": 21], "destination": ["texture": 50],
                 "copySize": ["width": 2, "height": 2]],
            ]),
            ("a render pass color attachment", [
                ["op": "createTextureView", "id": 22, "texture": 50],
                ["op": "beginRenderPass", "colorAttachments": [["view": 22, "loadOp": "clear", "storeOp": "store"]]],
                ["op": "endPass"],
            ]),
            ("a write-only storage texture binding", [
                ["op": "createBindGroupLayout", "id": 23, "entries": [
                    ["binding": 0, "visibility": 4,
                     "storageTexture": ["access": "write-only", "format": "rgba8unorm"]],
                ]],
                ["op": "createTextureView", "id": 24, "texture": 50],
                ["op": "createBindGroup", "id": 25, "layout": 23, "entries": [
                    ["binding": 0, "resource": ["textureView": 24]],
                ]],
            ]),
            ("a view of a shared texture must keep its format", [
                ["op": "createTextureView", "id": 26, "texture": 50, "format": "rgba8unorm-srgb"],
            ]),
        ]
        for (needle, commands) in writes {
            let result = a.execute(commands)
            let errors = a.describeErrors(result)
            XCTAssertEqual(result["ok"] as? Bool, false, needle)
            XCTAssertTrue(errors.contains("[validation]"), "\(needle): \(errors)")
            XCTAssertTrue(errors.contains(needle), "\(needle): \(errors)")
        }

        // Reading is fine — and the pixels survived every attempt.
        let readOnly = a.execute([
            ["op": "createBindGroupLayout", "id": 30, "entries": [
                ["binding": 0, "visibility": 4,
                 "storageTexture": ["access": "read-only", "format": "rgba8unorm"]],
            ]],
            ["op": "createTextureView", "id": 31, "texture": 50],
            ["op": "createBindGroup", "id": 32, "layout": 30, "entries": [["binding": 0, "resource": ["textureView": 31]]]],
        ])
        XCTAssertEqual(readOnly["ok"] as? Bool, true, a.describeErrors(readOnly))
        draw(a)
        try assertHalves(a, "after the rejected writes")
    }

    // MARK: - Decoding

    func test_bothSizesGivenAreTheExactSize() throws {
        let result = importShared(into: a, key: "halves@5x3", resizeWidth: 5, resizeHeight: 3)
        XCTAssertEqual(result["width"] as? Int, 5)
        XCTAssertEqual(result["height"] as? Int, 3)
        XCTAssertEqual(cache.entry(forKey: "halves@5x3", device: device)?.texture.width, 5)
    }

    /// One size given — the other follows the source's aspect ratio (8×8 here, so 4 → 4×4). The
    /// thumbnail path keeps the orientation and the colors.
    func test_oneSizeGivenKeepsTheAspectRatio() throws {
        let landscape = try Self.png(width: 40, height: 20, top: (255, 0, 0, 255), bottom: (0, 0, 255, 255))
        provider.register(landscape, for: "landscape.png")
        let result = importShared(into: a, key: "landscape@w10", name: "landscape.png", resizeWidth: 10)
        XCTAssertEqual(result["width"] as? Int, 10)
        XCTAssertEqual(result["height"] as? Int, 5)

        let half = importShared(into: b, handle: 51, key: "halves@w4", resizeWidth: 4)
        XCTAssertEqual(half["width"] as? Int, 4)
        XCTAssertEqual(half["height"] as? Int, 4)
        draw(b, texture: 51)
        try assertHalves(b, "a downscaled import")
    }

    /// Alpha comes out premultiplied — the documented contract (`docs/WEBGPU-API.md`).
    func test_alphaIsPremultiplied() throws {
        provider.register(
            try Self.png(width: 4, height: 4, top: (200, 100, 40, 128), bottom: (200, 100, 40, 128)),
            for: "translucent.png"
        )
        importShared(into: a, key: "translucent", name: "translucent.png")
        draw(a)
        try a.assertPixel(x: 32, y: 32, equals: (100, 50, 20, 128), "each channel × 128/255")
    }

    func test_aMissingAssetOrBrokenBytesFailWithoutAnEntry() throws {
        let missing = importShared(into: a, key: "missing", name: "no-such-image.png")
        XCTAssertEqual(missing["ok"] as? Bool, false)

        provider.register(Data([0, 1, 2, 3]), for: "broken.png")
        let broken = importShared(into: a, key: "broken", name: "broken.png")
        XCTAssertEqual(broken["ok"] as? Bool, false)
        let message = ((broken["errors"] as? [[String: Any]])?.first?["message"] as? String) ?? ""
        XCTAssertTrue(message.contains("could not decode"), message)

        XCTAssertEqual(cache.liveEntryCount, 0)
    }

    func test_aSizePastTheDeviceLimitIsAValidationError() {
        let limit = WGPUDeviceCapability.maxTextureDimension2D(device)
        let result = importShared(into: a, key: "huge", resizeWidth: limit + 1, resizeHeight: 4)
        XCTAssertEqual(result["ok"] as? Bool, false)
        let message = ((result["errors"] as? [[String: Any]])?.first?["message"] as? String) ?? ""
        XCTAssertTrue(message.contains("maxTextureDimension2D"), message)
    }

    /// One huge size on a sliver of an image — the derived axis would be past `Int`. It used to trap the
    /// process on the cache's queue; the requested axis is now checked against the device limit before
    /// anything is derived, the error names that axis, and nothing is cached.
    func test_aHugeOneAxisResizeOnASliverIsAValidationError() throws {
        provider.register(try Self.png(width: 1, height: 2000, top: (255, 0, 0, 255), bottom: (0, 0, 255, 255)),
                          for: "sliver.png")
        let result = importShared(into: a, key: "sliver@huge", name: "sliver.png", resizeWidth: (1 << 53) - 1)
        XCTAssertEqual(result["ok"] as? Bool, false)
        let message = ((result["errors"] as? [[String: Any]])?.first?["message"] as? String) ?? ""
        XCTAssertTrue(message.contains("maxTextureDimension2D"), message)
        XCTAssertTrue(message.contains("resizeWidth"), message)
        XCTAssertEqual(cache.liveEntryCount, 0)
    }

    /// A Swift caller skips the payload reader's `≥ 1` check — `makeTexture` applies the same bounds
    /// itself, so a zero, negative or oversized axis is a validation error, not a trap or a zero-byte buffer.
    func test_makeTextureRejectsARequestedAxisOutsideTheDeviceLimit() throws {
        let data = try Self.png(width: 1, height: 2000, top: (255, 0, 0, 255), bottom: (0, 0, 255, 255))
        let limit = WGPUDeviceCapability.maxTextureDimension2D(device)
        for request in [
            WGPUSharedTextureRequest(key: "k", name: "a", resizeWidth: Int.min),
            WGPUSharedTextureRequest(key: "k", name: "a", resizeHeight: -1),
            WGPUSharedTextureRequest(key: "k", name: "a", resizeWidth: 0, resizeHeight: 0),
            WGPUSharedTextureRequest(key: "k", name: "a", resizeWidth: 4, resizeHeight: limit + 1),
        ] {
            XCTAssertThrowsError(try WGPUSharedTextureUpload.makeTexture(from: data, request: request, device: device)) {
                let error = $0 as? WGPUError
                XCTAssertEqual(error?.kind, .validation, "\(request.resizeWidth as Any)x\(request.resizeHeight as Any)")
                XCTAssertTrue(error?.message.contains("maxTextureDimension2D") == true, error?.message ?? "")
            }
        }
    }

    /// The provider's scope applies to every import, cache hit or not.
    func test_theProviderIsAskedEvenOnACacheHit() throws {
        importShared(into: a)
        let narrowed = WGPUFileAssetProvider(allowedRoots: [FileManager.default.temporaryDirectory])
        let done = expectation(description: "import with a narrowed provider")
        var result: [String: Any] = [:]
        b.runtime.importSharedTexture(
            handle: 50, request: WGPUSharedTextureRequest(key: "halves", name: "/etc/hosts"), provider: narrowed
        ) {
            result = $0
            done.fulfill()
        }
        wait(for: [done], timeout: 5)
        XCTAssertEqual(result["ok"] as? Bool, false, "a path outside the provider's roots is refused before the cache")
        XCTAssertEqual(cache.makeCount, 1)
    }
}
