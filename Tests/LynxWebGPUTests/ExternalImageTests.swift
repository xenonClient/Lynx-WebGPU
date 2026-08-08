import XCTest
import Metal
import ImageIO
import CoreGraphics
import UniformTypeIdentifiers
import LynxWebGPUCore
@testable import LynxWebGPU

/// 외부 이미지 → 텍스처 (`createImageBitmap` + `copyExternalImageToTexture`).
///
/// 웹에서는 브라우저가 디코딩을 맡는 자리다. 여기서는 ImageIO가 하므로 **채널 순서와
/// 상하 방향**이 조용히 틀릴 수 있다 — 둘 다 화면에서만 드러나는 종류라 픽셀로 못 박는다.
final class ExternalImageTests: XCTestCase {
    var harness: RenderHarness!

    override func setUpWithError() throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "Metal 디바이스 없음")
        harness = try XCTUnwrap(RenderHarness.make())
    }

    override func tearDown() {
        harness = nil
        super.tearDown()
    }

    // MARK: - 이미지 만들기

    /// 위/아래 절반 색이 다른 PNG를 만든다 — flipY가 실제로 뒤집는지 보려면 비대칭이어야 한다.
    private func makePNG(
        width: Int, height: Int,
        top: (UInt8, UInt8, UInt8, UInt8), bottom: (UInt8, UInt8, UInt8, UInt8)
    ) throws -> Data {
        var pixels = [UInt8]()
        for row in 0..<height {
            let color = row < height / 2 ? top : bottom
            for _ in 0..<width { pixels.append(contentsOf: [color.0, color.1, color.2, color.3]) }
        }
        let provider = CGDataProvider(data: Data(pixels) as CFData)!
        let image = CGImage(
            width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
            bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue
                                     | CGBitmapInfo.byteOrder32Big.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent
        )!
        let output = NSMutableData()
        let destination = CGImageDestinationCreateWithData(
            output, UTType.png.identifier as CFString, 1, nil
        )!
        CGImageDestinationAddImage(destination, image, nil)
        XCTAssertTrue(CGImageDestinationFinalize(destination))
        return output as Data
    }

    /// `decodeImage`를 동기적으로 기다린다 (테스트 편의).
    @discardableResult
    private func decode(
        handle: Int, data: Data, options: WGPUImageDecoder.Options = .init()
    ) throws -> [String: Any] {
        let expectation = expectation(description: "decodeImage")
        var payload: [String: Any] = [:]
        harness.runtime.decodeImage(
            handle: handle, data: data, name: nil, options: options, provider: nil
        ) { result in
            payload = result
            expectation.fulfill()
        }
        wait(for: [expectation], timeout: 5)
        return payload
    }

    // MARK: - 디코딩

    func test_PNG를_풀면_크기와_채널_순서가_맞는다() throws {
        let png = try makePNG(width: 4, height: 4, top: (255, 0, 0, 255), bottom: (0, 0, 255, 255))
        let bitmap = try WGPUImageDecoder.decode(png)

        XCTAssertEqual(bitmap.width, 4)
        XCTAssertEqual(bitmap.height, 4)
        XCTAssertEqual(bitmap.bytesPerRow, 16)
        // 첫 행은 빨강 — RGBA 순서가 아니면 여기서 B가 255로 나온다.
        XCTAssertEqual(Array(bitmap.pixels.prefix(4)), [255, 0, 0, 255], "첫 픽셀이 빨강이 아니다")
        // 마지막 행은 파랑 — 위아래가 뒤집혔으면 여기서 걸린다.
        XCTAssertEqual(Array(bitmap.pixels.suffix(4)), [0, 0, 255, 255], "위아래가 뒤집혔다")
    }

    func test_flipY가_위아래를_뒤집는다() throws {
        let png = try makePNG(width: 4, height: 4, top: (255, 0, 0, 255), bottom: (0, 0, 255, 255))
        let flipped = try WGPUImageDecoder.decode(png, options: .init(flipY: true))

        XCTAssertEqual(Array(flipped.pixels.prefix(4)), [0, 0, 255, 255], "뒤집으면 첫 행이 파랑이다")
        XCTAssertEqual(Array(flipped.pixels.suffix(4)), [255, 0, 0, 255])
    }

    func test_premultiplyAlpha가_색에_알파를_곱한다() throws {
        // 알파 50%의 흰색. 곱하면 128 근처, 안 곱하면 255다.
        let png = try makePNG(width: 2, height: 2, top: (255, 255, 255, 128), bottom: (255, 255, 255, 128))

        let straight = try WGPUImageDecoder.decode(png)
        XCTAssertEqual(straight.pixels[0], 255, "곱하지 않으면 색은 그대로다")

        let premultiplied = try WGPUImageDecoder.decode(png, options: .init(premultiplyAlpha: true))
        XCTAssertEqual(Int(premultiplied.pixels[0]), 128, accuracy: 2)
        XCTAssertTrue(premultiplied.premultiplied)
    }

    func test_resize로_크기를_바꾼다() throws {
        let png = try makePNG(width: 8, height: 8, top: (255, 0, 0, 255), bottom: (255, 0, 0, 255))
        let bitmap = try WGPUImageDecoder.decode(png, options: .init(resize: (width: 4, height: 2)))

        XCTAssertEqual(bitmap.width, 4)
        XCTAssertEqual(bitmap.height, 2)
        XCTAssertEqual(bitmap.pixels.count, 4 * 2 * 4)
    }

    /// 깨진 데이터는 **오류로** 온다 — 조용히 빈 이미지를 만들면 화면이 검게 나오고 원인은 사라진다.
    func test_디코딩할_수_없는_데이터는_오류다() {
        XCTAssertThrowsError(try WGPUImageDecoder.decode(Data([0x00, 0x01, 0x02, 0x03]))) { error in
            XCTAssertTrue("\(error)".contains("디코딩"), "\(error)")
        }
        XCTAssertThrowsError(try WGPUImageDecoder.decode(Data()))
    }

    func test_decodeImage가_핸들에_등록하고_크기를_돌려준다() throws {
        let png = try makePNG(width: 8, height: 4, top: (0, 255, 0, 255), bottom: (0, 255, 0, 255))
        let result = try decode(handle: 40, data: png)

        XCTAssertEqual(result["ok"] as? Bool, true, "\(result)")
        XCTAssertEqual(result["width"] as? Int, 8)
        XCTAssertEqual(result["height"] as? Int, 4)
    }

    func test_이미지_바이트도_이름도_없으면_거부한다() throws {
        let expectation = expectation(description: "decodeImage")
        var payload: [String: Any] = [:]
        harness.runtime.decodeImage(
            handle: 40, data: nil, name: nil, options: .init(), provider: nil
        ) { payload = $0; expectation.fulfill() }
        wait(for: [expectation], timeout: 5)

        XCTAssertEqual(payload["ok"] as? Bool, false)
    }

    // MARK: - 텍스처로 올리기

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
        // 텍스처 좌표는 위가 0이다 — y를 뒤집어 이미지 첫 행이 화면 위로 가게 한다.
        out.uv = vec2f(positions[index].x * 0.5 + 0.5, 0.5 - positions[index].y * 0.5);
        return out;
    }

    @fragment
    fn fs_main(in: Out) -> @location(0) vec4f {
        return textureSample(tex, samp, in.uv);
    }
    """

    /// 이미지를 텍스처로 올려 화면에 펼치는 커맨드 스트림.
    private func drawImage(bitmap: Int, texture size: (Int, Int), copy: [String: Any]) {
        var commands: [[String: Any]] = [
            ["op": "configureCanvas", "canvas": "test", "format": "rgba8unorm"],
            ["op": "createShaderModule", "id": 1, "code": Self.shader],
            ["op": "createTexture", "id": 2, "size": ["width": size.0, "height": size.1],
             "format": "rgba8unorm",
             "usage": TestUsage.textureBinding | TestUsage.textureCopyDst],
        ]
        commands.append(copy)
        commands.append(contentsOf: [
            ["op": "createTextureView", "id": 3, "texture": 2],
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
        _ = bitmap
        harness.executeExpectingSuccess(commands)
    }

    func test_디코딩한_이미지를_텍스처로_올려_샘플링한다() throws {
        let png = try makePNG(width: 4, height: 4, top: (255, 0, 0, 255), bottom: (0, 0, 255, 255))
        try decode(handle: 40, data: png)

        drawImage(bitmap: 40, texture: (4, 4), copy: [
            "op": "copyExternalImageToTexture",
            "source": ["source": 40],
            "destination": ["texture": 2],
            "copySize": ["width": 4, "height": 4],
        ])

        // 화면 위쪽은 이미지 첫 행(빨강), 아래쪽은 마지막 행(파랑)이다.
        try harness.assertPixel(x: 32, y: 8, equals: (255, 0, 0, 255), "위 절반")
        try harness.assertPixel(x: 32, y: 56, equals: (0, 0, 255, 255), "아래 절반")
    }

    /// 소스 일부만 잘라 올린다 — 스트라이드를 이미지 폭 그대로 쓰면 여기서 색이 밀린다.
    func test_이미지의_일부만_잘라_올린다() throws {
        let png = try makePNG(width: 8, height: 8, top: (255, 0, 0, 255), bottom: (0, 0, 255, 255))
        try decode(handle: 40, data: png)

        drawImage(bitmap: 40, texture: (4, 4), copy: [
            "op": "copyExternalImageToTexture",
            "source": ["source": 40, "origin": ["x": 2, "y": 4]],   // 아래 절반(파랑)에서 4x4
            "destination": ["texture": 2],
            "copySize": ["width": 4, "height": 4],
        ])

        try harness.assertPixel(x: 32, y: 32, equals: (0, 0, 255, 255), "잘라낸 영역은 전부 파랑")
    }

    /// 명세 `GPUCopyExternalImageSourceInfo.flipY` — **복사 시점** 뒤집기.
    ///
    /// `createImageBitmap`의 flipY와는 다른 자리다. 웹 라이브러리는 이쪽을 쓴다
    /// (three.js의 `Texture.flipY`가 기본 true라, 무시하면 텍스처가 조용히 뒤집힌다).
    func test_복사_시점_flipY가_위아래를_뒤집는다() throws {
        let png = try makePNG(width: 4, height: 4, top: (255, 0, 0, 255), bottom: (0, 0, 255, 255))
        try decode(handle: 40, data: png)

        drawImage(bitmap: 40, texture: (4, 4), copy: [
            "op": "copyExternalImageToTexture",
            "source": ["source": 40, "flipY": true],
            "destination": ["texture": 2],
            "copySize": ["width": 4, "height": 4],
        ])

        // 뒤집었으니 화면 위쪽이 이미지의 **마지막** 행(파랑)이다.
        try harness.assertPixel(x: 32, y: 8, equals: (0, 0, 255, 255), "위 절반")
        try harness.assertPixel(x: 32, y: 56, equals: (255, 0, 0, 255), "아래 절반")
    }

    /// 디코딩 시점과 복사 시점을 **둘 다** 뒤집으면 제자리로 돌아온다 — 두 옵션이
    /// 서로 다른 층이라는 증거다.
    func test_디코딩과_복사에서_각각_뒤집으면_제자리다() throws {
        let png = try makePNG(width: 4, height: 4, top: (255, 0, 0, 255), bottom: (0, 0, 255, 255))
        try decode(handle: 40, data: png, options: .init(flipY: true))

        drawImage(bitmap: 40, texture: (4, 4), copy: [
            "op": "copyExternalImageToTexture",
            "source": ["source": 40, "flipY": true],
            "destination": ["texture": 2],
        ])

        try harness.assertPixel(x: 32, y: 8, equals: (255, 0, 0, 255), "원본 그대로")
        try harness.assertPixel(x: 32, y: 56, equals: (0, 0, 255, 255))
    }

    /// 부분 복사와 flipY가 함께 와도 **잘라낸 영역 안에서만** 뒤집힌다.
    func test_부분_복사에도_flipY가_영역_안에서_적용된다() throws {
        // 위 2행 빨강, 아래 2행 파랑. (0,1)에서 4x2를 뜨면 빨강 1행 + 파랑 1행이다.
        let png = try makePNG(width: 4, height: 4, top: (255, 0, 0, 255), bottom: (0, 0, 255, 255))
        try decode(handle: 40, data: png)

        drawImage(bitmap: 40, texture: (4, 2), copy: [
            "op": "copyExternalImageToTexture",
            "source": ["source": 40, "origin": ["x": 0, "y": 1], "flipY": true],
            "destination": ["texture": 2],
            "copySize": ["width": 4, "height": 2],
        ])

        // 원본 1행(빨강)·2행(파랑) → 뒤집으면 파랑이 먼저다.
        try harness.assertPixel(x: 32, y: 8, equals: (0, 0, 255, 255), "위")
        try harness.assertPixel(x: 32, y: 56, equals: (255, 0, 0, 255), "아래")
    }

    func test_copySize를_생략하면_이미지_전체다() throws {
        let png = try makePNG(width: 4, height: 4, top: (255, 0, 0, 255), bottom: (255, 0, 0, 255))
        try decode(handle: 40, data: png)

        drawImage(bitmap: 40, texture: (4, 4), copy: [
            "op": "copyExternalImageToTexture",
            "source": ["source": 40],
            "destination": ["texture": 2],
        ])

        try harness.assertPixel(x: 32, y: 32, equals: (255, 0, 0, 255))
    }

    func test_이미지를_넘는_복사를_거부한다() throws {
        let png = try makePNG(width: 4, height: 4, top: (255, 0, 0, 255), bottom: (255, 0, 0, 255))
        try decode(handle: 40, data: png)

        let result = harness.execute([
            ["op": "createTexture", "id": 2, "size": ["width": 8, "height": 8],
             "format": "rgba8unorm", "usage": TestUsage.textureBinding | TestUsage.textureCopyDst],
            ["op": "copyExternalImageToTexture",
             "source": ["source": 40],
             "destination": ["texture": 2],
             "copySize": ["width": 8, "height": 8]],
        ])
        XCTAssertEqual(result["ok"] as? Bool, false)
        XCTAssertTrue(harness.describeErrors(result).contains("이미지를 넘는다"), harness.describeErrors(result))
    }

    /// 압축 텍스처로는 올릴 수 없다 — GPU에 블록 인코더가 없다.
    func test_압축_텍스처로는_올릴_수_없다() throws {
        try XCTSkipUnless(
            WGPUDeviceCapability.supportsCompression(.astc4x4Unorm, on: harness.context!.device),
            "이 기기는 ASTC를 지원하지 않는다"
        )
        let png = try makePNG(width: 4, height: 4, top: (255, 0, 0, 255), bottom: (255, 0, 0, 255))
        try decode(handle: 40, data: png)

        let result = harness.execute([
            ["op": "createTexture", "id": 2, "size": ["width": 4, "height": 4],
             "format": "astc-4x4-unorm", "usage": TestUsage.textureBinding | TestUsage.textureCopyDst],
            ["op": "copyExternalImageToTexture",
             "source": ["source": 40], "destination": ["texture": 2]],
        ])
        XCTAssertEqual(result["ok"] as? Bool, false)
        XCTAssertTrue(harness.describeErrors(result).contains("압축"), harness.describeErrors(result))
    }

    /// 4바이트가 아닌 포맷은 조용히 어긋나느니 거부한다.
    func test_16비트_포맷은_거부한다() throws {
        let png = try makePNG(width: 4, height: 4, top: (255, 0, 0, 255), bottom: (255, 0, 0, 255))
        try decode(handle: 40, data: png)

        let result = harness.execute([
            ["op": "createTexture", "id": 2, "size": ["width": 4, "height": 4],
             "format": "rgba16float", "usage": TestUsage.textureBinding | TestUsage.textureCopyDst],
            ["op": "copyExternalImageToTexture",
             "source": ["source": 40], "destination": ["texture": 2]],
        ])
        XCTAssertEqual(result["ok"] as? Bool, false)
        XCTAssertTrue(harness.describeErrors(result).contains("4바이트"), harness.describeErrors(result))
    }

    func test_없는_이미지_핸들은_분명한_오류다() throws {
        let result = harness.execute([
            ["op": "createTexture", "id": 2, "size": ["width": 4, "height": 4],
             "format": "rgba8unorm", "usage": TestUsage.textureBinding | TestUsage.textureCopyDst],
            ["op": "copyExternalImageToTexture",
             "source": ["source": 99], "destination": ["texture": 2]],
        ])
        XCTAssertEqual(result["ok"] as? Bool, false)
        XCTAssertTrue(harness.describeErrors(result).contains("ImageBitmap"), harness.describeErrors(result))
    }

    /// `destroy` op으로 네이티브 픽셀이 실제로 사라지는지 (JS `bitmap.close()`).
    func test_destroy가_이미지를_지운다() throws {
        let png = try makePNG(width: 4, height: 4, top: (255, 0, 0, 255), bottom: (255, 0, 0, 255))
        try decode(handle: 40, data: png)

        let result = harness.execute([
            ["op": "destroy", "id": 40],
            ["op": "createTexture", "id": 2, "size": ["width": 4, "height": 4],
             "format": "rgba8unorm", "usage": TestUsage.textureBinding | TestUsage.textureCopyDst],
            ["op": "copyExternalImageToTexture",
             "source": ["source": 40], "destination": ["texture": 2]],
        ])
        XCTAssertEqual(result["ok"] as? Bool, false)
    }
}
