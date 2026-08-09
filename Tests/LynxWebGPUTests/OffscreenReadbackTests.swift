import XCTest
import Metal
import LynxWebGPUCore
@testable import LynxWebGPU

/// `WGPUOffscreenSurface.readPixels`의 계약 — 어떤 포맷을 읽고, 어디서 거부하는가.
///
/// 예전 구현은 픽셀당 4바이트를 가정해서, `rgba16float` 표면에서 **오류 없이** 길이도 해석도
/// 틀린 바이트를 돌려줬다. 여기서 못 박는 것은 "틀린 값 대신 오류가 난다"는 쪽이다.
final class OffscreenReadbackTests: XCTestCase {
    private var device: MTLDevice!

    override func setUpWithError() throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "Metal 디바이스 없음")
        device = MTLCreateSystemDefaultDevice()
    }

    override func tearDown() {
        device = nil
        super.tearDown()
    }

    private func surface(format: WGPUTextureFormat, width: Int = 4, height: Int = 3) throws
        -> WGPUOffscreenSurface
    {
        let surface = WGPUOffscreenSurface(
            size: CGSize(width: width, height: height), device: device
        )
        let configuration = try WGPUCanvasConfiguration(
            from: WGPUValueReader(["canvas": "offscreen", "format": format.rawValue])
        )
        try surface.configure(configuration, device: device)
        return surface
    }

    func test_readbackLengthAndRowStrideFollowTheFormat() throws {
        let queue = try XCTUnwrap(device.makeCommandQueue())

        for (format, bytesPerPixel) in [
            (WGPUTextureFormat.rgba8unorm, 4),
            (.bgra8unorm, 4),
            (.rgba16float, 8),
            (.rgba32float, 16),
            (.r8unorm, 1),
            // 팩된 32비트 — 채널 경계가 바이트에 맞지 않아도 픽셀은 4바이트다.
            (.rgb10a2uint, 4),
            (.rgb9e5ufloat, 4),
        ] as [(WGPUTextureFormat, Int)] {
            let readback = try surface(format: format).readPixels(queue: queue)
            XCTAssertEqual(readback.format, format)
            XCTAssertEqual(readback.width, 4)
            XCTAssertEqual(readback.height, 3)
            XCTAssertEqual(readback.bytesPerRow, 4 * bytesPerPixel, "\(format.rawValue)")
            XCTAssertEqual(readback.data.count, 4 * 3 * bytesPerPixel, "\(format.rawValue)")
        }
    }

    func test_depth_stencil_표면은_읽지_않고_오류를_낸다() throws {
        let queue = try XCTUnwrap(device.makeCommandQueue())

        for format: WGPUTextureFormat in [.depth32float, .depth16unorm, .depth32floatStencil8] {
            let surface = try surface(format: format)
            XCTAssertThrowsError(try surface.readPixels(queue: queue), format.rawValue) { error in
                let error = error as? WGPUError
                XCTAssertEqual(error?.kind, .validation)
                XCTAssertTrue(
                    error?.message.contains(format.rawValue) == true,
                    "오류 메시지에 포맷이 있어야 원인을 알 수 있다: \(error?.message ?? "")"
                )
            }
        }
    }

    func test_configure_전에는_읽을_수_없다() throws {
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let surface = WGPUOffscreenSurface(size: CGSize(width: 4, height: 4), device: device)
        XCTAssertThrowsError(try surface.readPixels(queue: queue)) { error in
            XCTAssertEqual((error as? WGPUError)?.kind, .validation)
        }
    }
}
