import XCTest
import Metal
import LynxWebGPUCore
@testable import LynxWebGPU

/// The contract of `WGPUOffscreenSurface.readPixels` — which formats it reads, and where it refuses.
///
/// The old implementation assumed 4 bytes per pixel and, on an `rgba16float` surface, returned bytes
/// wrong in both length and interpretation **with no error**. What is pinned here is that an error
/// comes back instead of a wrong value.
final class OffscreenReadbackTests: XCTestCase {
    private var device: MTLDevice!

    override func setUpWithError() throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "no Metal device")
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
            // Packed 32-bit — the pixel is 4 bytes even though channel boundaries do not fall on bytes.
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

    func test_aDepthStencilSurfaceErrorsInsteadOfReading() throws {
        let queue = try XCTUnwrap(device.makeCommandQueue())

        for format: WGPUTextureFormat in [.depth32float, .depth16unorm, .depth32floatStencil8] {
            let surface = try surface(format: format)
            XCTAssertThrowsError(try surface.readPixels(queue: queue), format.rawValue) { error in
                let error = error as? WGPUError
                XCTAssertEqual(error?.kind, .validation)
                XCTAssertTrue(
                    error?.message.contains(format.rawValue) == true,
                    "the error message needs the format for the cause to be knowable: \(error?.message ?? "")"
                )
            }
        }
    }

    func test_itCannotBeReadBeforeConfigure() throws {
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let surface = WGPUOffscreenSurface(size: CGSize(width: 4, height: 4), device: device)
        XCTAssertThrowsError(try surface.readPixels(queue: queue)) { error in
            XCTAssertEqual((error as? WGPUError)?.kind, .validation)
        }
    }
}
