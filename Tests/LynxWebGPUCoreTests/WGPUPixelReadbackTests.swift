import XCTest
@testable import LynxWebGPUCore

/// 되읽은 픽셀 블록의 해석 — GPU 없이 순수하게 검증한다.
///
/// 여기서 잡으려는 것은 "포맷을 잘못 알고 읽어도 오류 없이 그럴듯한 값이 나오는" 상황이다.
final class WGPUPixelReadbackTests: XCTestCase {
    // MARK: - half → float

    func test_half비트가_float으로_정확히_펴진다() {
        XCTAssertEqual(WGPUPixelReadback.float(fromHalf: 0x0000), 0)
        XCTAssertEqual(WGPUPixelReadback.float(fromHalf: 0x8000), -0.0)
        XCTAssertEqual(WGPUPixelReadback.float(fromHalf: 0x3C00), 1)
        XCTAssertEqual(WGPUPixelReadback.float(fromHalf: 0xBC00), -1)
        XCTAssertEqual(WGPUPixelReadback.float(fromHalf: 0x4100), 2.5)
        // half가 담을 수 있는 최대 정규수 — SDR 범위를 한참 넘는다.
        XCTAssertEqual(WGPUPixelReadback.float(fromHalf: 0x7BFF), 65504)
        XCTAssertEqual(WGPUPixelReadback.float(fromHalf: 0x0400), 0x1p-14, accuracy: 1e-12)
    }

    func test_half의_서브노멀과_무한대도_보존된다() {
        // 가장 작은 서브노멀 = 2^-24. float에서는 정규수라 지수를 밀어 올려야 한다.
        XCTAssertEqual(WGPUPixelReadback.float(fromHalf: 0x0001), 0x1p-24, accuracy: 1e-30)
        // 가장 큰 서브노멀 = 1023 * 2^-24.
        XCTAssertEqual(WGPUPixelReadback.float(fromHalf: 0x03FF), 1023 * 0x1p-24, accuracy: 1e-30)
        XCTAssertEqual(WGPUPixelReadback.float(fromHalf: 0x7C00), .infinity)
        XCTAssertEqual(WGPUPixelReadback.float(fromHalf: 0xFC00), -.infinity)
        XCTAssertTrue(WGPUPixelReadback.float(fromHalf: 0x7E00).isNaN)
    }

    // MARK: - 채널 해석

    func test_rgba16float은_SDR범위_밖의_값을_그대로_돌려준다() throws {
        // (2.5, 0.5, -1.0, 1.0) — HDR 되읽기가 존재하는 이유가 첫 채널과 셋째 채널이다.
        let pixel: [UInt16] = [0x4100, 0x3800, 0xBC00, 0x3C00]
        let readback = WGPUPixelReadback(
            data: pixel.withUnsafeBufferPointer { Data(buffer: $0) },
            format: .rgba16float, width: 1, height: 1, bytesPerRow: 8
        )

        let color = try readback.rgba(x: 0, y: 0)
        XCTAssertEqual(color.x, 2.5)
        XCTAssertEqual(color.y, 0.5)
        XCTAssertEqual(color.z, -1)
        XCTAssertEqual(color.w, 1)
    }

    func test_같은_바이트라도_포맷에_따라_다르게_읽힌다() throws {
        // rgba16float 한 픽셀(8B)을 rgba8unorm으로 읽으면 두 픽셀로 보인다 —
        // 예전 readPixels가 조용히 저지르던 착각이 바로 이것이다.
        let bytes = Data([0x00, 0x41, 0x00, 0x38, 0x00, 0xBC, 0x00, 0x3C])

        let asHalf = WGPUPixelReadback(
            data: bytes, format: .rgba16float, width: 1, height: 1, bytesPerRow: 8
        )
        XCTAssertEqual(try asHalf.rgba(x: 0, y: 0).x, 2.5)

        let asBytes = WGPUPixelReadback(
            data: bytes, format: .rgba8unorm, width: 2, height: 1, bytesPerRow: 8
        )
        XCTAssertEqual(try asBytes.rgba(x: 0, y: 0).x, 0, accuracy: 1e-6)
        XCTAssertEqual(try asBytes.rgba(x: 1, y: 0).x, 0, accuracy: 1e-6)
    }

    func test_bgra는_RGBA순서로_바꿔서_돌려준다() throws {
        let readback = WGPUPixelReadback(
            data: Data([10, 20, 30, 255]), format: .bgra8unorm, width: 1, height: 1, bytesPerRow: 4
        )
        let color = try readback.rgba(x: 0, y: 0)
        XCTAssertEqual(color.x, 30.0 / 255, accuracy: 1e-6)
        XCTAssertEqual(color.y, 20.0 / 255, accuracy: 1e-6)
        XCTAssertEqual(color.z, 10.0 / 255, accuracy: 1e-6)
        XCTAssertEqual(color.w, 1, accuracy: 1e-6)
    }

    func test_없는_채널은_RGB가0_알파가1로_채워진다() throws {
        let readback = WGPUPixelReadback(
            data: Data([128, 64]), format: .rg8unorm, width: 1, height: 1, bytesPerRow: 2
        )
        let color = try readback.rgba(x: 0, y: 0)
        XCTAssertEqual(color.x, 128.0 / 255, accuracy: 1e-6)
        XCTAssertEqual(color.y, 64.0 / 255, accuracy: 1e-6)
        XCTAssertEqual(color.z, 0)
        XCTAssertEqual(color.w, 1)
    }

    func test_행_패딩이_있으면_bytesPerRow를_따라_건너뛴다() throws {
        // 2×2, 픽셀당 4B인데 행 간격은 12B (행마다 4B 패딩).
        var data = Data()
        for row in 0..<2 {
            for column in 0..<2 {
                data.append(contentsOf: [UInt8(row * 10 + column), 0, 0, 255])
            }
            data.append(contentsOf: [0xFF, 0xFF, 0xFF, 0xFF])
        }
        let readback = WGPUPixelReadback(
            data: data, format: .rgba8unorm, width: 2, height: 2, bytesPerRow: 12
        )
        XCTAssertEqual(try readback.rgba(x: 1, y: 1).x, 11.0 / 255, accuracy: 1e-6)
    }

    // MARK: - 오류

    func test_범위_밖_좌표는_오류다() {
        let readback = WGPUPixelReadback(
            data: Data(count: 16), format: .rgba8unorm, width: 2, height: 2, bytesPerRow: 8
        )
        XCTAssertThrowsError(try readback.rgba(x: 2, y: 0)) { error in
            XCTAssertEqual((error as? WGPUError)?.kind, .validation)
        }
        XCTAssertThrowsError(try readback.rgba(x: 0, y: -1))
    }

    func test_채널로_풀_수_없는_포맷은_조용히_넘어가지_않는다() {
        // 팩된 포맷·정수 포맷은 정규화 float으로 펴면 값이 왜곡된다 → data를 직접 읽으라고 던진다.
        for format: WGPUTextureFormat in [
            .rgb10a2unorm, .rgb10a2uint, .rg11b10ufloat, .rgb9e5ufloat, .rgba8uint, .rgba16uint,
        ] {
            let readback = WGPUPixelReadback(
                data: Data(count: format.bytesPerPixel),
                format: format, width: 1, height: 1, bytesPerRow: format.bytesPerPixel
            )
            XCTAssertThrowsError(try readback.rgba(x: 0, y: 0), "\(format.rawValue)") { error in
                XCTAssertEqual((error as? WGPUError)?.kind, .validation)
            }
        }
    }

    func test_바이트가_모자라면_잘못된_값_대신_오류다() {
        let readback = WGPUPixelReadback(
            data: Data(count: 4), format: .rgba8unorm, width: 2, height: 2, bytesPerRow: 8
        )
        XCTAssertThrowsError(try readback.rgba(x: 0, y: 1))
    }
}
