import XCTest
@testable import LynxWebGPUCore

/// Interpreting a readback pixel block — verified purely, with no GPU.
///
/// What this catches is the situation where reading with the wrong format yields plausible values and no error.
final class WGPUPixelReadbackTests: XCTestCase {
    // MARK: - half → float

    func test_halfBitsExpandExactlyIntoFloat() {
        XCTAssertEqual(WGPUPixelReadback.float(fromHalf: 0x0000), 0)
        XCTAssertEqual(WGPUPixelReadback.float(fromHalf: 0x8000), -0.0)
        XCTAssertEqual(WGPUPixelReadback.float(fromHalf: 0x3C00), 1)
        XCTAssertEqual(WGPUPixelReadback.float(fromHalf: 0xBC00), -1)
        XCTAssertEqual(WGPUPixelReadback.float(fromHalf: 0x4100), 2.5)
        // The largest normal a half can hold — far beyond the SDR range.
        XCTAssertEqual(WGPUPixelReadback.float(fromHalf: 0x7BFF), 65504)
        XCTAssertEqual(WGPUPixelReadback.float(fromHalf: 0x0400), 0x1p-14, accuracy: 1e-12)
    }

    func test_halfSubnormalsAndInfinitySurvive() {
        // The smallest subnormal = 2^-24. It is normal in float, so the exponent must be shifted up.
        XCTAssertEqual(WGPUPixelReadback.float(fromHalf: 0x0001), 0x1p-24, accuracy: 1e-30)
        // The largest subnormal = 1023 * 2^-24.
        XCTAssertEqual(WGPUPixelReadback.float(fromHalf: 0x03FF), 1023 * 0x1p-24, accuracy: 1e-30)
        XCTAssertEqual(WGPUPixelReadback.float(fromHalf: 0x7C00), .infinity)
        XCTAssertEqual(WGPUPixelReadback.float(fromHalf: 0xFC00), -.infinity)
        XCTAssertTrue(WGPUPixelReadback.float(fromHalf: 0x7E00).isNaN)
    }

    // MARK: - Channel interpretation

    func test_rgba16floatReturnsValuesOutsideSDRUnchanged() throws {
        // (2.5, 0.5, -1.0, 1.0) — the first and third channels are why HDR readback exists.
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

    func test_sameBytesReadDifferentlyPerFormat() throws {
        // Reading one rgba16float pixel (8B) as rgba8unorm makes it look like two pixels —
        // exactly the confusion the old readPixels used to commit silently.
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

    func test_bgraComesBackReorderedIntoRGBA() throws {
        let readback = WGPUPixelReadback(
            data: Data([10, 20, 30, 255]), format: .bgra8unorm, width: 1, height: 1, bytesPerRow: 4
        )
        let color = try readback.rgba(x: 0, y: 0)
        XCTAssertEqual(color.x, 30.0 / 255, accuracy: 1e-6)
        XCTAssertEqual(color.y, 20.0 / 255, accuracy: 1e-6)
        XCTAssertEqual(color.z, 10.0 / 255, accuracy: 1e-6)
        XCTAssertEqual(color.w, 1, accuracy: 1e-6)
    }

    func test_missingChannelsFillRGBWith0AndAlphaWith1() throws {
        let readback = WGPUPixelReadback(
            data: Data([128, 64]), format: .rg8unorm, width: 1, height: 1, bytesPerRow: 2
        )
        let color = try readback.rgba(x: 0, y: 0)
        XCTAssertEqual(color.x, 128.0 / 255, accuracy: 1e-6)
        XCTAssertEqual(color.y, 64.0 / 255, accuracy: 1e-6)
        XCTAssertEqual(color.z, 0)
        XCTAssertEqual(color.w, 1)
    }

    func test_rowPaddingIsSkippedUsingBytesPerRow() throws {
        // 2×2 at 4B per pixel, but a 12B row stride (4B of padding per row).
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

    // MARK: - Errors

    func test_anOutOfRangeCoordinateIsAnError() {
        let readback = WGPUPixelReadback(
            data: Data(count: 16), format: .rgba8unorm, width: 2, height: 2, bytesPerRow: 8
        )
        XCTAssertThrowsError(try readback.rgba(x: 2, y: 0)) { error in
            XCTAssertEqual((error as? WGPUError)?.kind, .validation)
        }
        XCTAssertThrowsError(try readback.rgba(x: 0, y: -1))
    }

    func test_aFormatThatCannotExpandIntoChannelsDoesNotPassSilently() {
        // Packed and integer formats distort when expanded into normalized floats → throw, telling the caller to read data directly.
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

    func test_tooFewBytesIsAnErrorRatherThanAWrongValue() {
        let readback = WGPUPixelReadback(
            data: Data(count: 4), format: .rgba8unorm, width: 2, height: 2, bytesPerRow: 8
        )
        XCTAssertThrowsError(try readback.rgba(x: 0, y: 1))
    }
}
