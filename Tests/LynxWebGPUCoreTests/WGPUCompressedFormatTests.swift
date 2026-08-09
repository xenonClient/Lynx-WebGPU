import XCTest
@testable import LynxWebGPUCore

/// The **size arithmetic** of block-compressed formats. No GPU needed.
///
/// Every place that is easy to get wrong with compressed formats is gathered here: misreading the
/// block size from the name, guessing 8/16 bytes per block wrongly, counting rows in pixels. All
/// three upload misaligned pixels **with no error**, so the arithmetic is pinned down as values.
final class WGPUCompressedFormatTests: XCTestCase {
    /// Every compressed format must be `isCompressed` and every uncompressed one must not —
    /// that one distinction decides the `bytesPerRow`/`blockRows` path.
    func test_compressionFlagMatchesTheName() {
        for format in WGPUTextureFormat.allCases {
            let looksCompressed = format.rawValue.hasPrefix("bc")
                || format.rawValue.hasPrefix("etc2-")
                || format.rawValue.hasPrefix("eac-")
                || format.rawValue.hasPrefix("astc-")
            XCTAssertEqual(format.isCompressed, looksCompressed, "\(format.rawValue)")
        }
    }

    func test_uncompressedBlocksAre1x1AndBlockSizeEqualsPixelSize() {
        for format in WGPUTextureFormat.allCases where !format.isCompressed {
            XCTAssertEqual(format.blockWidth, 1, "\(format.rawValue)")
            XCTAssertEqual(format.blockHeight, 1, "\(format.rawValue)")
            XCTAssertEqual(format.bytesPerBlock, format.bytesPerPixel, "\(format.rawValue)")
            // With a 1×1 block the row calculation must match the old formula (the existing path is unchanged).
            XCTAssertEqual(format.bytesPerRow(width: 7), 7 * format.bytesPerPixel, "\(format.rawValue)")
            XCTAssertEqual(format.blockRows(height: 7), 7, "\(format.rawValue)")
        }
    }

    func test_bcAndETC2AreAll4x4Blocks() {
        for format in WGPUTextureFormat.allCases
        where format.rawValue.hasPrefix("bc") || format.rawValue.hasPrefix("etc2-")
            || format.rawValue.hasPrefix("eac-") {
            XCTAssertEqual(format.blockSize.width, 4, "\(format.rawValue)")
            XCTAssertEqual(format.blockSize.height, 4, "\(format.rawValue)")
        }
    }

    /// ASTC carries its block size **inside the name**. A parsing mistake silently falls back to (1,1)
    /// and computes a compressed texture as if it were uncompressed — so we check all of them.
    func test_astcBlockSizesAreReadFromTheName() {
        let expected: [(String, Int, Int)] = [
            ("astc-4x4", 4, 4), ("astc-5x4", 5, 4), ("astc-5x5", 5, 5),
            ("astc-6x5", 6, 5), ("astc-6x6", 6, 6), ("astc-8x5", 8, 5),
            ("astc-8x6", 8, 6), ("astc-8x8", 8, 8), ("astc-10x5", 10, 5),
            ("astc-10x6", 10, 6), ("astc-10x8", 10, 8), ("astc-10x10", 10, 10),
            ("astc-12x10", 12, 10), ("astc-12x12", 12, 12),
        ]
        var covered = 0
        for format in WGPUTextureFormat.allCases where format.rawValue.hasPrefix("astc-") {
            guard let want = expected.first(where: { format.rawValue.hasPrefix($0.0 + "-") }) else {
                XCTFail("the block size of the new ASTC format '\(format.rawValue)' is not in this table")
                continue
            }
            XCTAssertEqual(format.blockSize.width, want.1, "\(format.rawValue)")
            XCTAssertEqual(format.blockSize.height, want.2, "\(format.rawValue)")
            // ASTC is 16 bytes per block regardless of block size — which is why a larger block compresses more.
            XCTAssertEqual(format.bytesPerBlock, 16, "\(format.rawValue)")
            covered += 1
        }
        XCTAssertEqual(covered, expected.count * 2, "every unorm/srgb pair must be present")
    }

    /// Bytes per block is either 8 or 16. Guessing one wrong shifts the data by half.
    func test_bytesPerBlockAreCorrectPerFamily() {
        let eightByte: Set<String> = [
            "bc1-rgba-unorm", "bc1-rgba-unorm-srgb", "bc4-r-unorm", "bc4-r-snorm",
            "etc2-rgb8unorm", "etc2-rgb8unorm-srgb", "etc2-rgb8a1unorm", "etc2-rgb8a1unorm-srgb",
            "eac-r11unorm", "eac-r11snorm",
        ]
        for format in WGPUTextureFormat.allCases where format.isCompressed {
            let want = eightByte.contains(format.rawValue) ? 8 : 16
            XCTAssertEqual(format.bytesPerBlock, want, "\(format.rawValue)")
        }
    }

    /// One row of a compressed texture counts **rounded up in blocks**. Rounding down drops the edge
    /// block entirely and shifts the data — a common mistake, so it is pinned down as values.
    func test_rowBytesAndBlockRowsRoundUp() {
        let bc1 = WGPUTextureFormat.bc1RGBAUnorm      // 4x4 blocks, 8 bytes
        XCTAssertEqual(bc1.bytesPerRow(width: 16), 4 * 8)
        XCTAssertEqual(bc1.bytesPerRow(width: 17), 5 * 8, "17 pixels need 5 blocks")
        XCTAssertEqual(bc1.bytesPerRow(width: 1), 8, "even 1 pixel uses a whole block")
        XCTAssertEqual(bc1.blockRows(height: 16), 4)
        XCTAssertEqual(bc1.blockRows(height: 17), 5)

        let astc = WGPUTextureFormat.astc6x5Unorm     // 6x5 blocks, 16 bytes — not square
        XCTAssertEqual(astc.bytesPerRow(width: 12), 2 * 16)
        XCTAssertEqual(astc.bytesPerRow(width: 13), 3 * 16)
        XCTAssertEqual(astc.blockRows(height: 10), 2)
        XCTAssertEqual(astc.blockRows(height: 11), 3, "the height divides by 5 — not 6")
    }

    /// One 4×4 BC1 texture is exactly one block (8B). Shift this value and `writeTexture` either
    /// rejects it as "not enough data" or uploads short data to the GPU.
    func test_aTinyTextureIsOneWholeBlock() {
        let bc1 = WGPUTextureFormat.bc1RGBAUnorm
        XCTAssertEqual(bc1.bytesPerRow(width: 4) * bc1.blockRows(height: 4), 8)
        // The mip chain's tail (2x2, 1x1) is one block each too — the size must never become 0.
        XCTAssertEqual(bc1.bytesPerRow(width: 2) * bc1.blockRows(height: 2), 8)
        XCTAssertEqual(bc1.bytesPerRow(width: 1) * bc1.blockRows(height: 1), 8)
    }
}
