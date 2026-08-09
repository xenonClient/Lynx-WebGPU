import XCTest
@testable import LynxWebGPUCore

/// 블록 압축 포맷의 **크기 산수**. GPU가 필요 없다.
///
/// 압축 포맷에서 틀리기 쉬운 자리는 전부 여기 모여 있다: 블록 크기를 이름에서 잘못 읽거나,
/// 블록당 바이트를 8/16으로 잘못 짚거나, 행 수를 픽셀로 세는 것. 셋 다 **오류 없이**
/// 어긋난 픽셀을 올리는 종류의 버그라, 산수를 값으로 못 박아 둔다.
final class WGPUCompressedFormatTests: XCTestCase {
    /// 압축 포맷은 전부 `isCompressed`이고 비압축은 전부 아니어야 한다 —
    /// 이 구분 하나로 `bytesPerRow`/`blockRows`의 경로가 갈린다.
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
            // 블록이 1×1이면 행 계산은 예전 식과 같아야 한다 (기존 경로가 안 바뀐다).
            XCTAssertEqual(format.bytesPerRow(width: 7), 7 * format.bytesPerPixel, "\(format.rawValue)")
            XCTAssertEqual(format.blockRows(height: 7), 7, "\(format.rawValue)")
        }
    }

    func test_BC와_ETC2는_전부_4x4_블록이다() {
        for format in WGPUTextureFormat.allCases
        where format.rawValue.hasPrefix("bc") || format.rawValue.hasPrefix("etc2-")
            || format.rawValue.hasPrefix("eac-") {
            XCTAssertEqual(format.blockSize.width, 4, "\(format.rawValue)")
            XCTAssertEqual(format.blockSize.height, 4, "\(format.rawValue)")
        }
    }

    /// ASTC는 블록 크기가 **이름 안에** 있다. 파싱이 틀리면 조용히 (1,1)로 떨어져
    /// 압축 텍스처를 비압축처럼 계산한다 — 전수로 확인한다.
    func test_ASTC_블록_크기는_이름에서_읽는다() {
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
                XCTFail("새 ASTC 포맷 '\(format.rawValue)'의 블록 크기가 이 표에 없다")
                continue
            }
            XCTAssertEqual(format.blockSize.width, want.1, "\(format.rawValue)")
            XCTAssertEqual(format.blockSize.height, want.2, "\(format.rawValue)")
            // ASTC는 블록 크기와 무관하게 블록당 16바이트다 — 그래서 큰 블록일수록 압축률이 높다.
            XCTAssertEqual(format.bytesPerBlock, 16, "\(format.rawValue)")
            covered += 1
        }
        XCTAssertEqual(covered, expected.count * 2, "unorm/srgb 쌍이 전부 있어야 한다")
    }

    /// 블록당 바이트는 8이거나 16이다. 여기서 하나를 잘못 짚으면 데이터가 절반씩 밀린다.
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

    /// 압축 텍스처의 한 행은 **블록 수로 올림**해서 센다. 내림하면 가장자리 블록이
    /// 통째로 빠져 데이터가 밀린다 — 실제로 자주 나는 실수라 값으로 못 박는다.
    func test_rowBytesAndBlockRowsRoundUp() {
        let bc1 = WGPUTextureFormat.bc1RGBAUnorm      // 4x4 블록, 8바이트
        XCTAssertEqual(bc1.bytesPerRow(width: 16), 4 * 8)
        XCTAssertEqual(bc1.bytesPerRow(width: 17), 5 * 8, "17픽셀은 블록 5개가 필요하다")
        XCTAssertEqual(bc1.bytesPerRow(width: 1), 8, "1픽셀도 블록 하나를 다 쓴다")
        XCTAssertEqual(bc1.blockRows(height: 16), 4)
        XCTAssertEqual(bc1.blockRows(height: 17), 5)

        let astc = WGPUTextureFormat.astc6x5Unorm     // 6x5 블록, 16바이트 — 정사각이 아니다
        XCTAssertEqual(astc.bytesPerRow(width: 12), 2 * 16)
        XCTAssertEqual(astc.bytesPerRow(width: 13), 3 * 16)
        XCTAssertEqual(astc.blockRows(height: 10), 2)
        XCTAssertEqual(astc.blockRows(height: 11), 3, "높이는 5로 나눈다 — 6이 아니다")
    }

    /// 4×4 BC1 텍스처 하나는 정확히 블록 하나(8B)다. 이 값이 밀리면 `writeTexture`가
    /// "데이터가 부족하다"로 거부하거나, 반대로 모자란 데이터를 GPU에 올린다.
    func test_aTinyTextureIsOneWholeBlock() {
        let bc1 = WGPUTextureFormat.bc1RGBAUnorm
        XCTAssertEqual(bc1.bytesPerRow(width: 4) * bc1.blockRows(height: 4), 8)
        // 밉 사슬의 꼬리(2x2, 1x1)도 블록 하나씩이다 — 크기가 0이 되면 안 된다.
        XCTAssertEqual(bc1.bytesPerRow(width: 2) * bc1.blockRows(height: 2), 8)
        XCTAssertEqual(bc1.bytesPerRow(width: 1) * bc1.blockRows(height: 1), 8)
    }
}
