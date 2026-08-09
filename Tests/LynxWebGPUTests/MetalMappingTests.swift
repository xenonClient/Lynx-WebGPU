import XCTest
import Metal
import LynxWebGPUCore
@testable import LynxWebGPU

/// WebGPU enum → Metal enum mapping. **No GPU needed.**
///
/// Carrying the combinatorial explosion through GPU tests would be slow and gain nothing. Value
/// correspondence is checked exhaustively here; the GPU tests check only "does that value actually
/// change the render result" on representative combinations.
final class MetalMappingTests: XCTestCase {
    /// Whether **every** texture format has a Metal counterpart — adding a case and forgetting the mapping is caught here.
    ///
    /// Forget one and using that format raises `unsupported`, while its presence in the enum makes you
    /// believe it is supported. Running exhaustively is the point.
    func test_everyTextureFormatHasAMetalCounterpart() throws {
        for format in WGPUTextureFormat.allCases {
            let metal = try WGPUMetalMapping.pixelFormat(format)
            XCTAssertNotEqual(metal, .invalid, "'\(format.rawValue)' has no Metal counterpart")
        }
    }

    /// Whether the reverse mapping recovers **every** format — back when the table was written by hand
    /// it held only the few used by canvases, leaving `stencil8` and `rgba8snorm` silently nil.
    func test_everyTextureFormatSurvivesTheReverseMapping() throws {
        for format in WGPUTextureFormat.allCases {
            let metal = try WGPUMetalMapping.pixelFormat(format)
            XCTAssertNotNil(
                WGPUMetalMapping.textureFormat(from: metal),
                "cannot recover '\(format.rawValue)'"
            )
        }
    }

    /// Where formats collapse (`depth24plus` and `depth32float` are both `.depth32Float`), the one that
    /// **states the precision honestly** must come out. A weaker-sounding name deceives the reader.
    func test_depthFormatsCollapsingOntoOneMetalFormatReturnTheHigherPrecisionName() {
        XCTAssertEqual(WGPUMetalMapping.textureFormat(from: .depth32Float), .depth32float)
        XCTAssertEqual(WGPUMetalMapping.textureFormat(from: .depth32Float_stencil8), .depth32floatStencil8)
    }

    /// Whether bytes per pixel match the size Metal actually uses — a mismatch makes `writeTexture`'s
    /// default `bytesPerRow` wrong and uploads misaligned rows **with no error**.
    func test_packed32BitFormatsAre4BytesPerPixel() {
        for format: WGPUTextureFormat in [.rgb10a2unorm, .rgb10a2uint, .rg11b10ufloat, .rgb9e5ufloat] {
            XCTAssertEqual(format.bytesPerPixel, 4, "\(format.rawValue)")
        }
    }

    func test_everyStencilOpMapsToMetal() {
        let expected: [WGPUStencilOperation: MTLStencilOperation] = [
            .keep: .keep,
            .zero: .zero,
            .replace: .replace,
            .invert: .invert,
            .incrementClamp: .incrementClamp,
            .decrementClamp: .decrementClamp,
            .incrementWrap: .incrementWrap,
            .decrementWrap: .decrementWrap,
        ]
        // Running over CaseIterable is the point — add a case and this fails until the table is filled.
        for operation in WGPUStencilOperation.allCases {
            guard let want = expected[operation] else {
                XCTFail("the new stencil op '\(operation.rawValue)' has no Metal counterpart in this table")
                continue
            }
            XCTAssertEqual(WGPUMetalMapping.stencilOperation(operation), want, operation.rawValue)
        }
    }

    func test_everyCompareFunctionMapsToMetal() {
        let expected: [WGPUCompareFunction: MTLCompareFunction] = [
            .never: .never,
            .less: .less,
            .equal: .equal,
            .lessEqual: .lessEqual,
            .greater: .greater,
            .notEqual: .notEqual,
            .greaterEqual: .greaterEqual,
            .always: .always,
        ]
        for function in WGPUCompareFunction.allCases {
            guard let want = expected[function] else {
                XCTFail("the new compare function '\(function.rawValue)' has no Metal counterpart in this table")
                continue
            }
            XCTAssertEqual(WGPUMetalMapping.compareFunction(function), want, function.rawValue)
        }
    }

    /// Whether the four ops each land in their own slot — swapping `failOp` and `depthFailOp` would go
    /// unnoticed if they held the same value. So all four are given different values.
    func test_theStencilDescriptorPutsFourOpsInTheirSlots() {
        let descriptor = WGPUMetalMapping.stencilDescriptor(
            WGPUStencilFaceState(
                compare: .greater, failOp: .zero, depthFailOp: .invert, passOp: .replace
            ),
            readMask: 0x0F,
            writeMask: 0xF0
        )

        XCTAssertEqual(descriptor.stencilCompareFunction, .greater)
        XCTAssertEqual(descriptor.stencilFailureOperation, .zero)
        XCTAssertEqual(descriptor.depthFailureOperation, .invert)
        XCTAssertEqual(descriptor.depthStencilPassOperation, .replace)
        XCTAssertEqual(descriptor.readMask, 0x0F)
        XCTAssertEqual(descriptor.writeMask, 0xF0)
    }

    func test_theStencilMaskDefaultPassesAll32Bits() {
        let descriptor = WGPUMetalMapping.stencilDescriptor(
            WGPUStencilFaceState(), readMask: 0xFFFF_FFFF, writeMask: 0xFFFF_FFFF
        )
        XCTAssertEqual(descriptor.readMask, UInt32.max)
        XCTAssertEqual(descriptor.writeMask, UInt32.max)
    }
}
