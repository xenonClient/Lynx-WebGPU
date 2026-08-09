import Foundation

// WebGPU spec enums arrive from JS as **strings** ("rgba8unorm", "triangle-list", …).
// Raw values keep the spec spelling exactly — renaming one here breaks JS code.
// Converting to Metal types is the backend's job (LynxWebGPU); Core stays Metal-free.

public enum WGPUTextureFormat: String, CaseIterable, Sendable {
    // 8-bit
    case r8unorm, r8snorm, r8uint, r8sint
    // 16-bit
    case r16uint, r16sint, r16float
    case rg8unorm, rg8snorm, rg8uint, rg8sint
    // 32-bit
    case r32uint, r32sint, r32float
    case rg16uint, rg16sint, rg16float
    case rgba8unorm
    case rgba8unormSRGB = "rgba8unorm-srgb"
    case rgba8snorm, rgba8uint, rgba8sint
    case bgra8unorm
    case bgra8unormSRGB = "bgra8unorm-srgb"
    case rgb10a2unorm
    case rgb10a2uint
    case rg11b10ufloat
    /// Shared-exponent HDR — 9 mantissa bits per channel plus a 5-bit shared exponent. Cannot be a
    /// render target; used as a **read-only HDR source** (same dynamic range at half the size of `rgba16float`).
    case rgb9e5ufloat
    // 64-bit
    case rg32uint, rg32sint, rg32float
    case rgba16uint, rgba16sint, rgba16float
    // 128-bit
    case rgba32uint, rgba32sint, rgba32float
    // depth / stencil
    case stencil8
    case depth16unorm
    case depth24plus
    case depth24plusStencil8 = "depth24plus-stencil8"
    case depth32float
    case depth32floatStencil8 = "depth32float-stencil8"

    // --- Block compression (BC / ETC2 / ASTC) --------------------------------
    //
    // Stored in **blocks**, not pixels. Compute sizes with `blockWidth`/`blockHeight`/
    // `bytesPerBlock` — `bytesPerPixel` is meaningless for these formats.
    //
    // All three are **optional features** in the spec (`texture-compression-bc` / `-etc2` / `-astc`).
    // Check device support through `adapter.features`.

    // BC (desktop) — 4x4 blocks
    case bc1RGBAUnorm = "bc1-rgba-unorm"
    case bc1RGBAUnormSRGB = "bc1-rgba-unorm-srgb"
    case bc2RGBAUnorm = "bc2-rgba-unorm"
    case bc2RGBAUnormSRGB = "bc2-rgba-unorm-srgb"
    case bc3RGBAUnorm = "bc3-rgba-unorm"
    case bc3RGBAUnormSRGB = "bc3-rgba-unorm-srgb"
    case bc4RUnorm = "bc4-r-unorm"
    case bc4RSnorm = "bc4-r-snorm"
    case bc5RGUnorm = "bc5-rg-unorm"
    case bc5RGSnorm = "bc5-rg-snorm"
    case bc6hRGBUfloat = "bc6h-rgb-ufloat"
    case bc6hRGBFloat = "bc6h-rgb-float"
    case bc7RGBAUnorm = "bc7-rgba-unorm"
    case bc7RGBAUnormSRGB = "bc7-rgba-unorm-srgb"

    // ETC2 / EAC — 4x4 blocks
    case etc2RGB8Unorm = "etc2-rgb8unorm"
    case etc2RGB8UnormSRGB = "etc2-rgb8unorm-srgb"
    case etc2RGB8A1Unorm = "etc2-rgb8a1unorm"
    case etc2RGB8A1UnormSRGB = "etc2-rgb8a1unorm-srgb"
    case etc2RGBA8Unorm = "etc2-rgba8unorm"
    case etc2RGBA8UnormSRGB = "etc2-rgba8unorm-srgb"
    case eacR11Unorm = "eac-r11unorm"
    case eacR11Snorm = "eac-r11snorm"
    case eacRG11Unorm = "eac-rg11unorm"
    case eacRG11Snorm = "eac-rg11snorm"

    // ASTC — the block size is part of the format name (4x4 through 12x12), all 16 bytes/block
    case astc4x4Unorm = "astc-4x4-unorm"
    case astc4x4UnormSRGB = "astc-4x4-unorm-srgb"
    case astc5x4Unorm = "astc-5x4-unorm"
    case astc5x4UnormSRGB = "astc-5x4-unorm-srgb"
    case astc5x5Unorm = "astc-5x5-unorm"
    case astc5x5UnormSRGB = "astc-5x5-unorm-srgb"
    case astc6x5Unorm = "astc-6x5-unorm"
    case astc6x5UnormSRGB = "astc-6x5-unorm-srgb"
    case astc6x6Unorm = "astc-6x6-unorm"
    case astc6x6UnormSRGB = "astc-6x6-unorm-srgb"
    case astc8x5Unorm = "astc-8x5-unorm"
    case astc8x5UnormSRGB = "astc-8x5-unorm-srgb"
    case astc8x6Unorm = "astc-8x6-unorm"
    case astc8x6UnormSRGB = "astc-8x6-unorm-srgb"
    case astc8x8Unorm = "astc-8x8-unorm"
    case astc8x8UnormSRGB = "astc-8x8-unorm-srgb"
    case astc10x5Unorm = "astc-10x5-unorm"
    case astc10x5UnormSRGB = "astc-10x5-unorm-srgb"
    case astc10x6Unorm = "astc-10x6-unorm"
    case astc10x6UnormSRGB = "astc-10x6-unorm-srgb"
    case astc10x8Unorm = "astc-10x8-unorm"
    case astc10x8UnormSRGB = "astc-10x8-unorm-srgb"
    case astc10x10Unorm = "astc-10x10-unorm"
    case astc10x10UnormSRGB = "astc-10x10-unorm-srgb"
    case astc12x10Unorm = "astc-12x10-unorm"
    case astc12x10UnormSRGB = "astc-12x10-unorm-srgb"
    case astc12x12Unorm = "astc-12x12-unorm"
    case astc12x12UnormSRGB = "astc-12x12-unorm-srgb"

    /// Whether the format can only be a depth/stencil attachment.
    public var isDepthOrStencil: Bool {
        switch self {
        case .stencil8, .depth16unorm, .depth24plus, .depth24plusStencil8, .depth32float, .depth32floatStencil8:
            return true
        default:
            return false
        }
    }

    public var hasDepth: Bool {
        switch self {
        case .depth16unorm, .depth24plus, .depth24plusStencil8, .depth32float, .depth32floatStencil8: return true
        default: return false
        }
    }

    public var hasStencil: Bool {
        switch self {
        case .stencil8, .depth24plusStencil8, .depth32floatStencil8: return true
        default: return false
        }
    }

    /// Whether this is a block-compressed format — stored in **blocks**, not pixels.
    public var isCompressed: Bool { blockWidth > 1 || blockHeight > 1 }

    /// The area one texel block covers. Uncompressed formats are (1, 1).
    ///
    /// ASTC carries its block size in the name (`astc-8x6-unorm` → 8×6); BC and ETC2 are all 4×4.
    public var blockSize: (width: Int, height: Int) {
        guard rawValue.hasPrefix("astc-") else {
            switch self {
            case .bc1RGBAUnorm, .bc1RGBAUnormSRGB, .bc2RGBAUnorm, .bc2RGBAUnormSRGB,
                 .bc3RGBAUnorm, .bc3RGBAUnormSRGB, .bc4RUnorm, .bc4RSnorm,
                 .bc5RGUnorm, .bc5RGSnorm, .bc6hRGBUfloat, .bc6hRGBFloat,
                 .bc7RGBAUnorm, .bc7RGBAUnormSRGB,
                 .etc2RGB8Unorm, .etc2RGB8UnormSRGB, .etc2RGB8A1Unorm, .etc2RGB8A1UnormSRGB,
                 .etc2RGBA8Unorm, .etc2RGBA8UnormSRGB,
                 .eacR11Unorm, .eacR11Snorm, .eacRG11Unorm, .eacRG11Snorm:
                return (4, 4)
            default:
                return (1, 1)
            }
        }
        // "astc-<W>x<H>-unorm[-srgb]" — the name is the block size.
        let dimensions = rawValue.dropFirst("astc-".count).prefix { $0 != "-" }.split(separator: "x")
        guard dimensions.count == 2, let width = Int(dimensions[0]), let height = Int(dimensions[1]) else {
            return (1, 1)
        }
        return (width, height)
    }

    public var blockWidth: Int { blockSize.width }
    public var blockHeight: Int { blockSize.height }

    /// Bytes in one block. **For uncompressed formats this equals the bytes in one pixel.**
    ///
    /// Every size calculation uses this value — `bytesPerPixel` on a compressed format is silently wrong.
    public var bytesPerBlock: Int {
        switch self {
        case .r8unorm, .r8snorm, .r8uint, .r8sint, .stencil8: return 1
        case .r16uint, .r16sint, .r16float, .rg8unorm, .rg8snorm, .rg8uint, .rg8sint, .depth16unorm: return 2
        case .r32uint, .r32sint, .r32float, .rg16uint, .rg16sint, .rg16float,
             .rgba8unorm, .rgba8unormSRGB, .rgba8snorm, .rgba8uint, .rgba8sint,
             .bgra8unorm, .bgra8unormSRGB, .rgb10a2unorm, .rgb10a2uint, .rg11b10ufloat,
             .rgb9e5ufloat, .depth24plus, .depth32float: return 4
        case .rg32uint, .rg32sint, .rg32float, .rgba16uint, .rgba16sint, .rgba16float,
             .depth24plusStencil8, .depth32floatStencil8: return 8
        case .rgba32uint, .rgba32sint, .rgba32float: return 16
        // BC1/BC4 and the ETC2 RGB8 family are 8 bytes per block; every other compressed format is 16.
        case .bc1RGBAUnorm, .bc1RGBAUnormSRGB, .bc4RUnorm, .bc4RSnorm,
             .etc2RGB8Unorm, .etc2RGB8UnormSRGB, .etc2RGB8A1Unorm, .etc2RGB8A1UnormSRGB,
             .eacR11Unorm, .eacR11Snorm:
            return 8
        default:
            // BC2, BC3, BC5, BC6H, BC7, ETC2 RGBA8, EAC RG11 and every ASTC variant.
            return 16
        }
    }

    /// Bytes in one pixel. **Meaningful only for uncompressed formats** (use `bytesPerBlock` otherwise).
    public var bytesPerPixel: Int { bytesPerBlock }

    /// Bytes in one row holding `width` pixels (rounded up to whole blocks).
    public func bytesPerRow(width: Int) -> Int {
        (width + blockWidth - 1) / blockWidth * bytesPerBlock
    }

    /// Number of **block rows** covering `height` pixels.
    public func blockRows(height: Int) -> Int {
        (height + blockHeight - 1) / blockHeight
    }
}

public enum WGPUTextureDimension: String, CaseIterable, Sendable {
    case oneD = "1d"
    case twoD = "2d"
    case threeD = "3d"
}

public enum WGPUTextureViewDimension: String, CaseIterable, Sendable {
    case oneD = "1d"
    case twoD = "2d"
    case twoDArray = "2d-array"
    case cube
    case cubeArray = "cube-array"
    case threeD = "3d"
}

public enum WGPUTextureAspect: String, CaseIterable, Sendable {
    case all
    case stencilOnly = "stencil-only"
    case depthOnly = "depth-only"
}

public enum WGPUAddressMode: String, CaseIterable, Sendable {
    case clampToEdge = "clamp-to-edge"
    case repeatMode = "repeat"
    case mirrorRepeat = "mirror-repeat"
}

public enum WGPUFilterMode: String, CaseIterable, Sendable {
    case nearest, linear
}

public enum WGPUCompareFunction: String, CaseIterable, Sendable {
    case never
    case less
    case equal
    case lessEqual = "less-equal"
    case greater
    case notEqual = "not-equal"
    case greaterEqual = "greater-equal"
    case always
}

/// How to change the stencil value based on the stencil and depth test results.
///
/// `-clamp` stops at 0/255 while `-wrap` wraps around to the other end — when counting overlaps,
/// as with shadow volumes, that difference decides the result.
public enum WGPUStencilOperation: String, CaseIterable, Sendable {
    case keep
    case zero
    case replace
    case invert
    case incrementClamp = "increment-clamp"
    case decrementClamp = "decrement-clamp"
    case incrementWrap = "increment-wrap"
    case decrementWrap = "decrement-wrap"
}

public enum WGPUPrimitiveTopology: String, CaseIterable, Sendable {
    case pointList = "point-list"
    case lineList = "line-list"
    case lineStrip = "line-strip"
    case triangleList = "triangle-list"
    case triangleStrip = "triangle-strip"
}

public enum WGPUFrontFace: String, CaseIterable, Sendable {
    case ccw, cw
}

public enum WGPUCullMode: String, CaseIterable, Sendable {
    case none, front, back
}

public enum WGPUIndexFormat: String, CaseIterable, Sendable {
    case uint16, uint32
}

public enum WGPUVertexStepMode: String, CaseIterable, Sendable {
    case vertex, instance
}

public enum WGPUVertexFormat: String, CaseIterable, Sendable {
    case uint8x2, uint8x4
    case sint8x2, sint8x4
    case unorm8x2, unorm8x4
    case snorm8x2, snorm8x4
    case uint16x2, uint16x4
    case sint16x2, sint16x4
    case unorm16x2, unorm16x4
    case snorm16x2, snorm16x4
    case float16x2, float16x4
    case float32, float32x2, float32x3, float32x4
    case uint32, uint32x2, uint32x3, uint32x4
    case sint32, sint32x2, sint32x3, sint32x4

    public var byteSize: Int {
        switch self {
        case .uint8x2, .sint8x2, .unorm8x2, .snorm8x2: return 2
        case .uint8x4, .sint8x4, .unorm8x4, .snorm8x4,
             .uint16x2, .sint16x2, .unorm16x2, .snorm16x2, .float16x2,
             .float32, .uint32, .sint32: return 4
        case .uint16x4, .sint16x4, .unorm16x4, .snorm16x4, .float16x4,
             .float32x2, .uint32x2, .sint32x2: return 8
        case .float32x3, .uint32x3, .sint32x3: return 12
        case .float32x4, .uint32x4, .sint32x4: return 16
        }
    }
}

/// `GPUQuerySet.type`.
///
/// The two behave very differently. `occlusion` ("how many samples survived") is **deterministic**,
/// while `timestamp` is a GPU clock whose value differs on identical input. Never assert on the latter.
public enum WGPUQueryType: String, CaseIterable, Sendable {
    case occlusion
    case timestamp
}

public enum WGPULoadOp: String, CaseIterable, Sendable {
    case load, clear
}

public enum WGPUStoreOp: String, CaseIterable, Sendable {
    case store, discard
}

public enum WGPUBlendFactor: String, CaseIterable, Sendable {
    case zero
    case one
    case src
    case oneMinusSrc = "one-minus-src"
    case srcAlpha = "src-alpha"
    case oneMinusSrcAlpha = "one-minus-src-alpha"
    case dst
    case oneMinusDst = "one-minus-dst"
    case dstAlpha = "dst-alpha"
    case oneMinusDstAlpha = "one-minus-dst-alpha"
    case srcAlphaSaturated = "src-alpha-saturated"
    case constant
    case oneMinusConstant = "one-minus-constant"
}

public enum WGPUBlendOperation: String, CaseIterable, Sendable {
    case add
    case subtract
    case reverseSubtract = "reverse-subtract"
    case min
    case max
}

public enum WGPUBufferBindingType: String, CaseIterable, Sendable {
    case uniform
    case storage
    case readOnlyStorage = "read-only-storage"
}

public enum WGPUSamplerBindingType: String, CaseIterable, Sendable {
    case filtering
    case nonFiltering = "non-filtering"
    case comparison
}

public enum WGPUTextureSampleType: String, CaseIterable, Sendable {
    case float
    case unfilterableFloat = "unfilterable-float"
    case depth
    case sint
    case uint
}

public enum WGPUStorageTextureAccess: String, CaseIterable, Sendable {
    case writeOnly = "write-only"
    case readOnly = "read-only"
    case readWrite = "read-write"
}

public enum WGPUCanvasAlphaMode: String, CaseIterable, Sendable {
    case opaque
    case premultiplied
}

/// `GPUCanvasConfiguration.colorSpace` — the two the spec defines.
public enum WGPUPredefinedColorSpace: String, CaseIterable, Sendable {
    case srgb
    case displayP3 = "display-p3"
}

/// `GPUCanvasToneMappingMode`.
///
/// `standard` clips to the SDR range (0...1) on display. `extended` passes values through up to the
/// headroom (EDR) a display can push beyond SDR white.
///
/// Using `extended` requires two things to line up:
/// - the canvas format must hold values above 1.0 (`rgba16float`);
/// - the shader must emit **linear** values — the extended space is linear, so no sRGB encoding.
public enum WGPUCanvasToneMappingMode: String, CaseIterable, Sendable {
    case standard
    case extended
}

/// Shader source language. The WebGPU spec defines only WGSL, but this implementation offers an
/// escape hatch that bypasses the transpiler and takes Metal Shading Language directly (`docs/WGSL.md` §5).
public enum WGPUShaderLanguage: String, CaseIterable, Sendable {
    case wgsl
    case msl
}

// MARK: - Bitmask flags

/// `GPUBufferUsage` — values must match the JS constants.
public struct WGPUBufferUsage: OptionSet, Sendable {
    public let rawValue: Int
    public init(rawValue: Int) { self.rawValue = rawValue }

    public static let mapRead = WGPUBufferUsage(rawValue: 0x0001)
    public static let mapWrite = WGPUBufferUsage(rawValue: 0x0002)
    public static let copySrc = WGPUBufferUsage(rawValue: 0x0004)
    public static let copyDst = WGPUBufferUsage(rawValue: 0x0008)
    public static let index = WGPUBufferUsage(rawValue: 0x0010)
    public static let vertex = WGPUBufferUsage(rawValue: 0x0020)
    public static let uniform = WGPUBufferUsage(rawValue: 0x0040)
    public static let storage = WGPUBufferUsage(rawValue: 0x0080)
    public static let indirect = WGPUBufferUsage(rawValue: 0x0100)
    public static let queryResolve = WGPUBufferUsage(rawValue: 0x0200)
}

/// `GPUTextureUsage`.
public struct WGPUTextureUsage: OptionSet, Sendable {
    public let rawValue: Int
    public init(rawValue: Int) { self.rawValue = rawValue }

    public static let copySrc = WGPUTextureUsage(rawValue: 0x01)
    public static let copyDst = WGPUTextureUsage(rawValue: 0x02)
    public static let textureBinding = WGPUTextureUsage(rawValue: 0x04)
    public static let storageBinding = WGPUTextureUsage(rawValue: 0x08)
    public static let renderAttachment = WGPUTextureUsage(rawValue: 0x10)
}

/// `GPUShaderStage` — the `visibility` of a bind group layout entry.
public struct WGPUShaderStage: OptionSet, Sendable {
    public let rawValue: Int
    public init(rawValue: Int) { self.rawValue = rawValue }

    public static let vertex = WGPUShaderStage(rawValue: 0x1)
    public static let fragment = WGPUShaderStage(rawValue: 0x2)
    public static let compute = WGPUShaderStage(rawValue: 0x4)
}

/// `GPUColorWrite`.
public struct WGPUColorWriteMask: OptionSet, Sendable {
    public let rawValue: Int
    public init(rawValue: Int) { self.rawValue = rawValue }

    public static let red = WGPUColorWriteMask(rawValue: 0x1)
    public static let green = WGPUColorWriteMask(rawValue: 0x2)
    public static let blue = WGPUColorWriteMask(rawValue: 0x4)
    public static let alpha = WGPUColorWriteMask(rawValue: 0x8)
    public static let all: WGPUColorWriteMask = [.red, .green, .blue, .alpha]
}
