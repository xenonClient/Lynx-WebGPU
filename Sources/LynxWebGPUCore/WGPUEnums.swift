import Foundation

// WebGPU 명세의 열거형은 JS에서 **문자열**로 온다 ("rgba8unorm", "triangle-list" …).
// raw value는 명세 철자를 그대로 쓴다 — 여기서 이름을 바꾸면 JS 코드가 깨진다.
// Metal 타입으로의 변환은 백엔드(LynxWebGPU)가 담당한다 (Core는 Metal-free).

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
    /// 공유 지수 HDR — 채널마다 가수 9비트 + 공통 지수 5비트. 렌더 타깃은 될 수 없고
    /// **읽기 전용 HDR 소스**로 쓴다 (`rgba16float`의 절반 크기로 같은 동적 범위를 담는다).
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

    // --- 블록 압축 (BC / ETC2 / ASTC) ---------------------------------------
    //
    // 픽셀이 아니라 **블록** 단위로 저장된다. 크기 계산은 `blockWidth`/`blockHeight`/
    // `bytesPerBlock`을 쓸 것 — `bytesPerPixel`은 이들 포맷에서 뜻이 없다.
    //
    // 셋 다 명세에서 **선택 기능**이다 (`texture-compression-bc` / `-etc2` / `-astc`).
    // 기기 지원은 `adapter.features`로 확인한다.

    // BC (Desktop) — 4x4 블록
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

    // ETC2 / EAC — 4x4 블록
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

    // ASTC — 블록 크기가 포맷 이름에 들어 있다 (4x4 ~ 12x12), 전부 16바이트/블록
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

    /// 깊이/스텐실 어태치먼트로만 쓸 수 있는 포맷인가.
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

    /// 블록 압축 포맷인가 — 픽셀이 아니라 **블록** 단위로 저장된다.
    public var isCompressed: Bool { blockWidth > 1 || blockHeight > 1 }

    /// 텍셀 블록 하나가 덮는 크기. 비압축은 (1, 1)이다.
    ///
    /// ASTC는 블록 크기가 포맷 이름에 들어 있고(`astc-8x6-unorm` → 8×6), BC·ETC2는 전부 4×4다.
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
        // "astc-<W>x<H>-unorm[-srgb]" — 이름이 곧 블록 크기다.
        let dimensions = rawValue.dropFirst("astc-".count).prefix { $0 != "-" }.split(separator: "x")
        guard dimensions.count == 2, let width = Int(dimensions[0]), let height = Int(dimensions[1]) else {
            return (1, 1)
        }
        return (width, height)
    }

    public var blockWidth: Int { blockSize.width }
    public var blockHeight: Int { blockSize.height }

    /// 블록 하나의 바이트 수. **비압축 포맷에서는 픽셀 1개의 바이트 수와 같다.**
    ///
    /// 크기 계산은 전부 이 값을 쓴다 — 압축 포맷에 `bytesPerPixel`을 쓰면 조용히 틀린다.
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
        // BC1·BC4와 ETC2 RGB8 계열은 블록당 8바이트, 나머지 압축 포맷은 16바이트다.
        case .bc1RGBAUnorm, .bc1RGBAUnormSRGB, .bc4RUnorm, .bc4RSnorm,
             .etc2RGB8Unorm, .etc2RGB8UnormSRGB, .etc2RGB8A1Unorm, .etc2RGB8A1UnormSRGB,
             .eacR11Unorm, .eacR11Snorm:
            return 8
        default:
            // BC2·BC3·BC5·BC6H·BC7 · ETC2 RGBA8 · EAC RG11 · ASTC 전 종류.
            return 16
        }
    }

    /// 픽셀 1개의 바이트 수. **비압축 포맷에서만 뜻이 있다** (압축은 `bytesPerBlock`을 쓸 것).
    public var bytesPerPixel: Int { bytesPerBlock }

    /// `width` 픽셀을 담는 데 필요한 한 행의 바이트 수 (블록 단위로 올림).
    public func bytesPerRow(width: Int) -> Int {
        (width + blockWidth - 1) / blockWidth * bytesPerBlock
    }

    /// `height` 픽셀에 해당하는 **블록 행** 수.
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

/// 스텐실 테스트/깊이 테스트 결과에 따라 스텐실 값을 어떻게 바꿀지.
///
/// `-clamp`는 0/255에서 멈추고 `-wrap`은 넘어가면 반대쪽으로 감긴다 — 섀도 볼륨처럼
/// 겹침 횟수를 셀 때 둘의 차이가 결과를 가른다.
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
/// 둘은 성격이 아주 다르다. `occlusion`은 "몇 개의 샘플이 살아남았나"라 **결정적**이고,
/// `timestamp`는 GPU 시계라 같은 입력에도 값이 매번 다르다. 뒤엣것에 값 단언을 걸면 안 된다.
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

/// `GPUCanvasConfiguration.colorSpace` — 명세가 정의하는 두 가지.
public enum WGPUPredefinedColorSpace: String, CaseIterable, Sendable {
    case srgb
    case displayP3 = "display-p3"
}

/// `GPUCanvasToneMappingMode`.
///
/// `standard`는 표시할 때 SDR 범위(0~1)로 자른다. `extended`는 디스플레이가 SDR 흰색보다
/// 더 밝게 낼 수 있는 여유(EDR)까지 값을 그대로 내보낸다.
///
/// `extended`를 쓰려면 두 가지가 함께 맞아야 한다:
/// - 캔버스 포맷이 1.0을 넘는 값을 담을 수 있어야 한다 (`rgba16float`).
/// - 셰이더가 **선형** 값을 써야 한다. 확장 색공간은 선형이므로 sRGB 인코딩을 하면 안 된다.
public enum WGPUCanvasToneMappingMode: String, CaseIterable, Sendable {
    case standard
    case extended
}

/// 셰이더 소스 언어. WebGPU 명세는 WGSL만 정의하지만, 이 구현은 트랜스파일러를 우회해
/// Metal Shading Language를 직접 넣는 탈출구를 제공한다 (`docs/WGSL.md` §5).
public enum WGPUShaderLanguage: String, CaseIterable, Sendable {
    case wgsl
    case msl
}

// MARK: - 비트마스크 플래그

/// `GPUBufferUsage` — JS 상수와 값이 같아야 한다.
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

/// `GPUShaderStage` — 바인드 그룹 레이아웃의 `visibility`.
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
