import Foundation
import Metal
import LynxWebGPUCore

/// WebGPU 열거형 → Metal 열거형.
///
/// 대응하지 않는 값은 던진다 — 조용히 비슷한 값으로 바꾸면 렌더 결과가 미묘하게 달라져
/// 원인을 찾기 어려운 버그가 된다.
enum WGPUMetalMapping {
    // MARK: - 픽셀 포맷

    static func pixelFormat(_ format: WGPUTextureFormat) throws -> MTLPixelFormat {
        switch format {
        case .r8unorm: return .r8Unorm
        case .r8snorm: return .r8Snorm
        case .r8uint: return .r8Uint
        case .r8sint: return .r8Sint
        case .r16uint: return .r16Uint
        case .r16sint: return .r16Sint
        case .r16float: return .r16Float
        case .rg8unorm: return .rg8Unorm
        case .rg8snorm: return .rg8Snorm
        case .rg8uint: return .rg8Uint
        case .rg8sint: return .rg8Sint
        case .r32uint: return .r32Uint
        case .r32sint: return .r32Sint
        case .r32float: return .r32Float
        case .rg16uint: return .rg16Uint
        case .rg16sint: return .rg16Sint
        case .rg16float: return .rg16Float
        case .rgba8unorm: return .rgba8Unorm
        case .rgba8unormSRGB: return .rgba8Unorm_srgb
        case .rgba8snorm: return .rgba8Snorm
        case .rgba8uint: return .rgba8Uint
        case .rgba8sint: return .rgba8Sint
        case .bgra8unorm: return .bgra8Unorm
        case .bgra8unormSRGB: return .bgra8Unorm_srgb
        case .rgb10a2unorm: return .rgb10a2Unorm
        case .rgb10a2uint: return .rgb10a2Uint
        case .rg11b10ufloat: return .rg11b10Float
        case .rgb9e5ufloat: return .rgb9e5Float

        // --- 블록 압축 -------------------------------------------------------
        case .bc1RGBAUnorm: return .bc1_rgba
        case .bc1RGBAUnormSRGB: return .bc1_rgba_srgb
        case .bc2RGBAUnorm: return .bc2_rgba
        case .bc2RGBAUnormSRGB: return .bc2_rgba_srgb
        case .bc3RGBAUnorm: return .bc3_rgba
        case .bc3RGBAUnormSRGB: return .bc3_rgba_srgb
        case .bc4RUnorm: return .bc4_rUnorm
        case .bc4RSnorm: return .bc4_rSnorm
        case .bc5RGUnorm: return .bc5_rgUnorm
        case .bc5RGSnorm: return .bc5_rgSnorm
        case .bc6hRGBUfloat: return .bc6H_rgbuFloat
        case .bc6hRGBFloat: return .bc6H_rgbFloat
        case .bc7RGBAUnorm: return .bc7_rgbaUnorm
        case .bc7RGBAUnormSRGB: return .bc7_rgbaUnorm_srgb
        case .etc2RGB8Unorm: return .etc2_rgb8
        case .etc2RGB8UnormSRGB: return .etc2_rgb8_srgb
        case .etc2RGB8A1Unorm: return .etc2_rgb8a1
        case .etc2RGB8A1UnormSRGB: return .etc2_rgb8a1_srgb
        case .etc2RGBA8Unorm: return .eac_rgba8
        case .etc2RGBA8UnormSRGB: return .eac_rgba8_srgb
        case .eacR11Unorm: return .eac_r11Unorm
        case .eacR11Snorm: return .eac_r11Snorm
        case .eacRG11Unorm: return .eac_rg11Unorm
        case .eacRG11Snorm: return .eac_rg11Snorm
        case .astc4x4Unorm: return .astc_4x4_ldr
        case .astc4x4UnormSRGB: return .astc_4x4_srgb
        case .astc5x4Unorm: return .astc_5x4_ldr
        case .astc5x4UnormSRGB: return .astc_5x4_srgb
        case .astc5x5Unorm: return .astc_5x5_ldr
        case .astc5x5UnormSRGB: return .astc_5x5_srgb
        case .astc6x5Unorm: return .astc_6x5_ldr
        case .astc6x5UnormSRGB: return .astc_6x5_srgb
        case .astc6x6Unorm: return .astc_6x6_ldr
        case .astc6x6UnormSRGB: return .astc_6x6_srgb
        case .astc8x5Unorm: return .astc_8x5_ldr
        case .astc8x5UnormSRGB: return .astc_8x5_srgb
        case .astc8x6Unorm: return .astc_8x6_ldr
        case .astc8x6UnormSRGB: return .astc_8x6_srgb
        case .astc8x8Unorm: return .astc_8x8_ldr
        case .astc8x8UnormSRGB: return .astc_8x8_srgb
        case .astc10x5Unorm: return .astc_10x5_ldr
        case .astc10x5UnormSRGB: return .astc_10x5_srgb
        case .astc10x6Unorm: return .astc_10x6_ldr
        case .astc10x6UnormSRGB: return .astc_10x6_srgb
        case .astc10x8Unorm: return .astc_10x8_ldr
        case .astc10x8UnormSRGB: return .astc_10x8_srgb
        case .astc10x10Unorm: return .astc_10x10_ldr
        case .astc10x10UnormSRGB: return .astc_10x10_srgb
        case .astc12x10Unorm: return .astc_12x10_ldr
        case .astc12x10UnormSRGB: return .astc_12x10_srgb
        case .astc12x12Unorm: return .astc_12x12_ldr
        case .astc12x12UnormSRGB: return .astc_12x12_srgb
        case .rg32uint: return .rg32Uint
        case .rg32sint: return .rg32Sint
        case .rg32float: return .rg32Float
        case .rgba16uint: return .rgba16Uint
        case .rgba16sint: return .rgba16Sint
        case .rgba16float: return .rgba16Float
        case .rgba32uint: return .rgba32Uint
        case .rgba32sint: return .rgba32Sint
        case .rgba32float: return .rgba32Float
        case .stencil8: return .stencil8
        case .depth16unorm: return .depth16Unorm
        // Apple GPU에는 24비트 깊이 포맷이 없다. WebGPU의 depth24plus는 "24비트 **이상**"이므로
        // 32비트 float 깊이로 올려 주는 것이 명세에 맞는 처리다.
        case .depth24plus, .depth32float: return .depth32Float
        case .depth24plusStencil8, .depth32floatStencil8: return .depth32Float_stencil8
        }
    }

    /// 역방향 매핑 — 드로어블 텍스처가 실제로 어떤 포맷인지 JS에 돌려줄 때 쓴다.
    /// (캔버스 레이어 설정은 메인 스레드에 비동기로 반영되므로, 요청한 포맷이 아니라
    /// **실제 텍스처의 포맷**을 기준으로 삼아야 파이프라인 불일치를 정확히 진단할 수 있다.)
    ///
    /// 표는 `pixelFormat(_:)`에서 **자동으로 뒤집어** 만든다. 손으로 적어 두면 포맷을
    /// 늘릴 때마다 한쪽만 자라 조용히 비게 된다 (실제로 캔버스에 쓰이는 몇 개만 있었다).
    static func textureFormat(from pixelFormat: MTLPixelFormat) -> WGPUTextureFormat? {
        inverseFormatTable[pixelFormat]
    }

    /// 여러 WebGPU 포맷이 같은 Metal 포맷으로 접히는 자리가 있다 (`depth24plus`도
    /// `depth32float`도 `.depth32Float`이다). 그런 자리는 **정밀도를 그대로 말해 주는 쪽**을
    /// 고른다 — 되돌린 이름이 실제 텍스처보다 약하게 들리면 진단이 사람을 헷갈리게 한다.
    private static let inverseFormatTable: [MTLPixelFormat: WGPUTextureFormat] = {
        var table: [MTLPixelFormat: WGPUTextureFormat] = [:]
        for format in WGPUTextureFormat.allCases {
            guard let metal = try? Self.pixelFormat(format), table[metal] == nil else { continue }
            table[metal] = format
        }
        table[.depth32Float] = .depth32float
        table[.depth32Float_stencil8] = .depth32floatStencil8
        return table
    }()

    // MARK: - 정점

    static func vertexFormat(_ format: WGPUVertexFormat) -> MTLVertexFormat {
        switch format {
        case .uint8x2: return .uchar2
        case .uint8x4: return .uchar4
        case .sint8x2: return .char2
        case .sint8x4: return .char4
        case .unorm8x2: return .uchar2Normalized
        case .unorm8x4: return .uchar4Normalized
        case .snorm8x2: return .char2Normalized
        case .snorm8x4: return .char4Normalized
        case .uint16x2: return .ushort2
        case .uint16x4: return .ushort4
        case .sint16x2: return .short2
        case .sint16x4: return .short4
        case .unorm16x2: return .ushort2Normalized
        case .unorm16x4: return .ushort4Normalized
        case .snorm16x2: return .short2Normalized
        case .snorm16x4: return .short4Normalized
        case .float16x2: return .half2
        case .float16x4: return .half4
        case .float32: return .float
        case .float32x2: return .float2
        case .float32x3: return .float3
        case .float32x4: return .float4
        case .uint32: return .uint
        case .uint32x2: return .uint2
        case .uint32x3: return .uint3
        case .uint32x4: return .uint4
        case .sint32: return .int
        case .sint32x2: return .int2
        case .sint32x3: return .int3
        case .sint32x4: return .int4
        }
    }

    static func stepFunction(_ mode: WGPUVertexStepMode) -> MTLVertexStepFunction {
        mode == .instance ? .perInstance : .perVertex
    }

    static func primitiveType(_ topology: WGPUPrimitiveTopology) -> MTLPrimitiveType {
        switch topology {
        case .pointList: return .point
        case .lineList: return .line
        case .lineStrip: return .lineStrip
        case .triangleList: return .triangle
        case .triangleStrip: return .triangleStrip
        }
    }

    static func indexType(_ format: WGPUIndexFormat) -> MTLIndexType {
        format == .uint16 ? .uint16 : .uint32
    }

    static func cullMode(_ mode: WGPUCullMode) -> MTLCullMode {
        switch mode {
        case .none: return .none
        case .front: return .front
        case .back: return .back
        }
    }

    static func winding(_ face: WGPUFrontFace) -> MTLWinding {
        face == .ccw ? .counterClockwise : .clockwise
    }

    // MARK: - 깊이 / 블렌딩

    static func compareFunction(_ function: WGPUCompareFunction) -> MTLCompareFunction {
        switch function {
        case .never: return .never
        case .less: return .less
        case .equal: return .equal
        case .lessEqual: return .lessEqual
        case .greater: return .greater
        case .notEqual: return .notEqual
        case .greaterEqual: return .greaterEqual
        case .always: return .always
        }
    }

    static func stencilOperation(_ operation: WGPUStencilOperation) -> MTLStencilOperation {
        switch operation {
        case .keep: return .keep
        case .zero: return .zero
        case .replace: return .replace
        case .invert: return .invert
        case .incrementClamp: return .incrementClamp
        case .decrementClamp: return .decrementClamp
        case .incrementWrap: return .incrementWrap
        case .decrementWrap: return .decrementWrap
        }
    }

    /// 한 면(front/back)의 스텐실 상태 + 파이프라인 공통 마스크.
    ///
    /// 마스크는 명세상 `GPUDepthStencilState`에 하나씩만 있고 앞/뒤가 나뉘지 않는다.
    /// Metal은 면마다 들고 있으므로 같은 값을 양쪽에 넣는다.
    static func stencilDescriptor(
        _ face: WGPUStencilFaceState,
        readMask: Int,
        writeMask: Int
    ) -> MTLStencilDescriptor {
        let descriptor = MTLStencilDescriptor()
        descriptor.stencilCompareFunction = compareFunction(face.compare)
        descriptor.stencilFailureOperation = stencilOperation(face.failOp)
        descriptor.depthFailureOperation = stencilOperation(face.depthFailOp)
        descriptor.depthStencilPassOperation = stencilOperation(face.passOp)
        descriptor.readMask = UInt32(truncatingIfNeeded: readMask)
        descriptor.writeMask = UInt32(truncatingIfNeeded: writeMask)
        return descriptor
    }

    static func blendFactor(_ factor: WGPUBlendFactor) -> MTLBlendFactor {
        switch factor {
        case .zero: return .zero
        case .one: return .one
        case .src: return .sourceColor
        case .oneMinusSrc: return .oneMinusSourceColor
        case .srcAlpha: return .sourceAlpha
        case .oneMinusSrcAlpha: return .oneMinusSourceAlpha
        case .dst: return .destinationColor
        case .oneMinusDst: return .oneMinusDestinationColor
        case .dstAlpha: return .destinationAlpha
        case .oneMinusDstAlpha: return .oneMinusDestinationAlpha
        case .srcAlphaSaturated: return .sourceAlphaSaturated
        case .constant: return .blendColor
        case .oneMinusConstant: return .oneMinusBlendColor
        }
    }

    static func blendOperation(_ operation: WGPUBlendOperation) -> MTLBlendOperation {
        switch operation {
        case .add: return .add
        case .subtract: return .subtract
        case .reverseSubtract: return .reverseSubtract
        case .min: return .min
        case .max: return .max
        }
    }

    static func colorWriteMask(_ mask: WGPUColorWriteMask) -> MTLColorWriteMask {
        var result: MTLColorWriteMask = []
        if mask.contains(.red) { result.insert(.red) }
        if mask.contains(.green) { result.insert(.green) }
        if mask.contains(.blue) { result.insert(.blue) }
        if mask.contains(.alpha) { result.insert(.alpha) }
        return result
    }

    // MARK: - 샘플러

    static func addressMode(_ mode: WGPUAddressMode) -> MTLSamplerAddressMode {
        switch mode {
        case .clampToEdge: return .clampToEdge
        case .repeatMode: return .repeat
        case .mirrorRepeat: return .mirrorRepeat
        }
    }

    static func minMagFilter(_ filter: WGPUFilterMode) -> MTLSamplerMinMagFilter {
        filter == .linear ? .linear : .nearest
    }

    static func mipFilter(_ filter: WGPUFilterMode) -> MTLSamplerMipFilter {
        filter == .linear ? .linear : .nearest
    }

    // MARK: - 텍스처

    static func textureType(_ dimension: WGPUTextureDimension, arrayLayers: Int, sampleCount: Int) -> MTLTextureType {
        switch dimension {
        case .oneD: return .type1D
        case .threeD: return .type3D
        case .twoD:
            if sampleCount > 1 { return arrayLayers > 1 ? .type2DMultisampleArray : .type2DMultisample }
            return arrayLayers > 1 ? .type2DArray : .type2D
        }
    }

    static func textureUsage(_ usage: WGPUTextureUsage) -> MTLTextureUsage {
        var result: MTLTextureUsage = []
        if usage.contains(.textureBinding) { result.insert(.shaderRead) }
        if usage.contains(.storageBinding) { result.insert([.shaderRead, .shaderWrite]) }
        if usage.contains(.renderAttachment) { result.insert(.renderTarget) }
        // copySrc/copyDst는 Metal에서 별도 usage 비트가 아니라 blit 인코더로 처리된다.
        return result.isEmpty ? .shaderRead : result
    }

    static func loadAction(_ op: WGPULoadOp) -> MTLLoadAction {
        op == .clear ? .clear : .load
    }

    static func storeAction(_ op: WGPUStoreOp) -> MTLStoreAction {
        op == .store ? .store : .dontCare
    }

    static func viewType(_ dimension: WGPUTextureViewDimension) -> MTLTextureType {
        switch dimension {
        case .oneD: return .type1D
        case .twoD: return .type2D
        case .twoDArray: return .type2DArray
        case .cube: return .typeCube
        case .cubeArray: return .typeCubeArray
        case .threeD: return .type3D
        }
    }
}

/// Metal 기능 집합에서 오는 **기기 능력** — 없는 것을 부르면 Metal이 단언으로 죽는다.
///
/// 이 프로젝트의 검증 원칙은 "Metal에 맡길 수 있으면 맡긴다"지만, **단언으로 죽는 것은 예외**다.
/// 프로세스가 사라지면 진단할 기회조차 없으므로 여기서 미리 걸러 오류로 알린다.
public enum WGPUDeviceCapability {
    /// 간접 드로우·디스패치 인자(`drawIndirect` 계열)를 쓸 수 있는가.
    ///
    /// Metal 기능 집합표: **Apple family 3 이상** 또는 Mac family 2. iOS 17을 최소로 잡는
    /// 이 라이브러리에서 실기기는 A12(family 5) 이상이라 **항상 지원**하지만,
    /// **iOS 시뮬레이터는 family 2로 보고**해서 빠진다 — 지원하지 않는 기기에서 부르면
    /// `MTLValidateFeatureSupport ... failed assertion`으로 프로세스가 죽는다.
    public static func supportsIndirectArguments(_ device: MTLDevice) -> Bool {
        device.supportsFamily(.apple3) || device.supportsFamily(.mac2)
    }

    /// 포맷이 속한 압축 계열 — 명세의 선택 기능 이름과 1:1로 대응한다.
    public enum CompressionFamily {
        case none, bc, etc2, astc

        /// `adapter.features`에 싣는 이름 (명세 철자 그대로).
        public var featureName: String? {
            switch self {
            case .none: return nil
            case .bc: return "texture-compression-bc"
            case .etc2: return "texture-compression-etc2"
            case .astc: return "texture-compression-astc"
            }
        }
    }

    /// ETC2와 EAC는 명세에서 **같은 기능 비트**다 (`texture-compression-etc2`).
    public static func compressionFamily(_ format: WGPUTextureFormat) -> CompressionFamily {
        guard format.isCompressed else { return .none }
        if format.rawValue.hasPrefix("bc") { return .bc }
        if format.rawValue.hasPrefix("astc-") { return .astc }
        return .etc2
    }

    /// 블록 압축 포맷을 이 기기가 지원하는가.
    ///
    /// Metal은 **지원하지 않는 압축 포맷으로 텍스처를 만들면 단언으로 죽는다.** 그래서
    /// `adapter.features`로 미리 알려 주고, 없는 계열은 생성 시점에 오류로 한 번 더 막는다.
    ///
    /// - ETC2/EAC · ASTC: 모든 Apple GPU가 한다 (iOS 전 기종, Apple Silicon Mac).
    ///   Intel/AMD Mac(`mac2`)에는 없다.
    /// - BC(DXT/BPTC): Apple7(A14/M1) 이상, 또는 Intel/AMD Mac.
    public static func supportsCompression(_ format: WGPUTextureFormat, on device: MTLDevice) -> Bool {
        switch compressionFamily(format) {
        case .none: return true
        case .bc:
            if #available(iOS 16.4, macOS 11.0, *) { return device.supportsBCTextureCompression }
            return device.supportsFamily(.apple7) || device.supportsFamily(.mac2)
        case .etc2, .astc:
            return device.supportsFamily(.apple2)
        }
    }
}
