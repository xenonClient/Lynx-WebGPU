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
        case .rg11b10ufloat: return .rg11b10Float
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
    static func textureFormat(from pixelFormat: MTLPixelFormat) -> WGPUTextureFormat? {
        switch pixelFormat {
        case .bgra8Unorm: return .bgra8unorm
        case .bgra8Unorm_srgb: return .bgra8unormSRGB
        case .rgba8Unorm: return .rgba8unorm
        case .rgba8Unorm_srgb: return .rgba8unormSRGB
        case .rgba16Float: return .rgba16float
        case .rgb10a2Unorm: return .rgb10a2unorm
        case .depth32Float: return .depth32float
        case .depth32Float_stencil8: return .depth32floatStencil8
        case .depth16Unorm: return .depth16unorm
        default: return nil
        }
    }

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
