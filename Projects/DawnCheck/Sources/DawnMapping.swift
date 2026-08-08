import Foundation
import WebGPU
import LynxWebGPUCore

// Core의 문자열 열거형(명세 철자) → Dawn C 열거형.
//
// 타입 이름이 두 모듈에서 겹치므로 (Core `WGPUTextureFormat` ↔ C `WGPUTextureFormat`)
// 시그니처는 모듈을 명시한다. C 열거형 **값**은 전역 상수(`WGPUTextureFormat_RGBA8Unorm`)라
// 겹치지 않는다. 매핑이 없는 값은 `unsupported`로 던진다 — 비슷한 것으로 조용히 바꾸면
// 원인을 못 찾는 버그가 된다 (Metal 매핑과 같은 규칙).
enum DawnEnum {

    static func textureFormat(
        _ format: LynxWebGPUCore.WGPUTextureFormat
    ) throws -> WebGPU.WGPUTextureFormat {
        // Core의 전 케이스를 옮긴다 — default가 없어 Core에 포맷이 늘면 컴파일이 잡는다.
        switch format {
        case .r8unorm: return WGPUTextureFormat_R8Unorm
        case .r8snorm: return WGPUTextureFormat_R8Snorm
        case .r8uint: return WGPUTextureFormat_R8Uint
        case .r8sint: return WGPUTextureFormat_R8Sint
        case .r16uint: return WGPUTextureFormat_R16Uint
        case .r16sint: return WGPUTextureFormat_R16Sint
        case .r16float: return WGPUTextureFormat_R16Float
        case .rg8unorm: return WGPUTextureFormat_RG8Unorm
        case .rg8snorm: return WGPUTextureFormat_RG8Snorm
        case .rg8uint: return WGPUTextureFormat_RG8Uint
        case .rg8sint: return WGPUTextureFormat_RG8Sint
        case .r32uint: return WGPUTextureFormat_R32Uint
        case .r32sint: return WGPUTextureFormat_R32Sint
        case .r32float: return WGPUTextureFormat_R32Float
        case .rg16uint: return WGPUTextureFormat_RG16Uint
        case .rg16sint: return WGPUTextureFormat_RG16Sint
        case .rg16float: return WGPUTextureFormat_RG16Float
        case .rgba8unorm: return WGPUTextureFormat_RGBA8Unorm
        case .rgba8unormSRGB: return WGPUTextureFormat_RGBA8UnormSrgb
        case .rgba8snorm: return WGPUTextureFormat_RGBA8Snorm
        case .rgba8uint: return WGPUTextureFormat_RGBA8Uint
        case .rgba8sint: return WGPUTextureFormat_RGBA8Sint
        case .bgra8unorm: return WGPUTextureFormat_BGRA8Unorm
        case .bgra8unormSRGB: return WGPUTextureFormat_BGRA8UnormSrgb
        case .rgb10a2unorm: return WGPUTextureFormat_RGB10A2Unorm
        case .rgb10a2uint: return WGPUTextureFormat_RGB10A2Uint
        case .rg11b10ufloat: return WGPUTextureFormat_RG11B10Ufloat
        case .rgb9e5ufloat: return WGPUTextureFormat_RGB9E5Ufloat
        case .rg32uint: return WGPUTextureFormat_RG32Uint
        case .rg32sint: return WGPUTextureFormat_RG32Sint
        case .rg32float: return WGPUTextureFormat_RG32Float
        case .rgba16uint: return WGPUTextureFormat_RGBA16Uint
        case .rgba16sint: return WGPUTextureFormat_RGBA16Sint
        case .rgba16float: return WGPUTextureFormat_RGBA16Float
        case .rgba32uint: return WGPUTextureFormat_RGBA32Uint
        case .rgba32sint: return WGPUTextureFormat_RGBA32Sint
        case .rgba32float: return WGPUTextureFormat_RGBA32Float
        case .stencil8: return WGPUTextureFormat_Stencil8
        case .depth16unorm: return WGPUTextureFormat_Depth16Unorm
        case .depth24plus: return WGPUTextureFormat_Depth24Plus
        case .depth24plusStencil8: return WGPUTextureFormat_Depth24PlusStencil8
        case .depth32float: return WGPUTextureFormat_Depth32Float
        case .depth32floatStencil8: return WGPUTextureFormat_Depth32FloatStencil8
        case .bc1RGBAUnorm: return WGPUTextureFormat_BC1RGBAUnorm
        case .bc1RGBAUnormSRGB: return WGPUTextureFormat_BC1RGBAUnormSrgb
        case .bc2RGBAUnorm: return WGPUTextureFormat_BC2RGBAUnorm
        case .bc2RGBAUnormSRGB: return WGPUTextureFormat_BC2RGBAUnormSrgb
        case .bc3RGBAUnorm: return WGPUTextureFormat_BC3RGBAUnorm
        case .bc3RGBAUnormSRGB: return WGPUTextureFormat_BC3RGBAUnormSrgb
        case .bc4RUnorm: return WGPUTextureFormat_BC4RUnorm
        case .bc4RSnorm: return WGPUTextureFormat_BC4RSnorm
        case .bc5RGUnorm: return WGPUTextureFormat_BC5RGUnorm
        case .bc5RGSnorm: return WGPUTextureFormat_BC5RGSnorm
        case .bc6hRGBUfloat: return WGPUTextureFormat_BC6HRGBUfloat
        case .bc6hRGBFloat: return WGPUTextureFormat_BC6HRGBFloat
        case .bc7RGBAUnorm: return WGPUTextureFormat_BC7RGBAUnorm
        case .bc7RGBAUnormSRGB: return WGPUTextureFormat_BC7RGBAUnormSrgb
        case .etc2RGB8Unorm: return WGPUTextureFormat_ETC2RGB8Unorm
        case .etc2RGB8UnormSRGB: return WGPUTextureFormat_ETC2RGB8UnormSrgb
        case .etc2RGB8A1Unorm: return WGPUTextureFormat_ETC2RGB8A1Unorm
        case .etc2RGB8A1UnormSRGB: return WGPUTextureFormat_ETC2RGB8A1UnormSrgb
        case .etc2RGBA8Unorm: return WGPUTextureFormat_ETC2RGBA8Unorm
        case .etc2RGBA8UnormSRGB: return WGPUTextureFormat_ETC2RGBA8UnormSrgb
        case .eacR11Unorm: return WGPUTextureFormat_EACR11Unorm
        case .eacR11Snorm: return WGPUTextureFormat_EACR11Snorm
        case .eacRG11Unorm: return WGPUTextureFormat_EACRG11Unorm
        case .eacRG11Snorm: return WGPUTextureFormat_EACRG11Snorm
        case .astc4x4Unorm: return WGPUTextureFormat_ASTC4x4Unorm
        case .astc4x4UnormSRGB: return WGPUTextureFormat_ASTC4x4UnormSrgb
        case .astc5x4Unorm: return WGPUTextureFormat_ASTC5x4Unorm
        case .astc5x4UnormSRGB: return WGPUTextureFormat_ASTC5x4UnormSrgb
        case .astc5x5Unorm: return WGPUTextureFormat_ASTC5x5Unorm
        case .astc5x5UnormSRGB: return WGPUTextureFormat_ASTC5x5UnormSrgb
        case .astc6x5Unorm: return WGPUTextureFormat_ASTC6x5Unorm
        case .astc6x5UnormSRGB: return WGPUTextureFormat_ASTC6x5UnormSrgb
        case .astc6x6Unorm: return WGPUTextureFormat_ASTC6x6Unorm
        case .astc6x6UnormSRGB: return WGPUTextureFormat_ASTC6x6UnormSrgb
        case .astc8x5Unorm: return WGPUTextureFormat_ASTC8x5Unorm
        case .astc8x5UnormSRGB: return WGPUTextureFormat_ASTC8x5UnormSrgb
        case .astc8x6Unorm: return WGPUTextureFormat_ASTC8x6Unorm
        case .astc8x6UnormSRGB: return WGPUTextureFormat_ASTC8x6UnormSrgb
        case .astc8x8Unorm: return WGPUTextureFormat_ASTC8x8Unorm
        case .astc8x8UnormSRGB: return WGPUTextureFormat_ASTC8x8UnormSrgb
        case .astc10x5Unorm: return WGPUTextureFormat_ASTC10x5Unorm
        case .astc10x5UnormSRGB: return WGPUTextureFormat_ASTC10x5UnormSrgb
        case .astc10x6Unorm: return WGPUTextureFormat_ASTC10x6Unorm
        case .astc10x6UnormSRGB: return WGPUTextureFormat_ASTC10x6UnormSrgb
        case .astc10x8Unorm: return WGPUTextureFormat_ASTC10x8Unorm
        case .astc10x8UnormSRGB: return WGPUTextureFormat_ASTC10x8UnormSrgb
        case .astc10x10Unorm: return WGPUTextureFormat_ASTC10x10Unorm
        case .astc10x10UnormSRGB: return WGPUTextureFormat_ASTC10x10UnormSrgb
        case .astc12x10Unorm: return WGPUTextureFormat_ASTC12x10Unorm
        case .astc12x10UnormSRGB: return WGPUTextureFormat_ASTC12x10UnormSrgb
        case .astc12x12Unorm: return WGPUTextureFormat_ASTC12x12Unorm
        case .astc12x12UnormSRGB: return WGPUTextureFormat_ASTC12x12UnormSrgb
        }
    }

    static func vertexFormat(
        _ format: LynxWebGPUCore.WGPUVertexFormat
    ) throws -> WebGPU.WGPUVertexFormat {
        switch format {
        case .uint8x2: return WGPUVertexFormat_Uint8x2
        case .uint8x4: return WGPUVertexFormat_Uint8x4
        case .sint8x2: return WGPUVertexFormat_Sint8x2
        case .sint8x4: return WGPUVertexFormat_Sint8x4
        case .unorm8x2: return WGPUVertexFormat_Unorm8x2
        case .unorm8x4: return WGPUVertexFormat_Unorm8x4
        case .snorm8x2: return WGPUVertexFormat_Snorm8x2
        case .snorm8x4: return WGPUVertexFormat_Snorm8x4
        case .uint16x2: return WGPUVertexFormat_Uint16x2
        case .uint16x4: return WGPUVertexFormat_Uint16x4
        case .sint16x2: return WGPUVertexFormat_Sint16x2
        case .sint16x4: return WGPUVertexFormat_Sint16x4
        case .unorm16x2: return WGPUVertexFormat_Unorm16x2
        case .unorm16x4: return WGPUVertexFormat_Unorm16x4
        case .snorm16x2: return WGPUVertexFormat_Snorm16x2
        case .snorm16x4: return WGPUVertexFormat_Snorm16x4
        case .float16x2: return WGPUVertexFormat_Float16x2
        case .float16x4: return WGPUVertexFormat_Float16x4
        case .float32: return WGPUVertexFormat_Float32
        case .float32x2: return WGPUVertexFormat_Float32x2
        case .float32x3: return WGPUVertexFormat_Float32x3
        case .float32x4: return WGPUVertexFormat_Float32x4
        case .uint32: return WGPUVertexFormat_Uint32
        case .uint32x2: return WGPUVertexFormat_Uint32x2
        case .uint32x3: return WGPUVertexFormat_Uint32x3
        case .uint32x4: return WGPUVertexFormat_Uint32x4
        case .sint32: return WGPUVertexFormat_Sint32
        case .sint32x2: return WGPUVertexFormat_Sint32x2
        case .sint32x3: return WGPUVertexFormat_Sint32x3
        case .sint32x4: return WGPUVertexFormat_Sint32x4
        }
    }

    static func compare(_ function: LynxWebGPUCore.WGPUCompareFunction) -> WebGPU.WGPUCompareFunction {
        switch function {
        case .never: return WGPUCompareFunction_Never
        case .less: return WGPUCompareFunction_Less
        case .equal: return WGPUCompareFunction_Equal
        case .lessEqual: return WGPUCompareFunction_LessEqual
        case .greater: return WGPUCompareFunction_Greater
        case .notEqual: return WGPUCompareFunction_NotEqual
        case .greaterEqual: return WGPUCompareFunction_GreaterEqual
        case .always: return WGPUCompareFunction_Always
        }
    }

    static func stencilOperation(
        _ operation: LynxWebGPUCore.WGPUStencilOperation
    ) -> WebGPU.WGPUStencilOperation {
        switch operation {
        case .keep: return WGPUStencilOperation_Keep
        case .zero: return WGPUStencilOperation_Zero
        case .replace: return WGPUStencilOperation_Replace
        case .invert: return WGPUStencilOperation_Invert
        case .incrementClamp: return WGPUStencilOperation_IncrementClamp
        case .decrementClamp: return WGPUStencilOperation_DecrementClamp
        case .incrementWrap: return WGPUStencilOperation_IncrementWrap
        case .decrementWrap: return WGPUStencilOperation_DecrementWrap
        }
    }

    static func topology(
        _ topology: LynxWebGPUCore.WGPUPrimitiveTopology
    ) -> WebGPU.WGPUPrimitiveTopology {
        switch topology {
        case .pointList: return WGPUPrimitiveTopology_PointList
        case .lineList: return WGPUPrimitiveTopology_LineList
        case .lineStrip: return WGPUPrimitiveTopology_LineStrip
        case .triangleList: return WGPUPrimitiveTopology_TriangleList
        case .triangleStrip: return WGPUPrimitiveTopology_TriangleStrip
        }
    }

    static func frontFace(_ face: LynxWebGPUCore.WGPUFrontFace) -> WebGPU.WGPUFrontFace {
        face == .ccw ? WGPUFrontFace_CCW : WGPUFrontFace_CW
    }

    static func cullMode(_ mode: LynxWebGPUCore.WGPUCullMode) -> WebGPU.WGPUCullMode {
        switch mode {
        case .none: return WGPUCullMode_None
        case .front: return WGPUCullMode_Front
        case .back: return WGPUCullMode_Back
        }
    }

    static func indexFormat(_ format: LynxWebGPUCore.WGPUIndexFormat) -> WebGPU.WGPUIndexFormat {
        format == .uint16 ? WGPUIndexFormat_Uint16 : WGPUIndexFormat_Uint32
    }

    static func stepMode(_ mode: LynxWebGPUCore.WGPUVertexStepMode) -> WebGPU.WGPUVertexStepMode {
        mode == .vertex ? WGPUVertexStepMode_Vertex : WGPUVertexStepMode_Instance
    }

    static func loadOp(_ op: LynxWebGPUCore.WGPULoadOp?) -> WebGPU.WGPULoadOp {
        switch op {
        case .load: return WGPULoadOp_Load
        case .clear: return WGPULoadOp_Clear
        case nil: return WGPULoadOp_Undefined
        }
    }

    static func storeOp(_ op: LynxWebGPUCore.WGPUStoreOp?) -> WebGPU.WGPUStoreOp {
        switch op {
        case .store: return WGPUStoreOp_Store
        case .discard: return WGPUStoreOp_Discard
        case nil: return WGPUStoreOp_Undefined
        }
    }

    static func blendFactor(_ factor: LynxWebGPUCore.WGPUBlendFactor) -> WebGPU.WGPUBlendFactor {
        switch factor {
        case .zero: return WGPUBlendFactor_Zero
        case .one: return WGPUBlendFactor_One
        case .src: return WGPUBlendFactor_Src
        case .oneMinusSrc: return WGPUBlendFactor_OneMinusSrc
        case .srcAlpha: return WGPUBlendFactor_SrcAlpha
        case .oneMinusSrcAlpha: return WGPUBlendFactor_OneMinusSrcAlpha
        case .dst: return WGPUBlendFactor_Dst
        case .oneMinusDst: return WGPUBlendFactor_OneMinusDst
        case .dstAlpha: return WGPUBlendFactor_DstAlpha
        case .oneMinusDstAlpha: return WGPUBlendFactor_OneMinusDstAlpha
        case .srcAlphaSaturated: return WGPUBlendFactor_SrcAlphaSaturated
        case .constant: return WGPUBlendFactor_Constant
        case .oneMinusConstant: return WGPUBlendFactor_OneMinusConstant
        }
    }

    static func blendOperation(
        _ operation: LynxWebGPUCore.WGPUBlendOperation
    ) -> WebGPU.WGPUBlendOperation {
        switch operation {
        case .add: return WGPUBlendOperation_Add
        case .subtract: return WGPUBlendOperation_Subtract
        case .reverseSubtract: return WGPUBlendOperation_ReverseSubtract
        case .min: return WGPUBlendOperation_Min
        case .max: return WGPUBlendOperation_Max
        }
    }

    static func addressMode(_ mode: LynxWebGPUCore.WGPUAddressMode) -> WebGPU.WGPUAddressMode {
        switch mode {
        case .clampToEdge: return WGPUAddressMode_ClampToEdge
        case .repeatMode: return WGPUAddressMode_Repeat
        case .mirrorRepeat: return WGPUAddressMode_MirrorRepeat
        }
    }

    static func filter(_ mode: LynxWebGPUCore.WGPUFilterMode) -> WebGPU.WGPUFilterMode {
        mode == .nearest ? WGPUFilterMode_Nearest : WGPUFilterMode_Linear
    }

    static func mipmapFilter(_ mode: LynxWebGPUCore.WGPUFilterMode) -> WebGPU.WGPUMipmapFilterMode {
        mode == .nearest ? WGPUMipmapFilterMode_Nearest : WGPUMipmapFilterMode_Linear
    }

    static func textureDimension(
        _ dimension: LynxWebGPUCore.WGPUTextureDimension
    ) -> WebGPU.WGPUTextureDimension {
        switch dimension {
        case .oneD: return WGPUTextureDimension_1D
        case .twoD: return WGPUTextureDimension_2D
        case .threeD: return WGPUTextureDimension_3D
        }
    }

    static func viewDimension(
        _ dimension: LynxWebGPUCore.WGPUTextureViewDimension?
    ) -> WebGPU.WGPUTextureViewDimension {
        switch dimension {
        case .oneD: return WGPUTextureViewDimension_1D
        case .twoD: return WGPUTextureViewDimension_2D
        case .twoDArray: return WGPUTextureViewDimension_2DArray
        case .cube: return WGPUTextureViewDimension_Cube
        case .cubeArray: return WGPUTextureViewDimension_CubeArray
        case .threeD: return WGPUTextureViewDimension_3D
        case nil: return WGPUTextureViewDimension_Undefined
        }
    }

    static func aspect(_ aspect: LynxWebGPUCore.WGPUTextureAspect) -> WebGPU.WGPUTextureAspect {
        switch aspect {
        case .all: return WGPUTextureAspect_All
        case .stencilOnly: return WGPUTextureAspect_StencilOnly
        case .depthOnly: return WGPUTextureAspect_DepthOnly
        }
    }

    static func queryType(_ type: LynxWebGPUCore.WGPUQueryType) -> WebGPU.WGPUQueryType {
        type == .occlusion ? WGPUQueryType_Occlusion : WGPUQueryType_Timestamp
    }

    static func bufferBindingType(
        _ type: LynxWebGPUCore.WGPUBufferBindingType
    ) -> WebGPU.WGPUBufferBindingType {
        switch type {
        case .uniform: return WGPUBufferBindingType_Uniform
        case .storage: return WGPUBufferBindingType_Storage
        case .readOnlyStorage: return WGPUBufferBindingType_ReadOnlyStorage
        }
    }

    static func samplerBindingType(
        _ type: LynxWebGPUCore.WGPUSamplerBindingType
    ) -> WebGPU.WGPUSamplerBindingType {
        switch type {
        case .filtering: return WGPUSamplerBindingType_Filtering
        case .nonFiltering: return WGPUSamplerBindingType_NonFiltering
        case .comparison: return WGPUSamplerBindingType_Comparison
        }
    }

    static func sampleType(
        _ type: LynxWebGPUCore.WGPUTextureSampleType
    ) -> WebGPU.WGPUTextureSampleType {
        switch type {
        case .float: return WGPUTextureSampleType_Float
        case .unfilterableFloat: return WGPUTextureSampleType_UnfilterableFloat
        case .depth: return WGPUTextureSampleType_Depth
        case .sint: return WGPUTextureSampleType_Sint
        case .uint: return WGPUTextureSampleType_Uint
        }
    }

    static func storageAccess(
        _ access: LynxWebGPUCore.WGPUStorageTextureAccess
    ) -> WebGPU.WGPUStorageTextureAccess {
        switch access {
        case .writeOnly: return WGPUStorageTextureAccess_WriteOnly
        case .readOnly: return WGPUStorageTextureAccess_ReadOnly
        case .readWrite: return WGPUStorageTextureAccess_ReadWrite
        }
    }

    static func color(_ color: LynxWebGPUCore.WGPUColor) -> WebGPU.WGPUColor {
        WebGPU.WGPUColor(r: color.red, g: color.green, b: color.blue, a: color.alpha)
    }

    static func origin(_ origin: LynxWebGPUCore.WGPUOrigin3D) -> WebGPU.WGPUOrigin3D {
        WebGPU.WGPUOrigin3D(x: UInt32(origin.x), y: UInt32(origin.y), z: UInt32(origin.z))
    }

    static func extent(_ extent: LynxWebGPUCore.WGPUExtent3D) -> WebGPU.WGPUExtent3D {
        WebGPU.WGPUExtent3D(
            width: UInt32(extent.width),
            height: UInt32(extent.height),
            depthOrArrayLayers: UInt32(extent.depthOrArrayLayers)
        )
    }

    /// Dawn 기능 → 명세 철자. 모르는 기능은 nil — 광고하지도, 요청하지도 않는다.
    static func featureLabel(_ feature: WGPUFeatureName) -> String? {
        switch feature {
        case WGPUFeatureName_DepthClipControl: return "depth-clip-control"
        case WGPUFeatureName_Depth32FloatStencil8: return "depth32float-stencil8"
        case WGPUFeatureName_TextureCompressionBC: return "texture-compression-bc"
        case WGPUFeatureName_TextureCompressionETC2: return "texture-compression-etc2"
        case WGPUFeatureName_TextureCompressionASTC: return "texture-compression-astc"
        case WGPUFeatureName_TimestampQuery: return "timestamp-query"
        case WGPUFeatureName_IndirectFirstInstance: return "indirect-first-instance"
        case WGPUFeatureName_ShaderF16: return "shader-f16"
        case WGPUFeatureName_RG11B10UfloatRenderable: return "rg11b10ufloat-renderable"
        case WGPUFeatureName_BGRA8UnormStorage: return "bgra8unorm-storage"
        case WGPUFeatureName_Float32Filterable: return "float32-filterable"
        default: return nil
        }
    }

    static func errorType(_ type: WebGPU.WGPUErrorType, message: String) -> WGPUError {
        if type == WGPUErrorType_Validation { return .validation("Dawn: \(message)") }
        if type == WGPUErrorType_OutOfMemory { return .outOfMemory("Dawn: \(message)") }
        return .backend("Dawn: \(message)")
    }
}
