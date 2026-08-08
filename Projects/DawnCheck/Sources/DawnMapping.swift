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
        switch format {
        case .r8unorm: return WGPUTextureFormat_R8Unorm
        case .r8uint: return WGPUTextureFormat_R8Uint
        case .r8sint: return WGPUTextureFormat_R8Sint
        case .r16uint: return WGPUTextureFormat_R16Uint
        case .r16sint: return WGPUTextureFormat_R16Sint
        case .r16float: return WGPUTextureFormat_R16Float
        case .rg8unorm: return WGPUTextureFormat_RG8Unorm
        case .r32uint: return WGPUTextureFormat_R32Uint
        case .r32sint: return WGPUTextureFormat_R32Sint
        case .r32float: return WGPUTextureFormat_R32Float
        case .rg16float: return WGPUTextureFormat_RG16Float
        case .rgba8unorm: return WGPUTextureFormat_RGBA8Unorm
        case .rgba8unormSRGB: return WGPUTextureFormat_RGBA8UnormSrgb
        case .rgba8uint: return WGPUTextureFormat_RGBA8Uint
        case .rgba8sint: return WGPUTextureFormat_RGBA8Sint
        case .bgra8unorm: return WGPUTextureFormat_BGRA8Unorm
        case .bgra8unormSRGB: return WGPUTextureFormat_BGRA8UnormSrgb
        case .rgb10a2unorm: return WGPUTextureFormat_RGB10A2Unorm
        case .rg32float: return WGPUTextureFormat_RG32Float
        case .rg32uint: return WGPUTextureFormat_RG32Uint
        case .rg32sint: return WGPUTextureFormat_RG32Sint
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
        default:
            throw WGPUError.unsupported("Dawn 런타임이 아직 안 옮긴 텍스처 포맷 \(format.rawValue)")
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

    static func errorType(_ type: WebGPU.WGPUErrorType, message: String) -> WGPUError {
        if type == WGPUErrorType_Validation { return .validation("Dawn: \(message)") }
        if type == WGPUErrorType_OutOfMemory { return .outOfMemory("Dawn: \(message)") }
        return .backend("Dawn: \(message)")
    }
}
