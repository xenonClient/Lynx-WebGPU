import Foundation
import Metal
import LynxWebGPUCore

/// WebGPU enum → Metal enum.
///
/// Values without a counterpart throw — quietly substituting something similar changes the render
/// result subtly and becomes a bug that is hard to trace.
enum WGPUMetalMapping {
    // MARK: - Pixel formats

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

        // --- Block compression ------------------------------------------------
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
        // Apple GPUs have no 24-bit depth format. WebGPU's depth24plus means "24-bit **or more**", so
        // promoting it to 32-bit float depth is the spec-conforming handling.
        case .depth24plus, .depth32float: return .depth32Float
        case .depth24plusStencil8, .depth32floatStencil8: return .depth32Float_stencil8
        }
    }

    /// The reverse mapping — used when telling JS what format the drawable texture actually is.
    /// (Canvas layer configuration is applied asynchronously on the main thread, so diagnosing a
    /// pipeline mismatch accurately requires **the actual texture's format**, not the requested one.)
    ///
    /// The table is built by **inverting** `pixelFormat(_:)` automatically. Written by hand it would
    /// grow on one side only as formats were added, going quietly incomplete (it once held just the
    /// few used by canvases).
    static func textureFormat(from pixelFormat: MTLPixelFormat) -> WGPUTextureFormat? {
        inverseFormatTable[pixelFormat]
    }

    /// Several WebGPU formats collapse onto the same Metal format (`depth24plus` and `depth32float`
    /// are both `.depth32Float`). In those places we pick **the one that states the precision
    /// honestly** — a recovered name weaker than the real texture misleads the reader.
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

    // MARK: - Vertex

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

    // MARK: - Depth / blending

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

    /// One face's (front/back) stencil state plus the pipeline-wide masks.
    ///
    /// In the spec the masks live once on `GPUDepthStencilState` and are not split per face.
    /// Metal holds them per face, so the same value goes into both.
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

    // MARK: - Samplers

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

    // MARK: - Textures

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
        // copySrc/copyDst are not separate usage bits in Metal — they are handled by the blit encoder.
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

/// **Device capabilities** coming from Metal feature sets — calling into one that is absent kills Metal with an assertion.
///
/// This project's validation principle is "leave it to Metal where possible", but **an assertion is
/// the exception**. Once the process is gone there is no chance to diagnose, so we filter here and report an error.
public enum WGPUDeviceCapability {
    /// Whether indirect draw/dispatch arguments (the `drawIndirect` family) are available.
    ///
    /// Metal feature set table: **Apple family 3 or above**, or Mac family 2. With iOS 17 as the
    /// minimum, real devices in this library are A12 (family 5) or newer and therefore **always
    /// support it**, but **the iOS simulator reports family 2** and drops out — calling it on an
    /// unsupported device kills the process with `MTLValidateFeatureSupport ... failed assertion`.
    public static func supportsIndirectArguments(_ device: MTLDevice) -> Bool {
        device.supportsFamily(.apple3) || device.supportsFamily(.mac2)
    }

    /// The compression family a format belongs to — the classification itself is a spec fact, so it
    /// lives in Core (`WGPUTextureCompressionFamily`; the engine's creation check uses the same one).
    /// An alias and a forwarder keep the old spelling working.
    public typealias CompressionFamily = WGPUTextureCompressionFamily

    public static func compressionFamily(_ format: WGPUTextureFormat) -> CompressionFamily {
        format.compressionFamily
    }

    /// Whether this device supports a block-compressed format.
    ///
    /// Metal **dies on an assertion if you create a texture in an unsupported compressed format.** So
    /// we advertise support through `adapter.features` and block an absent family once more at creation.
    ///
    /// - ETC2/EAC and ASTC: every Apple GPU does these (all iOS devices, Apple Silicon Macs).
    ///   Intel/AMD Macs (`mac2`) do not.
    /// - BC (DXT/BPTC): Apple7 (A14/M1) and above, or an Intel/AMD Mac.
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
