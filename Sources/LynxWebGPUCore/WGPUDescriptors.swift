import Foundation

// The WebGPU spec's dictionaries (descriptors) carried over as Swift value types.
// Each type is built straight from the command stream's `[String: Any]` via `init(from:)`, and
// defaults follow the spec (only fields the spec gives no default are required).

// MARK: - Resources

public struct WGPUBufferDescriptor {
    public var size: Int
    public var usage: WGPUBufferUsage
    public var mappedAtCreation: Bool
    public var label: String?
    /// Extension beyond the spec: initial data to fill at creation. Collapses the
    /// `mappedAtCreation` + `getMappedRange` round trip into one command (the JS shim's
    /// `mappedAtCreation` arrives through this field).
    public var initialData: Data?

    public init(size: Int, usage: WGPUBufferUsage, mappedAtCreation: Bool = false, label: String? = nil, initialData: Data? = nil) {
        self.size = size
        self.usage = usage
        self.mappedAtCreation = mappedAtCreation
        self.label = label
        self.initialData = initialData
    }

    public init(from reader: WGPUValueReader) throws {
        let initialData = try reader.optionalData("data")
        // With size omitted, infer it from the initial data length.
        size = reader.optionalInt("size") ?? (initialData?.count ?? 0)
        usage = try reader.requiredFlags("usage", WGPUBufferUsage.self)
        mappedAtCreation = reader.bool("mappedAtCreation", default: false)
        label = reader.optionalString("label")
        self.initialData = initialData
        guard size > 0 else {
            throw WGPUError.validation("buffer size must be at least 1", path: reader.path)
        }
        if let initialData, initialData.count > size {
            throw WGPUError.validation(
                "initial data (\(initialData.count)B) exceeds the buffer size (\(size)B)", path: reader.path
            )
        }
        // Mapping usage may combine only with copies. Metal covers everything with `.storageModeShared`,
        // but unchecked here a combination like `QUERY_RESOLVE | MAP_READ` is rejected only in a browser.
        try Self.validateMapUsage(usage, path: reader.path)
    }

    /// The spec's mapping-usage exclusivity — `MAP_READ` only with `COPY_DST`, `MAP_WRITE` only with `COPY_SRC`.
    static func validateMapUsage(_ usage: WGPUBufferUsage, path: String?) throws {
        let rules: [(flag: WGPUBufferUsage, name: String, allowed: WGPUBufferUsage, allowedName: String)] = [
            (.mapRead, "MAP_READ", .copyDst, "COPY_DST"),
            (.mapWrite, "MAP_WRITE", .copySrc, "COPY_SRC"),
        ]
        for rule in rules where usage.contains(rule.flag) {
            let extra = usage.subtracting([rule.flag, rule.allowed])
            guard extra.isEmpty else {
                throw WGPUError.validation(
                    "\(rule.name) cannot combine with usages other than \(rule.allowedName) "
                        + "— for readback, copyBufferToBuffer into a \(rule.allowedName) buffer and map that",
                    path: path
                )
            }
        }
    }
}

public struct WGPUTextureDescriptor {
    public var size: WGPUExtent3D
    public var format: WGPUTextureFormat
    public var usage: WGPUTextureUsage
    public var dimension: WGPUTextureDimension
    public var mipLevelCount: Int
    public var sampleCount: Int
    public var label: String?

    public init(from reader: WGPUValueReader) throws {
        size = try reader.requiredExtent("size")
        format = try reader.requiredEnum("format", WGPUTextureFormat.self)
        usage = try reader.requiredFlags("usage", WGPUTextureUsage.self)
        dimension = try reader.enumValue("dimension", default: WGPUTextureDimension.twoD)
        mipLevelCount = reader.int("mipLevelCount", default: 1)
        sampleCount = reader.int("sampleCount", default: 1)
        label = reader.optionalString("label")
        guard size.width > 0, size.height > 0, size.depthOrArrayLayers > 0 else {
            throw WGPUError.validation("every component of a texture size must be at least 1", path: reader.path)
        }
    }
}

public struct WGPUTextureViewDescriptor {
    public var format: WGPUTextureFormat?
    public var dimension: WGPUTextureViewDimension?
    public var aspect: WGPUTextureAspect
    public var baseMipLevel: Int
    public var mipLevelCount: Int?
    public var baseArrayLayer: Int
    public var arrayLayerCount: Int?
    public var label: String?

    public init(from reader: WGPUValueReader) throws {
        format = try reader.optionalEnum("format", WGPUTextureFormat.self)
        dimension = try reader.optionalEnum("dimension", WGPUTextureViewDimension.self)
        aspect = try reader.enumValue("aspect", default: WGPUTextureAspect.all)
        baseMipLevel = reader.int("baseMipLevel", default: 0)
        mipLevelCount = reader.optionalInt("mipLevelCount")
        baseArrayLayer = reader.int("baseArrayLayer", default: 0)
        arrayLayerCount = reader.optionalInt("arrayLayerCount")
        label = reader.optionalString("label")
    }
}

public struct WGPUSamplerDescriptor {
    public var addressModeU: WGPUAddressMode
    public var addressModeV: WGPUAddressMode
    public var addressModeW: WGPUAddressMode
    public var magFilter: WGPUFilterMode
    public var minFilter: WGPUFilterMode
    public var mipmapFilter: WGPUFilterMode
    public var lodMinClamp: Double
    public var lodMaxClamp: Double
    public var compare: WGPUCompareFunction?
    public var maxAnisotropy: Int
    public var label: String?

    public init(from reader: WGPUValueReader) throws {
        addressModeU = try reader.enumValue("addressModeU", default: WGPUAddressMode.clampToEdge)
        addressModeV = try reader.enumValue("addressModeV", default: WGPUAddressMode.clampToEdge)
        addressModeW = try reader.enumValue("addressModeW", default: WGPUAddressMode.clampToEdge)
        magFilter = try reader.enumValue("magFilter", default: WGPUFilterMode.nearest)
        minFilter = try reader.enumValue("minFilter", default: WGPUFilterMode.nearest)
        mipmapFilter = try reader.enumValue("mipmapFilter", default: WGPUFilterMode.nearest)
        lodMinClamp = reader.double("lodMinClamp", default: 0)
        lodMaxClamp = reader.double("lodMaxClamp", default: 32)
        compare = try reader.optionalEnum("compare", WGPUCompareFunction.self)
        maxAnisotropy = reader.int("maxAnisotropy", default: 1)
        label = reader.optionalString("label")
    }
}

public struct WGPUShaderModuleDescriptor {
    public var code: String
    public var language: WGPUShaderLanguage
    public var label: String?

    public init(code: String, language: WGPUShaderLanguage = .wgsl, label: String? = nil) {
        self.code = code
        self.language = language
        self.label = label
    }

    public init(from reader: WGPUValueReader) throws {
        code = try reader.requiredString("code")
        language = try reader.enumValue("language", default: WGPUShaderLanguage.wgsl)
        label = reader.optionalString("label")
    }
}

// MARK: - Bindings

public struct WGPUBufferBindingLayout: Equatable {
    public var type: WGPUBufferBindingType
    public var hasDynamicOffset: Bool
    public var minBindingSize: Int

    public init(type: WGPUBufferBindingType = .uniform, hasDynamicOffset: Bool = false, minBindingSize: Int = 0) {
        self.type = type
        self.hasDynamicOffset = hasDynamicOffset
        self.minBindingSize = minBindingSize
    }
}

public struct WGPUSamplerBindingLayout: Equatable {
    public var type: WGPUSamplerBindingType

    public init(type: WGPUSamplerBindingType = .filtering) { self.type = type }
}

public struct WGPUTextureBindingLayout: Equatable {
    public var sampleType: WGPUTextureSampleType
    public var viewDimension: WGPUTextureViewDimension
    public var multisampled: Bool

    public init(
        sampleType: WGPUTextureSampleType = .float,
        viewDimension: WGPUTextureViewDimension = .twoD,
        multisampled: Bool = false
    ) {
        self.sampleType = sampleType
        self.viewDimension = viewDimension
        self.multisampled = multisampled
    }
}

public struct WGPUStorageTextureBindingLayout: Equatable {
    public var access: WGPUStorageTextureAccess
    public var format: WGPUTextureFormat
    public var viewDimension: WGPUTextureViewDimension

    public init(
        access: WGPUStorageTextureAccess = .writeOnly,
        format: WGPUTextureFormat,
        viewDimension: WGPUTextureViewDimension = .twoD
    ) {
        self.access = access
        self.format = format
        self.viewDimension = viewDimension
    }
}

/// The resource kind one binding slot takes. Per spec exactly one of `buffer`/`sampler`/`texture`/`storageTexture`.
public enum WGPUBindingLayout: Equatable {
    case buffer(WGPUBufferBindingLayout)
    case sampler(WGPUSamplerBindingLayout)
    case texture(WGPUTextureBindingLayout)
    case storageTexture(WGPUStorageTextureBindingLayout)

    /// Which Metal argument table it lands in — used when assigning binding indices.
    public var metalSlotKind: WGPUMetalSlotKind {
        switch self {
        case .buffer: return .buffer
        case .sampler: return .sampler
        case .texture, .storageTexture: return .texture
        }
    }
}

/// Metal gives buffers, textures and samplers independent index spaces.
public enum WGPUMetalSlotKind: String, Equatable, Sendable {
    case buffer, texture, sampler
}

public struct WGPUBindGroupLayoutEntry: Equatable {
    public var binding: Int
    public var visibility: WGPUShaderStage
    public var layout: WGPUBindingLayout

    public init(binding: Int, visibility: WGPUShaderStage, layout: WGPUBindingLayout) {
        self.binding = binding
        self.visibility = visibility
        self.layout = layout
    }

    public init(from reader: WGPUValueReader) throws {
        binding = try reader.requiredInt("binding")
        visibility = try reader.requiredFlags("visibility", WGPUShaderStage.self)

        if let buffer = reader.object("buffer") {
            layout = .buffer(WGPUBufferBindingLayout(
                type: try buffer.enumValue("type", default: WGPUBufferBindingType.uniform),
                hasDynamicOffset: buffer.bool("hasDynamicOffset", default: false),
                minBindingSize: buffer.int("minBindingSize", default: 0)
            ))
        } else if let sampler = reader.object("sampler") {
            layout = .sampler(WGPUSamplerBindingLayout(
                type: try sampler.enumValue("type", default: WGPUSamplerBindingType.filtering)
            ))
        } else if let texture = reader.object("texture") {
            layout = .texture(WGPUTextureBindingLayout(
                sampleType: try texture.enumValue("sampleType", default: WGPUTextureSampleType.float),
                viewDimension: try texture.enumValue("viewDimension", default: WGPUTextureViewDimension.twoD),
                multisampled: texture.bool("multisampled", default: false)
            ))
        } else if let storage = reader.object("storageTexture") {
            layout = .storageTexture(WGPUStorageTextureBindingLayout(
                access: try storage.enumValue("access", default: WGPUStorageTextureAccess.writeOnly),
                format: try storage.requiredEnum("format", WGPUTextureFormat.self),
                viewDimension: try storage.enumValue("viewDimension", default: WGPUTextureViewDimension.twoD)
            ))
        } else {
            throw WGPUError.validation(
                "specify exactly one of buffer / sampler / texture / storageTexture", path: reader.path
            )
        }
    }
}

public struct WGPUBindGroupLayoutDescriptor {
    public var entries: [WGPUBindGroupLayoutEntry]
    public var label: String?

    public init(entries: [WGPUBindGroupLayoutEntry], label: String? = nil) {
        self.entries = entries
        self.label = label
    }

    public init(from reader: WGPUValueReader) throws {
        entries = try reader.requiredObjects("entries").map(WGPUBindGroupLayoutEntry.init(from:))
        label = reader.optionalString("label")
    }
}

/// The resource a bind group actually plugs in.
public enum WGPUBindingResource {
    case buffer(handle: WGPUHandle, offset: Int, size: Int?)
    case sampler(WGPUHandle)
    case textureView(WGPUHandle)
}

public struct WGPUBindGroupEntry {
    public var binding: Int
    public var resource: WGPUBindingResource

    public init(from reader: WGPUValueReader) throws {
        binding = try reader.requiredInt("binding")
        let resourceReader = try reader.requiredObject("resource")
        if let buffer = resourceReader.optionalHandle("buffer") {
            let offset = resourceReader.int("offset", default: 0)
            let size = resourceReader.optionalInt("size")
            // A negative value traps the UInt32 conversion when filling the size table for
            // `arrayLength()` — filtering here is what keeps the "bad arguments never kill the
            // process" contract.
            guard offset >= 0, size.map({ $0 > 0 }) ?? true else {
                throw WGPUError.validation(
                    "a binding's offset must be >= 0 and its size >= 1", path: resourceReader.path
                )
            }
            resource = .buffer(handle: buffer, offset: offset, size: size)
        } else if let sampler = resourceReader.optionalHandle("sampler") {
            resource = .sampler(sampler)
        } else if let view = resourceReader.optionalHandle("textureView") {
            resource = .textureView(view)
        } else {
            throw WGPUError.validation(
                "a handle for one of buffer / sampler / textureView is required", path: resourceReader.path
            )
        }
    }
}

public struct WGPUBindGroupDescriptor {
    public var layout: WGPUHandle
    public var entries: [WGPUBindGroupEntry]
    public var label: String?

    public init(from reader: WGPUValueReader) throws {
        layout = try reader.requiredHandle("layout")
        entries = try reader.requiredObjects("entries").map(WGPUBindGroupEntry.init(from:))
        label = reader.optionalString("label")
    }
}

/// The layout a pipeline uses. `"auto"` derives it from shader reflection.
public enum WGPUPipelineLayoutRef {
    case auto
    case explicit(WGPUHandle)

    public init(from reader: WGPUValueReader, key: String = "layout") throws {
        if let handle = reader.optionalHandle(key) {
            self = .explicit(handle)
        } else if let string = reader.optionalString(key) {
            guard string == "auto" else {
                throw WGPUError.validation("layout must be \"auto\" or a GPUPipelineLayout handle", path: reader.path)
            }
            self = .auto
        } else {
            self = .auto
        }
    }
}

public struct WGPUPipelineLayoutDescriptor {
    public var bindGroupLayouts: [WGPUHandle]
    public var label: String?

    public init(from reader: WGPUValueReader) throws {
        bindGroupLayouts = try reader.handles("bindGroupLayouts")
        label = reader.optionalString("label")
    }
}

// MARK: - Pipelines

public struct WGPUVertexAttribute: Equatable {
    public var format: WGPUVertexFormat
    public var offset: Int
    public var shaderLocation: Int

    public init(from reader: WGPUValueReader) throws {
        format = try reader.requiredEnum("format", WGPUVertexFormat.self)
        offset = reader.int("offset", default: 0)
        shaderLocation = try reader.requiredInt("shaderLocation")
    }
}

public struct WGPUVertexBufferLayout: Equatable {
    public var arrayStride: Int
    public var stepMode: WGPUVertexStepMode
    public var attributes: [WGPUVertexAttribute]

    public init(from reader: WGPUValueReader) throws {
        arrayStride = try reader.requiredInt("arrayStride")
        stepMode = try reader.enumValue("stepMode", default: WGPUVertexStepMode.vertex)
        attributes = try reader.requiredObjects("attributes").map(WGPUVertexAttribute.init(from:))
    }
}

public struct WGPUVertexState {
    public var module: WGPUHandle
    /// May be omitted — then **the stage's only entry point** is used (the spec's "get the entry point").
    public var entryPoint: String?
    public var buffers: [WGPUVertexBufferLayout]
    /// Pipeline constant (`override`) values.
    public var constants: [String: Double]

    public init(from reader: WGPUValueReader) throws {
        module = try reader.requiredHandle("module")
        entryPoint = reader.optionalString("entryPoint")
        buffers = try reader.objects("buffers").map(WGPUVertexBufferLayout.init(from:))
        constants = reader.numberMap("constants")
    }
}

public struct WGPUBlendComponent: Equatable {
    public var operation: WGPUBlendOperation
    public var srcFactor: WGPUBlendFactor
    public var dstFactor: WGPUBlendFactor

    public init(from reader: WGPUValueReader?) throws {
        guard let reader else {
            operation = .add
            srcFactor = .one
            dstFactor = .zero
            return
        }
        operation = try reader.enumValue("operation", default: WGPUBlendOperation.add)
        srcFactor = try reader.enumValue("srcFactor", default: WGPUBlendFactor.one)
        dstFactor = try reader.enumValue("dstFactor", default: WGPUBlendFactor.zero)
    }
}

public struct WGPUBlendState: Equatable {
    public var color: WGPUBlendComponent
    public var alpha: WGPUBlendComponent

    public init(from reader: WGPUValueReader) throws {
        color = try WGPUBlendComponent(from: reader.object("color"))
        alpha = try WGPUBlendComponent(from: reader.object("alpha"))
    }
}

public struct WGPUColorTargetState {
    public var format: WGPUTextureFormat
    public var blend: WGPUBlendState?
    public var writeMask: WGPUColorWriteMask

    public init(from reader: WGPUValueReader) throws {
        format = try reader.requiredEnum("format", WGPUTextureFormat.self)
        blend = try reader.object("blend").map(WGPUBlendState.init(from:))
        writeMask = reader.flags("writeMask", WGPUColorWriteMask.self, default: .all)
    }
}

public struct WGPUFragmentState {
    public var module: WGPUHandle
    /// May be omitted — same rule as `WGPUVertexState.entryPoint`.
    public var entryPoint: String?
    public var targets: [WGPUColorTargetState]
    /// Pipeline constant (`override`) values.
    public var constants: [String: Double]

    public init(from reader: WGPUValueReader) throws {
        module = try reader.requiredHandle("module")
        entryPoint = reader.optionalString("entryPoint")
        targets = try reader.requiredObjects("targets").map(WGPUColorTargetState.init(from:))
        constants = reader.numberMap("constants")
    }
}

public struct WGPUPrimitiveState {
    public var topology: WGPUPrimitiveTopology
    public var stripIndexFormat: WGPUIndexFormat?
    public var frontFace: WGPUFrontFace
    public var cullMode: WGPUCullMode

    public init(from reader: WGPUValueReader?) throws {
        guard let reader else {
            topology = .triangleList
            stripIndexFormat = nil
            frontFace = .ccw
            cullMode = .none
            return
        }
        topology = try reader.enumValue("topology", default: WGPUPrimitiveTopology.triangleList)
        stripIndexFormat = try reader.optionalEnum("stripIndexFormat", WGPUIndexFormat.self)
        frontFace = try reader.enumValue("frontFace", default: WGPUFrontFace.ccw)
        cullMode = try reader.enumValue("cullMode", default: WGPUCullMode.none)
    }
}

/// Per-face stencil behaviour (`GPUStencilFaceState`).
///
/// The spec default is "do nothing" — compare `always` (always passes) and all three ops `keep`.
/// So omitting `stencilFront`/`stencilBack` leaves stencil with no effect on the result.
public struct WGPUStencilFaceState: Equatable {
    public var compare: WGPUCompareFunction
    public var failOp: WGPUStencilOperation
    public var depthFailOp: WGPUStencilOperation
    public var passOp: WGPUStencilOperation

    public init(
        compare: WGPUCompareFunction = .always,
        failOp: WGPUStencilOperation = .keep,
        depthFailOp: WGPUStencilOperation = .keep,
        passOp: WGPUStencilOperation = .keep
    ) {
        self.compare = compare
        self.failOp = failOp
        self.depthFailOp = depthFailOp
        self.passOp = passOp
    }

    public init(from reader: WGPUValueReader?) throws {
        guard let reader else {
            self.init()
            return
        }
        self.init(
            compare: try reader.enumValue("compare", default: WGPUCompareFunction.always),
            failOp: try reader.enumValue("failOp", default: WGPUStencilOperation.keep),
            depthFailOp: try reader.enumValue("depthFailOp", default: WGPUStencilOperation.keep),
            passOp: try reader.enumValue("passOp", default: WGPUStencilOperation.keep)
        )
    }

    /// Whether this state never **writes** stencil — the condition a `stencilReadOnly` pass allows.
    /// (Comparing is reading, so `compare` is not consulted.)
    public var isWriteFree: Bool {
        failOp == .keep && depthFailOp == .keep && passOp == .keep
    }

    /// Whether it neither touches nor reads stencil — at the default there is no state to build.
    public var isNoOp: Bool { self == WGPUStencilFaceState() }
}

public struct WGPUDepthStencilState {
    public var format: WGPUTextureFormat
    public var depthWriteEnabled: Bool
    public var depthCompare: WGPUCompareFunction
    public var depthBias: Int
    public var depthBiasSlopeScale: Double
    public var depthBiasClamp: Double
    public var stencilFront: WGPUStencilFaceState
    public var stencilBack: WGPUStencilFaceState
    public var stencilReadMask: Int
    public var stencilWriteMask: Int

    public init(from reader: WGPUValueReader) throws {
        format = try reader.requiredEnum("format", WGPUTextureFormat.self)
        depthWriteEnabled = reader.bool("depthWriteEnabled", default: false)
        depthCompare = try reader.enumValue("depthCompare", default: WGPUCompareFunction.always)
        depthBias = reader.int("depthBias", default: 0)
        depthBiasSlopeScale = reader.double("depthBiasSlopeScale", default: 0)
        depthBiasClamp = reader.double("depthBiasClamp", default: 0)
        stencilFront = try WGPUStencilFaceState(from: reader.object("stencilFront"))
        stencilBack = try WGPUStencilFaceState(from: reader.object("stencilBack"))
        stencilReadMask = reader.int("stencilReadMask", default: 0xFFFF_FFFF)
        stencilWriteMask = reader.int("stencilWriteMask", default: 0xFFFF_FFFF)
        guard format.isDepthOrStencil else {
            throw WGPUError.validation("depthStencil.format must be a depth/stencil format", path: reader.path)
        }
        // Give stencil state on a format with no stencil aspect and Metal ignores it silently — a
        // pipeline with a stencil test but no stencil attachment is created without error, leaving
        // you to debug "why isn't stencil masking working" with no clue at all.
        guard !usesStencil || format.hasStencil else {
            throw WGPUError.validation(
                "to give stencilFront/stencilBack beyond their defaults, format must have a stencil aspect "
                    + "(got: \(format.rawValue))",
                path: reader.path
            )
        }
    }

    /// Whether stencil testing is actually used — any non-default face means we build stencil state.
    public var usesStencil: Bool { !stencilFront.isNoOp || !stencilBack.isNoOp }
}

public struct WGPUMultisampleState {
    public var count: Int
    public var mask: Int
    public var alphaToCoverageEnabled: Bool

    public init(from reader: WGPUValueReader?) {
        count = reader?.int("count", default: 1) ?? 1
        mask = reader?.int("mask", default: 0xFFFF_FFFF) ?? 0xFFFF_FFFF
        alphaToCoverageEnabled = reader?.bool("alphaToCoverageEnabled", default: false) ?? false
    }
}

public struct WGPURenderPipelineDescriptor {
    public var layout: WGPUPipelineLayoutRef
    public var vertex: WGPUVertexState
    public var primitive: WGPUPrimitiveState
    public var depthStencil: WGPUDepthStencilState?
    public var multisample: WGPUMultisampleState
    public var fragment: WGPUFragmentState?
    public var label: String?

    public init(from reader: WGPUValueReader) throws {
        layout = try WGPUPipelineLayoutRef(from: reader)
        vertex = try WGPUVertexState(from: reader.requiredObject("vertex"))
        primitive = try WGPUPrimitiveState(from: reader.object("primitive"))
        depthStencil = try reader.object("depthStencil").map(WGPUDepthStencilState.init(from:))
        multisample = WGPUMultisampleState(from: reader.object("multisample"))
        fragment = try reader.object("fragment").map(WGPUFragmentState.init(from:))
        label = reader.optionalString("label")
    }
}

public struct WGPUComputePipelineDescriptor {
    public var layout: WGPUPipelineLayoutRef
    public var module: WGPUHandle
    /// May be omitted — same rule as `WGPUVertexState.entryPoint`.
    public var entryPoint: String?
    /// Pipeline constant (`override`) values.
    public var constants: [String: Double]
    public var label: String?

    public init(from reader: WGPUValueReader) throws {
        layout = try WGPUPipelineLayoutRef(from: reader)
        let compute = try reader.requiredObject("compute")
        module = try compute.requiredHandle("module")
        entryPoint = compute.optionalString("entryPoint")
        constants = compute.numberMap("constants")
        label = reader.optionalString("label")
    }
}

// MARK: - Render pass

public struct WGPURenderPassColorAttachment {
    public var view: WGPUHandle
    public var resolveTarget: WGPUHandle?
    public var clearValue: WGPUColor
    public var loadOp: WGPULoadOp
    public var storeOp: WGPUStoreOp

    public init(from reader: WGPUValueReader) throws {
        view = try reader.requiredHandle("view")
        resolveTarget = reader.optionalHandle("resolveTarget")
        clearValue = try reader.color("clearValue", default: .transparent)
        loadOp = try reader.enumValue("loadOp", default: WGPULoadOp.clear)
        storeOp = try reader.enumValue("storeOp", default: WGPUStoreOp.store)
    }
}

public struct WGPURenderPassDepthStencilAttachment {
    public var view: WGPUHandle
    public var depthClearValue: Double
    public var depthLoadOp: WGPULoadOp?
    public var depthStoreOp: WGPUStoreOp?
    public var depthReadOnly: Bool
    public var stencilClearValue: Int
    public var stencilLoadOp: WGPULoadOp?
    public var stencilStoreOp: WGPUStoreOp?
    public var stencilReadOnly: Bool

    public init(from reader: WGPUValueReader) throws {
        view = try reader.requiredHandle("view")
        depthClearValue = reader.double("depthClearValue", default: 1)
        depthLoadOp = try reader.optionalEnum("depthLoadOp", WGPULoadOp.self)
        depthStoreOp = try reader.optionalEnum("depthStoreOp", WGPUStoreOp.self)
        depthReadOnly = reader.bool("depthReadOnly", default: false)
        stencilClearValue = reader.int("stencilClearValue", default: 0)
        stencilLoadOp = try reader.optionalEnum("stencilLoadOp", WGPULoadOp.self)
        stencilStoreOp = try reader.optionalEnum("stencilStoreOp", WGPUStoreOp.self)
        stencilReadOnly = reader.bool("stencilReadOnly", default: false)
        // The spec forbids giving readOnly together with load/store ops — they contradict each other
        // (declaring "I will not write" while specifying how to store).
        guard !depthReadOnly || (depthLoadOp == nil && depthStoreOp == nil) else {
            throw WGPUError.validation(
                "depthReadOnly cannot be combined with depthLoadOp/depthStoreOp", path: reader.path
            )
        }
        guard !stencilReadOnly || (stencilLoadOp == nil && stencilStoreOp == nil) else {
            throw WGPUError.validation(
                "stencilReadOnly cannot be combined with stencilLoadOp/stencilStoreOp", path: reader.path
            )
        }
    }
}

/// Where timestamps are taken at pass boundaries (`GPURenderPassTimestampWrites`).
///
/// Either index may be omitted — sampling only the start or only the end is valid per spec.
public struct WGPUPassTimestampWrites {
    public var querySet: WGPUHandle
    public var beginningOfPassWriteIndex: Int?
    public var endOfPassWriteIndex: Int?

    public init(from reader: WGPUValueReader) throws {
        querySet = try reader.requiredHandle("querySet")
        beginningOfPassWriteIndex = reader.optionalInt("beginningOfPassWriteIndex")
        endOfPassWriteIndex = reader.optionalInt("endOfPassWriteIndex")
    }
}

public struct WGPURenderPassDescriptor {
    public var colorAttachments: [WGPURenderPassColorAttachment]
    public var depthStencilAttachment: WGPURenderPassDepthStencilAttachment?
    public var occlusionQuerySet: WGPUHandle?
    public var timestampWrites: WGPUPassTimestampWrites?
    public var label: String?

    public init(from reader: WGPUValueReader) throws {
        colorAttachments = try reader.requiredObjects("colorAttachments")
            .map(WGPURenderPassColorAttachment.init(from:))
        depthStencilAttachment = try reader.object("depthStencilAttachment")
            .map(WGPURenderPassDepthStencilAttachment.init(from:))
        occlusionQuerySet = reader.optionalHandle("occlusionQuerySet")
        timestampWrites = try reader.object("timestampWrites").map(WGPUPassTimestampWrites.init(from:))
        label = reader.optionalString("label")
    }
}

public struct WGPUComputePassDescriptor {
    public var timestampWrites: WGPUPassTimestampWrites?
    public var label: String?

    public init(from reader: WGPUValueReader?) throws {
        timestampWrites = try reader?.object("timestampWrites").map(WGPUPassTimestampWrites.init(from:))
        label = reader?.optionalString("label")
    }
}

// MARK: - Queries

public struct WGPUQuerySetDescriptor {
    /// The spec's cap on query set size (`GPUQuerySetDescriptor.count` <= 4096).
    public static let maxCount = 4096

    public var type: WGPUQueryType
    public var count: Int
    public var label: String?

    public init(from reader: WGPUValueReader) throws {
        type = try reader.requiredEnum("type", WGPUQueryType.self)
        count = try reader.requiredInt("count")
        label = reader.optionalString("label")
        // The cap comes from the spec. Metal accepts far larger query sets, but unchecked here
        // code ships that breaks only in a browser (same standard as `INDIRECT`/`QUERY_RESOLVE` usage).
        guard count > 0, count <= WGPUQuerySetDescriptor.maxCount else {
            throw WGPUError.validation(
                "query count must be between 1 and \(WGPUQuerySetDescriptor.maxCount) (got \(count))",
                path: reader.fieldPath("count")
            )
        }
    }
}

// MARK: - Render bundles

/// Descriptor for `GPURenderBundleEncoder`.
///
/// A bundle declares up front **what shape of pass it will run in**. A mismatch with the actual pass
/// is a spec error — and since this implementation replays a bundle as a command list, Metal will
/// not catch it, so `executeBundles` checks directly.
public struct WGPURenderBundleDescriptor {
    /// Format per color attachment. `null` means "that slot is empty".
    public var colorFormats: [WGPUTextureFormat?]
    public var depthStencilFormat: WGPUTextureFormat?
    public var sampleCount: Int
    /// Whether this bundle declared it will **not write** depth/stencil.
    /// Running in a read-only pass requires the bundle to be read-only too (spec requirement).
    public var depthReadOnly: Bool
    public var stencilReadOnly: Bool
    public var label: String?

    public init(from reader: WGPUValueReader) throws {
        colorFormats = try reader.strings("colorFormats").map { raw in
            guard let raw else { return nil }
            guard let format = WGPUTextureFormat(rawValue: raw) else {
                throw WGPUError.validation(
                    "unknown color format \"\(raw)\"", path: reader.fieldPath("colorFormats")
                )
            }
            return format
        }
        depthStencilFormat = try reader.optionalEnum("depthStencilFormat", WGPUTextureFormat.self)
        sampleCount = reader.int("sampleCount", default: 1)
        depthReadOnly = reader.bool("depthReadOnly", default: false)
        stencilReadOnly = reader.bool("stencilReadOnly", default: false)
        label = reader.optionalString("label")
        // The spec requires at least one attachment. Without one this implementation eventually fails
        // in `makeRenderCommandEncoder` too, but the error lands frames later somewhere unrelated.
        guard colorFormats.contains(where: { $0 != nil }) || depthStencilFormat != nil else {
            throw WGPUError.validation(
                "a bundle needs at least one attachment "
                    + "(one non-null entry in colorFormats, or depthStencilFormat)",
                path: reader.path
            )
        }
    }
}

/// `<webgpu-canvas>` surface configuration (`GPUCanvasContext.configure`).
public struct WGPUCanvasConfiguration {
    public var canvasId: String
    public var format: WGPUTextureFormat
    public var usage: WGPUTextureUsage
    public var alphaMode: WGPUCanvasAlphaMode
    public var colorSpace: WGPUPredefinedColorSpace
    public var toneMappingMode: WGPUCanvasToneMappingMode

    public init(from reader: WGPUValueReader) throws {
        canvasId = try reader.requiredString("canvas")
        format = try reader.enumValue("format", default: WGPUTextureFormat.bgra8unorm)
        usage = reader.flags("usage", WGPUTextureUsage.self, default: .renderAttachment)
        alphaMode = try reader.enumValue("alphaMode", default: WGPUCanvasAlphaMode.opaque)
        colorSpace = try reader.enumValue("colorSpace", default: WGPUPredefinedColorSpace.srgb)
        // In the spec toneMapping is a nested object of the form `{mode: …}`.
        if let toneMapping = reader.object("toneMapping") {
            toneMappingMode = try toneMapping.enumValue(
                "mode", default: WGPUCanvasToneMappingMode.standard
            )
        } else {
            toneMappingMode = .standard
        }
    }
}
