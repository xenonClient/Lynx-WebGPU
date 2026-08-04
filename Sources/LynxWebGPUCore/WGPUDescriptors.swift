import Foundation

// WebGPU 명세의 dictionary(descriptor)를 Swift 값 타입으로 옮긴 것.
// 각 타입은 `init(from:)`으로 커맨드 스트림의 `[String: Any]`에서 직접 만들어지며,
// 기본값은 명세의 default를 따른다 (명세에 default가 없는 필드만 required).

// MARK: - 리소스

public struct WGPUBufferDescriptor {
    public var size: Int
    public var usage: WGPUBufferUsage
    public var mappedAtCreation: Bool
    public var label: String?
    /// 명세 밖 확장: 생성과 동시에 채울 초기 데이터. `mappedAtCreation` + `getMappedRange` 왕복을
    /// 커맨드 하나로 줄인다 (JS shim의 `mappedAtCreation`이 이 필드로 내려온다).
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
        // size 생략 시 초기 데이터 길이로 유추한다.
        size = reader.optionalInt("size") ?? (initialData?.count ?? 0)
        usage = try reader.requiredFlags("usage", WGPUBufferUsage.self)
        mappedAtCreation = reader.bool("mappedAtCreation", default: false)
        label = reader.optionalString("label")
        self.initialData = initialData
        guard size > 0 else {
            throw WGPUError.validation("버퍼 크기는 1 이상이어야 한다", path: reader.path)
        }
        if let initialData, initialData.count > size {
            throw WGPUError.validation(
                "초기 데이터(\(initialData.count)B)가 버퍼 크기(\(size)B)를 넘는다", path: reader.path
            )
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
            throw WGPUError.validation("텍스처 크기의 모든 성분은 1 이상이어야 한다", path: reader.path)
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

// MARK: - 바인딩

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

/// 바인딩 슬롯 하나가 받는 리소스 종류. 명세상 `buffer`/`sampler`/`texture`/`storageTexture` 중 정확히 하나다.
public enum WGPUBindingLayout: Equatable {
    case buffer(WGPUBufferBindingLayout)
    case sampler(WGPUSamplerBindingLayout)
    case texture(WGPUTextureBindingLayout)
    case storageTexture(WGPUStorageTextureBindingLayout)

    /// Metal 인자 테이블 중 어디에 들어가는가 — 바인딩 인덱스 배정에 쓴다.
    public var metalSlotKind: WGPUMetalSlotKind {
        switch self {
        case .buffer: return .buffer
        case .sampler: return .sampler
        case .texture, .storageTexture: return .texture
        }
    }
}

/// Metal은 버퍼·텍스처·샘플러가 각각 독립된 인덱스 공간을 쓴다.
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
                "buffer / sampler / texture / storageTexture 중 하나를 지정해야 한다", path: reader.path
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

/// 바인드 그룹이 실제로 꽂는 리소스.
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
            resource = .buffer(
                handle: buffer,
                offset: resourceReader.int("offset", default: 0),
                size: resourceReader.optionalInt("size")
            )
        } else if let sampler = resourceReader.optionalHandle("sampler") {
            resource = .sampler(sampler)
        } else if let view = resourceReader.optionalHandle("textureView") {
            resource = .textureView(view)
        } else {
            throw WGPUError.validation(
                "buffer / sampler / textureView 중 하나의 핸들이 필요하다", path: resourceReader.path
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

/// 파이프라인이 쓰는 레이아웃. `"auto"`면 셰이더 리플렉션에서 유도한다.
public enum WGPUPipelineLayoutRef {
    case auto
    case explicit(WGPUHandle)

    public init(from reader: WGPUValueReader, key: String = "layout") throws {
        if let handle = reader.optionalHandle(key) {
            self = .explicit(handle)
        } else if let string = reader.optionalString(key) {
            guard string == "auto" else {
                throw WGPUError.validation("layout은 \"auto\" 또는 GPUPipelineLayout 핸들이어야 한다", path: reader.path)
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

// MARK: - 파이프라인

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
    public var entryPoint: String
    public var buffers: [WGPUVertexBufferLayout]
    /// 파이프라인 상수 (`override`) 값.
    public var constants: [String: Double]

    public init(from reader: WGPUValueReader) throws {
        module = try reader.requiredHandle("module")
        entryPoint = reader.string("entryPoint", default: "main")
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
    public var entryPoint: String
    public var targets: [WGPUColorTargetState]
    /// 파이프라인 상수 (`override`) 값.
    public var constants: [String: Double]

    public init(from reader: WGPUValueReader) throws {
        module = try reader.requiredHandle("module")
        entryPoint = reader.string("entryPoint", default: "main")
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

/// 앞면/뒷면 각각의 스텐실 동작 (`GPUStencilFaceState`).
///
/// 명세 기본값은 "아무것도 하지 않음"이다 — 비교는 `always`(항상 통과), 세 연산은 모두 `keep`.
/// 그래서 `stencilFront`/`stencilBack`을 주지 않으면 스텐실이 결과에 영향을 주지 않는다.
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

    /// 스텐실 값을 건드리지도 읽지도 않는 상태인가 — 기본값이면 상태를 만들 필요가 없다.
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
            throw WGPUError.validation("depthStencil.format은 깊이/스텐실 포맷이어야 한다", path: reader.path)
        }
        // 스텐실 상태를 줬는데 포맷에 스텐실 성분이 없으면 Metal은 조용히 무시한다 —
        // 스텐실 어태치먼트 없는 파이프라인에 스텐실 테스트만 붙은 물건이 오류 없이 생성되어
        // "스텐실 마스킹이 왜 안 먹지"를 단서 없이 디버깅하게 된다.
        guard !usesStencil || format.hasStencil else {
            throw WGPUError.validation(
                "stencilFront/stencilBack을 기본값 밖으로 주려면 format에 스텐실 성분이 있어야 한다 "
                    + "(받은 것: \(format.rawValue))",
                path: reader.path
            )
        }
    }

    /// 스텐실 테스트를 실제로 쓰는가 — 어느 면이든 기본값이 아니면 스텐실 상태를 만든다.
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
    public var entryPoint: String
    /// 파이프라인 상수 (`override`) 값.
    public var constants: [String: Double]
    public var label: String?

    public init(from reader: WGPUValueReader) throws {
        layout = try WGPUPipelineLayoutRef(from: reader)
        let compute = try reader.requiredObject("compute")
        module = try compute.requiredHandle("module")
        entryPoint = compute.string("entryPoint", default: "main")
        constants = compute.numberMap("constants")
        label = reader.optionalString("label")
    }
}

// MARK: - 렌더 패스

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

    public init(from reader: WGPUValueReader) throws {
        view = try reader.requiredHandle("view")
        depthClearValue = reader.double("depthClearValue", default: 1)
        depthLoadOp = try reader.optionalEnum("depthLoadOp", WGPULoadOp.self)
        depthStoreOp = try reader.optionalEnum("depthStoreOp", WGPUStoreOp.self)
        depthReadOnly = reader.bool("depthReadOnly", default: false)
        stencilClearValue = reader.int("stencilClearValue", default: 0)
        stencilLoadOp = try reader.optionalEnum("stencilLoadOp", WGPULoadOp.self)
        stencilStoreOp = try reader.optionalEnum("stencilStoreOp", WGPUStoreOp.self)
    }
}

/// 패스 경계에서 타임스탬프를 찍을 자리 (`GPURenderPassTimestampWrites`).
///
/// 두 인덱스는 각각 생략할 수 있다 — 시작만, 끝만 찍는 것도 명세상 유효하다.
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

// MARK: - 쿼리

public struct WGPUQuerySetDescriptor {
    public var type: WGPUQueryType
    public var count: Int
    public var label: String?

    public init(from reader: WGPUValueReader) throws {
        type = try reader.requiredEnum("type", WGPUQueryType.self)
        count = try reader.requiredInt("count")
        label = reader.optionalString("label")
        guard count > 0 else {
            throw WGPUError.validation("쿼리 개수는 1 이상이어야 한다", path: reader.fieldPath("count"))
        }
    }
}

// MARK: - 렌더 번들

/// `GPURenderBundleEncoder`의 디스크립터.
///
/// 번들은 **어떤 모양의 패스에서 실행될지**를 미리 선언한다. 그 선언이 실제 패스와 어긋나면
/// 명세상 오류다 — 이 구현은 번들을 명령 목록으로 되풀이하므로 Metal이 대신 잡아 주지 않아
/// `executeBundles`에서 직접 확인한다.
public struct WGPURenderBundleDescriptor {
    /// 컬러 어태치먼트별 포맷. `null`은 "그 슬롯은 비어 있다"는 뜻이다.
    public var colorFormats: [WGPUTextureFormat?]
    public var depthStencilFormat: WGPUTextureFormat?
    public var sampleCount: Int
    public var label: String?

    public init(from reader: WGPUValueReader) throws {
        colorFormats = try reader.strings("colorFormats").map { raw in
            guard let raw else { return nil }
            guard let format = WGPUTextureFormat(rawValue: raw) else {
                throw WGPUError.validation(
                    "알 수 없는 컬러 포맷 \"\(raw)\"", path: reader.fieldPath("colorFormats")
                )
            }
            return format
        }
        depthStencilFormat = try reader.optionalEnum("depthStencilFormat", WGPUTextureFormat.self)
        sampleCount = reader.int("sampleCount", default: 1)
        label = reader.optionalString("label")
    }
}

/// `<webgpu-canvas>` 표면 설정 (`GPUCanvasContext.configure`).
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
        // 명세에서 toneMapping은 `{mode: …}` 형태의 중첩 객체다.
        if let toneMapping = reader.object("toneMapping") {
            toneMappingMode = try toneMapping.enumValue(
                "mode", default: WGPUCanvasToneMappingMode.standard
            )
        } else {
            toneMappingMode = .standard
        }
    }
}
