import Foundation
import Metal
import LynxWebGPUCore
import LynxWebGPUShader

/// `GPUBindGroupLayout`.
public final class WGPUBindGroupLayoutObject {
    public let entries: [WGPUBindGroupLayoutEntry]

    init(entries: [WGPUBindGroupLayoutEntry]) {
        self.entries = entries.sorted { $0.binding < $1.binding }
    }

    func entry(binding: Int) -> WGPUBindGroupLayoutEntry? {
        entries.first { $0.binding == binding }
    }
}

/// `GPUPipelineLayout` — 그룹 목록과 그로부터 계산된 Metal 인덱스 배정을 함께 들고 있다.
public final class WGPUPipelineLayoutObject {
    public let groups: [WGPUBindGroupLayoutObject]
    let assignment: WGSLBindingAssignment

    init(groups: [WGPUBindGroupLayoutObject]) throws {
        self.groups = groups
        self.assignment = try WGSLBindingAssigner.assign(groups: groups.map(\.entries))
    }

    func group(at index: Int) -> WGPUBindGroupLayoutObject? {
        index >= 0 && index < groups.count ? groups[index] : nil
    }
}

/// 바인드 그룹이 실제로 가리키는 Metal 객체.
enum WGPUResolvedBinding {
    /// `boundSize`는 이 바인딩이 보는 바이트 수 — `arrayLength()`가 이 값을 쓴다.
    case buffer(MTLBuffer, offset: Int, boundSize: Int)
    case sampler(MTLSamplerState)
    case texture(MTLTexture)
}

/// `GPUBindGroup`.
public final class WGPUBindGroupObject {
    let layout: WGPUBindGroupLayoutObject
    let bindings: [(binding: Int, visibility: WGPUShaderStage, resource: WGPUResolvedBinding)]

    init(layout: WGPUBindGroupLayoutObject, descriptor: WGPUBindGroupDescriptor, registry: WGPUObjectRegistry) throws {
        self.layout = layout
        self.bindings = try descriptor.entries.map { entry in
            guard let layoutEntry = layout.entry(binding: entry.binding) else {
                throw WGPUError.validation("바인드 그룹 레이아웃에 binding \(entry.binding)이 없다")
            }
            let resolved: WGPUResolvedBinding
            switch entry.resource {
            case .buffer(let handle, let offset, let size):
                let object = try registry.lookup(handle, as: WGPUBufferObject.self, kind: "GPUBuffer")
                resolved = .buffer(
                    object.buffer, offset: offset, boundSize: size ?? max(object.size - offset, 0)
                )
            case .sampler(let handle):
                let object = try registry.lookup(handle, as: WGPUSamplerObject.self, kind: "GPUSampler")
                resolved = .sampler(object.sampler)
            case .textureView(let handle):
                let object = try registry.lookup(handle, as: WGPUTextureViewObject.self, kind: "GPUTextureView")
                resolved = .texture(object.texture)
            }
            return (entry.binding, layoutEntry.visibility, resolved)
        }
    }
}

/// `GPURenderPipeline`.
public final class WGPURenderPipelineObject {
    let state: MTLRenderPipelineState
    let depthStencilState: MTLDepthStencilState?
    let layout: WGPUPipelineLayoutObject
    let primitiveType: MTLPrimitiveType
    let cullMode: MTLCullMode
    let winding: MTLWinding
    let stripIndexFormat: WGPUIndexFormat?
    let depthBias: Float
    let depthBiasSlopeScale: Float
    let depthBiasClamp: Float
    /// 셰이더가 `arrayLength()`를 쓰는가 — 쓰면 버퍼 크기 표를 바인딩해야 한다.
    let needsBufferSizes: Bool

    init(
        device: MTLDevice,
        descriptor: WGPURenderPipelineDescriptor,
        layout: WGPUPipelineLayoutObject,
        vertexModule: WGPUShaderModuleObject,
        fragmentModule: WGPUShaderModuleObject?
    ) throws {
        self.layout = layout
        self.primitiveType = WGPUMetalMapping.primitiveType(descriptor.primitive.topology)
        self.cullMode = WGPUMetalMapping.cullMode(descriptor.primitive.cullMode)
        self.winding = WGPUMetalMapping.winding(descriptor.primitive.frontFace)
        self.stripIndexFormat = descriptor.primitive.stripIndexFormat
        self.depthBias = Float(descriptor.depthStencil?.depthBias ?? 0)
        self.depthBiasSlopeScale = Float(descriptor.depthStencil?.depthBiasSlopeScale ?? 0)
        self.depthBiasClamp = Float(descriptor.depthStencil?.depthBiasClamp ?? 0)
        var wantsBufferSizes = vertexModule.wgsl?.usesArrayLength(
            entryPoints: [descriptor.vertex.entryPoint]
        ) ?? false
        if let fragment = descriptor.fragment, let fragmentModule {
            wantsBufferSizes = wantsBufferSizes || (fragmentModule.wgsl?.usesArrayLength(
                entryPoints: [fragment.entryPoint]
            ) ?? false)
        }
        self.needsBufferSizes = wantsBufferSizes

        let metalDescriptor = MTLRenderPipelineDescriptor()
        // Metal 검증 레이어는 label에 nil을 넣으면 단언으로 죽는다 — 있을 때만 설정한다.
        if let label = descriptor.label { metalDescriptor.label = label }

        // 정점/프래그먼트가 같은 모듈이면 MSL 하나에 두 진입점을 담아 한 번만 컴파일한다.
        let sharesModule = fragmentModule === vertexModule
        let vertexEntryPoints = sharesModule && descriptor.fragment != nil
            ? [descriptor.vertex.entryPoint, descriptor.fragment!.entryPoint]
            : [descriptor.vertex.entryPoint]

        // 정점/프래그먼트가 같은 모듈이면 상수도 합쳐서 한 번에 방출한다.
        let sharedConstants = descriptor.vertex.constants.merging(
            descriptor.fragment?.constants ?? [:]
        ) { _, fragment in fragment }
        let vertexLibrary = try vertexModule.library(
            entryPoints: vertexEntryPoints,
            bindings: layout.assignment,
            constants: sharesModule ? sharedConstants : descriptor.vertex.constants,
            device: device
        )
        let vertexName = vertexModule.metalFunctionName(for: descriptor.vertex.entryPoint)
        guard let vertexFunction = vertexLibrary.makeFunction(name: vertexName) else {
            throw WGPUError.validation("정점 셰이더 진입점 '\(descriptor.vertex.entryPoint)'을(를) 찾을 수 없다")
        }
        metalDescriptor.vertexFunction = vertexFunction

        if let fragment = descriptor.fragment, let fragmentModule {
            let fragmentLibrary = sharesModule
                ? vertexLibrary
                : try fragmentModule.library(
                    entryPoints: [fragment.entryPoint],
                    bindings: layout.assignment,
                    constants: fragment.constants,
                    device: device
                )
            let fragmentName = fragmentModule.metalFunctionName(for: fragment.entryPoint)
            guard let fragmentFunction = fragmentLibrary.makeFunction(name: fragmentName) else {
                throw WGPUError.validation("프래그먼트 셰이더 진입점 '\(fragment.entryPoint)'을(를) 찾을 수 없다")
            }
            metalDescriptor.fragmentFunction = fragmentFunction

            for (index, target) in fragment.targets.enumerated() {
                let attachment = metalDescriptor.colorAttachments[index]!
                attachment.pixelFormat = try WGPUMetalMapping.pixelFormat(target.format)
                attachment.writeMask = WGPUMetalMapping.colorWriteMask(target.writeMask)
                if let blend = target.blend {
                    attachment.isBlendingEnabled = true
                    attachment.rgbBlendOperation = WGPUMetalMapping.blendOperation(blend.color.operation)
                    attachment.sourceRGBBlendFactor = WGPUMetalMapping.blendFactor(blend.color.srcFactor)
                    attachment.destinationRGBBlendFactor = WGPUMetalMapping.blendFactor(blend.color.dstFactor)
                    attachment.alphaBlendOperation = WGPUMetalMapping.blendOperation(blend.alpha.operation)
                    attachment.sourceAlphaBlendFactor = WGPUMetalMapping.blendFactor(blend.alpha.srcFactor)
                    attachment.destinationAlphaBlendFactor = WGPUMetalMapping.blendFactor(blend.alpha.dstFactor)
                }
            }
        }

        if let depthStencil = descriptor.depthStencil {
            let format = try WGPUMetalMapping.pixelFormat(depthStencil.format)
            // 깊이와 스텐실은 **각각** 확인한다. `stencil8` 단독 포맷에 깊이 어태치먼트 포맷을
            // 세팅하면 렌더 패스에 없는 깊이를 요구하게 되어 파이프라인 생성이 실패한다.
            if depthStencil.format.hasDepth { metalDescriptor.depthAttachmentPixelFormat = format }
            if depthStencil.format.hasStencil { metalDescriptor.stencilAttachmentPixelFormat = format }

            let depthDescriptor = MTLDepthStencilDescriptor()
            depthDescriptor.depthCompareFunction = WGPUMetalMapping.compareFunction(depthStencil.depthCompare)
            depthDescriptor.isDepthWriteEnabled = depthStencil.depthWriteEnabled
            if depthStencil.usesStencil {
                depthDescriptor.frontFaceStencil = WGPUMetalMapping.stencilDescriptor(
                    depthStencil.stencilFront,
                    readMask: depthStencil.stencilReadMask,
                    writeMask: depthStencil.stencilWriteMask
                )
                depthDescriptor.backFaceStencil = WGPUMetalMapping.stencilDescriptor(
                    depthStencil.stencilBack,
                    readMask: depthStencil.stencilReadMask,
                    writeMask: depthStencil.stencilWriteMask
                )
            }
            depthStencilState = device.makeDepthStencilState(descriptor: depthDescriptor)
        } else {
            depthStencilState = nil
        }

        metalDescriptor.rasterSampleCount = descriptor.multisample.count
        metalDescriptor.isAlphaToCoverageEnabled = descriptor.multisample.alphaToCoverageEnabled

        if !descriptor.vertex.buffers.isEmpty {
            metalDescriptor.vertexDescriptor = try Self.vertexDescriptor(for: descriptor.vertex.buffers)
        }

        do {
            state = try device.makeRenderPipelineState(descriptor: metalDescriptor)
        } catch {
            throw WGPUError.backend("렌더 파이프라인 생성 실패: \(error.localizedDescription)")
        }
    }

    private static func vertexDescriptor(for buffers: [WGPUVertexBufferLayout]) throws -> MTLVertexDescriptor {
        guard buffers.count <= WGSLMetalLimits.maxVertexBufferSlots else {
            throw WGPUError.validation(
                "정점 버퍼 슬롯은 최대 \(WGSLMetalLimits.maxVertexBufferSlots)개다 (요청 \(buffers.count)개)"
            )
        }
        let descriptor = MTLVertexDescriptor()
        for (slot, layout) in buffers.enumerated() {
            guard layout.arrayStride > 0 else {
                throw WGPUError.unsupported("arrayStride 0(모든 정점이 같은 값)은 지원하지 않는다 — 슬롯 \(slot)")
            }
            let bufferIndex = WGSLMetalLimits.vertexBufferIndex(slot: slot)
            descriptor.layouts[bufferIndex].stride = layout.arrayStride
            descriptor.layouts[bufferIndex].stepFunction = WGPUMetalMapping.stepFunction(layout.stepMode)
            descriptor.layouts[bufferIndex].stepRate = 1
            for attribute in layout.attributes {
                let target = descriptor.attributes[attribute.shaderLocation]!
                target.format = WGPUMetalMapping.vertexFormat(attribute.format)
                target.offset = attribute.offset
                target.bufferIndex = bufferIndex
            }
        }
        return descriptor
    }
}

/// `GPUComputePipeline`.
public final class WGPUComputePipelineObject {
    let state: MTLComputePipelineState
    let layout: WGPUPipelineLayoutObject
    let threadsPerThreadgroup: MTLSize
    let needsBufferSizes: Bool

    init(
        device: MTLDevice,
        descriptor: WGPUComputePipelineDescriptor,
        layout: WGPUPipelineLayoutObject,
        module: WGPUShaderModuleObject
    ) throws {
        self.layout = layout
        self.needsBufferSizes = module.wgsl?.usesArrayLength(entryPoints: [descriptor.entryPoint]) ?? false

        let library = try module.library(
            entryPoints: [descriptor.entryPoint],
            bindings: layout.assignment,
            constants: descriptor.constants,
            device: device
        )
        let name = module.metalFunctionName(for: descriptor.entryPoint)
        guard let function = library.makeFunction(name: name) else {
            throw WGPUError.validation("컴퓨트 진입점 '\(descriptor.entryPoint)'을(를) 찾을 수 없다")
        }
        do {
            state = try device.makeComputePipelineState(function: function)
        } catch {
            throw WGPUError.backend("컴퓨트 파이프라인 생성 실패: \(error.localizedDescription)")
        }

        // MSL에는 workgroup 크기 선언이 없다. WGSL의 `@workgroup_size`를 리플렉션에서 가져와
        // dispatch 시 threadsPerThreadgroup으로 쓴다.
        let size = module.wgsl?.workgroupSize(of: descriptor.entryPoint) ?? (x: 1, y: 1, z: 1)
        threadsPerThreadgroup = MTLSize(width: size.x, height: size.y, depth: size.z)
    }
}

/// `GPURenderBundle` — 명령 목록을 그대로 들고 있다가 렌더 패스에 되풀이한다.
///
/// Metal에는 대응하는 객체가 없다 (`MTLIndirectCommandBuffer`는 제약이 훨씬 크고 용도가 다르다).
/// 하지만 번들의 계약이 애초에 **"직접 인코딩과 같은 결과"**이므로, 명령을 저장했다가 현재
/// 인코더에 다시 흘리는 것으로 계약을 그대로 만족시킨다. 재사용해도 안전한 이유도 같다 —
/// 저장된 것은 값 타입인 리더뿐이라 실행이 원본을 바꾸지 않는다.
public final class WGPURenderBundleObject {
    /// 번들 안에서 쓸 수 있는 명령. 명세가 정한 목록 그대로다 — 뷰포트·시저·블렌드 상수·
    /// 스텐실 참조·복사·중첩 번들은 번들에 담을 수 없다.
    static let allowedOps: Set<String> = [
        "setPipeline", "setBindGroup", "setVertexBuffer", "setIndexBuffer",
        "draw", "drawIndexed", "drawIndirect", "drawIndexedIndirect",
    ]

    let commands: [WGPUValueReader]
    let descriptor: WGPURenderBundleDescriptor

    init(commands: [WGPUValueReader], descriptor: WGPURenderBundleDescriptor) throws {
        for command in commands {
            let op = try command.requiredString("op")
            guard Self.allowedOps.contains(op) else {
                throw WGPUError.validation(
                    "렌더 번들에는 '\(op)'을(를) 담을 수 없다 "
                        + "(가능: \(Self.allowedOps.sorted().joined(separator: ", ")))",
                    path: command.fieldPath("op")
                )
            }
        }
        self.commands = commands
        self.descriptor = descriptor
    }

    /// 이 번들이 지금 패스에서 실행될 수 있는가.
    ///
    /// 번들은 "어떤 모양의 패스에서 쓸 것"이라고 선언하고 만들어진다. 그 선언과 실제 패스가
    /// 어긋나면 브라우저는 오류를 내지만, 이 구현은 명령을 되풀이할 뿐이라 Metal이 못 잡는다
    /// (파이프라인이 패스와 맞기만 하면 그냥 그려진다). 여기서 막지 않으면 브라우저에서만
    /// 깨지는 코드가 나간다.
    func checkCompatibility(color: [WGPUTextureFormat], depthStencil: WGPUTextureFormat?, sampleCount: Int) throws {
        // 명세의 "render pass layout equals"는 **후행 null을 무시하고** colorFormats를 비교한다.
        // 자르지 않으면 `['bgra8unorm', null]` 번들이 컬러 1개짜리 패스에서 오탐으로 거부된다.
        let bundleFormats = Self.trimmingTrailingNulls(descriptor.colorFormats)
        guard bundleFormats.count == color.count else {
            throw WGPUError.validation(
                "번들의 컬러 어태치먼트 수(\(bundleFormats.count))가 "
                    + "패스(\(color.count))와 다르다"
            )
        }
        for (index, expected) in bundleFormats.enumerated() where expected != color[index] {
            throw WGPUError.validation(
                "번들의 colorFormats[\(index)]가 패스와 다르다 — "
                    + "번들 \(expected?.rawValue ?? "null"), 패스 \(color[index].rawValue)"
            )
        }
        guard descriptor.depthStencilFormat == depthStencil else {
            throw WGPUError.validation(
                "번들의 depthStencilFormat이 패스와 다르다 — "
                    + "번들 \(descriptor.depthStencilFormat?.rawValue ?? "없음"), "
                    + "패스 \(depthStencil?.rawValue ?? "없음")"
            )
        }
        guard descriptor.sampleCount == sampleCount else {
            throw WGPUError.validation(
                "번들의 sampleCount(\(descriptor.sampleCount))가 패스(\(sampleCount))와 다르다"
            )
        }
    }

    /// 후행 `null` 슬롯을 잘라낸다 — 명세의 레이아웃 동치 비교가 이것들을 무시한다.
    private static func trimmingTrailingNulls(_ formats: [WGPUTextureFormat?]) -> [WGPUTextureFormat?] {
        var trimmed = formats
        while let last = trimmed.last, last == nil { trimmed.removeLast() }
        return trimmed
    }
}

// MARK: - 레이아웃 유도

enum WGPUPipelineLayoutResolver {
    /// 명시적 레이아웃이면 핸들을 찾고, `"auto"`면 셰이더 선언에서 유도한다.
    static func resolve(
        _ reference: WGPUPipelineLayoutRef,
        stages: [(module: WGPUShaderModuleObject, entryPoints: [String])],
        registry: WGPUObjectRegistry
    ) throws -> WGPUPipelineLayoutObject {
        switch reference {
        case .explicit(let handle):
            return try registry.lookup(handle, as: WGPUPipelineLayoutObject.self, kind: "GPUPipelineLayout")
        case .auto:
            return try WGPUPipelineLayoutObject(groups: try derivedGroups(stages: stages))
        }
    }

    /// 여러 셰이더 모듈의 선언을 (그룹, 바인딩)으로 합친다. visibility는 합집합이다.
    private static func derivedGroups(
        stages: [(module: WGPUShaderModuleObject, entryPoints: [String])]
    ) throws -> [WGPUBindGroupLayoutObject] {
        var merged: [Int: [Int: WGPUBindGroupLayoutEntry]] = [:]

        for stage in stages {
            guard let wgsl = stage.module.wgsl else {
                throw WGPUError.validation(
                    "layout: \"auto\"는 WGSL 셰이더에만 쓸 수 있다 — MSL 셰이더는 GPUPipelineLayout을 명시할 것"
                )
            }
            for (groupIndex, entries) in wgsl.autoBindGroupLayouts(entryPoints: stage.entryPoints).enumerated() {
                for entry in entries {
                    if let existing = merged[groupIndex]?[entry.binding] {
                        guard existing.layout == entry.layout else {
                            throw WGPUError.validation(
                                "@group(\(groupIndex)) @binding(\(entry.binding))의 리소스 종류가 스테이지마다 다르다"
                            )
                        }
                        merged[groupIndex, default: [:]][entry.binding] = WGPUBindGroupLayoutEntry(
                            binding: entry.binding,
                            visibility: existing.visibility.union(entry.visibility),
                            layout: entry.layout
                        )
                    } else {
                        merged[groupIndex, default: [:]][entry.binding] = entry
                    }
                }
            }
        }

        guard let maximumGroup = merged.keys.max() else { return [] }
        return (0...maximumGroup).map { index in
            WGPUBindGroupLayoutObject(entries: Array(merged[index]?.values ?? [:].values))
        }
    }
}
