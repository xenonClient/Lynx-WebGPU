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
    case buffer(MTLBuffer, offset: Int)
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
            case .buffer(let handle, let offset, _):
                let object = try registry.lookup(handle, as: WGPUBufferObject.self, kind: "GPUBuffer")
                resolved = .buffer(object.buffer, offset: offset)
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
            metalDescriptor.depthAttachmentPixelFormat = format
            if depthStencil.format.hasStencil {
                metalDescriptor.stencilAttachmentPixelFormat = format
            }
            let depthDescriptor = MTLDepthStencilDescriptor()
            depthDescriptor.depthCompareFunction = WGPUMetalMapping.compareFunction(depthStencil.depthCompare)
            depthDescriptor.isDepthWriteEnabled = depthStencil.depthWriteEnabled
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

    init(
        device: MTLDevice,
        descriptor: WGPUComputePipelineDescriptor,
        layout: WGPUPipelineLayoutObject,
        module: WGPUShaderModuleObject
    ) throws {
        self.layout = layout

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
