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
    /// 드로우/디스패치 전에 반드시 바인드되어 있어야 하는 그룹 인덱스.
    /// 빈 그룹(선언에 구멍이 있어 생긴 자리)은 요구하지 않는다.
    let requiredGroups: Set<Int>

    init(groups: [WGPUBindGroupLayoutObject]) throws {
        self.groups = groups
        self.assignment = try WGSLBindingAssigner.assign(groups: groups.map(\.entries))
        self.requiredGroups = Set(groups.indices.filter { !groups[$0].entries.isEmpty })
    }

    func group(at index: Int) -> WGPUBindGroupLayoutObject? {
        index >= 0 && index < groups.count ? groups[index] : nil
    }
}

/// 바인드 그룹이 실제로 가리키는 Metal 객체 (`WGPUMetalBindGroup.Binding`이 담는다).
enum WGPUResolvedBinding {
    /// `boundSize`는 이 바인딩이 보는 바이트 수 — `arrayLength()`가 이 값을 쓴다.
    case buffer(MTLBuffer, offset: Int, boundSize: Int)
    case sampler(MTLSamplerState)
    case texture(MTLTexture)
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
    /// 드로우 전에 반드시 바인드되어 있어야 하는 정점 버퍼 슬롯 (`vertex.buffers`에 선언된 것).
    let requiredVertexSlots: Set<Int>
    /// 깊이/스텐실 값을 쓰는가 — `depthReadOnly`/`stencilReadOnly` 패스에서 거부할 때 쓴다.
    let writesDepth: Bool
    let writesStencil: Bool

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
        // 진입점 이름은 여기서 확정한다 — 명세상 생략할 수 있고, 그때는 그 스테이지의
        // 유일한 진입점을 쓴다 (`resolveEntryPoint`). 해석기가 이미 확정해 넘기지만
        // 같은 결과라 멱등이고, 이 타입만 봐도 계약이 닫힌다.
        let vertexEntry = try vertexModule.resolveEntryPoint(descriptor.vertex.entryPoint, stage: .vertex)
        let fragmentEntry = try descriptor.fragment.flatMap { fragment -> String? in
            try fragmentModule?.resolveEntryPoint(fragment.entryPoint, stage: .fragment)
        }

        var wantsBufferSizes = vertexModule.wgsl?.usesArrayLength(
            entryPoints: [vertexEntry]
        ) ?? false
        if let fragmentEntry, let fragmentModule {
            wantsBufferSizes = wantsBufferSizes || (fragmentModule.wgsl?.usesArrayLength(
                entryPoints: [fragmentEntry]
            ) ?? false)
        }
        self.needsBufferSizes = wantsBufferSizes
        self.requiredVertexSlots = Set(descriptor.vertex.buffers.indices)
        self.writesDepth = descriptor.depthStencil?.depthWriteEnabled ?? false
        // 명세의 기준은 **op**다 — 세 op이 전부 `keep`이면 쓰지 않는 것으로 본다
        // (writeMask 0으로 막아 둔 경우까지 허용하면 브라우저보다 느슨해진다).
        self.writesStencil = descriptor.depthStencil.map {
            !$0.stencilFront.isWriteFree || !$0.stencilBack.isWriteFree
        } ?? false

        let metalDescriptor = MTLRenderPipelineDescriptor()
        // Metal 검증 레이어는 label에 nil을 넣으면 단언으로 죽는다 — 있을 때만 설정한다.
        if let label = descriptor.label { metalDescriptor.label = label }

        // 정점/프래그먼트가 같은 모듈이면 MSL 하나에 두 진입점을 담아 한 번만 컴파일한다.
        let sharesModule = fragmentModule === vertexModule
        let vertexEntryPoints = sharesModule && fragmentEntry != nil
            ? [vertexEntry, fragmentEntry!]
            : [vertexEntry]

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
        let vertexName = vertexModule.metalFunctionName(for: vertexEntry)
        guard let vertexFunction = vertexLibrary.makeFunction(name: vertexName) else {
            throw WGPUError.validation("정점 셰이더 진입점 '\(vertexEntry)'을(를) 찾을 수 없다")
        }
        metalDescriptor.vertexFunction = vertexFunction

        if let fragment = descriptor.fragment, let fragmentModule, let fragmentEntry {
            let fragmentLibrary = sharesModule
                ? vertexLibrary
                : try fragmentModule.library(
                    entryPoints: [fragmentEntry],
                    bindings: layout.assignment,
                    constants: fragment.constants,
                    device: device
                )
            let fragmentName = fragmentModule.metalFunctionName(for: fragmentEntry)
            guard let fragmentFunction = fragmentLibrary.makeFunction(name: fragmentName) else {
                throw WGPUError.validation("프래그먼트 셰이더 진입점 '\(fragmentEntry)'을(를) 찾을 수 없다")
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
        // 렌더 쪽과 같다 — 이름을 생략하면 유일한 compute 진입점을 쓴다 (멱등).
        let entry = try module.resolveEntryPoint(descriptor.entryPoint, stage: .compute)
        self.needsBufferSizes = module.wgsl?.usesArrayLength(entryPoints: [entry]) ?? false

        let library = try module.library(
            entryPoints: [entry],
            bindings: layout.assignment,
            constants: descriptor.constants,
            device: device
        )
        let name = module.metalFunctionName(for: entry)
        guard let function = library.makeFunction(name: name) else {
            throw WGPUError.validation("컴퓨트 진입점 '\(entry)'을(를) 찾을 수 없다")
        }
        do {
            state = try device.makeComputePipelineState(function: function)
        } catch {
            throw WGPUError.backend("컴퓨트 파이프라인 생성 실패: \(error.localizedDescription)")
        }

        // MSL에는 workgroup 크기 선언이 없다. WGSL의 `@workgroup_size`를 리플렉션에서 가져와
        // dispatch 시 threadsPerThreadgroup으로 쓴다.
        let size = module.wgsl?.workgroupSize(of: entry) ?? (x: 1, y: 1, z: 1)
        threadsPerThreadgroup = MTLSize(width: size.x, height: size.y, depth: size.z)
    }
}

// MARK: - 레이아웃 유도

enum WGPUPipelineLayoutResolver {
    /// `layout: "auto"` — 여러 셰이더 모듈의 선언을 (그룹, 바인딩)으로 합친다.
    /// visibility는 합집합이다. (명시적 레이아웃의 핸들 해석은 엔진이 끝내고 온다.)
    static func derivedGroups(
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
