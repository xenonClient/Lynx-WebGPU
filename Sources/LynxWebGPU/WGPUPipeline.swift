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

/// `GPUPipelineLayout` — holds the group list together with the Metal index assignment derived from it.
public final class WGPUPipelineLayoutObject {
    public let groups: [WGPUBindGroupLayoutObject]
    let assignment: WGSLBindingAssignment
    /// Group indices that must be bound before a draw or dispatch.
    /// Empty groups (slots created by a hole in the declarations) are not required.
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

/// The Metal objects a bind group actually points at (carried by `WGPUMetalBindGroup.Binding`).
enum WGPUResolvedBinding {
    /// `boundSize` is how many bytes this binding sees — `arrayLength()` uses this value.
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
    /// Whether the shader uses `arrayLength()` — if so, the buffer size table must be bound.
    let needsBufferSizes: Bool
    /// Vertex buffer slots that must be bound before a draw (those declared in `vertex.buffers`).
    let requiredVertexSlots: Set<Int>
    /// Whether it writes depth/stencil — used when rejecting in a `depthReadOnly`/`stencilReadOnly` pass.
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
        // The entry point name is settled here — the spec allows omitting it, in which case the
        // stage's only entry point is used (`resolveEntryPoint`). The interpreter already settled and
        // passed it, but the result is identical so this is idempotent, and the contract closes within this type alone.
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
        // The spec's criterion is the **op** — all three ops being `keep` counts as not writing
        // (allowing a writeMask of 0 to count too would be looser than a browser).
        self.writesStencil = descriptor.depthStencil.map {
            !$0.stencilFront.isWriteFree || !$0.stencilBack.isWriteFree
        } ?? false

        let metalDescriptor = MTLRenderPipelineDescriptor()
        // The Metal validation layer dies on an assertion if label is nil — set it only when present.
        if let label = descriptor.label { metalDescriptor.label = label }

        // When vertex and fragment share a module, both entry points go into one MSL and compile once.
        let sharesModule = fragmentModule === vertexModule
        let vertexEntryPoints = sharesModule && fragmentEntry != nil
            ? [vertexEntry, fragmentEntry!]
            : [vertexEntry]

        // When vertex and fragment share a module, the constants are merged and emitted together too.
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
            throw WGPUError.validation("could not find vertex shader entry point '\(vertexEntry)'")
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
                throw WGPUError.validation("could not find fragment shader entry point '\(fragmentEntry)'")
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
            // Depth and stencil are checked **separately**. Setting a depth attachment format on a
            // stencil8-only format would demand a depth the render pass lacks, failing pipeline creation.
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
            throw WGPUError.backend("render pipeline creation failed: \(error.localizedDescription)")
        }
    }

    private static func vertexDescriptor(for buffers: [WGPUVertexBufferLayout]) throws -> MTLVertexDescriptor {
        guard buffers.count <= WGSLMetalLimits.maxVertexBufferSlots else {
            throw WGPUError.validation(
                "at most \(WGSLMetalLimits.maxVertexBufferSlots) vertex buffer slots are available (\(buffers.count) requested)"
            )
        }
        let descriptor = MTLVertexDescriptor()
        for (slot, layout) in buffers.enumerated() {
            guard layout.arrayStride > 0 else {
                throw WGPUError.unsupported("arrayStride 0 (every vertex the same value) is not supported — slot \(slot)")
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
        // Same as the render side — omitting the name uses the only compute entry point (idempotent).
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
            throw WGPUError.validation("could not find compute entry point '\(entry)'")
        }
        do {
            state = try device.makeComputePipelineState(function: function)
        } catch {
            throw WGPUError.backend("compute pipeline creation failed: \(error.localizedDescription)")
        }

        // MSL has no workgroup size declaration. WGSL's `@workgroup_size` is taken from reflection and
        // used as threadsPerThreadgroup at dispatch.
        let size = module.wgsl?.workgroupSize(of: entry) ?? (x: 1, y: 1, z: 1)
        threadsPerThreadgroup = MTLSize(width: size.x, height: size.y, depth: size.z)
    }
}

// MARK: - Deriving layouts

enum WGPUPipelineLayoutResolver {
    /// `layout: "auto"` — merges the declarations of several shader modules by (group, binding).
    /// Visibility is the union. (Resolving an explicit layout's handles is finished by the engine.)
    static func derivedGroups(
        stages: [(module: WGPUShaderModuleObject, entryPoints: [String])]
    ) throws -> [WGPUBindGroupLayoutObject] {
        var merged: [Int: [Int: WGPUBindGroupLayoutEntry]] = [:]

        for stage in stages {
            guard let wgsl = stage.module.wgsl else {
                throw WGPUError.validation(
                    "layout: \"auto\" can only be used with WGSL shaders — specify a GPUPipelineLayout for MSL shaders"
                )
            }
            for (groupIndex, entries) in wgsl.autoBindGroupLayouts(entryPoints: stage.entryPoints).enumerated() {
                for entry in entries {
                    if let existing = merged[groupIndex]?[entry.binding] {
                        guard existing.layout == entry.layout else {
                            throw WGPUError.validation(
                                "the resource kind of @group(\(groupIndex)) @binding(\(entry.binding)) differs between stages"
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
