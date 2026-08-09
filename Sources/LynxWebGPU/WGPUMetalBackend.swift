import Foundation
import CoreGraphics
import Metal
import QuartzCore
import LynxWebGPUCore
import LynxWebGPUShader

/// The **direct Metal implementation** of `WGPUBackend` — the backend this package ships by default.
///
/// Orchestration (decoding, validation, error scopes, frame lifetime, serialization) all lives in
/// `WGPUBackendEngine` (Core). What remains here is turning validated values into Metal encoding:
/// command buffer and encoder lifetime, staging pool uploads, MSL compilation (at pipeline creation —
/// `docs/ARCHITECTURE.md`), argument table assignment (`WGSLMetalLimits`), and the buffer size table
/// for `arrayLength()`.
///
/// Every verb is called under the engine's execution lock (see `WGPUBackend`) — this type takes no lock of its own.
public final class WGPUMetalBackend: WGPUBackend {
    public typealias Buffer = WGPUBufferObject
    public typealias Texture = WGPUTextureObject
    public typealias TextureView = WGPUTextureViewObject
    public typealias Sampler = WGPUSamplerObject
    public typealias ShaderModule = WGPUShaderModuleObject
    public typealias BindGroupLayout = WGPUBindGroupLayoutObject
    public typealias PipelineLayout = WGPUPipelineLayoutObject
    public typealias BindGroup = WGPUMetalBindGroup
    public typealias RenderPipeline = WGPURenderPipelineObject
    public typealias ComputePipeline = WGPUComputePipelineObject
    public typealias QuerySet = WGPUQuerySetObject
    /// Metal has no object corresponding to a render bundle (`MTLIndirectCommandBuffer` is far more
    /// constrained and serves a different purpose) — the engine does record/replay instead.
    public typealias RenderBundle = Never
    public typealias Surface = WGPUSurface

    let device: MTLDevice
    let queue: MTLCommandQueue
    /// Reuse pool for upload staging buffers (shared by writeBuffer and writeTexture).
    let stagingPool: WGPUStagingPool

    // Batch lifetime state — valid only between beginBatch and submit.
    private var commandBuffer: MTLCommandBuffer?
    private var renderEncoder: MTLRenderCommandEncoder?
    private var computeEncoder: MTLComputeCommandEncoder?
    private var blitEncoder: MTLBlitCommandEncoder?
    private var currentRenderPipeline: WGPURenderPipelineObject?
    private var currentComputePipeline: WGPUComputePipelineObject?
    /// Binding size per Metal buffer index — `arrayLength()` looks it up in this table.
    /// Refreshed whenever a bind group is applied, and uploaded to the encoder only when the shader uses it.
    private var bufferSizes = [UInt32](repeating: 0, count: WGSLMetalLimits.maxBindGroupBuffers)
    /// Staging buffers used for this frame's uploads — returned to the pool when the command buffer completes.
    private var frameStagingBuffers: [MTLBuffer] = []
    /// Drawables acquired this frame — `submit(present: true)` sends them to screen.
    private var acquiredDrawables: [WGPUDrawable] = []
    /// The last committed command buffer — used when `readBuffer` waits on GPU completion.
    private(set) var lastCommittedBuffer: MTLCommandBuffer?

    public init(device: MTLDevice, queue: MTLCommandQueue) {
        self.device = device
        self.queue = queue
        self.stagingPool = WGPUStagingPool(device: device)
    }

    // MARK: - Capabilities

    public var capabilities: WGPUBackendCapabilities {
        WGPUBackendCapabilities(
            supportsNativeRenderBundles: false,
            maxVertexBufferSlots: WGSLMetalLimits.maxVertexBufferSlots
        )
    }

    public func supportsTextureCompression(_ format: WGPUTextureFormat) -> Bool {
        WGPUDeviceCapability.supportsCompression(format, on: device)
    }

    /// **Stops here** when the device does not support indirect arguments. Passing it to Metal anyway
    /// kills the process with `MTLValidateFeatureSupport ... failed assertion`, leaving the app unable
    /// even to record why.
    public func ensureIndirectSupported() throws {
        guard WGPUDeviceCapability.supportsIndirectArguments(device) else {
            throw WGPUError.unsupported(
                "this device does not support indirect draw/dispatch arguments (Metal requires Apple GPU "
                    + "family 3 or above). **The iOS simulator falls here** — it works on a real device "
                    + "(A12 or newer), so substitute a direct draw or verify on hardware"
            )
        }
    }

    public func pumpEvents() {
        // Metal completions arrive on their own (completion handlers) — there is nothing to pump.
    }

    public func reset() {
        acquiredDrawables.removeAll()
        lastCommittedBuffer = nil
    }

    // MARK: - Batch lifetime

    public func beginBatch() {
        // If the previous batch ended without a submit (an errors-only batch, say), clear the leftovers.
        if commandBuffer == nil, !frameStagingBuffers.isEmpty {
            stagingPool.recycle(frameStagingBuffers)
            frameStagingBuffers.removeAll()
        }
        commandBuffer = nil
        renderEncoder = nil
        computeEncoder = nil
        blitEncoder = nil
        currentRenderPipeline = nil
        currentComputePipeline = nil
    }

    public func collectBatchDiagnostics() -> [WGPUError] { [] }

    public var hasPendingWork: Bool { commandBuffer != nil }

    public func ensureSubmittableWork() {
        _ = try? activeCommandBuffer()
    }

    public func submit(present: Bool, onCompleted: @escaping (WGPUError?) -> Void) {
        guard let commandBuffer else { return }
        if present {
            for drawable in acquiredDrawables { drawable.present(with: commandBuffer) }
        }
        // A completion handler can only be attached before commit (a Metal assertion).
        if !frameStagingBuffers.isEmpty {
            let buffers = frameStagingBuffers
            let pool = stagingPool
            commandBuffer.addCompletedHandler { _ in pool.recycle(buffers) }
        }
        // Collect GPU-side failures (.outOfMemory / .timeout / .deviceRemoved, …).
        commandBuffer.addCompletedHandler { buffer in
            onCompleted(buffer.status == .error ? Self.commandBufferError(buffer) : nil)
        }
        commandBuffer.commit()
        lastCommittedBuffer = commandBuffer
        if present { acquiredDrawables.removeAll() }
        frameStagingBuffers.removeAll()
        self.commandBuffer = nil
    }

    /// Releases a drawable we could not draw — called by the engine when the frame ended without present.
    /// A `CAMetalDrawable` returns to the pool once its last reference is gone. Held on to, the pool
    /// dries up in three frames and `nextDrawable()` stalls the JS thread.
    public func discardAcquiredFrames() {
        acquiredDrawables.removeAll()
    }

    /// Turns a failed command buffer into the best error we can name.
    static func commandBufferError(_ buffer: MTLCommandBuffer) -> WGPUError {
        .backend("GPU work failed: \(buffer.error?.localizedDescription ?? "cause unknown")")
    }

    // MARK: - Encoder lifetime

    private func activeCommandBuffer() throws -> MTLCommandBuffer {
        if let commandBuffer { return commandBuffer }
        guard let created = queue.makeCommandBuffer() else {
            throw WGPUError.backend("MTLCommandBuffer creation failed")
        }
        created.label = "webgpu.frame"
        commandBuffer = created
        return created
    }

    private func activeBlitEncoder() throws -> MTLBlitCommandEncoder {
        if let blitEncoder { return blitEncoder }
        guard renderEncoder == nil, computeEncoder == nil else {
            throw WGPUError.validation("copy and upload commands cannot be used inside a render/compute pass")
        }
        guard let encoder = try activeCommandBuffer().makeBlitCommandEncoder() else {
            throw WGPUError.backend("MTLBlitCommandEncoder creation failed")
        }
        blitEncoder = encoder
        return encoder
    }

    public func endPass() {
        renderEncoder?.endEncoding()
        renderEncoder = nil
        computeEncoder?.endEncoding()
        computeEncoder = nil
        blitEncoder?.endEncoding()
        blitEncoder = nil
    }

    /// The pass-scope target for debug groups and markers — blit is excluded as an internal encoder
    /// (`WGPUDebugScope` — deciding the scope itself is the engine's job).
    private var passEncoder: MTLCommandEncoder? {
        renderEncoder ?? computeEncoder
    }

    // MARK: - Resources

    public func makeBuffer(_ descriptor: WGPUBufferDescriptor) throws -> WGPUBufferObject {
        try WGPUBufferObject(device: device, descriptor: descriptor)
    }

    public func writeBuffer(_ buffer: WGPUBufferObject, offset: Int, data: Data) throws {
        let staging = try makeStagingBuffer(data)
        // A direct memcpy would race the previous frame's GPU work. A blit puts it in queue order.
        let encoder = try activeBlitEncoder()
        encoder.copy(
            from: staging, sourceOffset: 0,
            to: buffer.buffer, destinationOffset: offset, size: data.count
        )
    }

    public func readBuffer(
        _ buffer: WGPUBufferObject, offset: Int, length: Int,
        deliver: @escaping (Result<Data, WGPUError>) -> Void
    ) {
        let finish = { (failed: MTLCommandBuffer?) in
            if let failed {
                // `.error` means **failure, not completion.** Sending it down the success path would
                // read the contents of a buffer whose GPU work failed and return them as a success.
                deliver(.failure(Self.commandBufferError(failed)))
                return
            }
            deliver(.success(Data(
                bytes: buffer.buffer.contents().advanced(by: offset), count: length
            )))
        }

        // If submitted work is still running, read after it completes.
        // `addCompletedHandler` cannot be attached after commit (a Metal assertion), so we wait on a dedicated queue.
        let pending = lastCommittedBuffer
        if let pending, pending.status == .error {
            finish(pending)
            return
        }
        guard let pending, pending.status != .completed else {
            finish(nil)
            return
        }
        Self.readbackQueue.async {
            pending.waitUntilCompleted()
            finish(pending.status == .error ? pending : nil)
        }
    }

    /// The dedicated queue waiting on GPU completion — it never blocks the JS thread.
    private static let readbackQueue = DispatchQueue(label: "org.lynxwebgpu.readback")

    public func makeTexture(_ descriptor: WGPUTextureDescriptor) throws -> WGPUTextureObject {
        try WGPUTextureObject(device: device, descriptor: descriptor)
    }

    public func writeTexture(
        _ texture: WGPUTextureObject, data: Data, origin: WGPUOrigin3D, size: WGPUExtent3D,
        mipLevel: Int, bytesPerRow: Int, rowsPerImage: Int
    ) throws {
        let blockRows = texture.format.blockRows(height: size.height)
        let bytesPerImage = bytesPerRow * max(rowsPerImage, blockRows)
        let layers = max(size.depthOrArrayLayers, 1)
        // Staging reserves the full image stride — the Metal validation layer computes the last image
        // by bytesPerImage too. The leftover tail is never copied into the texture.
        let staging = try makeStagingBuffer(data, minimumLength: bytesPerImage * layers)
        // A blit puts it in queue order for the same reason as writeBuffer — serialized with earlier renders and copies.
        texture.encodeWrite(
            from: staging,
            origin: origin,
            size: size,
            mipLevel: mipLevel,
            bytesPerRow: bytesPerRow,
            rowsPerImage: rowsPerImage,
            blit: try activeBlitEncoder()
        )
    }

    /// Takes a staging buffer from the pool, fills it with data, and lists it for recycling at frame completion.
    private func makeStagingBuffer(_ data: Data, minimumLength: Int = 0) throws -> MTLBuffer {
        let staging = try stagingPool.acquire(data, minimumLength: minimumLength)
        frameStagingBuffers.append(staging)
        return staging
    }

    public func makeTextureView(
        _ texture: WGPUTextureObject, descriptor: WGPUTextureViewDescriptor,
        format: WGPUTextureFormat
    ) throws -> WGPUTextureViewObject {
        try WGPUTextureViewObject(source: texture, descriptor: descriptor, drawable: nil)
    }

    public func makeSampler(_ descriptor: WGPUSamplerDescriptor) throws -> WGPUSamplerObject {
        try WGPUSamplerObject(device: device, descriptor: descriptor)
    }

    public func makeShaderModule(
        _ descriptor: WGPUShaderModuleDescriptor, fieldPath: (String) -> String?
    ) -> WGPUShaderModuleCreation<WGPUMetalBackend> {
        let object = WGPUShaderModuleObject(descriptor: descriptor)
        let failure = object.isValid ? nil : object.compilationMessages.first
        return WGPUShaderModuleCreation(module: object, failure: failure)
    }

    public func unmapBuffer(_ buffer: WGPUBufferObject) {
        // Mapping on the Metal path is wire state only (shared memory is read directly) — there is nothing to release.
    }

    public func compilationMessages(of module: WGPUShaderModuleObject) -> [WGPUCompilationMessage] {
        module.compilationMessages.map { error in
            WGPUCompilationMessage(
                message: error.message,
                // Every diagnostic in this implementation is an error — the Metal runtime API gives no separate warnings.
                type: "error",
                lineNum: error.line ?? 0
            )
        }
    }

    public func makeBindGroupLayout(_ entries: [WGPUBindGroupLayoutEntry]) throws -> WGPUBindGroupLayoutObject {
        WGPUBindGroupLayoutObject(entries: entries)
    }

    public func makePipelineLayout(_ groups: [WGPUBindGroupLayoutObject]) throws -> WGPUPipelineLayoutObject {
        try WGPUPipelineLayoutObject(groups: groups)
    }

    public func makeBindGroup(
        layout: WGPUBindGroupLayoutObject,
        entries: [WGPUResolvedBindGroupEntry<WGPUMetalBackend>]
    ) throws -> WGPUMetalBindGroup {
        let bindings = try entries.map { entry -> WGPUMetalBindGroup.Binding in
            // A Metal backend layout always knows its entries (there is no native derived layout).
            guard let layoutEntry = entry.layoutEntry else {
                throw WGPUError.backend("a Metal bind group needs layout entry information")
            }
            let resource: WGPUResolvedBinding
            switch entry.resource {
            case .buffer(let buffer, let offset, let boundSize):
                resource = .buffer(buffer.buffer, offset: offset, boundSize: boundSize)
            case .sampler(let sampler):
                resource = .sampler(sampler.sampler)
            case .textureView(let view):
                resource = .texture(view.texture)
            }
            return WGPUMetalBindGroup.Binding(
                binding: entry.binding,
                visibility: layoutEntry.visibility,
                hasDynamicOffset: Self.hasDynamicOffset(layoutEntry),
                resource: resource
            )
        }
        return WGPUMetalBindGroup(bindings: bindings.sorted { $0.binding < $1.binding })
    }

    private static func hasDynamicOffset(_ entry: WGPUBindGroupLayoutEntry) -> Bool {
        if case .buffer(let buffer) = entry.layout { return buffer.hasDynamicOffset }
        return false
    }

    public func makeQuerySet(_ descriptor: WGPUQuerySetDescriptor) throws -> WGPUQuerySetObject {
        try WGPUQuerySetObject(device: device, descriptor: descriptor)
    }

    public func makeRenderPipeline(
        _ descriptor: WGPURenderPipelineDescriptor,
        vertexModule: WGPUShaderModuleObject, fragmentModule: WGPUShaderModuleObject?,
        layout: WGPUResolvedPipelineLayout<WGPUMetalBackend>,
        fieldPath: (String) -> String?
    ) throws -> WGPURenderPipelineCreation<WGPUMetalBackend> {
        var descriptor = descriptor
        // The spec's "get the entry point" — omitting the name uses the stage's only entry point.
        // Settling it once here leaves every layer below dealing only with a decided name.
        descriptor.vertex.entryPoint = try vertexModule.resolveEntryPoint(
            descriptor.vertex.entryPoint, stage: .vertex, path: fieldPath("vertex.entryPoint")
        )
        if let fragmentModule, let fragment = descriptor.fragment {
            descriptor.fragment?.entryPoint = try fragmentModule.resolveEntryPoint(
                fragment.entryPoint, stage: .fragment, path: fieldPath("fragment.entryPoint")
            )
        }
        let vertexEntry = descriptor.vertex.entryPoint!

        var stages: [(module: WGPUShaderModuleObject, entryPoints: [String])] = []
        if let fragmentModule, fragmentModule === vertexModule, let fragment = descriptor.fragment {
            stages = [(vertexModule, [vertexEntry, fragment.entryPoint!])]
        } else {
            stages = [(vertexModule, [vertexEntry])]
            if let fragmentModule, let fragment = descriptor.fragment {
                stages.append((fragmentModule, [fragment.entryPoint!]))
            }
        }
        let resolved = try resolveLayout(layout, stages: stages)
        let pipeline = try WGPURenderPipelineObject(
            device: device, descriptor: descriptor, layout: resolved,
            vertexModule: vertexModule, fragmentModule: fragmentModule
        )
        return WGPURenderPipelineCreation(
            pipeline: pipeline,
            info: WGPURenderPipelineInfo(
                requiredGroups: resolved.requiredGroups,
                requiredVertexSlots: pipeline.requiredVertexSlots,
                writesDepth: pipeline.writesDepth,
                writesStencil: pipeline.writesStencil
            )
        )
    }

    public func makeComputePipeline(
        _ descriptor: WGPUComputePipelineDescriptor,
        module: WGPUShaderModuleObject,
        layout: WGPUResolvedPipelineLayout<WGPUMetalBackend>,
        fieldPath: (String) -> String?
    ) throws -> WGPUComputePipelineCreation<WGPUMetalBackend> {
        var descriptor = descriptor
        descriptor.entryPoint = try module.resolveEntryPoint(
            descriptor.entryPoint, stage: .compute, path: fieldPath("compute.entryPoint")
        )
        let resolved = try resolveLayout(layout, stages: [(module, [descriptor.entryPoint!])])
        let pipeline = try WGPUComputePipelineObject(
            device: device, descriptor: descriptor, layout: resolved, module: module
        )
        return WGPUComputePipelineCreation(
            pipeline: pipeline,
            info: WGPUComputePipelineInfo(requiredGroups: resolved.requiredGroups)
        )
    }

    /// An explicit layout passes through; `"auto"` is derived from the shader declarations.
    private func resolveLayout(
        _ layout: WGPUResolvedPipelineLayout<WGPUMetalBackend>,
        stages: [(module: WGPUShaderModuleObject, entryPoints: [String])]
    ) throws -> WGPUPipelineLayoutObject {
        switch layout {
        case .explicit(let object):
            return object
        case .auto:
            return try WGPUPipelineLayoutObject(groups: WGPUPipelineLayoutResolver.derivedGroups(stages: stages))
        }
    }

    public func bindGroupLayout(
        of pipeline: WGPUResolvedPipeline<WGPUMetalBackend>, index: Int
    ) throws -> WGPUBindGroupLayoutCreation<WGPUMetalBackend>? {
        let layout: WGPUPipelineLayoutObject
        switch pipeline {
        case .render(let render): layout = render.layout
        case .compute(let compute): layout = compute.layout
        }
        guard let group = layout.group(at: index) else { return nil }
        return WGPUBindGroupLayoutCreation(layout: group, entries: group.entries)
    }

    public func makeRenderBundle(
        _ descriptor: WGPURenderBundleDescriptor, commands: [WGPUCommand],
        resolver: WGPUBundleResolver<WGPUMetalBackend>
    ) throws -> Never {
        // capabilities declares record/replay, so the engine never calls this verb.
        throw WGPUError.backend("the Metal backend has no native render bundles (use the engine's record/replay path)")
    }

    // MARK: - Surfaces

    public func makeLayerSurface(identifier: String, layer: CAMetalLayer) -> WGPUSurfaceCreation<WGPUMetalBackend> {
        let surface = WGPUMetalLayerSurface(identifier: identifier, layer: layer)
        return WGPUSurfaceCreation(surface: surface, pacesFrames: surface.pacesFrames)
    }

    public func makeOffscreenSurface(identifier: String, size: CGSize) throws -> WGPUSurfaceCreation<WGPUMetalBackend> {
        let surface = WGPUOffscreenSurface(identifier: identifier, size: size, device: device)
        return WGPUSurfaceCreation(surface: surface, pacesFrames: surface.pacesFrames)
    }

    public func configureSurface(_ surface: WGPUSurface, configuration: WGPUCanvasConfiguration) throws {
        try surface.configure(configuration, device: device)
    }

    public func resizeSurface(_ surface: WGPUSurface, size: CGSize) {
        surface.updateDrawableSize(size)
    }

    public func surfaceReport(_ surface: WGPUSurface) -> WGPUSurfaceReport {
        WGPUSurfaceReport(
            width: Int(surface.pixelSize.width),
            height: Int(surface.pixelSize.height),
            format: surface.configuredFormat
        )
    }

    public func acquireFrameTexture(_ surface: WGPUSurface) throws -> WGPUAcquiredSurfaceTexture<WGPUMetalBackend>? {
        guard let drawable = surface.nextDrawable() else { return nil }
        // Use the actual drawable texture's format — canvas configuration can land a frame late.
        let format = WGPUMetalMapping.textureFormat(from: drawable.texture.pixelFormat)
            ?? surface.configuredFormat
        let texture = WGPUTextureObject(drawableTexture: drawable.texture, format: format)
        acquiredDrawables.append(drawable)
        return WGPUAcquiredSurfaceTexture(
            texture: texture,
            format: format,
            width: drawable.texture.width,
            height: drawable.texture.height,
            sampleCount: drawable.texture.sampleCount
        )
    }

    public func readPixels(_ surface: WGPUSurface, identifier: String) throws -> WGPUPixelReadback {
        guard let offscreen = surface as? WGPUOffscreenSurface else {
            throw WGPUError.validation(
                "canvas '\(identifier)' is not an offscreen surface — its pixels cannot be read"
            )
        }
        return try offscreen.readPixels(queue: queue)
    }

    // MARK: - Passes

    public func beginRenderPass(_ pass: WGPUResolvedRenderPass<WGPUMetalBackend>) throws {
        let passDescriptor = MTLRenderPassDescriptor()

        for (index, attachment) in pass.colorAttachments.enumerated() {
            let target = passDescriptor.colorAttachments[index]!
            target.texture = attachment.view.texture
            target.loadAction = WGPUMetalMapping.loadAction(attachment.loadOp)
            target.storeAction = WGPUMetalMapping.storeAction(attachment.storeOp)
            target.clearColor = MTLClearColor(
                red: attachment.clearValue.red,
                green: attachment.clearValue.green,
                blue: attachment.clearValue.blue,
                alpha: attachment.clearValue.alpha
            )
            if let resolve = attachment.resolveTarget {
                target.resolveTexture = resolve.texture
                target.storeAction = .multisampleResolve
            }
        }

        if let depth = pass.depthStencil {
            if depth.format.hasDepth {
                let target = passDescriptor.depthAttachment!
                target.texture = depth.view.texture
                // readOnly means there are no load/store ops (decoding forbids it) — the combination
                // reads the contents as they are and leaves them.
                target.loadAction = WGPUMetalMapping.loadAction(depth.depthLoadOp ?? .load)
                target.storeAction = WGPUMetalMapping.storeAction(depth.depthStoreOp ?? .store)
                target.clearDepth = depth.depthClearValue
            }
            if depth.format.hasStencil {
                let target = passDescriptor.stencilAttachment!
                target.texture = depth.view.texture
                target.loadAction = WGPUMetalMapping.loadAction(depth.stencilLoadOp ?? .load)
                target.storeAction = WGPUMetalMapping.storeAction(depth.stencilStoreOp ?? .store)
                target.clearStencil = UInt32(truncatingIfNeeded: depth.stencilClearValue)
            }
        }

        if let querySet = pass.occlusionQuerySet {
            passDescriptor.visibilityResultBuffer = querySet.visibilityBuffer
        }

        if let writes = pass.timestampWrites {
            guard let counterBuffer = writes.querySet.counterBuffer else {
                throw WGPUError.backend("no timestamp sample buffer")
            }
            let attachment = passDescriptor.sampleBufferAttachments[0]!
            attachment.sampleBuffer = counterBuffer
            // Map WebGPU's "pass start/end" onto Metal's stage boundaries — start is entering the
            // vertex stage, end is leaving the fragment stage.
            attachment.startOfVertexSampleIndex = writes.beginningOfPassWriteIndex ?? MTLCounterDontSample
            attachment.endOfVertexSampleIndex = MTLCounterDontSample
            attachment.startOfFragmentSampleIndex = MTLCounterDontSample
            attachment.endOfFragmentSampleIndex = writes.endOfPassWriteIndex ?? MTLCounterDontSample
        }

        guard let encoder = try activeCommandBuffer().makeRenderCommandEncoder(descriptor: passDescriptor) else {
            throw WGPUError.backend("MTLRenderCommandEncoder creation failed — check the attachment configuration")
        }
        if let label = pass.label { encoder.label = label }
        renderEncoder = encoder
        currentRenderPipeline = nil
    }

    public func beginComputePass(_ pass: WGPUResolvedComputePass<WGPUMetalBackend>) throws {
        let buffer = try activeCommandBuffer()

        let encoder: MTLComputeCommandEncoder?
        if let writes = pass.timestampWrites {
            guard let counterBuffer = writes.querySet.counterBuffer else {
                throw WGPUError.backend("no timestamp sample buffer")
            }
            let passDescriptor = MTLComputePassDescriptor()
            let attachment = passDescriptor.sampleBufferAttachments[0]!
            attachment.sampleBuffer = counterBuffer
            attachment.startOfEncoderSampleIndex = writes.beginningOfPassWriteIndex ?? MTLCounterDontSample
            attachment.endOfEncoderSampleIndex = writes.endOfPassWriteIndex ?? MTLCounterDontSample
            encoder = buffer.makeComputeCommandEncoder(descriptor: passDescriptor)
        } else {
            encoder = buffer.makeComputeCommandEncoder()
        }
        guard let encoder else {
            throw WGPUError.backend("MTLComputeCommandEncoder creation failed")
        }
        if let label = pass.label { encoder.label = label }
        computeEncoder = encoder
        currentComputePipeline = nil
    }

    public func setRenderPipeline(_ pipeline: WGPURenderPipelineObject) {
        guard let encoder = renderEncoder else { return }
        encoder.setRenderPipelineState(pipeline.state)
        encoder.setCullMode(pipeline.cullMode)
        encoder.setFrontFacing(pipeline.winding)
        if let depthStencilState = pipeline.depthStencilState {
            encoder.setDepthStencilState(depthStencilState)
        }
        if pipeline.depthBias != 0 || pipeline.depthBiasSlopeScale != 0 {
            encoder.setDepthBias(
                pipeline.depthBias, slopeScale: pipeline.depthBiasSlopeScale, clamp: pipeline.depthBiasClamp
            )
        }
        currentRenderPipeline = pipeline
    }

    public func setComputePipeline(_ pipeline: WGPUComputePipelineObject) {
        guard let encoder = computeEncoder else { return }
        encoder.setComputePipelineState(pipeline.state)
        currentComputePipeline = pipeline
    }

    public func applyBindGroup(_ group: WGPUMetalBindGroup, at groupIndex: Int, dynamicOffsets: [Int]) throws {
        guard let layout = currentRenderPipeline?.layout ?? currentComputePipeline?.layout else { return }
        var offsetCursor = 0
        for binding in group.bindings {
            guard let metalIndex = layout.assignment.index(group: groupIndex, binding: binding.binding) else {
                throw WGPUError.validation(
                    "the pipeline layout has no @group(\(groupIndex)) @binding(\(binding.binding))"
                )
            }
            switch binding.resource {
            case .buffer(let buffer, let offset, let boundSize):
                // Fill the size table `arrayLength()` will read (uploaded only when the shader uses it).
                if metalIndex < bufferSizes.count { bufferSizes[metalIndex] = UInt32(boundSize) }
                var finalOffset = offset
                if binding.hasDynamicOffset {
                    guard offsetCursor < dynamicOffsets.count else {
                        throw WGPUError.validation("fewer dynamicOffsets than the layout declares")
                    }
                    finalOffset += dynamicOffsets[offsetCursor]
                    offsetCursor += 1
                }
                if let encoder = renderEncoder {
                    if binding.visibility.contains(.vertex) {
                        encoder.setVertexBuffer(buffer, offset: finalOffset, index: metalIndex)
                    }
                    if binding.visibility.contains(.fragment) {
                        encoder.setFragmentBuffer(buffer, offset: finalOffset, index: metalIndex)
                    }
                } else if let encoder = computeEncoder {
                    encoder.setBuffer(buffer, offset: finalOffset, index: metalIndex)
                }
            case .texture(let texture):
                if let encoder = renderEncoder {
                    if binding.visibility.contains(.vertex) { encoder.setVertexTexture(texture, index: metalIndex) }
                    if binding.visibility.contains(.fragment) {
                        encoder.setFragmentTexture(texture, index: metalIndex)
                    }
                } else if let encoder = computeEncoder {
                    encoder.setTexture(texture, index: metalIndex)
                }
            case .sampler(let sampler):
                if let encoder = renderEncoder {
                    if binding.visibility.contains(.vertex) {
                        encoder.setVertexSamplerState(sampler, index: metalIndex)
                    }
                    if binding.visibility.contains(.fragment) {
                        encoder.setFragmentSamplerState(sampler, index: metalIndex)
                    }
                } else if let encoder = computeEncoder {
                    encoder.setSamplerState(sampler, index: metalIndex)
                }
            }
        }
    }

    public func applyVertexBuffer(_ buffer: WGPUBufferObject, offset: Int, slot: Int) {
        renderEncoder?.setVertexBuffer(
            buffer.buffer,
            offset: offset,
            index: WGSLMetalLimits.vertexBufferIndex(slot: slot)
        )
    }

    public func setViewport(_ command: WGPUSetViewportCommand) throws {
        renderEncoder?.setViewport(MTLViewport(
            originX: command.x,
            originY: command.y,
            width: command.width,
            height: command.height,
            znear: command.minDepth,
            zfar: command.maxDepth
        ))
    }

    public func setScissorRect(_ command: WGPUSetScissorRectCommand) throws {
        renderEncoder?.setScissorRect(MTLScissorRect(
            x: command.x, y: command.y, width: command.width, height: command.height
        ))
    }

    public func setBlendConstant(_ color: WGPUColor) throws {
        renderEncoder?.setBlendColor(
            red: Float(color.red), green: Float(color.green), blue: Float(color.blue), alpha: Float(color.alpha)
        )
    }

    public func setStencilReference(_ reference: UInt32) throws {
        renderEncoder?.setStencilReferenceValue(reference)
    }

    public func beginOcclusionQuery(index: Int) throws {
        // `.counting` counts the **number of samples** that passed — the same meaning as the spec's occlusion result.
        renderEncoder?.setVisibilityResultMode(.counting, offset: index * WGPUQuerySetObject.resultStride)
    }

    public func endOcclusionQuery(index: Int) throws {
        renderEncoder?.setVisibilityResultMode(.disabled, offset: index * WGPUQuerySetObject.resultStride)
    }

    public func executeBundles(_ bundles: [Never]) throws {}

    public func pushDebugGroup(_ label: String, scope: WGPUDebugScope) throws {
        switch scope {
        case .pass:
            passEncoder?.pushDebugGroup(label)
        case .frame:
            // If there is no command buffer yet, make one. That is what makes the following pop match.
            try activeCommandBuffer().pushDebugGroup(label)
        }
    }

    public func popDebugGroup(scope: WGPUDebugScope) {
        switch scope {
        case .pass:
            passEncoder?.popDebugGroup()
        case .frame:
            commandBuffer?.popDebugGroup()
        }
    }

    public func popFrameDebugGroups(count: Int) {
        guard let commandBuffer else { return }
        for _ in 0..<count { commandBuffer.popDebugGroup() }
    }

    public func insertDebugMarker(_ label: String, scope: WGPUDebugScope) throws {
        switch scope {
        case .pass:
            // An encoder has signposts (point events).
            passEncoder?.insertDebugSignpost(label)
        case .frame:
            // A command buffer has only groups, so we imitate it by opening and closing one.
            let buffer = try activeCommandBuffer()
            buffer.pushDebugGroup(label)
            buffer.popDebugGroup()
        }
    }

    // MARK: - Draw / dispatch

    /// The buffer size table for `arrayLength()`. At 88 bytes, setBytes per draw costs nothing.
    private func uploadBufferSizesIfNeeded() {
        let needsSizes = renderEncoder != nil
            ? (currentRenderPipeline?.needsBufferSizes ?? false)
            : (currentComputePipeline?.needsBufferSizes ?? false)
        guard needsSizes else { return }
        let byteLength = bufferSizes.count * MemoryLayout<UInt32>.stride
        let index = WGSLMetalLimits.bufferSizesIndex
        bufferSizes.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return }
            if let encoder = renderEncoder {
                encoder.setVertexBytes(base, length: byteLength, index: index)
                encoder.setFragmentBytes(base, length: byteLength, index: index)
            } else if let encoder = computeEncoder {
                encoder.setBytes(base, length: byteLength, index: index)
            }
        }
    }

    public func draw(_ command: WGPUDrawCommand) throws {
        guard let encoder = renderEncoder, let pipeline = currentRenderPipeline else {
            throw WGPUError.backend("draw arrived with no render pass/pipeline")
        }
        uploadBufferSizesIfNeeded()
        encoder.drawPrimitives(
            type: pipeline.primitiveType,
            vertexStart: command.firstVertex,
            vertexCount: command.vertexCount,
            instanceCount: command.instanceCount,
            baseInstance: command.firstInstance
        )
    }

    public func drawIndexed(_ command: WGPUDrawIndexedCommand, index: WGPUResolvedIndexBinding<WGPUMetalBackend>) throws {
        guard let encoder = renderEncoder, let pipeline = currentRenderPipeline else {
            throw WGPUError.backend("drawIndexed arrived with no render pass/pipeline")
        }
        uploadBufferSizesIfNeeded()
        encoder.drawIndexedPrimitives(
            type: pipeline.primitiveType,
            indexCount: command.indexCount,
            indexType: WGPUMetalMapping.indexType(index.format),
            indexBuffer: index.buffer.buffer,
            indexBufferOffset: index.offset + command.firstIndex * index.stride,
            instanceCount: command.instanceCount,
            baseVertex: command.baseVertex,
            baseInstance: command.firstInstance
        )
    }

    public func drawIndirect(buffer: WGPUBufferObject, offset: Int) throws {
        guard let encoder = renderEncoder, let pipeline = currentRenderPipeline else {
            throw WGPUError.backend("drawIndirect arrived with no render pass/pipeline")
        }
        uploadBufferSizesIfNeeded()
        encoder.drawPrimitives(
            type: pipeline.primitiveType,
            indirectBuffer: buffer.buffer,
            indirectBufferOffset: offset
        )
    }

    public func drawIndexedIndirect(
        buffer: WGPUBufferObject, offset: Int, index: WGPUResolvedIndexBinding<WGPUMetalBackend>
    ) throws {
        guard let encoder = renderEncoder, let pipeline = currentRenderPipeline else {
            throw WGPUError.backend("drawIndexedIndirect arrived with no render pass/pipeline")
        }
        uploadBufferSizesIfNeeded()
        encoder.drawIndexedPrimitives(
            type: pipeline.primitiveType,
            indexType: WGPUMetalMapping.indexType(index.format),
            indexBuffer: index.buffer.buffer,
            // Unlike the direct path (`drawIndexed`), `firstIndex` is not added here — that value is
            // inside the argument buffer and the GPU reads it. Adding it would count twice and be silently wrong.
            indexBufferOffset: index.offset,
            indirectBuffer: buffer.buffer,
            indirectBufferOffset: offset
        )
    }

    public func dispatchWorkgroups(_ command: WGPUDispatchWorkgroupsCommand) throws {
        guard let encoder = computeEncoder, let pipeline = currentComputePipeline else {
            throw WGPUError.backend("dispatchWorkgroups arrived with no compute pass/pipeline")
        }
        uploadBufferSizesIfNeeded()
        encoder.dispatchThreadgroups(
            MTLSize(width: command.x, height: command.y, depth: command.z),
            threadsPerThreadgroup: pipeline.threadsPerThreadgroup
        )
    }

    public func dispatchWorkgroupsIndirect(buffer: WGPUBufferObject, offset: Int) throws {
        guard let encoder = computeEncoder, let pipeline = currentComputePipeline else {
            throw WGPUError.backend("dispatchWorkgroupsIndirect arrived with no compute pass/pipeline")
        }
        uploadBufferSizesIfNeeded()
        encoder.dispatchThreadgroups(
            indirectBuffer: buffer.buffer,
            indirectBufferOffset: offset,
            threadsPerThreadgroup: pipeline.threadsPerThreadgroup
        )
    }

    // MARK: - Copies

    public func copyBufferToBuffer(
        source: WGPUBufferObject, sourceOffset: Int,
        destination: WGPUBufferObject, destinationOffset: Int, size: Int
    ) throws {
        let encoder = try activeBlitEncoder()
        encoder.copy(
            from: source.buffer, sourceOffset: sourceOffset,
            to: destination.buffer, destinationOffset: destinationOffset,
            size: size
        )
    }

    public func clearBuffer(_ buffer: WGPUBufferObject, range: Range<Int>) throws {
        let encoder = try activeBlitEncoder()
        encoder.fill(buffer: buffer.buffer, range: range, value: 0)
    }

    public func copyTextureToBuffer(
        texture: WGPUTextureObject, slice: Int, mipLevel: Int, origin: WGPUOrigin3D,
        size: WGPUExtent3D, buffer: WGPUBufferObject, offset: Int,
        bytesPerRow: Int, bytesPerImage: Int
    ) throws {
        let encoder = try activeBlitEncoder()
        encoder.copy(
            from: texture.texture,
            sourceSlice: slice,
            sourceLevel: mipLevel,
            sourceOrigin: MTLOrigin(x: origin.x, y: origin.y, z: 0),
            sourceSize: MTLSize(width: size.width, height: size.height, depth: 1),
            to: buffer.buffer,
            destinationOffset: offset,
            destinationBytesPerRow: bytesPerRow,
            destinationBytesPerImage: bytesPerImage
        )
    }

    public func copyBufferToTexture(
        buffer: WGPUBufferObject, offset: Int, bytesPerRow: Int, bytesPerImage: Int,
        texture: WGPUTextureObject, slice: Int, mipLevel: Int, origin: WGPUOrigin3D,
        size: WGPUExtent3D
    ) throws {
        let encoder = try activeBlitEncoder()
        encoder.copy(
            from: buffer.buffer,
            sourceOffset: offset,
            sourceBytesPerRow: bytesPerRow,
            sourceBytesPerImage: bytesPerImage,
            sourceSize: MTLSize(width: size.width, height: size.height, depth: 1),
            to: texture.texture,
            destinationSlice: slice,
            destinationLevel: mipLevel,
            destinationOrigin: MTLOrigin(x: origin.x, y: origin.y, z: 0)
        )
    }

    public func copyTextureToTexture(
        source: WGPUTextureObject, sourceSlice: Int, sourceMipLevel: Int,
        sourceOrigin: WGPUOrigin3D,
        destination: WGPUTextureObject, destinationSlice: Int, destinationMipLevel: Int,
        destinationOrigin: WGPUOrigin3D, size: WGPUExtent3D
    ) throws {
        let encoder = try activeBlitEncoder()
        encoder.copy(
            from: source.texture,
            sourceSlice: sourceSlice,
            sourceLevel: sourceMipLevel,
            sourceOrigin: MTLOrigin(x: sourceOrigin.x, y: sourceOrigin.y, z: 0),
            sourceSize: MTLSize(width: size.width, height: size.height, depth: 1),
            to: destination.texture,
            destinationSlice: destinationSlice,
            destinationLevel: destinationMipLevel,
            destinationOrigin: MTLOrigin(x: destinationOrigin.x, y: destinationOrigin.y, z: 0)
        )
    }

    /// The blit command differs per query kind (see `WGPUQuerySetObject`).
    public func resolveQuerySet(
        _ querySet: WGPUQuerySetObject, first: Int, count: Int,
        destination: WGPUBufferObject, destinationOffset: Int
    ) throws {
        let encoder = try activeBlitEncoder()
        switch querySet.type {
        case .occlusion:
            guard let source = querySet.visibilityBuffer else {
                throw WGPUError.backend("no occlusion query buffer")
            }
            encoder.copy(
                from: source, sourceOffset: first * WGPUQuerySetObject.resultStride,
                to: destination.buffer, destinationOffset: destinationOffset,
                size: count * WGPUQuerySetObject.resultStride
            )
        case .timestamp:
            guard let source = querySet.counterBuffer else {
                throw WGPUError.backend("no timestamp sample buffer")
            }
            // Counters are a dedicated store rather than an ordinary buffer, extractable only through resolveCounters.
            encoder.resolveCounters(
                source, range: first..<(first + count),
                destinationBuffer: destination.buffer, destinationOffset: destinationOffset
            )
        }
    }

    // MARK: - Adapter info

    /// The adapter info and limits `navigator.gpu.requestAdapter()` returns.
    ///
    /// Keys use **the spec's `GPUSupportedLimits` spelling exactly**. Web libraries read them by those
    /// names to set their own budgets (Three.js reads `maxComputeWorkgroupsPerDimension` and
    /// `maxUniformBufferBindingSize`). Naming them our own way would have that code see `undefined` and
    /// build wrong assumptions — behaving as if the value were absent when it is not.
    ///
    /// Values are **read from the Metal device wherever possible**; where there is no runtime query we
    /// use the guarantee from the Metal feature set table (with the basis noted in the comments below).
    public func adapterInfo() -> [String: Any] {
        let threadgroup = device.maxThreadsPerThreadgroup
        // Apple GPU family 3 and above (A9+) and Mac2 allow 2D textures up to 16384. Below that, 8192.
        // This project's minimum (iOS 17 = A12+) is always the former, but we split for older macOS.
        let maxTexture2D = device.supportsFamily(.apple3) || device.supportsFamily(.mac2) ? 16384 : 8192

        let limits: [String: Any] = [
            // Textures
            "maxTextureDimension1D": maxTexture2D,
            "maxTextureDimension2D": maxTexture2D,
            "maxTextureDimension3D": 2048,
            "maxTextureArrayLayers": 2048,
            // Bindings — these fall straight out of our argument table assignment rules (`WGSLMetalLimits`)
            "maxBindGroups": 4,
            "maxBindGroupsPlusVertexBuffers": 4 + WGSLMetalLimits.maxVertexBufferSlots,
            "maxBindingsPerBindGroup": 1000,
            "maxSampledTexturesPerShaderStage": WGSLMetalLimits.textureSlotCount,
            "maxSamplersPerShaderStage": WGSLMetalLimits.samplerSlotCount,
            "maxStorageBuffersPerShaderStage": WGSLMetalLimits.maxBindGroupBuffers,
            "maxStorageTexturesPerShaderStage": WGSLMetalLimits.textureSlotCount,
            "maxUniformBuffersPerShaderStage": WGSLMetalLimits.maxBindGroupBuffers,
            "maxDynamicUniformBuffersPerPipelineLayout": 8,
            "maxDynamicStorageBuffersPerPipelineLayout": 4,
            // Buffers — offset alignment uses the spec default (256) as-is. It exceeds what Metal
            // requires (32B on Apple GPUs), so honouring it satisfies Metal too. Reporting 32 instead
            // would ship code that breaks only in a browser.
            "maxBufferSize": device.maxBufferLength,
            "maxUniformBufferBindingSize": 65536,
            "maxStorageBufferBindingSize": device.maxBufferLength,
            "minUniformBufferOffsetAlignment": 256,
            "minStorageBufferOffsetAlignment": 256,
            // Vertex
            "maxVertexBuffers": WGSLMetalLimits.maxVertexBufferSlots,
            "maxVertexAttributes": 30,
            "maxVertexBufferArrayStride": 2048,
            "maxInterStageShaderVariables": 16,
            // Attachments
            "maxColorAttachments": 8,
            "maxColorAttachmentBytesPerSample": 32,
            // Compute
            "maxComputeWorkgroupStorageSize": device.maxThreadgroupMemoryLength,
            "maxComputeInvocationsPerWorkgroup": threadgroup.width,
            "maxComputeWorkgroupSizeX": threadgroup.width,
            "maxComputeWorkgroupSizeY": threadgroup.height,
            "maxComputeWorkgroupSizeZ": threadgroup.depth,
            // Metal has no query for the dispatch grid cap. We use the same conservative value as Dawn.
            "maxComputeWorkgroupsPerDimension": 65535,
        ]

        // The spec's `GPUAdapterInfo` — the standard names web code reads when branching on GPU kind.
        // Fields whose value we do not know stay **empty strings** (as the spec directs) — inventing one
        // sends code branching on that string down a wrong detour.
        let info: [String: Any] = [
            "vendor": "apple",
            "architecture": architectureName(),
            // The spec's `device` is a vendor-specific identifier (a PCI device ID, say). Metal has none.
            "device": "",
            "description": device.name,
            "isFallbackAdapter": false,
            // We do not advertise the `subgroups` feature, so it is 0 as the spec directs.
            "subgroupMinSize": 0,
            "subgroupMaxSize": 0,
        ]

        return [
            "ok": true,
            "info": info,
            "name": device.name,
            "backend": "metal",
            "hasUnifiedMemory": device.hasUnifiedMemory,
            "supportsFamilyApple7": device.supportsFamily(.apple7),
            "preferredCanvasFormat": WGPUTextureFormat.bgra8unorm.rawValue,
            "limits": limits,
            "features": features(),
        ]
    }

    /// `GPUAdapterInfo.architecture` — the GPU family name.
    ///
    /// The spec says only "the family name, or an empty string if unknown". Metal has no API to ask for
    /// the family, only `supportsFamily` to **check upward from below**. We probe from the highest down
    /// and answer only as far as we learn — unknown means an empty string.
    private func architectureName() -> String {
        let families: [(MTLGPUFamily, String)] = [
            (.apple9, "apple-9"), (.apple8, "apple-8"), (.apple7, "apple-7"),
            (.apple6, "apple-6"), (.apple5, "apple-5"), (.apple4, "apple-4"),
            (.apple3, "apple-3"), (.apple2, "apple-2"), (.apple1, "apple-1"),
        ]
        for (family, name) in families where device.supportsFamily(family) { return name }
        return device.supportsFamily(.mac2) ? "mac-2" : ""
    }

    /// Features that vary per device (`adapter.features` — the spec spelling exactly).
    ///
    /// Only things JS must be able to ask about before creating something. Knowing in advance and
    /// taking another route beats trying to create the impossible and receiving an error.
    private func features() -> [String] {
        var result: [String] = []
        if device.supportsCounterSampling(.atStageBoundary),
           device.counterSets?.contains(where: { $0.name == MTLCommonCounterSet.timestamp.rawValue }) == true {
            result.append("timestamp-query")
        }
        // Whether indirect draw arguments honour `firstInstance`. The spec makes this an optional
        // feature and, without it, turns a draw with a non-zero value into **a complete no-op**. Metal's
        // argument layout matches WebGPU and honours `baseInstance` as-is, so here we stand where an
        // adapter with the feature enabled stands. The argument value lives in a GPU buffer and cannot
        // be checked at encoding time, so advertising the feature is the only way to convey the situation to an app.
        // A device that cannot do indirect arguments at all does not advertise this either — the simulator is one.
        // Advertising it and then refusing the first call would betray an app that checked before using it.
        if WGPUDeviceCapability.supportsIndirectArguments(device) {
            result.append("indirect-first-instance")
        }
        // Block compression families. Creating a texture in an absent family kills Metal with an
        // assertion, so we announce it here for apps to branch on (creation blocks it again with an error).
        for probe: WGPUTextureFormat in [.bc1RGBAUnorm, .etc2RGB8Unorm, .astc4x4Unorm] {
            guard let name = WGPUDeviceCapability.compressionFamily(probe).featureName else { continue }
            if WGPUDeviceCapability.supportsCompression(probe, on: device) { result.append(name) }
        }
        return result
    }
}

/// The Metal backend's `GPUBindGroup` — a binding list already resolved into Metal objects at creation.
///
/// Handle resolution, layout matching and the `boundSize` default are finished by the engine; what
/// remains here is only what `applyBindGroup` puts on the encoder right before a draw.
public final class WGPUMetalBindGroup {
    struct Binding {
        let binding: Int
        let visibility: WGPUShaderStage
        let hasDynamicOffset: Bool
        let resource: WGPUResolvedBinding
    }

    let bindings: [Binding]

    init(bindings: [Binding]) {
        self.bindings = bindings
    }
}
