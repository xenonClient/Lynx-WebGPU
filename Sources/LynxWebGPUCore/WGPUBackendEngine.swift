import Foundation
import CoreGraphics
import QuartzCore

/// The command-stream orchestration engine — the **backend-independent implementation** of `WebGPURuntime`.
///
/// It interprets one frame's worth of commands from JS and turns them into `WGPUBackend` verb calls.
/// Whatever the backend (Metal directly, Dawn, …), **the wire contract and spec validation are
/// discharged here exactly once**:
///
/// - the batch loop, decoding and dispatch (an exhaustive switch — a missing op is a compile error);
/// - error collection (never killing the frame), error scopes, deferred GPU errors, response
///   assembly (`WGPUBatchResult`);
/// - frame lifetime: expiring drawable handles at present, in-flight accounting (`WGPUFrameCoordinator`);
/// - the mapping gate (no queue use of a buffer during mapAsync), range/alignment/usage/completeness checks;
/// - draw-state shadowing (bind groups, vertex buffers) — binding invalidation at bundle boundaries lives here;
/// - render bundle record/replay (for backends without native bundles);
/// - serialization: `execute` (JS thread), `processEvents` (main tick) and `readBuffer` registration
///   all run under one execution lock — the discharge point of the pump-concurrency contract
///   (`docs/COMMAND-STREAM.md` §5-1).
///
/// The backend is left only with calling its own GPU API on values already resolved and validated
/// (see `WGPUBackend`). This layer was raised after duplicated orchestration in the Dawn prototype
/// turned into real defects — pump races, present order, scope drains.
public final class WGPUBackendEngine<B: WGPUBackend>: WebGPURuntime {
    public let backend: B
    /// In-flight frame accounting — present timing and saturation. Backend-independent policy, so the engine drives it.
    public let frameCoordinator: WGPUFrameCoordinator

    private let registry = WGPUObjectRegistry()
    /// The execution serialization lock. **Recursive** — on a backend where `readBuffer`'s completion
    /// arrives synchronously during registration (work already finished), the completion wrapper
    /// retakes the same lock.
    private let executionLock = NSRecursiveLock()
    private let canvasLock = NSLock()

    // MARK: Batch state — valid only for the lifetime of one batch

    private enum PassState { case render, compute }
    private var passState: PassState?
    /// Debug groups opened on the current encoder / in the frame region — a mismatch kills the backend with an assertion.
    private var encoderDebugDepth = 0
    private var bufferDebugDepth = 0
    private var currentRenderPipeline: WGPUEngineRenderPipeline<B>?
    private var currentComputePipeline: WGPUEngineComputePipeline<B>?
    private var boundGroups: [Int: (group: WGPUEngineBindGroup<B>, offsets: [Int])] = [:]
    private var dirtyGroups: Set<Int> = []
    private var indexBinding: (buffer: WGPUEngineBuffer<B>, offset: Int, format: WGPUIndexFormat, stride: Int)?
    /// Vertex buffer bindings per slot. **They do not go straight to the backend**; they gather here
    /// and go down right before a draw. That is what lets `resetPassBindings()` genuinely invalidate
    /// bindings at a bundle boundary — a backend encoder has no "unbind", so invalidation can only
    /// be expressed in shadow state.
    private var vertexBindings: [Int: (buffer: WGPUEngineBuffer<B>, offset: Int)] = [:]
    private var dirtyVertexSlots: Set<Int> = []
    /// The current render pass's attachment shape — used to decide whether a render bundle is valid here.
    private var passFormats: (color: [WGPUTextureFormat], depthStencil: WGPUTextureFormat?, sampleCount: Int)?
    /// Whether the current render pass declared it **will not write** depth/stencil (`depthReadOnly`/`stencilReadOnly`).
    private var passDepthReadOnly = false
    private var passStencilReadOnly = false
    private var passOcclusionQuerySet: WGPUEngineQuerySet<B>?
    /// The open occlusion query index — catches nesting and non-termination.
    private var openOcclusionQuery: Int?
    /// Occlusion query indices already used in this pass — the spec forbids reuse within a pass.
    private var usedOcclusionQueries: Set<Int> = []
    /// The (handle, canvas id) pairs handed a drawable this frame — subject to expiry and accounting at present.
    private var acquiredFrames: [(handle: WGPUHandle, canvas: String)] = []
    /// Canvases handed a drawable **in this batch**. Used to tell a repeat acquisition within one
    /// batch (same frame) from one across batches (a new frame) — see `getCurrentTexture`.
    private var acquiredThisBatch: Set<String> = []
    /// Handles that become invalid once the frame ends (the drawable texture and its views).
    private var frameScopedHandles: [WGPUHandle] = []
    private var touchedCanvases: [String: B.Surface] = [:]
    private var errors: [WGPUError] = []

    /// Report that a previous batch's GPU execution failed — filled by the completion callback
    /// (arbitrary thread) and drained by the next batch (see `WGPUDeferredErrorQueue`).
    private let gpuFailures = WGPUDeferredErrorQueue()
    /// Open error scopes — all the rules live in `WGPUErrorScopeStack`.
    private var errorScopes = WGPUErrorScopeStack()
    /// Results of scopes popped this batch (in pop order — matched 1:1 with the JS promise order).
    private var poppedScopes: [WGPUPoppedErrorScope] = []

    /// Whether the engine performs spec checks — false on a backend carrying a complete validator
    /// (Dawn), leaving this layer with bridging and **minimal exception handling** only (handle
    /// lookup, wire mapping state, pass state guards, CPU path protection)
    /// (`WGPUBackendCapabilities.validatesNatively`).
    private let specValidation: Bool

    public init(backend: B, frameCoordinator: WGPUFrameCoordinator = WGPUFrameCoordinator()) {
        self.backend = backend
        self.frameCoordinator = frameCoordinator
        self.specValidation = !backend.capabilities.validatesNatively
    }

    // MARK: - WebGPURuntime: execution

    public func execute(_ payload: [String: Any]) -> [String: Any] {
        executionLock.lock()
        defer { executionLock.unlock() }

        let reader = WGPUValueReader(payload)
        let commands: [WGPUValueReader]
        do {
            commands = try reader.requiredObjects("commands")
        } catch let error as WGPUError {
            return WGPUBatchResult.failure([error])
        } catch {
            return WGPUBatchResult.failure([WGPUError.validation("\(error)")])
        }
        return run(commands, present: reader.bool("present", default: true))
    }

    private func run(_ commands: [WGPUValueReader], present: Bool) -> [String: Any] {
        resetBatchState()

        // Flush the previous batch's GPU execution failures first — an open error scope catches them.
        for failure in gpuFailures.drain() { record(failure) }
        backend.beginBatch()

        for (index, command) in commands.enumerated() {
            do {
                try perform(command)
            } catch let error as WGPUError {
                // Fill in the path and **carry everything else across unchanged** — dropping a field
                // here (the line number, say) silently loses a clue a lower layer worked to attach.
                record(WGPUError(
                    kind: error.kind,
                    message: error.message,
                    path: error.path ?? "commands[\(index)].\(command.optionalString("op") ?? "?")",
                    line: error.line
                ))
            } catch {
                record(.backend(error.localizedDescription, path: "commands[\(index)]"))
            }
        }

        // No commands but present means **the tick's closing batch** (the end of a frame-loop callback).
        finish(WGPUFrameBoundary(requestedPresent: present, commandCount: commands.count))

        // Per-batch backend diagnostics (Dawn's device scope, …) — collected after submit so this
        // batch's GPU validation errors are included too.
        for diagnostic in backend.collectBatchDiagnostics() { record(diagnostic) }

        return WGPUBatchResult(
            commandCount: commands.count,
            liveObjectCount: registry.count,
            errors: errors,
            canvases: touchedCanvases.mapValues { surface in
                let report = backend.surfaceReport(surface)
                return WGPUCanvasReport(width: report.width, height: report.height)
            },
            poppedScopes: poppedScopes
        ).payload
    }

    /// Files one error with the innermost matching scope, or emits it in the batch result if there is none.
    private func record(_ error: WGPUError) {
        if errorScopes.capture(error) { return }
        errors.append(error)
    }

    private func resetBatchState() {
        passState = nil
        encoderDebugDepth = 0
        bufferDebugDepth = 0
        currentRenderPipeline = nil
        currentComputePipeline = nil
        boundGroups.removeAll()
        dirtyGroups.removeAll()
        indexBinding = nil
        // Clear the vertex bindings too — the next `beginRenderPass`'s `resetPassBindings()` would
        // overwrite them anyway, but not releasing here holds a `destroy`ed buffer until then.
        vertexBindings.removeAll()
        dirtyVertexSlots.removeAll()
        passFormats = nil
        passOcclusionQuerySet = nil
        passDepthReadOnly = false
        passStencilReadOnly = false
        openOcclusionQuery = nil
        usedOcclusionQueries.removeAll()
        touchedCanvases.removeAll()
        acquiredThisBatch.removeAll()
        errors.removeAll()
        poppedScopes.removeAll()
        // `errorScopes` is deliberately not cleared — it is device state and spans batches.
        // The same goes for `acquiredFrames` and `frameScopedHandles` — a frame's boundary is
        // **present**, not the batch, and one frame can be split across several (see finish() below).
    }

    /// Finishes one batch.
    ///
    /// `present: false` means an internal submit from the **middle** of a frame — a batch the shim
    /// flushed early to collect a `popErrorScope`/`mapAsync` result. Submit the GPU work (readback
    /// waits on the completion) but **defer the drawable present and frame-scoped handle expiry to
    /// the real frame submit that follows.** Without deferring: if that batch produced GPU work from
    /// even a single `writeBuffer`, the just-acquired drawable is presented before anything is drawn
    /// and its handle expires, so the following `beginRenderPass` is rejected outright with
    /// "GPUTextureView does not exist" — Three.js's lazy pipeline creation (flush right after pop)
    /// walked exactly this path.
    private func finish(_ boundary: WGPUFrameBoundary) {
        let present = boundary.presents
        closePass()
        // Groups opened in the frame region are closed before submit too (same reason as encoders —
        // Metal dies on an assertion). A natively-validating backend raises its own error at Finish,
        // so we do not close for it.
        if specValidation, backend.hasPendingWork, bufferDebugDepth > 0 {
            record(.validation(
                "\(bufferDebugDepth) debug group(s) were still open at submit (a popDebugGroup is missing)"
            ))
            backend.popFrameDebugGroups(count: bufferDebugDepth)
        }
        bufferDebugDepth = 0
        // A tick's closing batch must **put the drawable out** even with no commands — passing over
        // it for lack of submittable work freezes the screen with nothing said.
        //
        // Narrowing the condition to "no commands" matters. A batch that has commands but produced no
        // GPU work (one that only acquired a drawable, say) does not present — putting out a drawable
        // that was never drawn ships that frame as a blank screen.
        if boundary.closesFrame, !backend.hasPendingWork, !acquiredFrames.isEmpty {
            backend.ensureSubmittableWork()
        }
        // With nothing to submit there is nothing to present. The acquired drawable **stays as it is**
        // — this state may mean "the frame is still going", not "the frame failed" (Three.js's lazy
        // pipeline creation does exactly this), and releasing here would break the following
        // `beginRenderPass` with a "missing handle". A drawable held this way is reclaimed by the
        // next frame's acquisition (see `getCurrentTexture`).
        guard backend.hasPendingWork else { return }

        // In-flight accounting — the frame ticker reads this count and skips a tick when saturated.
        // A batch that does not present is not a frame, so it is not counted.
        let presentedCanvases = present ? uniquePresentedCanvases() : []
        let coordinator = frameCoordinator
        for canvas in presentedCanvases { coordinator.noteCommitted(canvas: canvas) }
        // The completion callback captures values (the queue and coordinator), not the engine — safe
        // even if the engine is released first.
        let failures = gpuFailures
        backend.submit(present: present) { failure in
            if let failure { failures.report(failure) }
            for canvas in presentedCanvases { coordinator.noteCompleted(canvas: canvas) }
        }
        // The drawable texture and its views become invalid **at present** (the moment the spec's
        // "Expire the current texture" fixes). Reclaiming at the end of every batch would let a
        // mid-frame submit erase that frame's swapchain handles and break the following
        // `beginRenderPass` with a "missing handle". (The drawable itself was already released by
        // `submit(present: true)` as it put it out.)
        if present, !acquiredFrames.isEmpty { expireFrame() }
    }

    /// The frame ended — expire the drawable texture and view handles (the spec's
    /// "Expire the current texture" in `GPUCanvasContext`).
    ///
    /// Factored out because a presented frame and **a frame that ended with nothing to submit** must
    /// do the same thing. Fix only one and handles live forever on the other path.
    private func expireFrame() {
        for handle in frameScopedHandles { registry.remove(handle) }
        frameScopedHandles.removeAll()
        acquiredFrames.removeAll()
    }

    /// Canvases handed a drawable this frame (deduplicated — several acquisitions on one surface are still one frame).
    private func uniquePresentedCanvases() -> [String] {
        var seen = Set<String>()
        var canvases: [String] = []
        for acquired in acquiredFrames where seen.insert(acquired.canvas).inserted {
            canvases.append(acquired.canvas)
        }
        return canvases
    }

    /// Closes the open pass — reports an unterminated occlusion query or leftover debug groups as
    /// errors but **closes them and carries on.** Dying on a backend assertion here would remove any
    /// chance to diagnose.
    private func closePass() {
        if passState == .render {
            // The spec requires no occlusion query to be open when a pass closes. Metal simply writes
            // the value, so not catching it here leaves **even the value looking correct** while a
            // browser drops the whole frame. The pass is already closing, so we record instead of
            // throwing. (A natively-validating backend raises its own error at End.)
            if specValidation, let index = openOcclusionQuery {
                record(.validation(
                    "occlusion query \(index) was still open when the render pass ended "
                        + "(an endOcclusionQuery is missing)"
                ))
            }
            openOcclusionQuery = nil
            usedOcclusionQueries.removeAll()
            passOcclusionQuerySet = nil
            passFormats = nil
            passDepthReadOnly = false
            passStencilReadOnly = false
        }
        // Closing an encoder with debug groups open kills Metal with an assertion — close them and report.
        // (A natively-validating backend turns it into an error at End/Finish without dying.)
        if specValidation, passState != nil, encoderDebugDepth > 0 {
            record(.validation(
                "\(encoderDebugDepth) debug group(s) were still open when the pass ended (a popDebugGroup is missing)"
            ))
            while encoderDebugDepth > 0 {
                backend.popDebugGroup(scope: .pass)
                encoderDebugDepth -= 1
            }
        }
        encoderDebugDepth = 0
        passState = nil
        backend.endPass()
    }

    // MARK: - Command dispatch

    /// Decoding and the branch table are finished by `WGPUCommand` — what remains here is the
    /// **exhaustive switch** (no `default`) that validates and resolves decoded values into backend
    /// verbs. Forget a case when adding an op and the compile breaks — a missing backend verb is
    /// caught by `WGPUBackend`'s protocol requirements.
    private func perform(_ command: WGPUValueReader) throws {
        try dispatch(WGPUCommand(from: command))
    }

    private func dispatch(_ command: WGPUCommand) throws {
        switch command {
        // Resources
        case .createBuffer(let c): try createBuffer(c)
        case .writeBuffer(let c): try writeBuffer(c)
        case .unmapBuffer(let c): try unmapBuffer(c)
        case .createTexture(let c): try createTexture(c)
        case .writeTexture(let c): try writeTexture(c)
        case .copyExternalImageToTexture(let c): try copyExternalImageToTexture(c)
        case .createTextureView(let c): try createTextureView(c)
        case .createSampler(let c): try createSampler(c)
        case .createShaderModule(let c): try createShaderModule(c)
        case .createBindGroupLayout(let c): try createBindGroupLayout(c)
        case .createPipelineLayout(let c): try createPipelineLayout(c)
        case .createBindGroup(let c): try createBindGroup(c)
        case .createQuerySet(let c): try createQuerySet(c)
        case .createRenderBundle(let c): try createRenderBundle(c)
        case .createRenderPipeline(let c): try createRenderPipeline(c)
        case .createComputePipeline(let c): try createComputePipeline(c)
        case .getBindGroupLayout(let c): try getBindGroupLayout(c)
        case .destroy(let c): registry.remove(c.id)

        // Error scopes
        case .pushErrorScope(let filter, let decodeFailure):
            // Push first even on failure — the depth-keeping contract (see `WGPUCommand`).
            errorScopes.push(filter)
            if let decodeFailure { throw decodeFailure }
        case .popErrorScope: poppedScopes.append(errorScopes.pop())

        // Canvas
        case .configureCanvas(let c): try configureCanvas(c)
        case .getCurrentTexture(let c): try getCurrentTexture(c)

        // Render pass
        case .beginRenderPass(let c): try beginRenderPass(c)
        case .setPipeline(let c): try setPipeline(c)
        case .setBindGroup(let c): try setBindGroup(c)
        case .setVertexBuffer(let c): try setVertexBuffer(c)
        case .setIndexBuffer(let c): try setIndexBuffer(c)
        case .setViewport(let c):
            try requireRenderPass()
            try backend.setViewport(c)
        case .setScissorRect(let c):
            try requireRenderPass()
            try backend.setScissorRect(c)
        case .setBlendConstant(let c):
            try requireRenderPass()
            try backend.setBlendConstant(c.color)
        case .setStencilReference(let c):
            try requireRenderPass()
            try backend.setStencilReference(c.reference)
        case .draw(let c): try draw(c)
        case .drawIndexed(let c): try drawIndexed(c)
        case .drawIndirect(let c): try drawIndirect(c)
        case .drawIndexedIndirect(let c): try drawIndexedIndirect(c)
        case .executeBundles(let c): try executeBundles(c)
        case .beginOcclusionQuery(let c): try beginOcclusionQuery(c)
        case .endOcclusionQuery: try endOcclusionQuery()

        // Compute pass
        case .beginComputePass(let c): try beginComputePass(c)
        case .dispatchWorkgroups(let c): try dispatchWorkgroups(c)
        case .dispatchWorkgroupsIndirect(let c): try dispatchWorkgroupsIndirect(c)

        case .endPass: closePass()

        // Copies
        case .copyBufferToBuffer(let c): try copyBufferToBuffer(c)
        case .clearBuffer(let c): try clearBuffer(c)
        case .copyTextureToBuffer(let c): try copyTextureToBuffer(c)
        case .copyBufferToTexture(let c): try copyBufferToTexture(c)
        case .copyTextureToTexture(let c): try copyTextureToTexture(c)

        // Queries
        case .resolveQuerySet(let c): try resolveQuerySet(c)

        // Debug markers
        case .pushDebugGroup(let c): try pushDebugGroup(c)
        case .popDebugGroup: popDebugGroup()
        case .insertDebugMarker(let c):
            try backend.insertDebugMarker(c.markerLabel, scope: passState != nil ? .pass : .frame)
        }
    }

    // MARK: - Lookup helpers

    private func buffer(_ handle: WGPUHandle, path: String? = nil) throws -> WGPUEngineBuffer<B> {
        try registry.lookup(handle, as: WGPUEngineBuffer<B>.self, kind: "GPUBuffer", path: path)
    }

    /// Fetches a buffer for queue work — **rejects it while mapped.**
    ///
    /// The spec makes `mapAsync` mark a buffer "unavailable" so it cannot be used in queue work until
    /// `unmap()`, removing the race entirely. If a write from the next frame overlaps the same memory
    /// while a read waits on GPU completion, **which frame's values JS receives is not guaranteed.**
    ///
    /// Every command that uses a buffer must pass through here — miss one and the race leaks in by that path.
    private func unmappedBuffer(_ handle: WGPUHandle, path: String? = nil) throws -> WGPUEngineBuffer<B> {
        let object = try buffer(handle, path: path)
        guard !object.isMapped else {
            throw WGPUError.validation(
                "GPUBuffer \(handle) is mapped and cannot be used in queue work "
                    + "(read it with mapAsync, then call unmap())",
                path: path
            )
        }
        return object
    }

    private func texture(_ handle: WGPUHandle, path: String? = nil) throws -> WGPUEngineTexture<B> {
        try registry.lookup(handle, as: WGPUEngineTexture<B>.self, kind: "GPUTexture", path: path)
    }

    private func textureView(_ handle: WGPUHandle, path: String? = nil) throws -> WGPUEngineTextureView<B> {
        try registry.lookup(handle, as: WGPUEngineTextureView<B>.self, kind: "GPUTextureView", path: path)
    }

    private func requireRenderPass() throws {
        guard passState == .render else {
            throw WGPUError.validation("no render pass has begun (call beginRenderPass first)")
        }
    }

    private func requireComputePass() throws {
        guard passState == .compute else {
            throw WGPUError.validation("no compute pass has begun (call beginComputePass first)")
        }
    }

    /// Copies and uploads only outside a pass — they are command-encoder-level commands (and the JS
    /// shim only sends them that way). On a natively-validating backend the backend's validator
    /// catches it (encoder copies — at Dawn's Finish).
    private func requireNoOpenPass() throws {
        guard specValidation else { return }
        guard passState == nil else {
            throw WGPUError.validation("copy and upload commands cannot be used inside a render/compute pass")
        }
    }

    // MARK: - Resource creation

    private func createBuffer(_ command: WGPUCreateCommand<WGPUBufferDescriptor>) throws {
        let raw = try backend.makeBuffer(command.descriptor)
        let object = WGPUEngineBuffer<B>(
            raw: raw, size: command.descriptor.size, usage: command.descriptor.usage
        )
        // Spec: a mappedAtCreation buffer is "unavailable" until unmap.
        // (The JS shim folds this flag on the client and sends initialData instead — this path
        //  belongs to native users building the command stream directly.)
        object.isMapped = command.descriptor.mappedAtCreation
        registry.insert(object, at: command.id)
    }

    private func unmapBuffer(_ command: WGPUUnmapBufferCommand) throws {
        let object = try registry.lookup(
            command.buffer, as: WGPUEngineBuffer<B>.self, kind: "GPUBuffer"
        )
        object.isMapped = false
        backend.unmapBuffer(object.raw)
    }

    private func writeBuffer(_ command: WGPUWriteBufferCommand) throws {
        let target = try unmappedBuffer(command.buffer, path: command.fieldPath("buffer"))
        let offset = command.bufferOffset
        let data = command.data
        if specValidation {
            guard offset >= 0, offset + data.count <= target.size else {
                throw WGPUError.validation(
                    "writeBuffer out of range — offset \(offset) + \(data.count)B > buffer size \(target.size)B"
                )
            }
            guard target.usage.contains(.copyDst) else {
                throw WGPUError.validation(
                    "the destination of writeBuffer must be created with GPUBufferUsage.COPY_DST",
                    path: command.fieldPath("buffer")
                )
            }
            // The multiple-of-4 requirement is a spec rule. Metal accepts byte-granular blits too, so
            // leaving it unchecked ships code that is rejected only in a browser (same as `clearBuffer`).
            guard offset % 4 == 0, data.count % 4 == 0 else {
                throw WGPUError.validation(
                    "writeBuffer's bufferOffset and size must be multiples of 4 "
                        + "(got \(offset), \(data.count)B)",
                    path: command.fieldPath("bufferOffset")
                )
            }
        }
        guard !data.isEmpty else { return }   // a zero-size write is a no-op
        try requireNoOpenPass()
        try backend.writeBuffer(target.raw, offset: offset, data: data)
    }

    private func createTexture(_ command: WGPUCreateCommand<WGPUTextureDescriptor>) throws {
        if specValidation {
            try validateTextureDescriptor(command.descriptor, path: { command.fieldPath($0) })
            try validateCompressedTexture(command.descriptor)
        }
        let raw = try backend.makeTexture(command.descriptor)
        registry.insert(
            WGPUEngineTexture<B>(
                raw: raw,
                format: command.descriptor.format,
                size: command.descriptor.size,
                sampleCount: command.descriptor.sampleCount,
                isDrawable: false
            ),
            at: command.id
        )
    }

    /// Numeric constraints of `GPUTextureDescriptor` (the spec's "validating GPUTextureDescriptor").
    ///
    /// **A negative value kills the process.** JS can send any integer, and a negative `mipLevelCount`
    /// crossing into an unsigned Metal descriptor field folds to `UInt.max`, ending the process with
    /// the assertion `MTLTextureDescriptor requests 18446744073709551615 mipmap levels` — the app
    /// never gets a chance to say why. Size and sample count have the same shape, so they are guarded
    /// together — all of these are rules the spec already fixed.
    private func validateTextureDescriptor(
        _ descriptor: WGPUTextureDescriptor,
        path: (String) -> String
    ) throws {
        let size = descriptor.size
        guard size.width >= 1, size.height >= 1, size.depthOrArrayLayers >= 1 else {
            throw WGPUError.validation(
                "every axis of a texture size must be at least 1 "
                    + "(\(size.width)x\(size.height)x\(size.depthOrArrayLayers))",
                path: path("size")
            )
        }
        // Spec: 1 <= mipLevelCount <= the maximum number of levels this size can hold.
        let largest = descriptor.dimension == .threeD
            ? max(size.width, size.height, size.depthOrArrayLayers)
            : max(size.width, size.height)
        let maxLevels = Int(log2(Double(largest))) + 1
        guard descriptor.mipLevelCount >= 1, descriptor.mipLevelCount <= maxLevels else {
            throw WGPUError.validation(
                "mipLevelCount must be between 1 and \(maxLevels) "
                    + "(got \(descriptor.mipLevelCount), size \(size.width)x\(size.height))",
                path: path("mipLevelCount")
            )
        }
        // The spec allows only 1 and 4.
        guard descriptor.sampleCount == 1 || descriptor.sampleCount == 4 else {
            throw WGPUError.validation(
                "sampleCount must be 1 or 4 (got \(descriptor.sampleCount))",
                path: path("sampleCount")
            )
        }
        if descriptor.sampleCount == 4 {
            guard descriptor.mipLevelCount == 1, descriptor.dimension == .twoD,
                  size.depthOrArrayLayers == 1 else {
                throw WGPUError.validation(
                    "a multisample texture must be 2d with 1 mip level and 1 layer "
                        + "(mipLevelCount \(descriptor.mipLevelCount), "
                        + "\(descriptor.dimension.rawValue), \(size.depthOrArrayLayers) layer(s))",
                    path: path("sampleCount")
                )
            }
        }
    }

    /// Limits on block-compressed textures. **These combinations kill the backend with an assertion**,
    /// so we catch them first and return the spec's validation error instead.
    private func validateCompressedTexture(_ descriptor: WGPUTextureDescriptor) throws {
        let format = descriptor.format
        guard format.isCompressed else { return }
        guard backend.supportsTextureCompression(format) else {
            let feature = format.compressionFamily.featureName ?? "?"
            throw WGPUError.validation(
                "this device does not support \(format.rawValue) — check adapter.features for '\(feature)' first"
            )
        }
        // Compressed formats can only be sampled and copied (spec: no RENDER_ATTACHMENT or STORAGE_BINDING).
        let forbidden: WGPUTextureUsage = [.renderAttachment, .storageBinding]
        guard descriptor.usage.isDisjoint(with: forbidden) else {
            throw WGPUError.validation(
                "a compressed texture (\(format.rawValue)) cannot be a render target or storage (usage \(descriptor.usage))"
            )
        }
        guard descriptor.dimension == .twoD else {
            throw WGPUError.validation("compressed textures must be 2d (\(descriptor.dimension.rawValue) requested)")
        }
        guard descriptor.sampleCount == 1 else {
            throw WGPUError.validation("a compressed texture cannot be multisampled (sampleCount \(descriptor.sampleCount))")
        }
    }

    private func writeTexture(_ command: WGPUWriteTextureCommand) throws {
        let target = try texture(command.texture, path: command.fieldPath("texture"))
        let data = command.data
        let size = command.size
        let format = target.format
        // The default for an omitted stride **requires knowing the format** — in a compressed format
        // a row is measured in **blocks**, not pixels (spec GPUTexelCopyBufferLayout). So it is filled
        // in here rather than during decoding.
        let bytesPerRow = command.bytesPerRow ?? format.bytesPerRow(width: size.width)
        let blockRows = format.blockRows(height: size.height)
        let rowsPerImage = command.rowsPerImage ?? blockRows
        guard size.width > 0, size.height > 0, size.depthOrArrayLayers > 0 else { return }   // no-op
        if specValidation {
            try validateBlockAlignment(format: format, origin: command.origin, size: size,
                                       textureSize: target.size, mipLevel: command.mipLevel,
                                       label: "writeTexture")
            let bytesPerImage = bytesPerRow * max(rowsPerImage, blockRows)
            let layers = max(size.depthOrArrayLayers, 1)
            let required = bytesPerImage * (layers - 1) + bytesPerRow * blockRows
            guard data.count >= required else {
                throw WGPUError.validation("writeTexture data is too short (\(data.count)B, at least \(required)B needed)")
            }
        }
        try requireNoOpenPass()
        try backend.writeTexture(
            target.raw, data: data, origin: command.origin, size: size,
            mipLevel: command.mipLevel, bytesPerRow: bytesPerRow, rowsPerImage: rowsPerImage
        )
    }

    /// Uploads an already-decoded image (`ImageBitmap`) into a texture — the spec's
    /// `queue.copyExternalImageToTexture()`.
    ///
    /// The pixels are already RGBA8, so all that happens here is cropping and converging on the `writeTexture` verb.
    private func copyExternalImageToTexture(_ command: WGPUCopyExternalImageCommand) throws {
        let bitmap = try registry.lookup(
            command.source.image, as: WGPUImageBitmapObject.self, kind: "ImageBitmap",
            path: command.source.fieldPath("source")
        )
        let target = try texture(command.destination.texture, path: command.destination.fieldPath("texture"))
        guard !target.format.isCompressed else {
            throw WGPUError.validation(
                "copyExternalImageToTexture cannot target a compressed texture (\(target.format.rawValue)) "
                + "— the GPU has no block encoder"
            )
        }
        // The spec requires the source and destination byte widths to match. The decode result is
        // RGBA8, so only 4-byte formats are accepted — better to stop here than to let the screen go
        // quietly wrong.
        guard target.format.bytesPerBlock == 4, !target.format.rawValue.hasPrefix("depth"),
              !target.format.rawValue.hasPrefix("stencil") else {
            throw WGPUError.validation(
                "the destination of copyExternalImageToTexture must be a 4-byte color format "
                + "(\(target.format.rawValue)) — upload anything else directly with writeTexture"
            )
        }

        let sourceOrigin = command.source.origin
        // An omitted copy size means **the whole remainder of the image** — that needs the image size, so it is filled here.
        let size = command.copySize
            ?? WGPUExtent3D(width: bitmap.width - sourceOrigin.x, height: bitmap.height - sourceOrigin.y)
        guard size.width > 0, size.height > 0 else { return }   // no-op
        guard sourceOrigin.x + size.width <= bitmap.width,
              sourceOrigin.y + size.height <= bitmap.height else {
            throw WGPUError.validation(
                "the copy region exceeds the image — (\(sourceOrigin.x), \(sourceOrigin.y)) + "
                + "\(size.width)x\(size.height) > \(bitmap.width)x\(bitmap.height)"
            )
        }

        // Spec `GPUCopyExternalImageSourceInfo.flipY` — flips top to bottom **at copy time**.
        // (`createImageBitmap`'s flipY happens at decode time and is separate. Web libraries use this
        // one — three.js's `Texture.flipY` defaults to true, so ignoring it flips textures silently.)
        let flipY = command.source.flipY

        // Copy only what is needed. Using the full width would make bytesPerRow wrong for a partial copy.
        let rowBytes = size.width * 4
        var slice = Data(count: rowBytes * size.height)
        bitmap.pixels.withUnsafeBytes { source in
            slice.withUnsafeMutableBytes { destination in
                guard let sourceBase = source.baseAddress,
                      let destinationBase = destination.baseAddress else { return }
                for row in 0..<size.height {
                    let sourceRow = flipY ? (sourceOrigin.y + size.height - 1 - row)
                                          : (sourceOrigin.y + row)
                    let from = sourceRow * bitmap.bytesPerRow + sourceOrigin.x * 4
                    memcpy(destinationBase + row * rowBytes, sourceBase + from, rowBytes)
                }
            }
        }

        try requireNoOpenPass()
        try backend.writeTexture(
            target.raw, data: slice,
            origin: command.destination.origin,
            size: WGPUExtent3D(width: size.width, height: size.height),
            mipLevel: command.destination.mipLevel,
            bytesPerRow: rowBytes, rowsPerImage: size.height
        )
    }

    /// Block alignment for compressed texture copies (the spec's "validating texel copy range").
    ///
    /// The origin must sit on a block boundary, and the size must be a multiple of the block size or
    /// **reach the end of the mip level** (edge blocks are cut off, hence the exception). Violations
    /// kill the backend with an assertion, so we stop them first. Uncompressed formats have 1×1
    /// blocks, so this check always passes for them.
    private func validateBlockAlignment(
        format: WGPUTextureFormat,
        origin: WGPUOrigin3D,
        size: WGPUExtent3D,
        textureSize: WGPUExtent3D,
        mipLevel: Int,
        label: String
    ) throws {
        guard format.isCompressed else { return }
        let (blockWidth, blockHeight) = format.blockSize
        guard origin.x % blockWidth == 0, origin.y % blockHeight == 0 else {
            throw WGPUError.validation(
                "\(label): the origin of a compressed texture must be on a block boundary "
                + "(\(origin.x), \(origin.y)) / block \(blockWidth)x\(blockHeight)"
            )
        }
        let levelWidth = max(textureSize.width >> mipLevel, 1)
        let levelHeight = max(textureSize.height >> mipLevel, 1)
        guard size.width % blockWidth == 0 || origin.x + size.width == levelWidth,
              size.height % blockHeight == 0 || origin.y + size.height == levelHeight else {
            throw WGPUError.validation(
                "\(label): a compressed copy size must be a multiple of the block size or reach the mip level's end "
                + "(\(size.width)x\(size.height) @ level \(mipLevel), size \(levelWidth)x\(levelHeight))"
            )
        }
    }

    /// Row layout rules for buffer↔texture copies (the spec's "validating GPUTexelCopyBufferInfo" and
    /// "validating linear texture data").
    ///
    /// **`bytesPerRow` must be a multiple of 256.** Metal is far looser (it takes anything whose row
    /// size fits), so leaving it unchecked lets code through that only a browser or Dawn rejects — two
    /// demo scenes really did use 32 here until the Dawn work exposed it (the demo scene table in
    /// `docs/TESTING.md`).
    ///
    /// `queue.writeTexture` carries **no such limit** (the spec treats queue uploads and encoder
    /// copies differently), so this check attaches to the two copy ops only.
    private func validateTexelCopyBufferLayout(
        bytesPerRow: Int?,
        format: WGPUTextureFormat,
        size: WGPUExtent3D,
        label: String,
        path: String?
    ) throws {
        let blockRows = format.blockRows(height: size.height)
        let layers = max(size.depthOrArrayLayers, 1)
        // With several rows the stride cannot be derived — the spec requires it to be stated.
        guard let bytesPerRow else {
            guard blockRows <= 1, layers <= 1 else {
                throw WGPUError.validation(
                    "\(label): bytesPerRow is required when a copy spans several rows or layers "
                        + "(\(blockRows) block rows, \(layers) layer(s))",
                    path: path
                )
            }
            return
        }
        guard bytesPerRow % 256 == 0 else {
            throw WGPUError.validation(
                "\(label): bytesPerRow must be a multiple of 256 (got \(bytesPerRow))",
                path: path
            )
        }
    }

    private func createTextureView(_ command: WGPUCreateTextureViewCommand) throws {
        let source = try texture(command.texture, path: command.fieldPath("texture"))
        let format = command.descriptor.format ?? source.format
        let raw = try backend.makeTextureView(source.raw, descriptor: command.descriptor, format: format)
        registry.insert(
            WGPUEngineTextureView<B>(raw: raw, format: format, sampleCount: source.sampleCount),
            at: command.id
        )
        // Views of a drawable texture expire with the frame (the spec's "Expire the current texture").
        if source.isDrawable { frameScopedHandles.append(command.id) }
    }

    private func createSampler(_ command: WGPUCreateCommand<WGPUSamplerDescriptor>) throws {
        let raw = try backend.makeSampler(command.descriptor)
        registry.insert(WGPUEngineSampler<B>(raw: raw), at: command.id)
    }

    /// In the spec **a shader module is created even when compilation fails** — the error surfaces
    /// through `getCompilationInfo()` and a pipeline creation failure. So we register it even on a
    /// broken parse and carry the diagnostic.
    ///
    /// With no handle at all, every later command breaks with only "does not exist" and **the real
    /// cause (the parse failure) vanishes from view.** Here the cause is reported on the spot too.
    private func createShaderModule(_ command: WGPUCreateCommand<WGPUShaderModuleDescriptor>) throws {
        let creation = try backend.makeShaderModule(command.descriptor, fieldPath: { command.fieldPath($0) })
        registry.insert(WGPUEngineShaderModule<B>(raw: creation.module), at: command.id)
        if let failure = creation.failure {
            throw WGPUError(
                kind: failure.kind, message: failure.message,
                path: failure.path ?? command.fieldPath("code"), line: failure.line
            )
        }
    }

    private func createBindGroupLayout(_ command: WGPUCreateCommand<WGPUBindGroupLayoutDescriptor>) throws {
        let entries = command.descriptor.entries
        let raw = try backend.makeBindGroupLayout(entries)
        registry.insert(WGPUEngineBindGroupLayout<B>(raw: raw, entries: entries), at: command.id)
    }

    private func createPipelineLayout(_ command: WGPUCreateCommand<WGPUPipelineLayoutDescriptor>) throws {
        let groups = try command.descriptor.bindGroupLayouts.map {
            try registry.lookup(
                $0, as: WGPUEngineBindGroupLayout<B>.self, kind: "GPUBindGroupLayout",
                path: command.fieldPath("bindGroupLayouts")
            )
        }
        let raw = try backend.makePipelineLayout(groups.map(\.raw))
        registry.insert(WGPUEnginePipelineLayout<B>(raw: raw), at: command.id)
    }

    private func createBindGroup(_ command: WGPUCreateCommand<WGPUBindGroupDescriptor>) throws {
        let layout = try registry.lookup(
            command.descriptor.layout, as: WGPUEngineBindGroupLayout<B>.self, kind: "GPUBindGroupLayout",
            path: command.fieldPath("layout")
        )
        var buffers: [WGPUEngineBuffer<B>] = []
        let entries: [WGPUResolvedBindGroupEntry<B>] = try command.descriptor.entries.map { entry in
            // Check matching when the layout's entries are known. A natively-validating backend and a
            // derived layout (entries unknown) are left to backend validation — it knows visibility itself.
            let layoutEntry: WGPUBindGroupLayoutEntry?
            if specValidation, layout.entries != nil {
                guard let matched = layout.entry(binding: entry.binding) else {
                    throw WGPUError.validation("the bind group layout has no binding \(entry.binding)")
                }
                layoutEntry = matched
            } else {
                layoutEntry = nil
            }
            let resource: WGPUResolvedBindGroupEntry<B>.Resource
            switch entry.resource {
            case .buffer(let handle, let offset, let size):
                let object = try registry.lookup(handle, as: WGPUEngineBuffer<B>.self, kind: "GPUBuffer")
                buffers.append(object)
                resource = .buffer(
                    object.raw, offset: offset, boundSize: size ?? max(object.size - offset, 0)
                )
            case .sampler(let handle):
                let object = try registry.lookup(handle, as: WGPUEngineSampler<B>.self, kind: "GPUSampler")
                resource = .sampler(object.raw)
            case .textureView(let handle):
                let object = try registry.lookup(
                    handle, as: WGPUEngineTextureView<B>.self, kind: "GPUTextureView"
                )
                resource = .textureView(object.raw)
            }
            return WGPUResolvedBindGroupEntry<B>(
                binding: entry.binding, layoutEntry: layoutEntry, resource: resource
            )
        }
        let raw = try backend.makeBindGroup(layout: layout.raw, entries: entries)
        registry.insert(WGPUEngineBindGroup<B>(raw: raw, buffers: buffers), at: command.id)
    }

    private func createQuerySet(_ command: WGPUCreateCommand<WGPUQuerySetDescriptor>) throws {
        let raw = try backend.makeQuerySet(command.descriptor)
        registry.insert(
            WGPUEngineQuerySet<B>(raw: raw, type: command.descriptor.type, count: command.descriptor.count),
            at: command.id
        )
    }

    /// `bundleEncoder.finish()` — registers the command list JS gathered as a bundle object.
    ///
    /// The bundle encoder itself is not on the wire. JS gathers commands into an array and sends them
    /// down at once in `finish()`, so there is no reason to align the encoder's lifetime on both
    /// sides. A backend with native bundles finishes recording right here; otherwise we store the
    /// readers and repeat them at execution.
    private func createRenderBundle(_ command: WGPUCreateRenderBundleCommand) throws {
        // **Runs regardless of `specValidation`.** "Can this op go in a bundle?" is wire contract, not
        // something the backend knows — a native validator handed an op absent from its bundle encoder
        // simply breaks in its own way, and a replay backend just executes it.
        try WGPUEngineRenderBundle<B>.validateOps(command.commands)
        let native: B.RenderBundle?
        if backend.capabilities.supportsNativeRenderBundles {
            let decoded = try command.commands.map { try WGPUCommand(from: $0) }
            native = try backend.makeRenderBundle(
                command.descriptor, commands: decoded, resolver: bundleResolver()
            )
        } else {
            native = nil
        }
        registry.insert(
            WGPUEngineRenderBundle<B>(
                commands: command.commands, native: native, descriptor: command.descriptor
            ),
            at: command.id
        )
    }

    private func bundleResolver() -> WGPUBundleResolver<B> {
        let registry = self.registry
        return WGPUBundleResolver<B>(
            renderPipeline: { handle, path in
                try registry.lookup(
                    handle, as: WGPUEngineRenderPipeline<B>.self, kind: "GPURenderPipeline", path: path
                ).raw
            },
            bindGroup: { handle, path in
                try registry.lookup(
                    handle, as: WGPUEngineBindGroup<B>.self, kind: "GPUBindGroup", path: path
                ).raw
            },
            buffer: { handle, path in
                let object = try registry.lookup(
                    handle, as: WGPUEngineBuffer<B>.self, kind: "GPUBuffer", path: path
                )
                guard !object.isMapped else {
                    throw WGPUError.validation(
                        "GPUBuffer \(handle) is mapped and cannot be used in queue work "
                            + "(read it with mapAsync, then call unmap())",
                        path: path
                    )
                }
                return object.raw
            }
        )
    }

    private func createRenderPipeline(_ command: WGPUCreateCommand<WGPURenderPipelineDescriptor>) throws {
        let descriptor = command.descriptor
        let vertexModule = try registry.lookup(
            descriptor.vertex.module, as: WGPUEngineShaderModule<B>.self, kind: "GPUShaderModule",
            path: command.fieldPath("vertex.module")
        )
        let fragmentModule = try descriptor.fragment.map {
            try registry.lookup(
                $0.module, as: WGPUEngineShaderModule<B>.self, kind: "GPUShaderModule",
                path: command.fieldPath("fragment.module")
            )
        }
        let layout = try resolvePipelineLayout(descriptor.layout)
        let creation = try backend.makeRenderPipeline(
            descriptor,
            vertexModule: vertexModule.raw,
            fragmentModule: fragmentModule?.raw,
            layout: layout,
            fieldPath: { command.fieldPath($0) }
        )
        registry.insert(
            WGPUEngineRenderPipeline<B>(raw: creation.pipeline, info: creation.info), at: command.id
        )
    }

    private func createComputePipeline(_ command: WGPUCreateCommand<WGPUComputePipelineDescriptor>) throws {
        let descriptor = command.descriptor
        let module = try registry.lookup(
            descriptor.module, as: WGPUEngineShaderModule<B>.self, kind: "GPUShaderModule",
            path: command.fieldPath("compute.module")
        )
        let layout = try resolvePipelineLayout(descriptor.layout)
        let creation = try backend.makeComputePipeline(
            descriptor, module: module.raw, layout: layout, fieldPath: { command.fieldPath($0) }
        )
        registry.insert(
            WGPUEngineComputePipeline<B>(raw: creation.pipeline, info: creation.info), at: command.id
        )
    }

    private func resolvePipelineLayout(_ reference: WGPUPipelineLayoutRef) throws -> WGPUResolvedPipelineLayout<B> {
        switch reference {
        case .auto:
            return .auto
        case .explicit(let handle):
            let layout = try registry.lookup(
                handle, as: WGPUEnginePipelineLayout<B>.self, kind: "GPUPipelineLayout"
            )
            return .explicit(layout.raw)
        }
    }

    /// `pipeline.getBindGroupLayout(index)` — pulls out a layout derived by `layout: "auto"` as a handle.
    private func getBindGroupLayout(_ command: WGPUGetBindGroupLayoutCommand) throws {
        let pipelineHandle = command.pipeline
        let pipeline: WGPUResolvedPipeline<B>
        if let render = try? registry.lookup(
            pipelineHandle, as: WGPUEngineRenderPipeline<B>.self, kind: "x"
        ) {
            pipeline = .render(render.raw)
        } else {
            pipeline = .compute(try registry.lookup(
                pipelineHandle, as: WGPUEngineComputePipeline<B>.self, kind: "GPUPipeline",
                path: command.fieldPath("pipeline")
            ).raw)
        }
        guard let creation = try backend.bindGroupLayout(of: pipeline, index: command.index) else {
            throw WGPUError.validation("the pipeline has no bind group \(command.index)")
        }
        registry.insert(
            WGPUEngineBindGroupLayout<B>(raw: creation.layout, entries: creation.entries), at: command.id
        )
    }

    // MARK: - Canvas (command stream)

    private func configureCanvas(_ configuration: WGPUCanvasConfiguration) throws {
        guard let entry = surfaceEntry(for: configuration.canvasId) else {
            throw WGPUError.validation(
                "canvas '\(configuration.canvasId)' is not registered "
                    + "(check that <webgpu-canvas canvas-id=\"…\"> is attached on screen)"
            )
        }
        try backend.configureSurface(entry.raw, configuration: configuration)
        touchedCanvases[configuration.canvasId] = entry.raw
    }

    private func getCurrentTexture(_ command: WGPUGetCurrentTextureCommand) throws {
        let handle = command.id
        let canvasId = command.canvas
        guard let entry = surfaceEntry(for: canvasId) else {
            throw WGPUError.validation("canvas '\(canvasId)' is not registered")
        }
        // If a drawable obtained in **a previous batch** was never presented and this canvas is asked
        // again, that frame is over — reclaim it here.
        //
        // A batch ending without a submit leaving its drawable behind is intentional: the frame may
        // still be going (see `finish()` — Three.js's lazy pipeline creation). The problem is a frame
        // that will **never be drawn**, which is what happens when a validation error hits before the
        // first encoder. Without reclaiming, one piles up per frame and an on-screen surface drains
        // its drawable pool in three, after which acquisition stalls the JS thread up to a second and
        // then fails forever.
        //
        // **Repeat acquisition within one batch is excluded** — that is the same frame (the spec says
        // to return the same texture), and the view already handed out is still in use that frame.
        if !acquiredThisBatch.contains(canvasId),
           acquiredFrames.contains(where: { $0.canvas == canvasId }) {
            backend.discardAcquiredFrames()
            expireFrame()
        }
        acquiredThisBatch.insert(canvasId)
        touchedCanvases[canvasId] = entry.raw
        guard let acquired = try backend.acquireFrameTexture(entry.raw) else {
            throw WGPUError.validation(
                "could not obtain a drawable for canvas '\(canvasId)' (size is zero, or drawables are exhausted)"
            )
        }
        registry.insert(
            WGPUEngineTexture<B>(
                raw: acquired.texture,
                format: acquired.format,
                size: WGPUExtent3D(width: acquired.width, height: acquired.height),
                sampleCount: acquired.sampleCount,
                isDrawable: true
            ),
            at: handle
        )
        acquiredFrames.append((handle, canvasId))
        frameScopedHandles.append(handle)
    }

    // MARK: - Render pass

    private func beginRenderPass(_ descriptor: WGPURenderPassDescriptor) throws {
        closePass()
        var colorFormats: [WGPUTextureFormat] = []
        var sampleCount = 1
        var colors: [WGPUResolvedRenderPass<B>.ColorAttachment] = []

        for attachment in descriptor.colorAttachments {
            let view = try textureView(attachment.view)
            colorFormats.append(view.format)
            sampleCount = max(sampleCount, view.sampleCount)
            var resolveTarget: B.TextureView?
            if let resolveHandle = attachment.resolveTarget {
                resolveTarget = try textureView(resolveHandle).raw
            }
            colors.append(WGPUResolvedRenderPass<B>.ColorAttachment(
                view: view.raw,
                resolveTarget: resolveTarget,
                loadOp: attachment.loadOp,
                storeOp: attachment.storeOp,
                clearValue: attachment.clearValue
            ))
        }

        var depthStencilFormat: WGPUTextureFormat?
        var depthReadOnly = false
        var stencilReadOnly = false
        var depthStencil: WGPUResolvedRenderPass<B>.DepthStencilAttachment?
        if let depth = descriptor.depthStencilAttachment {
            let view = try textureView(depth.view)
            depthStencilFormat = view.format
            // The depth view feeds the pass layout's sampleCount too — omit it and an MSAA pass with no
            // color attachment (a shadow map or depth prepass) rejects a correctly declared bundle.
            sampleCount = max(sampleCount, view.sampleCount)
            depthStencil = WGPUResolvedRenderPass<B>.DepthStencilAttachment(
                view: view.raw,
                format: view.format,
                depthLoadOp: depth.depthLoadOp,
                depthStoreOp: depth.depthStoreOp,
                depthClearValue: depth.depthClearValue,
                stencilLoadOp: depth.stencilLoadOp,
                stencilStoreOp: depth.stencilStoreOp,
                stencilClearValue: depth.stencilClearValue,
                depthReadOnly: depth.depthReadOnly,
                stencilReadOnly: depth.stencilReadOnly
            )
            depthReadOnly = depth.depthReadOnly
            stencilReadOnly = depth.stencilReadOnly
        }

        // An occlusion query set can only be attached **when opening the pass** (WebGPU and the backends agree).
        var occlusionQuerySet: WGPUEngineQuerySet<B>?
        if let handle = descriptor.occlusionQuerySet {
            let querySet = try registry.lookup(
                handle, as: WGPUEngineQuerySet<B>.self, kind: "GPUQuerySet"
            )
            if specValidation {
                guard querySet.type == .occlusion else {
                    throw WGPUError.validation(
                        "occlusionQuerySet must be type: \"occlusion\" (got: \(querySet.type.rawValue))"
                    )
                }
            }
            occlusionQuerySet = querySet
        }

        var timestampWrites: WGPUResolvedTimestampWrites<B>?
        if let writes = descriptor.timestampWrites {
            timestampWrites = try resolveTimestampWrites(writes)
        }

        try backend.beginRenderPass(WGPUResolvedRenderPass<B>(
            label: descriptor.label,
            colorAttachments: colors,
            depthStencil: depthStencil,
            occlusionQuerySet: occlusionQuerySet?.raw,
            timestampWrites: timestampWrites
        ))
        passState = .render
        passFormats = (colorFormats, depthStencilFormat, sampleCount)
        passOcclusionQuerySet = occlusionQuerySet
        passDepthReadOnly = depthReadOnly
        passStencilReadOnly = stencilReadOnly
        openOcclusionQuery = nil
        usedOcclusionQueries.removeAll()
        resetPassBindings()
    }

    /// Checks where timestamps are written.
    private func resolveTimestampWrites(_ writes: WGPUPassTimestampWrites) throws -> WGPUResolvedTimestampWrites<B> {
        let querySet = try registry.lookup(writes.querySet, as: WGPUEngineQuerySet<B>.self, kind: "GPUQuerySet")
        if specValidation {
            guard querySet.type == .timestamp else {
                throw WGPUError.validation(
                    "the query set for timestampWrites must be type: \"timestamp\" (got: \(querySet.type.rawValue))"
                )
            }
            // Omitting both makes it a **silent no-op pass**. The query set's initial value (0) resolves
            // without error, so the app reads a GPU time of 0ns.
            guard writes.beginningOfPassWriteIndex != nil || writes.endOfPassWriteIndex != nil else {
                throw WGPUError.validation(
                    "timestampWrites must supply at least one of beginningOfPassWriteIndex "
                        + "and endOfPassWriteIndex",
                    path: "timestampWrites"
                )
            }
            // Pointing both at the same slot lets the later sample overwrite the earlier, so the delta loses meaning.
            if let begin = writes.beginningOfPassWriteIndex, begin == writes.endOfPassWriteIndex {
                throw WGPUError.validation(
                    "the two timestampWrites indices must differ (both are \(begin))",
                    path: "timestampWrites"
                )
            }
            for index in [writes.beginningOfPassWriteIndex, writes.endOfPassWriteIndex].compactMap({ $0 }) {
                try querySet.checkRange(first: index, count: 1, path: "timestampWrites")
            }
        }
        return WGPUResolvedTimestampWrites<B>(
            querySet: querySet.raw,
            beginningOfPassWriteIndex: writes.beginningOfPassWriteIndex,
            endOfPassWriteIndex: writes.endOfPassWriteIndex
        )
    }

    private func beginComputePass(_ descriptor: WGPUComputePassDescriptor) throws {
        closePass()
        var timestampWrites: WGPUResolvedTimestampWrites<B>?
        if let writes = descriptor.timestampWrites {
            timestampWrites = try resolveTimestampWrites(writes)
        }
        try backend.beginComputePass(WGPUResolvedComputePass<B>(
            label: descriptor.label, timestampWrites: timestampWrites
        ))
        passState = .compute
        currentComputePipeline = nil
        boundGroups.removeAll()
        dirtyGroups.removeAll()
    }

    // MARK: - Pipeline and binding state

    private func setPipeline(_ command: WGPUSetPipelineCommand) throws {
        let handle = command.pipeline
        switch passState {
        case .render:
            let pipeline = try registry.lookup(
                handle, as: WGPUEngineRenderPipeline<B>.self, kind: "GPURenderPipeline",
                path: command.fieldPath("pipeline")
            )
            // A pipeline that writes an attachment declared read-only is stopped here — let the backend
            // just write and the depth buffer marked read-only is genuinely modified. A backend that
            // supplies no metadata (nil) validates this itself.
            guard !passDepthReadOnly || !(pipeline.info.writesDepth ?? false) else {
                throw WGPUError.validation(
                    "a depthReadOnly pass cannot use a pipeline with depthWriteEnabled: true"
                )
            }
            guard !passStencilReadOnly || !(pipeline.info.writesStencil ?? false) else {
                throw WGPUError.validation(
                    "a stencilReadOnly pass cannot use a pipeline that writes stencil "
                        + "(failOp, depthFailOp and passOp must all be \"keep\")"
                )
            }
            backend.setRenderPipeline(pipeline.raw)
            currentRenderPipeline = pipeline
        case .compute:
            let pipeline = try registry.lookup(
                handle, as: WGPUEngineComputePipeline<B>.self, kind: "GPUComputePipeline",
                path: command.fieldPath("pipeline")
            )
            backend.setComputePipeline(pipeline.raw)
            currentComputePipeline = pipeline
        case nil:
            throw WGPUError.validation("setPipeline can only be used inside a pass")
        }
        // A pipeline change can change the layout, so bind groups are applied again.
        dirtyGroups = Set(boundGroups.keys)
    }

    private func setBindGroup(_ command: WGPUSetBindGroupCommand) throws {
        let group = try registry.lookup(
            command.bindGroup, as: WGPUEngineBindGroup<B>.self, kind: "GPUBindGroup",
            path: command.fieldPath("bindGroup")
        )
        boundGroups[command.index] = (group, command.dynamicOffsets)
        dirtyGroups.insert(command.index)
    }

    /// Returns pipeline, bind group and vertex/index buffer bindings to "unspecified".
    ///
    /// Used when opening a pass and on both sides of `executeBundles`. The spec states that executing
    /// a bundle **invalidates the previous state rather than restoring it** — a bundle does not
    /// inherit pass state, and once it finishes the pass does not inherit what the bundle left. So
    /// both sides are reset. (Viewport, scissor, blend constant and stencil reference are not on this
    /// list — they persist.)
    private func resetPassBindings() {
        currentRenderPipeline = nil
        boundGroups.removeAll()
        dirtyGroups.removeAll()
        indexBinding = nil
        vertexBindings.removeAll()
        dirtyVertexSlots.removeAll()
    }

    /// Right before a draw or dispatch, verifies every piece of state the pipeline requires and sends it down.
    ///
    /// Bind groups and vertex buffers are handled in one place because both are **state invalidated at
    /// bundle boundaries**, so they must be checked at the same moment. Calling this one function when
    /// adding a new draw op brings the isolation contract along automatically.
    ///
    /// Each draw op raises its own pipeline guard, with its name in the message, **before** this
    /// function — the identical check below is a safety net for an op that forgot that guard, which is
    /// why its message is generic.
    private func applyDrawState() throws {
        let requiredGroups: Set<Int>?
        if passState == .render {
            guard let pipeline = currentRenderPipeline else {
                throw WGPUError.validation("setPipeline is required before draw")
            }
            requiredGroups = pipeline.info.requiredGroups
        } else {
            guard let pipeline = currentComputePipeline else {
                throw WGPUError.validation("setPipeline is required before dispatch")
            }
            requiredGroups = pipeline.info.requiredGroups
        }

        // Every group the layout requires must be bound. Without this check a draw silently uses
        // bindings a bundle left behind (or ones the pass set up earlier) — an encoder has no
        // "unbind", so `resetPassBindings()` alone does not actually isolate.
        // (A backend that supplies no metadata validates this itself.)
        if let requiredGroups {
            for groupIndex in requiredGroups.sorted() where boundGroups[groupIndex] == nil {
                throw WGPUError.validation(
                    "@group(\(groupIndex)) required by the pipeline layout is not bound "
                        + "(bindings are invalidated around bundle execution — call setBindGroup again)"
                )
            }
        }

        // If a buffer held by a bind group is mapped, this draw is queue work too, so it is rejected.
        // (A group pins its buffers at creation, so a buffer mapped afterwards is caught here.)
        for (_, bound) in boundGroups {
            for buffer in bound.group.buffers where buffer.isMapped {
                throw WGPUError.validation(
                    "cannot draw with a bind group holding a mapped buffer (call unmap() first)"
                )
            }
        }

        for groupIndex in dirtyGroups.sorted() {
            guard let bound = boundGroups[groupIndex] else { continue }
            try backend.applyBindGroup(bound.group.raw, at: groupIndex, dynamicOffsets: bound.offsets)
        }
        dirtyGroups.removeAll()

        try applyVertexBuffers()
    }

    /// Right before a draw, checks the vertex buffers the pipeline requires are present and sends them down.
    ///
    /// The spec states that "if `vertex.buffers[slot]` is not null, `[[vertex_buffers]]` must contain
    /// that slot". Without this check a bundle draws with vertex buffers the pass set up earlier, and
    /// the pass draws with what the bundle set — both invalid code in a browser.
    private func applyVertexBuffers() throws {
        guard passState == .render, let pipeline = currentRenderPipeline else { return }
        if let required = pipeline.info.requiredVertexSlots {
            for slot in required.sorted() where vertexBindings[slot] == nil {
                throw WGPUError.validation(
                    "vertex buffer slot \(slot) required by the pipeline is not bound "
                        + "(bindings are invalidated around bundle execution — call setVertexBuffer again)"
                )
            }
        }
        for slot in dirtyVertexSlots.sorted() {
            guard let binding = vertexBindings[slot] else { continue }
            try backend.applyVertexBuffer(binding.buffer.raw, offset: binding.offset, slot: slot)
        }
        dirtyVertexSlots.removeAll()
    }

    private func setVertexBuffer(_ command: WGPUSetVertexBufferCommand) throws {
        try requireRenderPass()
        let slot = command.slot
        let buffer = try unmappedBuffer(command.buffer, path: command.fieldPath("buffer"))
        let offset = command.offset
        if specValidation {
            let maxSlots = backend.capabilities.maxVertexBufferSlots
            guard slot >= 0, slot < maxSlots else {
                throw WGPUError.validation("vertex buffer slots range from 0 to \(maxSlots - 1)")
            }
            guard offset >= 0, offset <= buffer.size else {
                throw WGPUError.validation(
                    "vertex buffer offset (\(offset)) is outside the buffer size (\(buffer.size)B)",
                    path: command.fieldPath("offset")
                )
            }
        }
        // Not sent to the backend immediately — sending right before the draw is what makes bundle-boundary invalidation hold.
        vertexBindings[slot] = (buffer, offset)
        dirtyVertexSlots.insert(slot)
    }

    private func setIndexBuffer(_ command: WGPUSetIndexBufferCommand) throws {
        try requireRenderPass()
        let buffer = try unmappedBuffer(command.buffer, path: command.fieldPath("buffer"))
        indexBinding = (buffer, command.offset, command.format, command.indexStride)
    }

    private func resolvedIndexBinding(
        _ binding: (buffer: WGPUEngineBuffer<B>, offset: Int, format: WGPUIndexFormat, stride: Int)
    ) -> WGPUResolvedIndexBinding<B> {
        WGPUResolvedIndexBinding<B>(
            buffer: binding.buffer.raw, offset: binding.offset,
            format: binding.format, stride: binding.stride
        )
    }

    // MARK: - Draw / dispatch

    private func draw(_ command: WGPUDrawCommand) throws {
        try requireRenderPass()
        // The pipeline guard sits **before** `applyDrawState()` — if the identical check inside it
        // threw first, this op-named message would be dead code that never ships (same for the draw family below).
        guard currentRenderPipeline != nil else {
            throw WGPUError.validation("setPipeline is required before draw")
        }
        try applyDrawState()
        try backend.draw(command)
    }

    private func drawIndexed(_ command: WGPUDrawIndexedCommand) throws {
        try requireRenderPass()
        guard currentRenderPipeline != nil else {
            throw WGPUError.validation("setPipeline is required before drawIndexed")
        }
        guard let indexBinding else {
            throw WGPUError.validation("setIndexBuffer is required before drawIndexed")
        }
        try applyDrawState()
        try backend.drawIndexed(command, index: resolvedIndexBinding(indexBinding))
    }

    /// Finds the indirect argument buffer and validates the offset.
    ///
    /// - 4-byte alignment and range are caught here because **the backend may handle them with an
    ///   assertion (process death)**.
    /// - `INDIRECT` usage is not checked by the backend, which may have no corresponding concept.
    ///   Unchecked, code that runs here but breaks only in a browser ships.
    /// - Device capability is filtered first by the backend with its own context (`ensureIndirectSupported`).
    private func indirectArguments(
        _ command: WGPUIndirectCommand,
        argumentSize: Int
    ) throws -> (buffer: B.Buffer, offset: Int) {
        try backend.ensureIndirectSupported()
        let object = try unmappedBuffer(command.indirectBuffer, path: command.fieldPath("indirectBuffer"))
        let offset = command.indirectOffset
        if specValidation {
            guard offset >= 0, offset % 4 == 0 else {
                throw WGPUError.validation(
                    "indirectOffset must be a multiple of 4 (got \(offset))",
                    path: command.fieldPath("indirectOffset")
                )
            }
            guard offset + argumentSize <= object.size else {
                throw WGPUError.validation(
                    "\(argumentSize)B of indirect arguments exceed the buffer — "
                        + "offset \(offset) + \(argumentSize)B > buffer size \(object.size)B",
                    path: command.fieldPath("indirectOffset")
                )
            }
            guard object.usage.contains(.indirect) else {
                throw WGPUError.validation(
                    "the argument buffer of an indirect draw/dispatch must be created with GPUBufferUsage.INDIRECT",
                    path: command.fieldPath("indirectBuffer")
                )
            }
        }
        return (object.raw, offset)
    }

    private func drawIndirect(_ command: WGPUIndirectCommand) throws {
        try requireRenderPass()
        // Argument validation runs **before** `applyDrawState()` — a command that will be rejected must
        // not have already changed encoder state (all the more so because errors accumulate without
        // killing the frame). vertexCount, instanceCount, firstVertex, firstInstance — four u32s.
        let arguments = try indirectArguments(command, argumentSize: 16)
        guard currentRenderPipeline != nil else {
            throw WGPUError.validation("setPipeline is required before drawIndirect")
        }
        try applyDrawState()
        try backend.drawIndirect(buffer: arguments.buffer, offset: arguments.offset)
    }

    private func drawIndexedIndirect(_ command: WGPUIndirectCommand) throws {
        try requireRenderPass()
        // indexCount, instanceCount, firstIndex, baseVertex(i32), firstInstance — five slots.
        let arguments = try indirectArguments(command, argumentSize: 20)
        guard currentRenderPipeline != nil else {
            throw WGPUError.validation("setPipeline is required before drawIndexedIndirect")
        }
        guard let indexBinding else {
            throw WGPUError.validation("setIndexBuffer is required before drawIndexedIndirect")
        }
        try applyDrawState()
        try backend.drawIndexedIndirect(
            buffer: arguments.buffer, offset: arguments.offset,
            index: resolvedIndexBinding(indexBinding)
        )
    }

    private func dispatchWorkgroups(_ command: WGPUDispatchWorkgroupsCommand) throws {
        try requireComputePass()
        guard currentComputePipeline != nil else {
            throw WGPUError.validation("setPipeline is required before dispatchWorkgroups")
        }
        try applyDrawState()
        try backend.dispatchWorkgroups(command)
    }

    private func dispatchWorkgroupsIndirect(_ command: WGPUIndirectCommand) throws {
        try requireComputePass()
        // x, y, z — three u32s.
        let arguments = try indirectArguments(command, argumentSize: 12)
        guard currentComputePipeline != nil else {
            throw WGPUError.validation("setPipeline is required before dispatchWorkgroupsIndirect")
        }
        try applyDrawState()
        try backend.dispatchWorkgroupsIndirect(buffer: arguments.buffer, offset: arguments.offset)
    }

    // MARK: - Occlusion queries

    private func beginOcclusionQuery(_ command: WGPUBeginOcclusionQueryCommand) throws {
        try requireRenderPass()
        let index = command.queryIndex
        if specValidation {
            guard let querySet = passOcclusionQuerySet else {
                throw WGPUError.validation(
                    "to use beginOcclusionQuery, beginRenderPass must be given an occlusionQuerySet"
                )
            }
            guard openOcclusionQuery == nil else {
                throw WGPUError.validation("occlusion queries cannot nest (close the previous one with endOcclusionQuery)")
            }
            try querySet.checkRange(first: index, count: 1, path: command.fieldPath("queryIndex"))
            // Using the same index twice in one pass makes two regions share one 8-byte slot — the final
            // value then depends on the backend's accumulate/overwrite behaviour and diverges from a browser.
            guard usedOcclusionQueries.insert(index).inserted else {
                throw WGPUError.validation(
                    "occlusion query index \(index) was already used in this pass",
                    path: command.fieldPath("queryIndex")
                )
            }
        } else {
            usedOcclusionQueries.insert(index)
        }
        try backend.beginOcclusionQuery(index: index)
        openOcclusionQuery = index
    }

    private func endOcclusionQuery() throws {
        try requireRenderPass()
        guard let index = openOcclusionQuery else {
            throw WGPUError.validation("endOcclusionQuery: no occlusion query is open")
        }
        try backend.endOcclusionQuery(index: index)
        openOcclusionQuery = nil
    }

    // MARK: - Render bundle execution

    private func executeBundles(_ command: WGPUExecuteBundlesCommand) throws {
        try requireRenderPass()
        guard let formats = passFormats else {
            throw WGPUError.validation("executeBundles can only be used inside a render pass")
        }
        let bundles = try command.bundles.map {
            try registry.lookup(
                $0, as: WGPUEngineRenderBundle<B>.self, kind: "GPURenderBundle",
                path: command.fieldPath("bundles")
            )
        }

        if specValidation {
            for bundle in bundles {
                try bundle.checkCompatibility(
                    color: formats.color,
                    depthStencil: formats.depthStencil,
                    sampleCount: formats.sampleCount,
                    depthReadOnly: passDepthReadOnly,
                    stencilReadOnly: passStencilReadOnly
                )
            }
        }
        // The spec's "Reset the render pass binding state" (step 4) runs **unconditionally** once
        // compatibility validation passes — including when a bundle command throws. Omit it here and the
        // pass commands that follow draw holding the pipeline and bind groups the bundle left, shipping wrong pixels.
        defer { resetPassBindings() }

        if backend.capabilities.supportsNativeRenderBundles {
            let natives = try bundles.map { bundle -> B.RenderBundle in
                guard let native = bundle.native else {
                    throw WGPUError.backend("no native render bundle (a handle reused across a backend change)")
                }
                return native
            }
            try backend.executeBundles(natives)
        } else {
            // If even one does not match, nothing runs — we do not leave a half-drawn frame behind.
            for bundle in bundles {
                resetPassBindings()
                for bundleCommand in bundle.commands {
                    try perform(bundleCommand)
                }
            }
        }
    }

    // MARK: - Copies

    private func copyBufferToBuffer(_ command: WGPUCopyBufferToBufferCommand) throws {
        let source = try unmappedBuffer(command.source, path: command.fieldPath("source"))
        let destination = try unmappedBuffer(command.destination, path: command.fieldPath("destination"))
        let sourceOffset = command.sourceOffset
        let destinationOffset = command.destinationOffset
        // The spec's short form `copyBufferToBuffer(src, dst)` means "the rest of the source".
        // The JS shim fills it in, but the same default is given to anyone building the command stream
        // directly (native-only use) — matching the rule of `clearBuffer`.
        let size = command.size ?? max(0, source.size - sourceOffset)
        if specValidation {
            // A copy past the end **kills Metal with an assertion.** Turn it into a validation error here.
            guard sourceOffset >= 0, destinationOffset >= 0, size >= 0 else {
                throw WGPUError.validation(
                    "copyBufferToBuffer offsets and size cannot be negative "
                    + "(sourceOffset \(sourceOffset), destinationOffset \(destinationOffset), size \(size))"
                )
            }
            guard sourceOffset + size <= source.size else {
                throw WGPUError.validation(
                    "copyBufferToBuffer source range exceeds the buffer — "
                    + "\(sourceOffset) + \(size)B > size \(source.size)B",
                    path: command.fieldPath("size")
                )
            }
            guard destinationOffset + size <= destination.size else {
                throw WGPUError.validation(
                    "copyBufferToBuffer destination range exceeds the buffer — "
                    + "\(destinationOffset) + \(size)B > size \(destination.size)B",
                    path: command.fieldPath("size")
                )
            }
            guard source.usage.contains(.copySrc) else {
                throw WGPUError.validation(
                    "the source of copyBufferToBuffer must be created with GPUBufferUsage.COPY_SRC",
                    path: command.fieldPath("source")
                )
            }
            guard destination.usage.contains(.copyDst) else {
                throw WGPUError.validation(
                    "the destination of copyBufferToBuffer must be created with GPUBufferUsage.COPY_DST",
                    path: command.fieldPath("destination")
                )
            }
            // A spec rule — Metal's blit copies at byte granularity too, so leaving it unchecked ships
            // code that only a browser rejects.
            guard sourceOffset % 4 == 0, destinationOffset % 4 == 0, size % 4 == 0 else {
                throw WGPUError.validation(
                    "copyBufferToBuffer offsets and size must be multiples of 4 "
                    + "(got \(sourceOffset), \(destinationOffset), \(size))",
                    path: command.fieldPath("size")
                )
            }
        }
        if size == 0 { return }   // a zero-byte copy is a no-op
        // Negatives are not swallowed quietly — the Metal path never reaches here because the check
        // above throws first, and the natively-validating path rejects with the same wording (moving a
        // negative into a GPU argument width is the trap, so this rejection is minimal exception
        // handling regardless of who validates).
        guard sourceOffset >= 0, destinationOffset >= 0, size > 0 else {
            throw WGPUError.validation(
                "copyBufferToBuffer offsets and size cannot be negative "
                + "(sourceOffset \(sourceOffset), destinationOffset \(destinationOffset), size \(size))"
            )
        }
        try requireNoOpenPass()
        try backend.copyBufferToBuffer(
            source: source.raw, sourceOffset: sourceOffset,
            destination: destination.raw, destinationOffset: destinationOffset, size: size
        )
    }

    /// `clearBuffer` — fills a range of a buffer with zeros.
    ///
    /// The result matches pushing zeros through `writeBuffer`, but **it never builds a zero array on
    /// the CPU and ships it across the bridge.** That difference is large on a compute path clearing a
    /// big storage buffer every frame.
    private func clearBuffer(_ command: WGPUClearBufferCommand) throws {
        let object = try unmappedBuffer(command.buffer, path: command.fieldPath("buffer"))
        let offset = command.offset
        // Spec: omitting size means through the end of the buffer.
        let size = command.size ?? max(0, object.size - offset)

        if specValidation {
            guard object.usage.contains(.copyDst) else {
                throw WGPUError.validation(
                    "the destination of clearBuffer must be created with GPUBufferUsage.COPY_DST",
                    path: command.fieldPath("buffer")
                )
            }
            // The multiple-of-4 requirement is a spec rule. Metal fills at byte granularity too, so
            // leaving it unchecked ships code that only a browser rejects.
            guard offset % 4 == 0, size % 4 == 0 else {
                throw WGPUError.validation(
                    "clearBuffer's offset and size must be multiples of 4 (got \(offset), \(size))",
                    path: command.fieldPath("offset")
                )
            }
            guard offset >= 0, size >= 0, offset + size <= object.size else {
                throw WGPUError.validation(
                    "clearBuffer range exceeds the buffer — offset \(offset) + \(size)B > size \(object.size)B",
                    path: command.fieldPath("size")
                )
            }
        }
        if size == 0 { return }   // no-op
        // Building the Range itself traps on a negative — trap prevention is ours regardless of who
        // validates (the Metal path never reaches here because the check above throws first).
        guard offset >= 0, size > 0 else {
            throw WGPUError.validation(
                "clearBuffer's offset and size cannot be negative (got \(offset), \(size))"
            )
        }
        try requireNoOpenPass()
        try backend.clearBuffer(object.raw, range: offset..<(offset + size))
    }

    private func copyTextureToBuffer(_ command: WGPUCopyTextureToBufferCommand) throws {
        let source = command.source
        let destination = command.destination
        let texture = try self.texture(source.texture, path: source.fieldPath("texture"))
        let buffer = try unmappedBuffer(destination.buffer, path: destination.fieldPath("buffer"))
        let size = command.copySize
        let format = texture.format
        let bytesPerRow = destination.bytesPerRow ?? format.bytesPerRow(width: size.width)
        if specValidation {
            try validateBlockAlignment(format: format, origin: source.origin, size: size,
                                       textureSize: texture.size, mipLevel: source.mipLevel,
                                       label: "copyTextureToBuffer")
            try validateTexelCopyBufferLayout(
                bytesPerRow: destination.bytesPerRow, format: format, size: size,
                label: "copyTextureToBuffer", path: destination.fieldPath("bytesPerRow")
            )
        }
        try requireNoOpenPass()
        try backend.copyTextureToBuffer(
            texture: texture.raw, slice: source.origin.z, mipLevel: source.mipLevel,
            origin: source.origin, size: size,
            buffer: buffer.raw, offset: destination.offset,
            bytesPerRow: bytesPerRow,
            // Only one slice is copied, so `rowsPerImage` goes unused
            // (a known difference in `docs/COMMAND-STREAM.md`).
            bytesPerImage: bytesPerRow * format.blockRows(height: size.height)
        )
    }

    private func copyBufferToTexture(_ command: WGPUCopyBufferToTextureCommand) throws {
        let source = command.source
        let destination = command.destination
        let buffer = try unmappedBuffer(source.buffer, path: source.fieldPath("buffer"))
        let texture = try self.texture(destination.texture, path: destination.fieldPath("texture"))
        let size = command.copySize
        let format = texture.format
        let bytesPerRow = source.bytesPerRow ?? format.bytesPerRow(width: size.width)
        if specValidation {
            try validateBlockAlignment(format: format, origin: destination.origin, size: size,
                                       textureSize: texture.size, mipLevel: destination.mipLevel,
                                       label: "copyBufferToTexture")
            try validateTexelCopyBufferLayout(
                bytesPerRow: source.bytesPerRow, format: format, size: size,
                label: "copyBufferToTexture", path: source.fieldPath("bytesPerRow")
            )
        }
        try requireNoOpenPass()
        try backend.copyBufferToTexture(
            buffer: buffer.raw, offset: source.offset,
            bytesPerRow: bytesPerRow,
            bytesPerImage: bytesPerRow * format.blockRows(height: size.height),
            texture: texture.raw, slice: destination.origin.z, mipLevel: destination.mipLevel,
            origin: destination.origin, size: size
        )
    }

    private func copyTextureToTexture(_ command: WGPUCopyTextureToTextureCommand) throws {
        let source = try texture(command.source.texture, path: command.source.fieldPath("texture"))
        let destination = try texture(
            command.destination.texture, path: command.destination.fieldPath("texture")
        )
        try requireNoOpenPass()
        try backend.copyTextureToTexture(
            source: source.raw,
            sourceSlice: command.source.origin.z,
            sourceMipLevel: command.source.mipLevel,
            sourceOrigin: command.source.origin,
            destination: destination.raw,
            destinationSlice: command.destination.origin.z,
            destinationMipLevel: command.destination.mipLevel,
            destinationOrigin: command.destination.origin,
            size: command.copySize
        )
    }

    /// Resolves query results into a buffer.
    private func resolveQuerySet(_ command: WGPUResolveQuerySetCommand) throws {
        let querySet = try registry.lookup(
            command.querySet, as: WGPUEngineQuerySet<B>.self, kind: "GPUQuerySet",
            path: command.fieldPath("querySet")
        )
        let destination = try unmappedBuffer(command.destination, path: command.fieldPath("destination"))
        let first = command.firstQuery
        // Omitted means the rest of the query set — that needs its size, so it is filled here.
        let count = command.queryCount ?? (querySet.count - first)
        let offset = command.destinationOffset
        if specValidation {
            try querySet.checkRange(first: first, count: count, path: command.fieldPath("firstQuery"))

            // Alignment the spec requires. Metal is looser, so leaving it unchecked breaks only in a browser.
            guard offset >= 0, offset % 256 == 0 else {
                throw WGPUError.validation(
                    "destinationOffset must be a multiple of 256 (got \(offset))",
                    path: command.fieldPath("destinationOffset")
                )
            }
            let byteCount = count * WGPUEngineQuerySet<B>.resultStride
            guard offset + byteCount <= destination.size else {
                throw WGPUError.validation(
                    "\(byteCount)B of query results exceed the buffer — "
                        + "offset \(offset) + \(byteCount)B > buffer size \(destination.size)B",
                    path: command.fieldPath("destinationOffset")
                )
            }
            guard destination.usage.contains(.queryResolve) else {
                throw WGPUError.validation(
                    "the destination buffer of resolveQuerySet must be created with GPUBufferUsage.QUERY_RESOLVE",
                    path: command.fieldPath("destination")
                )
            }
        }
        if count == 0 { return }   // no-op (the backend rejects a zero-byte copy)
        // A negative count passes through — the Metal path already threw in checkRange above, and the
        // natively-validating path filters it as validation in the backend's safe conversion (dawnU32).
        try requireNoOpenPass()
        try backend.resolveQuerySet(
            querySet.raw, first: first, count: count,
            destination: destination.raw, destinationOffset: offset
        )
    }

    // MARK: - Debug markers

    private func pushDebugGroup(_ command: WGPUPushDebugGroupCommand) throws {
        if passState != nil {
            try backend.pushDebugGroup(command.groupLabel, scope: .pass)
            encoderDebugDepth += 1
        } else {
            try backend.pushDebugGroup(command.groupLabel, scope: .frame)
            bufferDebugDepth += 1
        }
    }

    /// An unmatched `pop` **kills the process with a Metal assertion.** So we count the depth and stop
    /// it here — the spec also defines this case as an error, so behaviour matches and the app
    /// survives. A natively-validating backend raises an error instead of dying, so we pass it through.
    private func popDebugGroup() {
        if passState != nil {
            guard specValidation else {
                encoderDebugDepth = max(0, encoderDebugDepth - 1)
                backend.popDebugGroup(scope: .pass)
                return
            }
            guard encoderDebugDepth > 0 else {
                record(.validation("popDebugGroup: no matching pushDebugGroup (inside a pass)"))
                return
            }
            encoderDebugDepth -= 1
            backend.popDebugGroup(scope: .pass)
        } else {
            guard specValidation else {
                bufferDebugDepth = max(0, bufferDebugDepth - 1)
                // Call the backend only when a command buffer exists — otherwise there is nothing to attach to.
                if backend.hasPendingWork { backend.popDebugGroup(scope: .frame) }
                return
            }
            // **Reporting does not depend on `hasPendingWork`.** Previously, with no command buffer this
            // branch passed over entirely, so a function whose job is catching unmatched pops missed the
            // most common mistake of all: popping before any work has been done.
            guard bufferDebugDepth > 0 else {
                record(.validation("popDebugGroup: no matching pushDebugGroup"))
                return
            }
            bufferDebugDepth -= 1
            if backend.hasPendingWork { backend.popDebugGroup(scope: .frame) }
        }
    }

    // MARK: - WebGPURuntime: queries and async

    public func adapterInfo() -> [String: Any] {
        executionLock.lock()
        defer { executionLock.unlock() }
        return backend.adapterInfo()
    }

    /// `GPUShaderModule.getCompilationInfo()` — that module's compilation diagnostics.
    public func shaderCompilationInfo(handle: Int) -> [String: Any] {
        executionLock.lock()
        defer { executionLock.unlock() }

        guard let module = try? registry.lookup(
            WGPUHandle(handle), as: WGPUEngineShaderModule<B>.self, kind: "GPUShaderModule"
        ) else {
            return ["ok": false, "errors": [
                WGPUError.validation("GPUShaderModule #\(handle) does not exist").payload,
            ]]
        }
        let messages = backend.compilationMessages(of: module.raw).map { message -> [String: Any] in
            [
                "message": message.message,
                "type": message.type,
                "lineNum": message.lineNum,
                "linePos": message.linePos,
                "offset": message.offset,
                "length": message.length,
            ]
        }
        return ["ok": true, "messages": messages]
    }

    /// Reads buffer contents (corresponding to `GPUBuffer.mapAsync` + `getMappedRange`).
    ///
    /// While reading, this buffer is "unavailable" — if a write from the next frame overlaps the same
    /// memory, which frame's values JS receives is not guaranteed (`WGPUEngineBuffer.isMapped`).
    public func readBuffer(
        handle: Int,
        offset: Int = 0,
        size: Int? = nil,
        completion: @escaping ([String: Any]) -> Void
    ) {
        let target: WGPUEngineBuffer<B>
        do {
            target = try registry.lookup(WGPUHandle(handle), as: WGPUEngineBuffer<B>.self, kind: "GPUBuffer")
        } catch let error as WGPUError {
            completion(["ok": false, "errors": [error.payload]])
            return
        } catch {
            completion(["ok": false, "errors": [WGPUError.backend("\(error)").payload]])
            return
        }

        executionLock.lock()
        guard !target.isMapped else {
            executionLock.unlock()
            completion([
                "ok": false,
                "errors": [WGPUError.validation(
                    "GPUBuffer \(WGPUHandle(handle)) is already mapped (call unmap() first)"
                ).payload],
            ])
            return
        }
        target.isMapped = true

        let length = size ?? (target.size - offset)
        guard offset >= 0, length >= 0, offset + length <= target.size else {
            // On failure we do not set up the mapping — a failed mapAsync leaves the buffer unmapped in the spec too.
            target.isMapped = false
            executionLock.unlock()
            completion([
                "ok": false,
                "errors": [WGPUError.validation(
                    "readBuffer out of range — offset \(offset) + \(length)B > buffer size \(target.size)B"
                ).payload],
            ])
            return
        }

        // Waiting on previously submitted GPU work is the backend's job. Completion arrives on an
        // arbitrary thread, so the wrapper retakes the lock — safe even when it arrives synchronously
        // during registration, because the lock is recursive.
        noteReadbackStarted()
        backend.readBuffer(target.raw, offset: offset, length: length) { [weak self] result in
            guard let self else { return }
            self.executionLock.lock()
            defer { self.executionLock.unlock() }
            self.noteReadbackFinished()
            switch result {
            case .failure(let error):
                target.isMapped = false
                completion(["ok": false, "errors": [error.payload]])
            case .success(let data):
                // Carry `Data` straight through — Lynx turns `NSData` into a JS `ArrayBuffer`.
                // base64 would add 33% inflation plus a decoding loop on the JS side.
                completion(["ok": true, "data": data, "byteLength": data.count])
            }
        }
        executionLock.unlock()
    }

    // MARK: - Readback self-pump

    /// Outstanding readback count — read and written only under `executionLock`.
    private var pendingReadbacks = 0
    private var readbackPumpRunning = false

    /// Spin cap for the self-pump (1ms per turn × this count = about 30 seconds).
    ///
    /// A safeguard so the detached thread does not spin forever when a completion **never arrives**
    /// (device loss, or a backend dropping the callback). Without a cap that thread burns CPU and
    /// takes `executionLock` every turn, slowing the JS thread too — the cause is elsewhere but the
    /// symptom spreads. (A computed property because a generic type cannot hold a static stored one.)
    private static var readbackPumpMaxSpins: Int { 30_000 }

    /// For backends whose completions come only from `pumpEvents()` (Dawn), runs a self-pump while
    /// readbacks are outstanding.
    ///
    /// The host's frame ticker runs **only in scenes where JS started the frame loop.** When a scene
    /// without animation (a static check screen) issues `mapAsync`, nobody steps the pump and the
    /// completion never arrived — not an error but **an endless wait**, with nothing happening on
    /// screen. This is the discharge of what `WebGPURuntime.processEvents` requires: "completions must
    /// arrive even with no ticker — have your own waiting mechanism". (Metal's completion handler
    /// arrives on its own, so `needsEventPump` is false and this whole path drops out at no cost.)
    ///
    /// Called under `executionLock`.
    private func noteReadbackStarted() {
        guard backend.capabilities.needsEventPump else { return }
        pendingReadbacks += 1
        guard !readbackPumpRunning else { return }
        readbackPumpRunning = true
        Thread.detachNewThread { [weak self] in
            var spins = 0
            while true {
                guard let self else { return }
                self.executionLock.lock()
                let outstanding = self.pendingReadbacks
                spins += 1
                let exhausted = spins >= Self.readbackPumpMaxSpins
                if outstanding > 0, !exhausted {
                    self.backend.pumpEvents()
                } else {
                    // We put it down here — the outstanding counter is **deliberately left alone.**
                    // The next `readBuffer` stands the pump back up, so this is recoverable, and
                    // faking the counter to 0 would hide whether anything is really outstanding.
                    self.readbackPumpRunning = false
                }
                self.executionLock.unlock()
                if outstanding == 0 { return }
                if exhausted {
                    WGPULog.runtime.error(
                        """
                        Stopping the readback self-pump — \(outstanding, privacy: .public) outstanding \
                        completion(s) did not arrive within 30s (device loss, or the backend is dropping completion callbacks)
                        """
                    )
                    return
                }
                // Rest with the lock released — it competes with the JS thread's execute only at 1ms intervals.
                usleep(1_000)
            }
        }
    }

    /// Called under `executionLock` (inside the completion wrapper).
    private func noteReadbackFinished() {
        guard backend.capabilities.needsEventPump else { return }
        pendingReadbacks -= 1
    }

    /// Decodes an encoded image and registers it as the object standing in for `ImageBitmap` (JS `createImageBitmap`).
    ///
    /// **The handle is issued by JS** — the same rule as the command stream. Decoding is slow, so it
    /// happens on a background queue and only registration runs inside the execution lock.
    public func decodeImage(
        handle: Int,
        data: Data?,
        name: String?,
        options: WGPUImageDecodeOptions,
        provider: WGPUAssetProvider?,
        completion: @escaping ([String: Any]) -> Void
    ) {
        let fail = { (error: WGPUError) in completion(["ok": false, "errors": [error.payload]]) }
        let finish = { [weak self] (bytes: Data) in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let bitmap = try WGPUImageDecoder.decode(bytes, options: options)
                    // **The callback must fire exactly once** even if the runtime was released first.
                    // Slipping out would leave the JS `createImageBitmap()` promise unsettled forever,
                    // stopping the app there without even an error (every failure path calls back —
                    // this was the one hole).
                    guard let self else {
                        return fail(WGPUError.validation(
                            "the runtime has already been released — cannot register the createImageBitmap result"
                        ))
                    }
                    self.executionLock.lock()
                    self.registry.insert(bitmap, at: WGPUHandle(handle))
                    self.executionLock.unlock()
                    completion(["ok": true, "width": bitmap.width, "height": bitmap.height])
                } catch let error as WGPUError {
                    fail(error)
                } catch {
                    fail(WGPUError.backend("\(error)"))
                }
            }
        }

        if let data {
            finish(data)
        } else if let name {
            guard let provider else {
                return fail(WGPUError.validation("no asset provider — pass the image bytes directly"))
            }
            provider.loadAsset(named: name) { result in
                switch result {
                case .success(let bytes): finish(bytes)
                case .failure(let error): fail(error)
                }
            }
        } else {
            fail(WGPUError.validation("createImageBitmap needs image bytes or an asset name"))
        }
    }

    // MARK: - WebGPURuntime: canvas

    private struct CanvasEntry {
        let raw: B.Surface
        let pacesFrames: Bool
    }

    private var canvases: [String: CanvasEntry] = [:]

    public func attachCanvas(identifier: String, layer: CAMetalLayer) {
        let creation = backend.makeLayerSurface(identifier: identifier, layer: layer)
        registerSurface(creation.surface, identifier: identifier, pacesFrames: creation.pacesFrames)
    }

    public func attachOffscreenCanvas(identifier: String, size: CGSize) throws {
        guard Self.isUsableDrawableSize(size) else {
            throw WGPUError.validation(
                "the offscreen canvas size is not valid (\(size.width)x\(size.height))"
            )
        }
        let creation = try backend.makeOffscreenSurface(identifier: identifier, size: size)
        registerSurface(creation.surface, identifier: identifier, pacesFrames: creation.pacesFrames)
    }

    /// Whether a value is usable as a surface size — **finite and non-negative.**
    ///
    /// NaN, infinity and negatives are **the seed of a trap** in every later integer conversion.
    /// `Int(CGFloat.nan)` is a Swift runtime trap (process death), and folded into an unsigned GPU
    /// argument width it becomes an enormous value. Sizes come from UI layout (`bounds × pixelRatio`),
    /// so the moment an app leaks a strange value is a crash — not something to guard per backend, so
    /// it is guarded once here.
    private static func isUsableDrawableSize(_ size: CGSize) -> Bool {
        size.width.isFinite && size.height.isFinite && size.width >= 0 && size.height >= 0
    }

    /// Registers a backend-built surface directly — the channel for custom surfaces (test doubles, …).
    public func registerSurface(_ surface: B.Surface, identifier: String, pacesFrames: Bool) {
        canvasLock.lock()
        canvases[identifier] = CanvasEntry(raw: surface, pacesFrames: pacesFrames)
        canvasLock.unlock()
        // Only surfaces with a drawable pool are paced — an offscreen one cannot fall behind.
        if pacesFrames { frameCoordinator.track(canvas: identifier) }
    }

    public func detachCanvas(identifier: String) {
        canvasLock.lock()
        canvases.removeValue(forKey: identifier)
        canvasLock.unlock()
        // Leaving a dead canvas's counter behind would block frame ticks forever.
        frameCoordinator.forget(canvas: identifier)
    }

    /// Updates the drawable size — **unusable values are ignored silently.**
    ///
    /// NaN and negatives are common during layout (a frame before measurement, say), and passing them
    /// straight down kills the process at `Int(CGFloat.nan)` (`isUsableDrawableSize`). There is no
    /// channel to return an error here either (no return value), so ignoring is the only right
    /// handling — the next layout comes back with a proper value.
    public func resizeCanvas(identifier: String, drawableSize: CGSize) {
        guard Self.isUsableDrawableSize(drawableSize) else { return }
        guard let entry = surfaceEntry(for: identifier) else { return }
        backend.resizeSurface(entry.raw, size: drawableSize)
    }

    public func surface(for identifier: String) -> B.Surface? {
        surfaceEntry(for: identifier)?.raw
    }

    private func surfaceEntry(for identifier: String) -> CanvasEntry? {
        canvasLock.lock()
        defer { canvasLock.unlock() }
        return canvases[identifier]
    }

    public var registeredSurfaceIdentifiers: [String] {
        canvasLock.lock()
        defer { canvasLock.unlock() }
        return Array(canvases.keys).sorted()
    }

    public func canvasInfo(identifier: String) -> [String: Any] {
        guard let entry = surfaceEntry(for: identifier) else {
            return [
                "ok": false,
                "errors": [WGPUError.validation(
                    "canvas '\(identifier)' does not exist (registered: "
                        + "\(registeredSurfaceIdentifiers.joined(separator: ", ")))"
                ).payload],
            ]
        }
        let report = backend.surfaceReport(entry.raw)
        return [
            "ok": true,
            "width": report.width,
            "height": report.height,
            "format": report.format.rawValue,
        ]
    }

    public func readCanvasPixels(identifier: String) throws -> WGPUPixelReadback {
        guard let entry = surfaceEntry(for: identifier) else {
            throw WGPUError.validation(
                "canvas '\(identifier)' is not an offscreen surface — its pixels cannot be read"
            )
        }
        return try backend.readPixels(entry.raw, identifier: identifier)
    }

    // MARK: - WebGPURuntime: frames and lifetime

    /// Whether every registered surface can take a new frame — `frameCoordinator` does the accounting.
    public var isReadyForNextFrame: Bool { frameCoordinator.isReadyForNextFrame }

    /// Asynchronous completion pump — the frame ticker calls it **every tick, on the main thread.**
    ///
    /// **Using `tryLock` is the point.** The backend API may not be thread-safe, so it has to be
    /// serialized under the same lock as `execute` (see `WebGPURuntime`); but taking `lock()` outright
    /// makes the main thread wait for the JS thread's entire batch encode — a UI hitch on any heavy
    /// frame. The contract states the pump is "not something correctness depends on, only a way to
    /// narrow the latency bound" (same document), so a held lock means skipping this tick. `execute`
    /// is driving the backend at that moment anyway, and the next tick is close behind.
    ///
    /// Readback starvation in a configuration with no ticker at all is prevented not by this path but
    /// by the self-pump in `noteReadbackStarted()`.
    public func processEvents() {
        guard executionLock.try() else { return }
        backend.pumpEvents()
        executionLock.unlock()
    }

    /// Discards every GPU object (leaving the page, and so on).
    public func reset() {
        executionLock.lock()
        registry.removeAll()
        errorScopes.discardAll()
        // Mid-frame state goes too — leaving it would make the next device's first frame try to
        // present a dead drawable.
        acquiredFrames.removeAll()
        frameScopedHandles.removeAll()
        _ = gpuFailures.drain()
        backend.reset()
        executionLock.unlock()
    }

    public var liveObjectCount: Int { registry.count }
}
