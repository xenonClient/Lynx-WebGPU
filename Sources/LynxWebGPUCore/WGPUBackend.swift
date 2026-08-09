import Foundation
import CoreGraphics
import QuartzCore

/// The **verb protocol** a GPU backend implements — the far side of `WGPUBackendEngine`.
///
/// ## Why it looks like this
///
/// `WebGPURuntime` is a contract over the whole wire (the command stream), so every implementation
/// had to rebuild the batch loop, error scopes, frame lifetime, the mapping gate and serialization
/// from scratch. Every defect that actually bit in the Dawn prototype — pump races, present order,
/// a missed scope drain, crash-prone conversions — **came out of that duplicated orchestration**,
/// not out of encoding.
///
/// So orchestration was lifted into one place, `WGPUBackendEngine` (Core), leaving the backend only
/// **the job of calling its GPU API with values that are already validated and resolved**:
///
/// - Handle → object resolution is done by the engine. A verb receives **its own types**, `Buffer`,
///   `Texture` and so on.
/// - Spec-level validation (ranges, alignment, usage, completeness, bundle isolation, occlusion
///   nesting) is done by the engine. Values reaching a verb have already cleared the spec — a verb
///   throws only for backend-specific limits (device capability, API conversion limits).
/// - Decoding and dispatch live in the engine's one exhaustive switch. **Adding an op means adding
///   a verb to this protocol, and the compiler catches the omission in every backend** — the
///   protocol requirement now gives the guarantee a per-backend switch used to.
///
/// ## Threading
///
/// Every verb is called under the engine's execution lock — `execute` (JS thread), `pumpEvents`
/// (main tick) and `readBuffer` registration are all serialized by that one lock (the engine
/// discharges the pump-concurrency contract of `docs/COMMAND-STREAM.md` §5-1 on the backend's
/// behalf). A backend needs no lock of its own, but completion callbacks (the closures of `submit`
/// and `readBuffer`) may be invoked **from any thread** — the engine-side wrapper retakes its lock.
public protocol WGPUBackend: AnyObject {

    // MARK: - Handle types

    associatedtype Buffer
    associatedtype Texture
    associatedtype TextureView
    associatedtype Sampler
    associatedtype ShaderModule
    associatedtype BindGroupLayout
    associatedtype PipelineLayout
    associatedtype BindGroup
    associatedtype RenderPipeline
    associatedtype ComputePipeline
    associatedtype QuerySet
    associatedtype RenderBundle
    associatedtype Surface

    // MARK: - Capabilities

    var capabilities: WGPUBackendCapabilities { get }

    /// Whether this device supports that compression family — asked by the engine's compressed
    /// texture creation check.
    func supportsTextureCompression(_ format: WGPUTextureFormat) -> Bool

    /// On a device that cannot do indirect draw/dispatch, **the backend throws with its own
    /// context** (Metal: the simulator = Apple GPU family 2 note). The rest of indirect-argument
    /// validation (alignment, range, usage) is the engine's.
    func ensureIndirectSupported() throws

    /// Info, limits and features used by `navigator.gpu.requestAdapter()`. Keys use the spec
    /// spelling (see `WebGPURuntime.adapterInfo`). Every value is device- and API-specific, so the
    /// whole thing is a verb.
    func adapterInfo() -> [String: Any]

    /// Asynchronous completion pump (`WebGPURuntime.processEvents`). Called under the engine lock.
    func pumpEvents()

    /// On discarding the device (`reset`) — drop mid-frame state (drawables, last command buffer, …).
    func reset()

    // MARK: - Batch lifetime

    /// Batch start — the seat for backends needing per-batch diagnostic collection (Dawn: push a
    /// device error scope). A backend whose failures arrive by completion handler, like Metal, has
    /// nothing to do.
    func beginBatch()

    /// After the batch is submitted, the backend hands over the diagnostics it gathered (Dawn: pop
    /// the scope, pump, collect errors). The engine routes them into error scopes / the batch result.
    func collectBatchDiagnostics() -> [WGPUError]

    /// Whether GPU work to submit was produced in this batch (Metal: a command buffer exists).
    /// The engine decides present, accounting and frame expiry from this value.
    var hasPendingWork: Bool { get }

    /// Produces empty submittable work for a batch that must present without commands (a tick's
    /// tail). Failure passes silently — the next batch tries again.
    func ensureSubmittableWork()

    /// Submits the batch's GPU work. Called only when `hasPendingWork` is true.
    ///
    /// - When present is true, send the surface textures acquired this frame to screen **at the
    ///   moment the backend's rules require** (Metal: `present(drawable)` before commit; Dawn:
    ///   `wgpuSurfacePresent` after submit). Afterwards the backend clears the acquired list it held.
    /// - `onCompleted` is called **from any thread** when GPU execution finishes, carrying an error
    ///   on failure — the engine routes it through the deferred error queue into the next batch.
    func submit(present: Bool, onCompleted: @escaping (WGPUError?) -> Void)

    /// Hands back an acquisition we decided not to present — called by the engine when a drawable
    /// was obtained but the frame ended with **no GPU work to submit at all** (a frame that hit a
    /// validation error before the first encoder looks like this).
    ///
    /// Simply holding on drains the swapchain pool, and on an on-screen surface **the next
    /// acquisition stalls the JS thread for up to a second** and then fails permanently. A drawable
    /// that was never drawn cannot be presented (a blank frame would go out), so releasing it
    /// without presenting is the only way out.
    ///
    /// A no-op when nothing was acquired. Must stay safe when called after `submit(present: true)`
    /// already cleared the list.
    func discardAcquiredFrames()

    // MARK: - Resources

    func makeBuffer(_ descriptor: WGPUBufferDescriptor) throws -> Buffer
    func writeBuffer(_ buffer: Buffer, offset: Int, data: Data) throws

    /// `unmap()` — the wire mapping state is the engine's; only a backend with **real mapping**
    /// (Dawn's `mappedAtCreation`) releases its own here. A no-op where there is none.
    func unmapBuffer(_ buffer: Buffer)

    /// Reads buffer contents — guarantee the values are those after previously submitted GPU work.
    /// `deliver` may be called from any thread, and synchronously when the work already finished.
    func readBuffer(_ buffer: Buffer, offset: Int, length: Int,
                    deliver: @escaping (Result<Data, WGPUError>) -> Void)

    func makeTexture(_ descriptor: WGPUTextureDescriptor) throws -> Texture

    /// Uploads CPU data into a texture — `writeTexture` and `copyExternalImageToTexture` converge
    /// here (for the latter the engine finishes decoding, cropping and flipY and passes pixels only).
    /// Omitted defaults for `bytesPerRow`/`rowsPerImage` arrive already filled in by the engine.
    func writeTexture(_ texture: Texture, data: Data, origin: WGPUOrigin3D, size: WGPUExtent3D,
                      mipLevel: Int, bytesPerRow: Int, rowsPerImage: Int) throws

    /// `format` is the view format the engine settled on (`descriptor.format ?? source format`).
    func makeTextureView(_ texture: Texture, descriptor: WGPUTextureViewDescriptor,
                         format: WGPUTextureFormat) throws -> TextureView

    func makeSampler(_ descriptor: WGPUSamplerDescriptor) throws -> Sampler

    /// In the spec a shader module **is created even when compilation fails** — compilation
    /// diagnostics are not thrown but carried in the result's `failure` (the engine registers the
    /// module and reports the diagnostic on the spot). Throw only when **the module itself cannot
    /// be created** (an unsupported language, say).
    func makeShaderModule(_ descriptor: WGPUShaderModuleDescriptor,
                          fieldPath: (String) -> String?) throws -> WGPUShaderModuleCreation<Self>

    /// `getCompilationInfo()` — diagnostics accumulated so far (including pipeline creation failure).
    func compilationMessages(of module: ShaderModule) -> [WGPUCompilationMessage]

    func makeBindGroupLayout(_ entries: [WGPUBindGroupLayoutEntry]) throws -> BindGroupLayout
    func makePipelineLayout(_ groups: [BindGroupLayout]) throws -> PipelineLayout

    /// Resource resolution (handle → object, the `boundSize` default, layout entry matching) is done
    /// by the engine. `layoutEntry` arrives only when the layout entries are known — it is nil for a
    /// native derived layout (a backend whose `getBindGroupLayout` cannot hand back entries).
    func makeBindGroup(layout: BindGroupLayout,
                       entries: [WGPUResolvedBindGroupEntry<Self>]) throws -> BindGroup

    func makeQuerySet(_ descriptor: WGPUQuerySetDescriptor) throws -> QuerySet

    /// Entry-point resolution (the sole entry point when omitted) is **the backend's job** — Metal
    /// does it through WGSL reflection, Dawn through its native default rule. `info` carries the
    /// metadata the engine's pre-draw checks use — leaving an item nil drops that engine check
    /// (meaning the backend validates it natively). `fieldPath` builds the command-stream path to
    /// attach to errors (`fieldPath("vertex.entryPoint")` → `commands[3].vertex.entryPoint`) — used
    /// when a backend reports a diagnostic down to the individual field.
    func makeRenderPipeline(_ descriptor: WGPURenderPipelineDescriptor,
                            vertexModule: ShaderModule, fragmentModule: ShaderModule?,
                            layout: WGPUResolvedPipelineLayout<Self>,
                            fieldPath: (String) -> String?) throws -> WGPURenderPipelineCreation<Self>
    func makeComputePipeline(_ descriptor: WGPUComputePipelineDescriptor,
                             module: ShaderModule,
                             layout: WGPUResolvedPipelineLayout<Self>,
                             fieldPath: (String) -> String?) throws -> WGPUComputePipelineCreation<Self>

    /// `pipeline.getBindGroupLayout(index)`. Return nil when no group sits at that index — the
    /// engine turns it into the spec-worded error. Supply `entries` when known (the engine's bind
    /// group entry matching uses them).
    func bindGroupLayout(of pipeline: WGPUResolvedPipeline<Self>,
                         index: Int) throws -> WGPUBindGroupLayoutCreation<Self>?

    /// Native render bundle creation — called only on a backend whose
    /// `capabilities.supportsNativeRenderBundles` is true. The commands are values the engine already
    /// decoded, and handles inside them are resolved through `resolver`. Where it is false (Metal)
    /// the engine does record/replay instead and this is never called.
    func makeRenderBundle(_ descriptor: WGPURenderBundleDescriptor, commands: [WGPUCommand],
                          resolver: WGPUBundleResolver<Self>) throws -> RenderBundle

    // MARK: - Surfaces

    /// `CAMetalLayer` is the common denominator of both backends — on Apple platforms Dawn takes the
    /// same layer (see `WebGPURuntime.attachCanvas`).
    func makeLayerSurface(identifier: String, layer: CAMetalLayer) -> WGPUSurfaceCreation<Self>
    func makeOffscreenSurface(identifier: String, size: CGSize) throws -> WGPUSurfaceCreation<Self>
    func configureSurface(_ surface: Surface, configuration: WGPUCanvasConfiguration) throws
    /// Arrives from the main thread (a layout change).
    func resizeSurface(_ surface: Surface, size: CGSize)
    /// Measured current pixel size and format — used by `canvasInfo` and the batch result's
    /// `canvases` report.
    func surfaceReport(_ surface: Surface) -> WGPUSurfaceReport

    /// Obtains the surface texture to draw this frame. Nil becomes the engine's "size is zero or
    /// drawables exhausted" error. Remembering the present target is the backend's job (used by
    /// `submit(present:)`).
    func acquireFrameTexture(_ surface: Surface) throws -> WGPUAcquiredSurfaceTexture<Self>?

    /// Reads an offscreen surface's pixels — on any other surface the backend throws in its own words.
    func readPixels(_ surface: Surface, identifier: String) throws -> WGPUPixelReadback

    // MARK: - Passes

    func beginRenderPass(_ pass: WGPUResolvedRenderPass<Self>) throws
    func beginComputePass(_ pass: WGPUResolvedComputePass<Self>) throws
    /// Closes the open pass and any internal encoder. Must be safe to call repeatedly.
    func endPass()

    /// Called only while a pass is open (the engine guarantees it). Backend state that follows a
    /// pipeline change (culling, winding, depth state, …) is applied here too.
    func setRenderPipeline(_ pipeline: RenderPipeline)
    func setComputePipeline(_ pipeline: ComputePipeline)

    /// Right before a draw, **only the dirty groups** come down from the engine's shadow state —
    /// because binding invalidation at bundle boundaries is expressed in that shadow state (see
    /// `WGPUBackendEngine`).
    func applyBindGroup(_ group: BindGroup, at index: Int, dynamicOffsets: [Int]) throws
    /// Right before a draw, only the dirty slots come down. A `validatesNatively` backend must
    /// convert slot and offset safely itself (the engine's range check is skipped).
    func applyVertexBuffer(_ buffer: Buffer, offset: Int, slot: Int) throws

    func setViewport(_ command: WGPUSetViewportCommand) throws
    func setScissorRect(_ command: WGPUSetScissorRectCommand) throws
    func setBlendConstant(_ color: WGPUColor) throws
    func setStencilReference(_ reference: UInt32) throws

    /// `index` is a value the engine already checked for range, nesting and reuse.
    func beginOcclusionQuery(index: Int) throws
    func endOcclusionQuery(index: Int) throws

    /// Native bundle execution — called only on a backend where `supportsNativeRenderBundles` is
    /// true. The engine finished compatibility validation first.
    func executeBundles(_ bundles: [RenderBundle]) throws

    /// `scope` mirrors the spec's two layers — inside a pass (.pass) or the frame region (.frame).
    /// Pairing them up (the depth count) is the engine's job.
    func pushDebugGroup(_ label: String, scope: WGPUDebugScope) throws
    func popDebugGroup(scope: WGPUDebugScope)
    /// Right before submit, close this many groups left open in the frame region (after the engine
    /// has reported the error).
    func popFrameDebugGroups(count: Int)
    func insertDebugMarker(_ label: String, scope: WGPUDebugScope) throws

    // MARK: - Draw / dispatch

    func draw(_ command: WGPUDrawCommand) throws
    func drawIndexed(_ command: WGPUDrawIndexedCommand, index: WGPUResolvedIndexBinding<Self>) throws
    func drawIndirect(buffer: Buffer, offset: Int) throws
    func drawIndexedIndirect(buffer: Buffer, offset: Int, index: WGPUResolvedIndexBinding<Self>) throws
    func dispatchWorkgroups(_ command: WGPUDispatchWorkgroupsCommand) throws
    func dispatchWorkgroupsIndirect(buffer: Buffer, offset: Int) throws

    // MARK: - Copies

    func copyBufferToBuffer(source: Buffer, sourceOffset: Int,
                            destination: Buffer, destinationOffset: Int, size: Int) throws
    /// `range` has been checked by the engine for alignment and bounds, and is non-empty. Fill with zeros.
    func clearBuffer(_ buffer: Buffer, range: Range<Int>) throws
    func copyTextureToBuffer(texture: Texture, slice: Int, mipLevel: Int, origin: WGPUOrigin3D,
                             size: WGPUExtent3D, buffer: Buffer, offset: Int,
                             bytesPerRow: Int, bytesPerImage: Int) throws
    func copyBufferToTexture(buffer: Buffer, offset: Int, bytesPerRow: Int, bytesPerImage: Int,
                             texture: Texture, slice: Int, mipLevel: Int, origin: WGPUOrigin3D,
                             size: WGPUExtent3D) throws
    func copyTextureToTexture(source: Texture, sourceSlice: Int, sourceMipLevel: Int,
                              sourceOrigin: WGPUOrigin3D,
                              destination: Texture, destinationSlice: Int, destinationMipLevel: Int,
                              destinationOrigin: WGPUOrigin3D, size: WGPUExtent3D) throws
    /// Choosing the blit by kind (occlusion/timestamp) is the backend's job. Range, alignment and
    /// usage were checked by the engine, and `count > 0`.
    func resolveQuerySet(_ querySet: QuerySet, first: Int, count: Int,
                         destination: Buffer, destinationOffset: Int) throws
}

// MARK: - Capabilities

public struct WGPUBackendCapabilities {
    /// Whether the backend records and executes render bundles natively.
    /// False makes the engine do record/replay instead (Metal — it has no corresponding object).
    public let supportsNativeRenderBundles: Bool
    /// Cap on vertex buffer slots — used by the engine's `setVertexBuffer` slot check (Metal derives
    /// it from its argument table assignment rules, Dawn from the spec default of 8).
    public let maxVertexBufferSlots: Int
    /// Whether asynchronous completions arrive only from `pumpEvents()` (Dawn's
    /// `wgpuInstanceProcessEvents`). True makes the engine run a **self-pump while readbacks are
    /// outstanding** — the place where `WebGPURuntime.processEvents`'s contract, that completions
    /// must arrive even with no frame ticker (static scenes, headless), is discharged. Leaving it
    /// false on a backend like Metal, whose completions arrive on their own, costs nothing.
    public let needsEventPump: Bool
    /// Whether the backend carries a **complete** WebGPU spec validator (Dawn — the same validator
    /// browsers use).
    ///
    /// True makes the engine skip spec-level checks (range, alignment, usage, occlusion nesting,
    /// bundle compatibility, compression limits) and keep only **bridging and minimal exception
    /// handling** — handle lookup, wire mapping state, pass state guards, CPU path protection. Bad
    /// values are rejected by the backend's own validator (its device error scope), and trap
    /// prevention falls to the backend's safe conversions (the `dawnU32` family) — the same split
    /// react-native-webgpu uses over Dawn (`docs/extra/RN-WEBGPU-LAYERING.md`).
    ///
    /// False (Metal — a direct implementation over a permissive API) runs all of the engine's spec
    /// checks — conceptually those checks are **part of the direct implementation**.
    public let validatesNatively: Bool

    public init(supportsNativeRenderBundles: Bool, maxVertexBufferSlots: Int,
                needsEventPump: Bool = false, validatesNatively: Bool = false) {
        self.supportsNativeRenderBundles = supportsNativeRenderBundles
        self.maxVertexBufferSlots = maxVertexBufferSlots
        self.needsEventPump = needsEventPump
        self.validatesNatively = validatesNatively
    }
}

// MARK: - Verb argument value types

/// Scope of a debug group or marker — inside a pass, or the frame region (`docs/COMMAND-STREAM.md`).
public enum WGPUDebugScope {
    case pass
    case frame
}

public struct WGPUShaderModuleCreation<B: WGPUBackend> {
    public let module: B.ShaderModule
    /// A diagnostic that makes the module unusable (a parse failure, say). The module is registered anyway.
    public let failure: WGPUError?

    public init(module: B.ShaderModule, failure: WGPUError? = nil) {
        self.module = module
        self.failure = failure
    }
}

/// One `getCompilationInfo()` message — the shape of the spec's `GPUCompilationMessage`.
public struct WGPUCompilationMessage {
    public let message: String
    /// `"error"` / `"warning"` / `"info"`.
    public let type: String
    public let lineNum: Int
    public let linePos: Int
    public let offset: Int
    public let length: Int

    public init(message: String, type: String = "error",
                lineNum: Int = 0, linePos: Int = 0, offset: Int = 0, length: Int = 0) {
        self.message = message
        self.type = type
        self.lineNum = lineNum
        self.linePos = linePos
        self.offset = offset
        self.length = length
    }
}

/// Pipeline metadata used by the engine's pre-draw/dispatch checks.
/// **Nil means "the backend validates this itself"** — that engine check is skipped.
public struct WGPURenderPipelineInfo {
    /// Group indices that must be bound before a draw (empty groups excluded).
    public let requiredGroups: Set<Int>?
    /// Vertex buffer slots that must be bound before a draw (declared by `vertex.buffers`).
    public let requiredVertexSlots: Set<Int>?
    /// Used when rejecting in a `depthReadOnly`/`stencilReadOnly` pass.
    public let writesDepth: Bool?
    public let writesStencil: Bool?

    public init(requiredGroups: Set<Int>? = nil, requiredVertexSlots: Set<Int>? = nil,
                writesDepth: Bool? = nil, writesStencil: Bool? = nil) {
        self.requiredGroups = requiredGroups
        self.requiredVertexSlots = requiredVertexSlots
        self.writesDepth = writesDepth
        self.writesStencil = writesStencil
    }
}

public struct WGPUComputePipelineInfo {
    public let requiredGroups: Set<Int>?

    public init(requiredGroups: Set<Int>? = nil) {
        self.requiredGroups = requiredGroups
    }
}

public struct WGPURenderPipelineCreation<B: WGPUBackend> {
    public let pipeline: B.RenderPipeline
    public let info: WGPURenderPipelineInfo

    public init(pipeline: B.RenderPipeline, info: WGPURenderPipelineInfo) {
        self.pipeline = pipeline
        self.info = info
    }
}

public struct WGPUComputePipelineCreation<B: WGPUBackend> {
    public let pipeline: B.ComputePipeline
    public let info: WGPUComputePipelineInfo

    public init(pipeline: B.ComputePipeline, info: WGPUComputePipelineInfo) {
        self.pipeline = pipeline
        self.info = info
    }
}

public struct WGPUBindGroupLayoutCreation<B: WGPUBackend> {
    public let layout: B.BindGroupLayout
    /// Supplied when the layout entries are known — used by the engine's bind group entry matching
    /// and visibility decision. Nil for a native derived layout with unknown entries (that check is
    /// then left to backend validation).
    public let entries: [WGPUBindGroupLayoutEntry]?

    public init(layout: B.BindGroupLayout, entries: [WGPUBindGroupLayoutEntry]?) {
        self.layout = layout
        self.entries = entries
    }
}

public enum WGPUResolvedPipelineLayout<B: WGPUBackend> {
    case auto
    case explicit(B.PipelineLayout)
}

public enum WGPUResolvedPipeline<B: WGPUBackend> {
    case render(B.RenderPipeline)
    case compute(B.ComputePipeline)
}

public struct WGPUResolvedBindGroupEntry<B: WGPUBackend> {
    public enum Resource {
        /// `boundSize` is how many bytes this binding sees — the omitted default (to the end of the
        /// buffer) was filled in by the engine.
        case buffer(B.Buffer, offset: Int, boundSize: Int)
        case sampler(B.Sampler)
        case textureView(B.TextureView)
    }

    public let binding: Int
    /// The matched layout entry — visibility and whether it has a dynamic offset live here.
    /// Nil for a layout with unknown entries (natively derived).
    public let layoutEntry: WGPUBindGroupLayoutEntry?
    public let resource: Resource

    public init(binding: Int, layoutEntry: WGPUBindGroupLayoutEntry?, resource: Resource) {
        self.binding = binding
        self.layoutEntry = layoutEntry
        self.resource = resource
    }
}

public struct WGPUResolvedIndexBinding<B: WGPUBackend> {
    public let buffer: B.Buffer
    public let offset: Int
    public let format: WGPUIndexFormat
    /// Bytes per index — used to convert `drawIndexed`'s `firstIndex` into bytes.
    public let stride: Int

    public init(buffer: B.Buffer, offset: Int, format: WGPUIndexFormat, stride: Int) {
        self.buffer = buffer
        self.offset = offset
        self.format = format
        self.stride = stride
    }
}

public struct WGPUResolvedTimestampWrites<B: WGPUBackend> {
    public let querySet: B.QuerySet
    public let beginningOfPassWriteIndex: Int?
    public let endOfPassWriteIndex: Int?

    public init(querySet: B.QuerySet, beginningOfPassWriteIndex: Int?, endOfPassWriteIndex: Int?) {
        self.querySet = querySet
        self.beginningOfPassWriteIndex = beginningOfPassWriteIndex
        self.endOfPassWriteIndex = endOfPassWriteIndex
    }
}

/// A render pass with every handle resolved into backend objects — the argument of the `beginRenderPass` verb.
public struct WGPUResolvedRenderPass<B: WGPUBackend> {
    public struct ColorAttachment {
        public let view: B.TextureView
        public let resolveTarget: B.TextureView?
        public let loadOp: WGPULoadOp
        public let storeOp: WGPUStoreOp
        public let clearValue: WGPUColor

        public init(view: B.TextureView, resolveTarget: B.TextureView?,
                    loadOp: WGPULoadOp, storeOp: WGPUStoreOp, clearValue: WGPUColor) {
            self.view = view
            self.resolveTarget = resolveTarget
            self.loadOp = loadOp
            self.storeOp = storeOp
            self.clearValue = clearValue
        }
    }

    public struct DepthStencilAttachment {
        public let view: B.TextureView
        /// The view's WebGPU format — used by the backend to tell depth/stencil aspects apart.
        public let format: WGPUTextureFormat
        /// Nil when readOnly — read the contents as they are and leave them (matching load/store).
        public let depthLoadOp: WGPULoadOp?
        public let depthStoreOp: WGPUStoreOp?
        public let depthClearValue: Double
        public let stencilLoadOp: WGPULoadOp?
        public let stencilStoreOp: WGPUStoreOp?
        public let stencilClearValue: Int
        public let depthReadOnly: Bool
        public let stencilReadOnly: Bool

        public init(view: B.TextureView, format: WGPUTextureFormat,
                    depthLoadOp: WGPULoadOp?, depthStoreOp: WGPUStoreOp?, depthClearValue: Double,
                    stencilLoadOp: WGPULoadOp?, stencilStoreOp: WGPUStoreOp?, stencilClearValue: Int,
                    depthReadOnly: Bool, stencilReadOnly: Bool) {
            self.view = view
            self.format = format
            self.depthLoadOp = depthLoadOp
            self.depthStoreOp = depthStoreOp
            self.depthClearValue = depthClearValue
            self.stencilLoadOp = stencilLoadOp
            self.stencilStoreOp = stencilStoreOp
            self.stencilClearValue = stencilClearValue
            self.depthReadOnly = depthReadOnly
            self.stencilReadOnly = stencilReadOnly
        }
    }

    public let label: String?
    public let colorAttachments: [ColorAttachment]
    public let depthStencil: DepthStencilAttachment?
    public let occlusionQuerySet: B.QuerySet?
    public let timestampWrites: WGPUResolvedTimestampWrites<B>?

    public init(label: String?, colorAttachments: [ColorAttachment],
                depthStencil: DepthStencilAttachment?, occlusionQuerySet: B.QuerySet?,
                timestampWrites: WGPUResolvedTimestampWrites<B>?) {
        self.label = label
        self.colorAttachments = colorAttachments
        self.depthStencil = depthStencil
        self.occlusionQuerySet = occlusionQuerySet
        self.timestampWrites = timestampWrites
    }
}

public struct WGPUResolvedComputePass<B: WGPUBackend> {
    public let label: String?
    public let timestampWrites: WGPUResolvedTimestampWrites<B>?

    public init(label: String?, timestampWrites: WGPUResolvedTimestampWrites<B>?) {
        self.label = label
        self.timestampWrites = timestampWrites
    }
}

// MARK: - Surface value types

public struct WGPUSurfaceCreation<B: WGPUBackend> {
    public let surface: B.Surface
    /// Whether the surface has a drawable pool and can fall behind — the subject of frame accounting
    /// (`WGPUFrameCoordinator`).
    public let pacesFrames: Bool

    public init(surface: B.Surface, pacesFrames: Bool) {
        self.surface = surface
        self.pacesFrames = pacesFrames
    }
}

public struct WGPUSurfaceReport {
    public let width: Int
    public let height: Int
    public let format: WGPUTextureFormat

    public init(width: Int, height: Int, format: WGPUTextureFormat) {
        self.width = width
        self.height = height
        self.format = format
    }
}

public struct WGPUAcquiredSurfaceTexture<B: WGPUBackend> {
    public let texture: B.Texture
    /// The actual texture's format — canvas configuration can land a frame late, so this is measured,
    /// not the configured value.
    public let format: WGPUTextureFormat
    public let width: Int
    public let height: Int
    public let sampleCount: Int

    public init(texture: B.Texture, format: WGPUTextureFormat, width: Int, height: Int,
                sampleCount: Int = 1) {
        self.texture = texture
        self.format = format
        self.width = width
        self.height = height
        self.sampleCount = sampleCount
    }
}

/// The lookup window for resolving handles inside commands while recording a native bundle — a
/// narrow channel so the engine registry is not handed to the backend wholesale.
public struct WGPUBundleResolver<B: WGPUBackend> {
    public let renderPipeline: (WGPUHandle, String?) throws -> B.RenderPipeline
    public let bindGroup: (WGPUHandle, String?) throws -> B.BindGroup
    public let buffer: (WGPUHandle, String?) throws -> B.Buffer

    public init(renderPipeline: @escaping (WGPUHandle, String?) throws -> B.RenderPipeline,
                bindGroup: @escaping (WGPUHandle, String?) throws -> B.BindGroup,
                buffer: @escaping (WGPUHandle, String?) throws -> B.Buffer) {
        self.renderPipeline = renderPipeline
        self.bindGroup = bindGroup
        self.buffer = buffer
    }
}

// MARK: - Compression families

/// The compression family a format belongs to — 1:1 with the spec's optional feature names.
/// (Whether the device supports it is the backend's knowledge — `WGPUBackend.supportsTextureCompression`.)
public enum WGPUTextureCompressionFamily {
    case none, bc, etc2, astc

    /// The name carried in `adapter.features` (the spec spelling exactly).
    public var featureName: String? {
        switch self {
        case .none: return nil
        case .bc: return "texture-compression-bc"
        case .etc2: return "texture-compression-etc2"
        case .astc: return "texture-compression-astc"
        }
    }
}

public extension WGPUTextureFormat {
    /// ETC2 and EAC are **the same feature bit** in the spec (`texture-compression-etc2`).
    var compressionFamily: WGPUTextureCompressionFamily {
        guard isCompressed else { return .none }
        if rawValue.hasPrefix("bc") { return .bc }
        if rawValue.hasPrefix("astc-") { return .astc }
        return .etc2
    }
}
