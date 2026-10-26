import Foundation

/// The composite objects the engine stores in the registry — **a backend handle plus the metadata
/// spec validation needs.**
///
/// For the engine to do the validating it has to know spec-level facts such as size, usage and
/// format. Rather than ask the backend object, we carve them in here at creation time — a backend
/// may have no query API at all (Dawn cannot be asked a buffer's usage back) and the contracts
/// differ where one exists.
///
/// Mutable state such as `isMapped` is read and written **only under the engine's execution lock.**

public final class WGPUEngineBuffer<B: WGPUBackend> {
    public let raw: B.Buffer
    public let size: Int
    public let usage: WGPUBufferUsage
    /// Whether it is currently mapped to the CPU (between `mapAsync` and `unmap`) — the spec's
    /// "unavailable". The engine's mapping gate is what keeps such a buffer out of queue work.
    public var isMapped = false
    /// The buffer range of the current **write** mapping (`mapAsync(GPUMapMode.WRITE)` or
    /// `mappedAtCreation`) — the only range `unmapBuffer`'s `data` may be written back into.
    /// Nil while unmapped or mapped for reading.
    public var writeMappedRange: Range<Int>?
    /// Bumped at every mapping request and every unmap — a `readBuffer` completion carrying an older
    /// value belongs to a mapping that was already released (unmap while pending, then mapped again),
    /// so it must not touch the current mapping's state.
    public var mapGeneration = 0

    public init(raw: B.Buffer, size: Int, usage: WGPUBufferUsage) {
        self.raw = raw
        self.size = size
        self.usage = usage
    }
}

public final class WGPUEngineTexture<B: WGPUBackend> {
    public let raw: B.Texture
    public let format: WGPUTextureFormat
    public let size: WGPUExtent3D
    public let sampleCount: Int
    /// Needed to bound a copy's `mipLevel` and its per-level extent before the backend sees it.
    public let mipLevelCount: Int
    public let dimension: WGPUTextureDimension
    /// The canvas this drawable texture came from — nil for a regular texture. Non-nil means the
    /// handle expires once the frame is presented, and views of it inherit the same canvas scope
    /// (so a per-canvas expiry can pick them out).
    public let drawableCanvas: String?
    public var isDrawable: Bool { drawableCanvas != nil }

    public init(raw: B.Texture, format: WGPUTextureFormat, size: WGPUExtent3D,
                sampleCount: Int, mipLevelCount: Int = 1, dimension: WGPUTextureDimension = .twoD,
                drawableCanvas: String? = nil) {
        self.raw = raw
        self.format = format
        self.size = size
        self.sampleCount = sampleCount
        self.mipLevelCount = mipLevelCount
        self.dimension = dimension
        self.drawableCanvas = drawableCanvas
    }
}

public final class WGPUEngineTextureView<B: WGPUBackend> {
    public let raw: B.TextureView
    public let format: WGPUTextureFormat
    public let sampleCount: Int
    /// The extent of the view's base mip level — what a render pass measures its attachments,
    /// viewport and scissor against. Zero when unknown.
    public let width: Int
    public let height: Int

    public init(raw: B.TextureView, format: WGPUTextureFormat, sampleCount: Int,
                width: Int = 0, height: Int = 0) {
        self.raw = raw
        self.format = format
        self.sampleCount = sampleCount
        self.width = width
        self.height = height
    }
}

public final class WGPUEngineSampler<B: WGPUBackend> {
    public let raw: B.Sampler

    public init(raw: B.Sampler) { self.raw = raw }
}

public final class WGPUEngineShaderModule<B: WGPUBackend> {
    public let raw: B.ShaderModule

    public init(raw: B.ShaderModule) { self.raw = raw }
}

public final class WGPUEngineBindGroupLayout<B: WGPUBackend> {
    public let raw: B.BindGroupLayout
    /// For a layout whose entries we know, kept sorted by binding. Nil for a native derived layout
    /// (a backend whose `getBindGroupLayout` cannot hand back entries) — entry matching is then
    /// skipped and left to backend validation.
    public let entries: [WGPUBindGroupLayoutEntry]?

    public init(raw: B.BindGroupLayout, entries: [WGPUBindGroupLayoutEntry]?) {
        self.raw = raw
        self.entries = entries?.sorted { $0.binding < $1.binding }
    }

    public func entry(binding: Int) -> WGPUBindGroupLayoutEntry? {
        entries?.first { $0.binding == binding }
    }
}

public final class WGPUEnginePipelineLayout<B: WGPUBackend> {
    public let raw: B.PipelineLayout
    /// Each group's layout entries (nil where they are unknown) — pipeline creation checks the
    /// layout's `minBindingSize`s against the shader here.
    public let groupEntries: [[WGPUBindGroupLayoutEntry]?]

    public init(raw: B.PipelineLayout, groupEntries: [[WGPUBindGroupLayoutEntry]?] = []) {
        self.raw = raw
        self.groupEntries = groupEntries
    }
}

/// One buffer entry of a bind group, as the draw-time binding size check needs it.
public struct WGPUEngineBufferBinding {
    public let binding: Int
    /// The layout entry's type — picks the dynamic offset alignment limit.
    public let type: WGPUBufferBindingType
    public let bufferSize: Int
    public let offset: Int
    /// The entry's explicit `size` — nil means "to the end of the buffer", which a dynamic offset
    /// then shortens (the spec's effective binding size is taken after adding it).
    public let size: Int?
    public let hasDynamicOffset: Bool
    /// The layout entry's `minBindingSize` was non-zero, so `createBindGroup` already checked the
    /// size and pipeline creation checked the shader against it — the spec's `[[prevalidatedSize]]`.
    public let prevalidatedSize: Bool

    public init(binding: Int, type: WGPUBufferBindingType, bufferSize: Int, offset: Int, size: Int?,
                hasDynamicOffset: Bool, prevalidatedSize: Bool) {
        self.binding = binding
        self.type = type
        self.bufferSize = bufferSize
        self.offset = offset
        self.size = size
        self.hasDynamicOffset = hasDynamicOffset
        self.prevalidatedSize = prevalidatedSize
    }

    /// The spec's "effective buffer binding size" once `dynamicOffset` is applied.
    public func effectiveSize(dynamicOffset: Int) -> Int {
        if let size { return size }
        let (end, overflow) = offset.addingReportingOverflow(dynamicOffset)
        return overflow ? 0 : max(bufferSize - end, 0)
    }
}

public final class WGPUEngineBindGroup<B: WGPUBackend> {
    public let raw: B.BindGroup
    /// Buffers this group holds — needed to check "is it mapped?" right before a draw (the group
    /// pins its buffers at creation, so a buffer mapped afterwards is caught there).
    public let buffers: [WGPUEngineBuffer<B>]
    /// Its buffer entries, sorted by binding — the size and dynamic offset checks read them. Nil when
    /// the layout's entries were unknown (or the backend validates natively): those checks are skipped.
    public let bufferBindings: [WGPUEngineBufferBinding]?

    public init(raw: B.BindGroup, buffers: [WGPUEngineBuffer<B>],
                bufferBindings: [WGPUEngineBufferBinding]? = nil) {
        self.raw = raw
        self.buffers = buffers
        self.bufferBindings = bufferBindings?.sorted { $0.binding < $1.binding }
    }
}

public final class WGPUEngineRenderPipeline<B: WGPUBackend> {
    public let raw: B.RenderPipeline
    public let info: WGPURenderPipelineInfo

    public init(raw: B.RenderPipeline, info: WGPURenderPipelineInfo) {
        self.raw = raw
        self.info = info
    }
}

public final class WGPUEngineComputePipeline<B: WGPUBackend> {
    public let raw: B.ComputePipeline
    public let info: WGPUComputePipelineInfo

    public init(raw: B.ComputePipeline, info: WGPUComputePipelineInfo) {
        self.raw = raw
        self.info = info
    }
}

public final class WGPUEngineQuerySet<B: WGPUBackend> {
    /// Size of one result — both kinds are `u64` (spec and backends agree).
    public static var resultStride: Int { 8 }

    public let raw: B.QuerySet
    public let type: WGPUQueryType
    public let count: Int

    public init(raw: B.QuerySet, type: WGPUQueryType, count: Int) {
        self.raw = raw
        self.type = type
        self.count = count
    }

    /// Whether a query range fits this set. Overrun can kill the backend with an assertion, so we
    /// stop it here.
    public func checkRange(first: Int, count queryCount: Int, path: String?) throws {
        guard first >= 0, queryCount >= 0, first + queryCount <= count else {
            throw WGPUError.validation(
                "query range out of bounds — \(queryCount) queries from \(first), query set holds \(count)",
                path: path
            )
        }
    }
}

/// `GPURenderBundle` — a command list (for replay backends), a native bundle, or both.
///
/// A bundle's contract is "same result as encoding directly". On a backend without native bundles
/// (Metal) we store the commands and replay them into the current pass, which satisfies that
/// contract exactly — and it is also why reuse is safe: all we stored are value-type readers, so
/// executing never mutates the original.
public final class WGPUEngineRenderBundle<B: WGPUBackend> {
    /// Commands allowed inside a bundle, exactly as the spec lists them — viewport, scissor, blend
    /// constant, stencil reference, copies and nested bundles cannot go in a bundle.
    ///
    /// Debug markers **can** — the spec's `GPURenderBundleEncoder` includes `GPUDebugCommandsMixin`.
    /// Omit them and a single marker rejects the whole bundle, with the user unlikely to guess that
    /// the marker was the cause.
    public static var allowedOps: Set<String> {
        [
            "setPipeline", "setBindGroup", "setVertexBuffer", "setIndexBuffer",
            "draw", "drawIndexed", "drawIndirect", "drawIndexedIndirect",
            "pushDebugGroup", "popDebugGroup", "insertDebugMarker",
        ]
    }

    /// For replay backends — re-decoded on every execution (decode errors surface at execution time).
    public let commands: [WGPUValueReader]
    /// For native backends — a bundle already recorded at creation.
    public let native: B.RenderBundle?
    public let descriptor: WGPURenderBundleDescriptor

    /// Only pass commands that already cleared `validateOps(_:)` — native recording needs the op
    /// list validated even before decoding, and validation inside init could not enforce that order.
    public init(commands: [WGPUValueReader], native: B.RenderBundle?,
                descriptor: WGPURenderBundleDescriptor) {
        self.commands = commands
        self.native = native
        self.descriptor = descriptor
    }

    /// Checks the command list against the ops a bundle may contain (the spec's
    /// `GPURenderBundleEncoder` op set).
    public static func validateOps(_ commands: [WGPUValueReader]) throws {
        for command in commands {
            let op = try command.requiredString("op")
            guard allowedOps.contains(op) else {
                throw WGPUError.validation(
                    "'\(op)' cannot go in a render bundle "
                        + "(allowed: \(allowedOps.sorted().joined(separator: ", ")))",
                    path: command.fieldPath("op")
                )
            }
        }
    }

    /// Whether this bundle may execute in the current pass.
    ///
    /// A bundle is created declaring the shape of pass it is for. If that declaration and the actual
    /// pass disagree a browser raises an error, but a replay implementation merely repeats the
    /// commands and the backend cannot catch it (as long as the pipeline matches the pass, it just
    /// draws). Without this check, code that only breaks in a browser ships.
    public func checkCompatibility(
        color: [WGPUTextureFormat],
        depthStencil: WGPUTextureFormat?,
        sampleCount: Int,
        depthReadOnly: Bool,
        stencilReadOnly: Bool
    ) throws {
        // The spec's "render pass layout equals" compares colorFormats **ignoring trailing nulls**.
        // Without trimming, a `['bgra8unorm', null]` bundle is falsely rejected in a one-color pass.
        let bundleFormats = Self.trimmingTrailingNulls(descriptor.colorFormats)
        guard bundleFormats.count == color.count else {
            throw WGPUError.validation(
                "bundle color attachment count (\(bundleFormats.count)) differs from the pass "
                    + "(\(color.count))"
            )
        }
        for (index, expected) in bundleFormats.enumerated() where expected != color[index] {
            throw WGPUError.validation(
                "bundle colorFormats[\(index)] differs from the pass — "
                    + "bundle \(expected?.rawValue ?? "null"), pass \(color[index].rawValue)"
            )
        }
        guard descriptor.depthStencilFormat == depthStencil else {
            throw WGPUError.validation(
                "bundle depthStencilFormat differs from the pass — "
                    + "bundle \(descriptor.depthStencilFormat?.rawValue ?? "none"), "
                    + "pass \(depthStencil?.rawValue ?? "none")"
            )
        }
        guard descriptor.sampleCount == sampleCount else {
            throw WGPUError.validation(
                "bundle sampleCount (\(descriptor.sampleCount)) differs from the pass (\(sampleCount))"
            )
        }
        // Only one direction is required — a read-only pass takes only read-only bundles, but a
        // writable pass has no problem with a read-only bundle.
        guard !depthReadOnly || descriptor.depthReadOnly else {
            throw WGPUError.validation(
                "a depthReadOnly pass can only execute bundles created with depthReadOnly: true"
            )
        }
        guard !stencilReadOnly || descriptor.stencilReadOnly else {
            throw WGPUError.validation(
                "a stencilReadOnly pass can only execute bundles created with stencilReadOnly: true"
            )
        }
    }

    /// Trims trailing `null` slots — the spec's layout equality ignores them.
    private static func trimmingTrailingNulls(_ formats: [WGPUTextureFormat?]) -> [WGPUTextureFormat?] {
        var trimmed = formats
        while let last = trimmed.last, last == nil { trimmed.removeLast() }
        return trimmed
    }
}
