import Foundation

/// The **decoded result** of one command-stream op.
///
/// Where `WGPUDescriptors.swift` carries the spec's `GPU*Descriptor` dictionaries, this file
/// carries **the op's own arguments** (`draw`'s `vertexCount`, `copyBufferToBuffer`'s offsets, …).
///
/// Why they are separate — if the interpreter read fields straight from the reader, **decoding and
/// backend encoding would fuse into one function.** Swapping the backend (for Dawn, say) would then
/// mean rewriting the decoding too, and the field names shared by JS and Swift would scatter across
/// the code, so fixing one side still compiles on both — silent drift (`CLAUDE.md`: "type checking
/// does not catch command-stream field names"). Gathered here, names have a single source and the
/// backend receives **values only**.
///
/// Two rules:
/// - **Defaults that depend on an object are not filled here.** Anything needing a registry lookup
///   — `copyBufferToBuffer`'s `size` (the rest of the source), say — stays `nil` for the interpreter.
/// - **Value conversions the spec mandates happen here.** Backend-independent rules such as WebIDL's
///   `u32` modulo conversion or the workgroup-count floor must be identical on every backend.
public protocol WGPUCommandFields {
    /// Position within the command stream (`commands[3]`). The root of the path a late validation attaches.
    var path: String { get }
}

public extension WGPUCommandFields {
    /// Path of one field under this command (`commands[3].size`).
    func fieldPath(_ key: String) -> String { path.isEmpty ? key : "\(path).\(key)" }
}

// MARK: - Object creation

/// A descriptor buildable from a single reader in the command stream.
///
/// `WGPUCreateCommand` builds descriptors knowing only this requirement — which is why adding a
/// `create*` op needs no interpreter change.
public protocol WGPUDecodableDescriptor {
    init(from reader: WGPUValueReader) throws
}

/// One `create*` op — a **JS-issued handle** paired with the spec descriptor.
///
/// Client-issued handles are the premise of this design (see `WGPUHandle`), so every creation op
/// carries an extra `id` field the spec does not have — and that name lives in **this one place**.
public struct WGPUCreateCommand<Descriptor: WGPUDecodableDescriptor>: WGPUCommandFields {
    public let id: WGPUHandle
    public let descriptor: Descriptor
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        id = try reader.requiredHandle("id")
        descriptor = try Descriptor(from: reader)
        path = reader.path
    }
}

extension WGPUBufferDescriptor: WGPUDecodableDescriptor {}
extension WGPUTextureDescriptor: WGPUDecodableDescriptor {}
extension WGPUSamplerDescriptor: WGPUDecodableDescriptor {}
extension WGPUShaderModuleDescriptor: WGPUDecodableDescriptor {}
extension WGPUBindGroupLayoutDescriptor: WGPUDecodableDescriptor {}
extension WGPUPipelineLayoutDescriptor: WGPUDecodableDescriptor {}
extension WGPUBindGroupDescriptor: WGPUDecodableDescriptor {}
extension WGPUQuerySetDescriptor: WGPUDecodableDescriptor {}
extension WGPURenderPipelineDescriptor: WGPUDecodableDescriptor {}
extension WGPUComputePipelineDescriptor: WGPUDecodableDescriptor {}

/// `device.createTextureView()` — the source texture handle rides outside the descriptor.
public struct WGPUCreateTextureViewCommand: WGPUCommandFields {
    public let id: WGPUHandle
    public let texture: WGPUHandle
    public let descriptor: WGPUTextureViewDescriptor
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        id = try reader.requiredHandle("id")
        texture = try reader.requiredHandle("texture")
        descriptor = try WGPUTextureViewDescriptor(from: reader)
        path = reader.path
    }
}

/// `bundleEncoder.finish()` — **stores the command list as values** and replays it into a render pass.
///
/// This is the only place holding readers directly. The bundle contract is "same result as encoding
/// directly", so what must be stored is **the commands themselves**, not decoded values. Readers are
/// value types, so replaying never mutates the original.
public struct WGPUCreateRenderBundleCommand: WGPUCommandFields {
    public let id: WGPUHandle
    public let commands: [WGPUValueReader]
    public let descriptor: WGPURenderBundleDescriptor
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        id = try reader.requiredHandle("id")
        commands = try reader.requiredObjects("commands")
        descriptor = try WGPURenderBundleDescriptor(from: reader)
        path = reader.path
    }
}

// MARK: - Shared copy arguments

/// Spec `GPUTexelCopyTextureInfo` — the texture end of a copy.
public struct WGPUTexelCopyTextureInfo: WGPUCommandFields {
    public let texture: WGPUHandle
    public let mipLevel: Int
    public let origin: WGPUOrigin3D
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        texture = try reader.requiredHandle("texture")
        mipLevel = reader.int("mipLevel", default: 0)
        origin = try reader.origin("origin")
        path = reader.path
    }
}

/// Spec `GPUTexelCopyBufferInfo` — the buffer end of a copy.
///
/// `bytesPerRow` and `rowsPerImage` may be omitted, and their defaults **require knowing the texture
/// format** (in block formats a row is blocks, not pixels). So they stay nil here.
public struct WGPUTexelCopyBufferInfo: WGPUCommandFields {
    public let buffer: WGPUHandle
    public let offset: Int
    public let bytesPerRow: Int?
    public let rowsPerImage: Int?
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        buffer = try reader.requiredHandle("buffer")
        offset = reader.int("offset", default: 0)
        bytesPerRow = reader.optionalInt("bytesPerRow")
        rowsPerImage = reader.optionalInt("rowsPerImage")
        path = reader.path
    }
}

// MARK: - Resources and queue

/// `queue.writeBuffer()`.
public struct WGPUWriteBufferCommand: WGPUCommandFields {
    public let buffer: WGPUHandle
    public let data: Data
    public let bufferOffset: Int
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        buffer = try reader.requiredHandle("buffer")
        data = try reader.requiredData("data")
        bufferOffset = reader.int("bufferOffset", default: 0)
        path = reader.path
    }
}

/// `queue.writeTexture()`.
public struct WGPUWriteTextureCommand: WGPUCommandFields {
    public let texture: WGPUHandle
    public let data: Data
    public let mipLevel: Int
    public let origin: WGPUOrigin3D
    public let size: WGPUExtent3D
    public let bytesPerRow: Int?
    public let rowsPerImage: Int?
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        texture = try reader.requiredHandle("texture")
        data = try reader.requiredData("data")
        mipLevel = reader.int("mipLevel", default: 0)
        origin = try reader.origin("origin")
        size = try reader.requiredExtent("size")
        bytesPerRow = reader.optionalInt("bytesPerRow")
        rowsPerImage = reader.optionalInt("rowsPerImage")
        path = reader.path
    }
}

/// The source of `queue.copyExternalImageToTexture()` (spec `GPUCopyExternalImageSourceInfo`).
///
/// `flipY` here flips at **copy time** — separate from `createImageBitmap`'s `flipY` (decode time).
public struct WGPUExternalImageSource: WGPUCommandFields {
    public let image: WGPUHandle
    public let origin: WGPUOrigin3D
    public let flipY: Bool
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        image = try reader.requiredHandle("source")
        origin = try reader.origin("origin")
        flipY = reader.bool("flipY", default: false)
        path = reader.path
    }
}

/// `queue.copyExternalImageToTexture()`.
public struct WGPUCopyExternalImageCommand: WGPUCommandFields {
    public let source: WGPUExternalImageSource
    public let destination: WGPUTexelCopyTextureInfo
    /// Omitted means **the whole remainder of the image** — that needs the image size, so it stays nil.
    public let copySize: WGPUExtent3D?
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        source = try WGPUExternalImageSource(from: try reader.requiredObject("source"))
        destination = try WGPUTexelCopyTextureInfo(from: try reader.requiredObject("destination"))
        copySize = reader.extent("copySize")
        path = reader.path
    }
}

/// An op taking just one handle (`destroy`).
public struct WGPUDestroyCommand: WGPUCommandFields {
    public let id: WGPUHandle
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        id = try reader.requiredHandle("id")
        path = reader.path
    }
}

/// `buffer.unmap()`.
public struct WGPUUnmapBufferCommand: WGPUCommandFields {
    public let buffer: WGPUHandle
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        buffer = try reader.requiredHandle("buffer")
        path = reader.path
    }
}

/// `pipeline.getBindGroupLayout(index)`.
public struct WGPUGetBindGroupLayoutCommand: WGPUCommandFields {
    public let id: WGPUHandle
    public let pipeline: WGPUHandle
    public let index: Int
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        id = try reader.requiredHandle("id")
        pipeline = try reader.requiredHandle("pipeline")
        index = try reader.requiredInt("index")
        path = reader.path
    }
}

// MARK: - Error scopes

/// `device.pushErrorScope(filter)`.
public struct WGPUPushErrorScopeCommand: WGPUCommandFields {
    public let filter: WGPUErrorFilter
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        path = reader.path
        filter = try reader.requiredEnum("filter", WGPUErrorFilter.self)
    }
}

// MARK: - Canvas

/// `context.getCurrentTexture()`.
public struct WGPUGetCurrentTextureCommand: WGPUCommandFields {
    public let id: WGPUHandle
    public let canvas: String
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        id = try reader.requiredHandle("id")
        canvas = try reader.requiredString("canvas")
        path = reader.path
    }
}

// MARK: - Pass state

/// `pass.setPipeline()` — shared by render and compute (the open pass decides which).
public struct WGPUSetPipelineCommand: WGPUCommandFields {
    public let pipeline: WGPUHandle
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        pipeline = try reader.requiredHandle("pipeline")
        path = reader.path
    }
}

/// `pass.setBindGroup()`.
public struct WGPUSetBindGroupCommand: WGPUCommandFields {
    public let index: Int
    public let bindGroup: WGPUHandle
    public let dynamicOffsets: [Int]
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        index = try reader.requiredInt("index")
        bindGroup = try reader.requiredHandle("bindGroup")
        // Unreadable means **treat it as empty.** Layouts without dynamic offsets are overwhelmingly
        // common, and if one really was needed, applying the bind group catches it as "too few".
        dynamicOffsets = (try? reader.integers("dynamicOffsets")) ?? []
        path = reader.path
    }
}

/// `pass.setVertexBuffer()`.
public struct WGPUSetVertexBufferCommand: WGPUCommandFields {
    public let slot: Int
    public let buffer: WGPUHandle
    public let offset: Int
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        slot = try reader.requiredInt("slot")
        buffer = try reader.requiredHandle("buffer")
        offset = reader.int("offset", default: 0)
        path = reader.path
    }
}

/// `pass.setIndexBuffer()`.
public struct WGPUSetIndexBufferCommand: WGPUCommandFields {
    public let buffer: WGPUHandle
    public let format: WGPUIndexFormat
    public let offset: Int
    public let path: String

    /// Bytes per index — used to turn `firstIndex` into a byte offset.
    public var indexStride: Int { format == .uint16 ? 2 : 4 }

    public init(from reader: WGPUValueReader) throws {
        buffer = try reader.requiredHandle("buffer")
        format = try reader.requiredEnum("format", WGPUIndexFormat.self)
        offset = reader.int("offset", default: 0)
        path = reader.path
    }
}

/// `pass.setViewport()`.
public struct WGPUSetViewportCommand: WGPUCommandFields {
    public let x: Double
    public let y: Double
    public let width: Double
    public let height: Double
    public let minDepth: Double
    public let maxDepth: Double
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        x = reader.double("x", default: 0)
        y = reader.double("y", default: 0)
        width = try reader.requiredDouble("width")
        height = try reader.requiredDouble("height")
        minDepth = reader.double("minDepth", default: 0)
        maxDepth = reader.double("maxDepth", default: 1)
        path = reader.path
    }
}

/// `pass.setScissorRect()`.
public struct WGPUSetScissorRectCommand: WGPUCommandFields {
    public let x: Int
    public let y: Int
    public let width: Int
    public let height: Int
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        x = reader.int("x", default: 0)
        y = reader.int("y", default: 0)
        width = try reader.requiredInt("width")
        height = try reader.requiredInt("height")
        path = reader.path
    }
}

/// `pass.setBlendConstant()`.
public struct WGPUSetBlendConstantCommand: WGPUCommandFields {
    public let color: WGPUColor
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        color = try reader.color("color", default: .transparent)
        path = reader.path
    }
}

/// `pass.setStencilReference()`.
public struct WGPUSetStencilReferenceCommand: WGPUCommandFields {
    /// **WebIDL's `u32` conversion (modulo) is completed here.** With a non-truncating initializer a
    /// single `setStencilReference(-1)` traps the Swift runtime — violating this library's contract
    /// that "bad arguments never kill the process" (`WGPUError.swift`).
    public let reference: UInt32
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        reference = UInt32(truncatingIfNeeded: reader.int("reference", default: 0))
        path = reader.path
    }
}

// MARK: - Draw and dispatch

/// `pass.draw()`.
public struct WGPUDrawCommand: WGPUCommandFields {
    public let vertexCount: Int
    public let instanceCount: Int
    public let firstVertex: Int
    public let firstInstance: Int
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        vertexCount = try reader.requiredInt("vertexCount")
        instanceCount = reader.int("instanceCount", default: 1)
        firstVertex = reader.int("firstVertex", default: 0)
        firstInstance = reader.int("firstInstance", default: 0)
        path = reader.path
    }
}

/// `pass.drawIndexed()`.
public struct WGPUDrawIndexedCommand: WGPUCommandFields {
    public let indexCount: Int
    public let instanceCount: Int
    public let firstIndex: Int
    public let baseVertex: Int
    public let firstInstance: Int
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        indexCount = try reader.requiredInt("indexCount")
        instanceCount = reader.int("instanceCount", default: 1)
        firstIndex = reader.int("firstIndex", default: 0)
        baseVertex = reader.int("baseVertex", default: 0)
        firstInstance = reader.int("firstInstance", default: 0)
        path = reader.path
    }
}

/// Shared arguments of the three indirect ops (`drawIndirect`, `drawIndexedIndirect`,
/// `dispatchWorkgroupsIndirect`). The actual argument values live in a GPU buffer, invisible here.
public struct WGPUIndirectCommand: WGPUCommandFields {
    public let indirectBuffer: WGPUHandle
    public let indirectOffset: Int
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        indirectBuffer = try reader.requiredHandle("indirectBuffer")
        indirectOffset = reader.int("indirectOffset", default: 0)
        path = reader.path
    }
}

/// `pass.dispatchWorkgroups()`.
public struct WGPUDispatchWorkgroupsCommand: WGPUCommandFields {
    public let x: Int
    public let y: Int
    public let z: Int
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        // Passing 0 makes Metal assert on an empty grid. The spec disallows 0 too, so we raise it to 1.
        x = max(reader.int("x", default: 1), 1)
        y = max(reader.int("y", default: 1), 1)
        z = max(reader.int("z", default: 1), 1)
        path = reader.path
    }
}

/// `pass.executeBundles()`.
public struct WGPUExecuteBundlesCommand: WGPUCommandFields {
    public let bundles: [WGPUHandle]
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        bundles = try reader.handles("bundles")
        path = reader.path
    }
}

// MARK: - Queries

/// `pass.beginOcclusionQuery()`.
public struct WGPUBeginOcclusionQueryCommand: WGPUCommandFields {
    public let queryIndex: Int
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        queryIndex = try reader.requiredInt("queryIndex")
        path = reader.path
    }
}

/// `encoder.resolveQuerySet()`.
public struct WGPUResolveQuerySetCommand: WGPUCommandFields {
    public let querySet: WGPUHandle
    public let firstQuery: Int
    /// Omitted means **the rest of the query set** — that needs its size, so it stays nil.
    public let queryCount: Int?
    public let destination: WGPUHandle
    public let destinationOffset: Int
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        querySet = try reader.requiredHandle("querySet")
        firstQuery = reader.int("firstQuery", default: 0)
        queryCount = reader.optionalInt("queryCount")
        destination = try reader.requiredHandle("destination")
        destinationOffset = reader.int("destinationOffset", default: 0)
        path = reader.path
    }
}

// MARK: - Copies

/// `encoder.copyBufferToBuffer()`.
public struct WGPUCopyBufferToBufferCommand: WGPUCommandFields {
    public let source: WGPUHandle
    public let sourceOffset: Int
    public let destination: WGPUHandle
    public let destinationOffset: Int
    /// Omitted means **the rest of the source** (the spec's short overload). Needs the buffer size, so nil.
    public let size: Int?
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        source = try reader.requiredHandle("source")
        sourceOffset = reader.int("sourceOffset", default: 0)
        destination = try reader.requiredHandle("destination")
        destinationOffset = reader.int("destinationOffset", default: 0)
        size = reader.optionalInt("size")
        path = reader.path
    }
}

/// `encoder.clearBuffer()`.
public struct WGPUClearBufferCommand: WGPUCommandFields {
    public let buffer: WGPUHandle
    public let offset: Int
    /// Omitted means **through the end of the buffer**.
    public let size: Int?
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        buffer = try reader.requiredHandle("buffer")
        offset = reader.int("offset", default: 0)
        size = reader.optionalInt("size")
        path = reader.path
    }
}

/// `encoder.copyTextureToBuffer()`.
public struct WGPUCopyTextureToBufferCommand: WGPUCommandFields {
    public let source: WGPUTexelCopyTextureInfo
    public let destination: WGPUTexelCopyBufferInfo
    public let copySize: WGPUExtent3D
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        source = try WGPUTexelCopyTextureInfo(from: try reader.requiredObject("source"))
        destination = try WGPUTexelCopyBufferInfo(from: try reader.requiredObject("destination"))
        copySize = try reader.requiredExtent("copySize")
        path = reader.path
    }
}

/// `encoder.copyBufferToTexture()`.
public struct WGPUCopyBufferToTextureCommand: WGPUCommandFields {
    public let source: WGPUTexelCopyBufferInfo
    public let destination: WGPUTexelCopyTextureInfo
    public let copySize: WGPUExtent3D
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        source = try WGPUTexelCopyBufferInfo(from: try reader.requiredObject("source"))
        destination = try WGPUTexelCopyTextureInfo(from: try reader.requiredObject("destination"))
        copySize = try reader.requiredExtent("copySize")
        path = reader.path
    }
}

/// `encoder.copyTextureToTexture()`.
public struct WGPUCopyTextureToTextureCommand: WGPUCommandFields {
    public let source: WGPUTexelCopyTextureInfo
    public let destination: WGPUTexelCopyTextureInfo
    public let copySize: WGPUExtent3D
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        source = try WGPUTexelCopyTextureInfo(from: try reader.requiredObject("source"))
        destination = try WGPUTexelCopyTextureInfo(from: try reader.requiredObject("destination"))
        copySize = try reader.requiredExtent("copySize")
        path = reader.path
    }
}

// MARK: - Debug markers

/// `pushDebugGroup()`.
public struct WGPUPushDebugGroupCommand: WGPUCommandFields {
    public let groupLabel: String
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        groupLabel = try reader.requiredString("groupLabel")
        path = reader.path
    }
}

/// `insertDebugMarker()`.
public struct WGPUInsertDebugMarkerCommand: WGPUCommandFields {
    public let markerLabel: String
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        markerLabel = try reader.requiredString("markerLabel")
        path = reader.path
    }
}
