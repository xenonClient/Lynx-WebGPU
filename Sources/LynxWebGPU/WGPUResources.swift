import Foundation
import Metal
import LynxWebGPUCore
import LynxWebGPUShader

/// `GPUBuffer`.
public final class WGPUBufferObject {
    public let buffer: MTLBuffer
    public let size: Int
    public let usage: WGPUBufferUsage
    public let label: String?

    /// Whether it is currently mapped to the CPU (between `mapAsync` and `unmap`).
    ///
    /// The spec marks a mapped buffer "unavailable" so it **cannot be used in queue work**. This
    /// implementation reads a `.storageModeShared` buffer directly without staging, so without this
    /// state a write from the next frame could overlap the same memory while a readback waits on GPU
    /// completion — and the guarantee that JS receives the frame it waited for disappears.
    ///
    /// Both reads and writes happen only under `LynxWebGPUContext.executionLock`.
    public var isMapped = false

    init(device: MTLDevice, descriptor: WGPUBufferDescriptor) throws {
        // On unified memory (Apple GPUs) shared is visible to both CPU and GPU with no copy.
        // That is why writeBuffer / readBuffer finish with a plain memcpy and no blit.
        guard let buffer = device.makeBuffer(length: max(descriptor.size, 1), options: .storageModeShared) else {
            throw WGPUError.outOfMemory("failed to create a \(descriptor.size)B buffer")
        }
        self.buffer = buffer
        self.size = descriptor.size
        self.usage = descriptor.usage
        self.label = descriptor.label
        if let label = descriptor.label { buffer.label = label }

        // No `withUnsafeBytes` — same reason as `WGPUStagingPool.acquire`: inside a throwing function the
        // optimizer may park `contents()` in the error-return register across the rethrows closure.
        if let data = descriptor.initialData, !data.isEmpty {
            data.copyBytes(to: buffer.contents().assumingMemoryBound(to: UInt8.self),
                           count: min(data.count, size))
        }
    }

    /// Writes into the buffer from the CPU. Out of range is a validation error.
    func write(_ data: Data, offset: Int) throws {
        guard offset >= 0, offset + data.count <= size else {
            throw WGPUError.validation(
                "writeBuffer out of range — offset \(offset) + \(data.count)B > buffer size \(size)B"
            )
        }
        guard !data.isEmpty else { return }
        // Closure-free for the reason spelled out in `WGPUStagingPool.acquire` — `writeBuffer` reaches this
        // through the same throwing path `writeTexture` does.
        data.copyBytes(to: buffer.contents().advanced(by: offset).assumingMemoryBound(to: UInt8.self),
                       count: data.count)
    }

    func read(offset: Int, length: Int) throws -> Data {
        guard offset >= 0, length >= 0, offset + length <= size else {
            throw WGPUError.validation("readBuffer out of range — offset \(offset) + \(length)B > buffer size \(size)B")
        }
        return Data(bytes: buffer.contents().advanced(by: offset), count: length)
    }
}

/// `GPUQuerySet`.
///
/// The two kinds land as completely different things in Metal:
/// - `occlusion` — an ordinary `MTLBuffer`. The render pass holds it as `visibilityResultBuffer` and
///   accumulates the samples a draw passed, 8 bytes per query index.
/// - `timestamp` — an `MTLCounterSampleBuffer`. Not a buffer but a counter sample store, extractable
///   only through a blit's `resolveCounters`.
///
/// So `resolveQuerySet` uses a different blit command per kind too.
public final class WGPUQuerySetObject {
    /// Size of one result — both kinds are `u64` (spec and Metal agree).
    static let resultStride = 8

    public let type: WGPUQueryType
    public let count: Int
    /// The result buffer when this is an `occlusion` set.
    let visibilityBuffer: MTLBuffer?
    /// The counter sample buffer when this is a `timestamp` set.
    let counterBuffer: MTLCounterSampleBuffer?

    init(device: MTLDevice, descriptor: WGPUQuerySetDescriptor) throws {
        self.type = descriptor.type
        self.count = descriptor.count

        switch descriptor.type {
        case .occlusion:
            let length = descriptor.count * Self.resultStride
            guard let buffer = device.makeBuffer(length: length, options: .storageModeShared) else {
                throw WGPUError.outOfMemory("failed to create a \(length)B occlusion query buffer")
            }
            // Zero it so unused query slots do not hand back garbage.
            memset(buffer.contents(), 0, length)
            if let label = descriptor.label { buffer.label = label }
            visibilityBuffer = buffer
            counterBuffer = nil

        case .timestamp:
            guard device.supportsCounterSampling(.atStageBoundary),
                  let counterSet = device.counterSets?.first(
                    where: { $0.name == MTLCommonCounterSet.timestamp.rawValue }
                  ) else {
                throw WGPUError.unsupported(
                    "this device does not support pass-boundary timestamp sampling "
                        + "(check adapter.features.has('timestamp-query') first)"
                )
            }
            let counterDescriptor = MTLCounterSampleBufferDescriptor()
            counterDescriptor.counterSet = counterSet
            counterDescriptor.sampleCount = descriptor.count
            counterDescriptor.storageMode = .shared
            if let label = descriptor.label { counterDescriptor.label = label }
            do {
                counterBuffer = try device.makeCounterSampleBuffer(descriptor: counterDescriptor)
            } catch {
                throw WGPUError.backend("timestamp sample buffer creation failed: \(error.localizedDescription)")
            }
            visibilityBuffer = nil
        }
    }

    /// Whether a query range fits this set. Overrun kills Metal with an assertion.
    func checkRange(first: Int, count queryCount: Int, path: String?) throws {
        guard first >= 0, queryCount >= 0, first + queryCount <= count else {
            throw WGPUError.validation(
                "query range out of bounds — \(queryCount) queries from \(first), query set holds \(count)",
                path: path
            )
        }
    }
}

/// `GPUTexture`.
public final class WGPUTextureObject {
    public let texture: MTLTexture
    public let format: WGPUTextureFormat
    public let size: WGPUExtent3D
    public let sampleCount: Int
    /// Whether the texture came from a swapchain drawable (it must not be destroyed directly).
    let isDrawable: Bool

    init(device: MTLDevice, descriptor: WGPUTextureDescriptor) throws {
        try WGPUTextureObject.validateCompressed(descriptor, on: device)
        let metalDescriptor = MTLTextureDescriptor()
        metalDescriptor.pixelFormat = try WGPUMetalMapping.pixelFormat(descriptor.format)
        metalDescriptor.width = descriptor.size.width
        metalDescriptor.height = descriptor.size.height
        metalDescriptor.textureType = WGPUMetalMapping.textureType(
            descriptor.dimension,
            arrayLayers: descriptor.dimension == .threeD ? 1 : descriptor.size.depthOrArrayLayers,
            sampleCount: descriptor.sampleCount
        )
        if descriptor.dimension == .threeD {
            metalDescriptor.depth = descriptor.size.depthOrArrayLayers
        } else {
            metalDescriptor.arrayLength = descriptor.size.depthOrArrayLayers
        }
        metalDescriptor.mipmapLevelCount = descriptor.mipLevelCount
        metalDescriptor.sampleCount = descriptor.sampleCount
        metalDescriptor.usage = WGPUMetalMapping.textureUsage(descriptor.usage)
        // Render targets and multisample textures are far faster in private storage the CPU cannot reach.
        // When readback is needed we blit down into a shared buffer (`copyTextureToBuffer`).
        metalDescriptor.storageMode = descriptor.sampleCount > 1 ? .private : .private

        guard let texture = device.makeTexture(descriptor: metalDescriptor) else {
            throw WGPUError.outOfMemory(
                "texture creation failed (\(descriptor.size.width)x\(descriptor.size.height) \(descriptor.format.rawValue))"
            )
        }
        if let label = descriptor.label { texture.label = label }
        self.texture = texture
        self.format = descriptor.format
        self.size = descriptor.size
        self.sampleCount = descriptor.sampleCount
        self.isDrawable = false
    }

    /// Limits on block-compressed textures. **Metal handles these with an assertion and kills the
    /// process** — we catch them first and return the spec's validation error instead.
    private static func validateCompressed(_ descriptor: WGPUTextureDescriptor, on device: MTLDevice) throws {
        let format = descriptor.format
        guard format.isCompressed else { return }
        guard WGPUDeviceCapability.supportsCompression(format, on: device) else {
            let feature = WGPUDeviceCapability.compressionFamily(format).featureName ?? "?"
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

    /// Wraps a canvas drawable texture.
    init(drawableTexture: MTLTexture, format: WGPUTextureFormat) {
        self.texture = drawableTexture
        self.format = format
        self.size = WGPUExtent3D(width: drawableTexture.width, height: drawableTexture.height)
        self.sampleCount = drawableTexture.sampleCount
        self.isDrawable = true
    }

    /// Encodes the blit copying staging buffer CPU data into a texture **onto the frame command buffer**.
    ///
    /// The old approach of making its own command buffer and waiting for it to finish (1) stalled the
    /// JS thread until the GPU drained on every call and (2) ran **before** the not-yet-committed frame
    /// buffer, breaking stream order. Commit order on one queue is execution order, so riding the frame
    /// blit solves both.
    func encodeWrite(
        from staging: MTLBuffer,
        origin: WGPUOrigin3D,
        size: WGPUExtent3D,
        mipLevel: Int,
        bytesPerRow: Int,
        rowsPerImage: Int,
        blit: MTLBlitCommandEncoder
    ) {
        // rowsPerImage is measured in **block rows** (spec GPUTexelCopyBufferLayout). Uncompressed formats match pixel rows.
        let bytesPerImage = bytesPerRow * max(rowsPerImage, format.blockRows(height: size.height))
        if texture.textureType == .type3D {
            blit.copy(
                from: staging,
                sourceOffset: 0,
                sourceBytesPerRow: bytesPerRow,
                sourceBytesPerImage: bytesPerImage,
                sourceSize: MTLSize(width: size.width, height: size.height, depth: size.depthOrArrayLayers),
                to: texture,
                destinationSlice: 0,
                destinationLevel: mipLevel,
                destinationOrigin: MTLOrigin(x: origin.x, y: origin.y, z: origin.z)
            )
        } else {
            // An array texture can only copy one slice at a time (a Metal rule — depth is 3D only).
            for layer in 0..<max(size.depthOrArrayLayers, 1) {
                blit.copy(
                    from: staging,
                    sourceOffset: layer * bytesPerImage,
                    sourceBytesPerRow: bytesPerRow,
                    sourceBytesPerImage: bytesPerImage,
                    sourceSize: MTLSize(width: size.width, height: size.height, depth: 1),
                    to: texture,
                    destinationSlice: origin.z + layer,
                    destinationLevel: mipLevel,
                    destinationOrigin: MTLOrigin(x: origin.x, y: origin.y, z: 0)
                )
            }
        }
    }
}

/// `GPUTextureView`.
public final class WGPUTextureViewObject {
    public let texture: MTLTexture
    public let format: WGPUTextureFormat
    public let sampleCount: Int
    /// If this view came from a canvas drawable, it is a present target.
    let drawable: WGPUDrawable?

    init(source: WGPUTextureObject, descriptor: WGPUTextureViewDescriptor, drawable: WGPUDrawable?) throws {
        let format = descriptor.format ?? source.format
        let needsReinterpretation = descriptor.format != nil && descriptor.format != source.format
        let needsSlice = descriptor.baseMipLevel != 0 || descriptor.baseArrayLayer != 0
            || descriptor.mipLevelCount != nil || descriptor.arrayLayerCount != nil
            || descriptor.dimension != nil

        if !needsReinterpretation && !needsSlice {
            self.texture = source.texture
        } else {
            let levels = descriptor.mipLevelCount ?? (source.texture.mipmapLevelCount - descriptor.baseMipLevel)
            let layers = descriptor.arrayLayerCount ?? max(source.texture.arrayLength - descriptor.baseArrayLayer, 1)
            let viewType = descriptor.dimension.map(WGPUMetalMapping.viewType) ?? source.texture.textureType
            guard let view = source.texture.makeTextureView(
                pixelFormat: try WGPUMetalMapping.pixelFormat(format),
                textureType: viewType,
                levels: descriptor.baseMipLevel..<(descriptor.baseMipLevel + max(levels, 1)),
                slices: descriptor.baseArrayLayer..<(descriptor.baseArrayLayer + max(layers, 1))
            ) else {
                throw WGPUError.validation("createTextureView failed — the format/dimension is not compatible with the source texture")
            }
            self.texture = view
        }
        self.format = format
        self.sampleCount = source.sampleCount
        self.drawable = drawable
    }
}

/// `GPUSampler`.
public final class WGPUSamplerObject {
    public let sampler: MTLSamplerState

    init(device: MTLDevice, descriptor: WGPUSamplerDescriptor) throws {
        let metalDescriptor = MTLSamplerDescriptor()
        metalDescriptor.sAddressMode = WGPUMetalMapping.addressMode(descriptor.addressModeU)
        metalDescriptor.tAddressMode = WGPUMetalMapping.addressMode(descriptor.addressModeV)
        metalDescriptor.rAddressMode = WGPUMetalMapping.addressMode(descriptor.addressModeW)
        metalDescriptor.magFilter = WGPUMetalMapping.minMagFilter(descriptor.magFilter)
        metalDescriptor.minFilter = WGPUMetalMapping.minMagFilter(descriptor.minFilter)
        metalDescriptor.mipFilter = WGPUMetalMapping.mipFilter(descriptor.mipmapFilter)
        metalDescriptor.lodMinClamp = Float(descriptor.lodMinClamp)
        metalDescriptor.lodMaxClamp = Float(descriptor.lodMaxClamp)
        metalDescriptor.maxAnisotropy = max(descriptor.maxAnisotropy, 1)
        if let label = descriptor.label { metalDescriptor.label = label }
        if let compare = descriptor.compare {
            metalDescriptor.compareFunction = WGPUMetalMapping.compareFunction(compare)
        }
        guard let sampler = device.makeSamplerState(descriptor: metalDescriptor) else {
            throw WGPUError.backend("sampler creation failed")
        }
        self.sampler = sampler
    }
}

/// `GPUShaderModule` — parses WGSL up front; MSL emission and compilation happen at pipeline creation.
public final class WGPUShaderModuleObject {
    public let language: WGPUShaderLanguage
    public let label: String?
    /// Present only for WGSL.
    public let wgsl: WGSLShaderModule?
    private let rawSource: String
    /// (entry point combination + binding assignment) → the compiled library.
    private var libraryCache: [String: MTLLibrary] = [:]
    private let lock = NSLock()

    /// Compilation diagnostics (returned by `getCompilationInfo()`).
    ///
    /// In the spec **a shader module is created even when compilation fails** — the error surfaces
    /// through this list and a pipeline creation failure. So we register the object even on a broken
    /// parse and carry the reason here. (With no handle at all, later commands break only with "does
    /// not exist" and the cause is unknowable.)
    public private(set) var compilationMessages: [WGPUError] = []

    /// Whether this module is usable (whether WGSL parsing succeeded).
    public var isValid: Bool { language != .wgsl || wgsl != nil }

    init(descriptor: WGPUShaderModuleDescriptor) {
        self.language = descriptor.language
        self.label = descriptor.label
        self.rawSource = descriptor.code
        guard descriptor.language == .wgsl else {
            self.wgsl = nil
            return
        }
        do {
            self.wgsl = try WGSLShaderModule(source: descriptor.code)
        } catch let error as WGPUError {
            self.wgsl = nil
            self.compilationMessages = [error]
        } catch {
            self.wgsl = nil
            self.compilationMessages = [.validation("WGSL parse failed: \(error.localizedDescription)")]
        }
    }

    /// Appends diagnostics produced by pipeline creation (MSL compilation failure, and the like).
    func record(_ error: WGPUError) {
        lock.lock()
        defer { lock.unlock() }
        // Building several pipelines from one shader repeats the same error — duplicates are filtered.
        if !compilationMessages.contains(error) { compilationMessages.append(error) }
    }

    /// The spec's **"get the entry point"** — omitting the name uses the stage's **only** entry point.
    ///
    /// `entryPoint` is not a required member in the spec. Guessing `"main"` would reject entire shaders
    /// whose entry points are named otherwise — three.js's mipmap shaders (`mainVS` + `main_2d`, …) broke that way.
    ///
    /// An MSL module has no reflection, so stages cannot be counted. There `"main"` is the convention,
    /// and **if the function really is absent Metal catches it on the spot** (`makeFunction` returns nil).
    func resolveEntryPoint(_ requested: String?, stage: WGSLStage, path: String? = nil) throws -> String {
        // For a module that failed to parse, report the real cause again — restating it as "no entry
        // point" sends the user off suspecting the shader name and fixing the wrong thing.
        if !isValid, let failure = compilationMessages.first {
            throw WGPUError(
                kind: failure.kind,
                message: "this shader module failed to compile — \(failure.message)",
                path: path, line: failure.line
            )
        }
        guard let wgsl else { return requested ?? "main" }
        do {
            return try wgsl.resolveEntryPoint(requested, stage: stage)
        } catch let error as WGPUError {
            throw WGPUError(kind: error.kind, message: error.message, path: error.path ?? path)
        }
    }

    /// Obtains the `MTLLibrary` matching an entry point list and binding assignment (identical combinations are reused).
    func library(
        entryPoints: [String],
        bindings: WGSLBindingAssignment,
        constants: [String: Double] = [:],
        device: MTLDevice
    ) throws -> MTLLibrary {
        // Pipeline constants go into the cache key too — the same shader with different constants yields different MSL.
        let constantsKey = constants.sorted { $0.key < $1.key }.map { "\($0.key)=\($0.value)" }.joined(separator: ",")
        let key = "\(entryPoints.sorted().joined(separator: "|"))#\(bindings.signature)#\(constantsKey)"
        lock.lock()
        let cached = libraryCache[key]
        lock.unlock()
        if let cached { return cached }

        let source: String
        if let wgsl {
            source = try wgsl.translateToMSL(entryPoints: entryPoints, bindings: bindings, constants: constants)
        } else {
            source = rawSource
        }

        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: source, options: nil)
        } catch {
            throw WGPUError.backend(
                "MSL compilation failed: \(error.localizedDescription)\n--- generated MSL ---\n\(numbered(source))"
            )
        }
        lock.lock()
        libraryCache[key] = library
        lock.unlock()
        return library
    }

    /// WGSL entry point name → MSL function name (names MSL rejects, such as `main`, are changed).
    func metalFunctionName(for entryPoint: String) -> String {
        language == .wgsl ? WGSLShaderModule.mslFunctionName(for: entryPoint) : entryPoint
    }

    private func numbered(_ source: String) -> String {
        source.split(separator: "\n", omittingEmptySubsequences: false)
            .enumerated()
            .map { String(format: "%3d| %@", $0.offset + 1, String($0.element)) }
            .joined(separator: "\n")
    }
}
