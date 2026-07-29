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

    init(device: MTLDevice, descriptor: WGPUBufferDescriptor) throws {
        // 통합 메모리(Apple GPU)에서는 shared가 CPU/GPU 양쪽에서 보이며 복사가 없다.
        // writeBuffer / readBuffer 가 blit 없이 바로 memcpy로 끝나는 이유다.
        guard let buffer = device.makeBuffer(length: max(descriptor.size, 1), options: .storageModeShared) else {
            throw WGPUError.outOfMemory("버퍼 \(descriptor.size)B 생성 실패")
        }
        self.buffer = buffer
        self.size = descriptor.size
        self.usage = descriptor.usage
        self.label = descriptor.label
        if let label = descriptor.label { buffer.label = label }

        if let data = descriptor.initialData {
            data.withUnsafeBytes { raw in
                guard let base = raw.baseAddress else { return }
                buffer.contents().copyMemory(from: base, byteCount: min(data.count, size))
            }
        }
    }

    /// CPU에서 버퍼에 쓴다. 범위를 벗어나면 validation 오류.
    func write(_ data: Data, offset: Int) throws {
        guard offset >= 0, offset + data.count <= size else {
            throw WGPUError.validation(
                "writeBuffer 범위 초과 — offset \(offset) + \(data.count)B > 버퍼 크기 \(size)B"
            )
        }
        data.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return }
            buffer.contents().advanced(by: offset).copyMemory(from: base, byteCount: data.count)
        }
    }

    func read(offset: Int, length: Int) throws -> Data {
        guard offset >= 0, length >= 0, offset + length <= size else {
            throw WGPUError.validation("readBuffer 범위 초과 — offset \(offset) + \(length)B > 버퍼 크기 \(size)B")
        }
        return Data(bytes: buffer.contents().advanced(by: offset), count: length)
    }
}

/// `GPUTexture`.
public final class WGPUTextureObject {
    public let texture: MTLTexture
    public let format: WGPUTextureFormat
    public let size: WGPUExtent3D
    public let sampleCount: Int
    /// 스왑체인 드로어블에서 온 텍스처인가 (직접 파괴하면 안 된다).
    let isDrawable: Bool

    init(device: MTLDevice, descriptor: WGPUTextureDescriptor) throws {
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
        // 렌더 타깃/멀티샘플은 CPU가 접근할 수 없는 private 저장소가 훨씬 빠르다.
        // readback이 필요하면 blit으로 shared 버퍼에 내린다 (`copyTextureToBuffer`).
        metalDescriptor.storageMode = descriptor.sampleCount > 1 ? .private : .private

        guard let texture = device.makeTexture(descriptor: metalDescriptor) else {
            throw WGPUError.outOfMemory(
                "텍스처 생성 실패 (\(descriptor.size.width)x\(descriptor.size.height) \(descriptor.format.rawValue))"
            )
        }
        if let label = descriptor.label { texture.label = label }
        self.texture = texture
        self.format = descriptor.format
        self.size = descriptor.size
        self.sampleCount = descriptor.sampleCount
        self.isDrawable = false
    }

    /// 캔버스 드로어블 텍스처를 감싼다.
    init(drawableTexture: MTLTexture, format: WGPUTextureFormat) {
        self.texture = drawableTexture
        self.format = format
        self.size = WGPUExtent3D(width: drawableTexture.width, height: drawableTexture.height)
        self.sampleCount = drawableTexture.sampleCount
        self.isDrawable = true
    }

    /// CPU 데이터를 텍스처에 올린다. private 저장소이므로 staging 버퍼를 거친다.
    func write(
        _ data: Data,
        origin: WGPUOrigin3D,
        size: WGPUExtent3D,
        mipLevel: Int,
        bytesPerRow: Int,
        rowsPerImage: Int,
        device: MTLDevice,
        queue: MTLCommandQueue
    ) throws {
        let expected = bytesPerRow * max(rowsPerImage, size.height) * size.depthOrArrayLayers
        guard data.count >= bytesPerRow * size.height else {
            throw WGPUError.validation("writeTexture 데이터가 부족하다 (\(data.count)B, 최소 \(expected)B 필요)")
        }
        guard let staging = device.makeBuffer(length: data.count, options: .storageModeShared) else {
            throw WGPUError.outOfMemory("writeTexture staging 버퍼 생성 실패")
        }
        data.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return }
            staging.contents().copyMemory(from: base, byteCount: data.count)
        }
        guard let commandBuffer = queue.makeCommandBuffer(), let blit = commandBuffer.makeBlitCommandEncoder() else {
            throw WGPUError.backend("writeTexture blit 인코더 생성 실패")
        }
        blit.copy(
            from: staging,
            sourceOffset: 0,
            sourceBytesPerRow: bytesPerRow,
            sourceBytesPerImage: bytesPerRow * max(rowsPerImage, size.height),
            sourceSize: MTLSize(width: size.width, height: size.height, depth: size.depthOrArrayLayers),
            to: texture,
            destinationSlice: texture.textureType == .type3D ? 0 : origin.z,
            destinationLevel: mipLevel,
            destinationOrigin: MTLOrigin(x: origin.x, y: origin.y, z: texture.textureType == .type3D ? origin.z : 0)
        )
        blit.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
}

/// `GPUTextureView`.
public final class WGPUTextureViewObject {
    public let texture: MTLTexture
    public let format: WGPUTextureFormat
    public let sampleCount: Int
    /// 이 뷰가 캔버스 드로어블에서 왔다면 present 대상이다.
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
                throw WGPUError.validation("createTextureView 실패 — 포맷/차원이 원본 텍스처와 호환되지 않는다")
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
            throw WGPUError.backend("샘플러 생성 실패")
        }
        self.sampler = sampler
    }
}

/// `GPUShaderModule` — WGSL을 파싱해 두고, MSL 방출·컴파일은 파이프라인 생성 시점에 한다.
public final class WGPUShaderModuleObject {
    public let language: WGPUShaderLanguage
    public let label: String?
    /// WGSL일 때만 존재한다.
    public let wgsl: WGSLShaderModule?
    private let rawSource: String
    /// (진입점 조합 + 바인딩 배정) → 컴파일된 라이브러리.
    private var libraryCache: [String: MTLLibrary] = [:]
    private let lock = NSLock()

    init(descriptor: WGPUShaderModuleDescriptor) throws {
        self.language = descriptor.language
        self.label = descriptor.label
        self.rawSource = descriptor.code
        self.wgsl = descriptor.language == .wgsl ? try WGSLShaderModule(source: descriptor.code) : nil
    }

    /// 진입점 목록과 바인딩 배정에 맞는 `MTLLibrary`를 얻는다 (같은 조합은 재사용).
    func library(
        entryPoints: [String],
        bindings: WGSLBindingAssignment,
        device: MTLDevice
    ) throws -> MTLLibrary {
        let key = "\(entryPoints.sorted().joined(separator: "|"))#\(bindings.signature)"
        lock.lock()
        let cached = libraryCache[key]
        lock.unlock()
        if let cached { return cached }

        let source: String
        if let wgsl {
            source = try wgsl.translateToMSL(entryPoints: entryPoints, bindings: bindings)
        } else {
            source = rawSource
        }

        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: source, options: nil)
        } catch {
            throw WGPUError.backend(
                "MSL 컴파일 실패: \(error.localizedDescription)\n--- 생성된 MSL ---\n\(numbered(source))"
            )
        }
        lock.lock()
        libraryCache[key] = library
        lock.unlock()
        return library
    }

    /// WGSL 진입점 이름 → MSL 함수 이름 (`main`처럼 MSL이 거부하는 이름은 바뀐다).
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
