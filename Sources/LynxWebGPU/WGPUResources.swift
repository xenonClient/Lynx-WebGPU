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

    /// 지금 CPU에 매핑되어 있는가 (`mapAsync` ~ `unmap` 사이).
    ///
    /// 명세는 매핑 중인 버퍼를 "unavailable"로 두어 **큐 작업에 쓰지 못하게** 한다. 이 구현은
    /// `.storageModeShared` 버퍼를 스테이징 없이 그대로 읽으므로, 이 상태가 없으면 리드백이
    /// GPU 완료를 기다리는 동안 다음 프레임의 쓰기가 같은 메모리에 겹칠 수 있다 — JS가 받는
    /// 값이 기다린 프레임의 것이라는 보장이 사라진다.
    ///
    /// 읽기·쓰기 모두 `LynxWebGPUContext.executionLock` 아래에서만 일어난다.
    public var isMapped = false

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

/// `GPUQuerySet`.
///
/// 두 종류가 Metal에서 전혀 다른 물건으로 내려간다:
/// - `occlusion` — 평범한 `MTLBuffer`다. 렌더 패스가 `visibilityResultBuffer`로 물고,
///   드로우가 통과시킨 샘플 수를 쿼리 인덱스마다 8바이트로 쌓는다.
/// - `timestamp` — `MTLCounterSampleBuffer`다. 버퍼가 아니라 카운터 샘플 저장소라
///   blit의 `resolveCounters`로만 꺼낼 수 있다.
///
/// 그래서 `resolveQuerySet`도 종류에 따라 다른 blit 명령을 쓴다.
public final class WGPUQuerySetObject {
    /// 결과 하나의 크기 — 두 종류 다 `u64`다 (명세와 Metal이 일치한다).
    static let resultStride = 8

    public let type: WGPUQueryType
    public let count: Int
    /// `occlusion`일 때의 결과 버퍼.
    let visibilityBuffer: MTLBuffer?
    /// `timestamp`일 때의 카운터 샘플 버퍼.
    let counterBuffer: MTLCounterSampleBuffer?

    init(device: MTLDevice, descriptor: WGPUQuerySetDescriptor) throws {
        self.type = descriptor.type
        self.count = descriptor.count

        switch descriptor.type {
        case .occlusion:
            let length = descriptor.count * Self.resultStride
            guard let buffer = device.makeBuffer(length: length, options: .storageModeShared) else {
                throw WGPUError.outOfMemory("occlusion 쿼리 버퍼 \(length)B 생성 실패")
            }
            // 쓰이지 않은 쿼리 슬롯이 쓰레기 값을 돌려주지 않도록 0으로 깐다.
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
                    "이 기기는 패스 경계 타임스탬프 샘플링을 지원하지 않는다 "
                        + "(adapter 정보의 timestampQuery로 미리 확인할 것)"
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
                throw WGPUError.backend("타임스탬프 샘플 버퍼 생성 실패: \(error.localizedDescription)")
            }
            visibilityBuffer = nil
        }
    }

    /// 쿼리 구간이 이 셋 안에 들어오는지. 넘으면 Metal이 단언으로 죽는다.
    func checkRange(first: Int, count queryCount: Int, path: String?) throws {
        guard first >= 0, queryCount >= 0, first + queryCount <= count else {
            throw WGPUError.validation(
                "쿼리 범위를 벗어났다 — \(first)부터 \(queryCount)개, 쿼리셋 크기 \(count)",
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

    /// staging 버퍼의 CPU 데이터를 텍스처로 복사하는 blit을 **프레임 커맨드 버퍼에** 인코딩한다.
    ///
    /// 자체 커맨드 버퍼를 만들어 완주를 기다리던 방식은 (1) 호출마다 GPU가 빌 때까지 JS 스레드를
    /// 세웠고 (2) 아직 커밋되지 않은 프레임 버퍼보다 **먼저** 실행되어 스트림 순서를 깼다.
    /// 같은 큐의 커밋 순서가 곧 실행 순서이므로, 프레임 blit에 태우면 둘 다 해결된다.
    func encodeWrite(
        from staging: MTLBuffer,
        origin: WGPUOrigin3D,
        size: WGPUExtent3D,
        mipLevel: Int,
        bytesPerRow: Int,
        rowsPerImage: Int,
        blit: MTLBlitCommandEncoder
    ) {
        let bytesPerImage = bytesPerRow * max(rowsPerImage, size.height)
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
            // 배열 텍스처는 한 번에 한 슬라이스만 복사할 수 있다 (Metal 규칙 — depth는 3D 전용).
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
        constants: [String: Double] = [:],
        device: MTLDevice
    ) throws -> MTLLibrary {
        // 파이프라인 상수까지 캐시 키에 넣는다 — 같은 셰이더라도 상수가 다르면 다른 MSL이 나온다.
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
