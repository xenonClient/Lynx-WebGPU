import Foundation
import CoreGraphics
import QuartzCore
import WebGPU
import LynxWebGPUCore

// Dawn C 핸들의 Swift 래퍼.
//
// `WGPUObjectRegistry`(Core)는 `AnyObject`를 저장하므로, C의 불투명 포인터를 클래스로 감싼다.
// 래퍼가 참조 +1을 소유하고 `deinit`에서 릴리스한다 — 레지스트리에서 빠지면(destroy·reset)
// Dawn 객체도 따라 죽는 수명 모델이다. 핸들 타입 불일치가 경로 붙은 validation 오류가 되는
// 것도 레지스트리 계약 그대로다 (`handle-type-mismatch` 검사).

final class DawnBufferObject {
    let buffer: WGPUBuffer
    let size: Int
    let usage: LynxWebGPUCore.WGPUBufferUsage
    /// Dawn 수준에서 실제로 매핑돼 있는가 (mappedAtCreation·직접 map 경로).
    /// 와이어 계약의 매핑 상태는 엔진(`WGPUEngineBuffer.isMapped`)이 관리한다.
    var dawnMapped = false

    init(buffer: WGPUBuffer, size: Int, usage: LynxWebGPUCore.WGPUBufferUsage) {
        self.buffer = buffer
        self.size = size
        self.usage = usage
    }

    deinit { wgpuBufferRelease(buffer) }
}

final class DawnTextureObject {
    let texture: WGPUTexture
    let format: LynxWebGPUCore.WGPUTextureFormat
    let width: Int
    let height: Int

    /// - Parameter retain: 이미 소유된 참조를 감싸면 false(생성 반환값),
    ///   남의 참조를 나눠 가지면 true(캔버스 텍스처 — 캔버스도 계속 쓴다).
    init(
        texture: WGPUTexture,
        format: LynxWebGPUCore.WGPUTextureFormat,
        width: Int,
        height: Int,
        retain: Bool = false
    ) {
        if retain { wgpuTextureAddRef(texture) }
        self.texture = texture
        self.format = format
        self.width = width
        self.height = height
    }

    deinit { wgpuTextureRelease(texture) }
}

final class DawnTextureViewObject {
    let view: WGPUTextureView
    init(view: WGPUTextureView) { self.view = view }
    deinit { wgpuTextureViewRelease(view) }
}

final class DawnSamplerObject {
    let sampler: WGPUSampler
    init(sampler: WGPUSampler) { self.sampler = sampler }
    deinit { wgpuSamplerRelease(sampler) }
}

final class DawnShaderModuleObject {
    let module: WGPUShaderModule
    init(module: WGPUShaderModule) { self.module = module }
    deinit { wgpuShaderModuleRelease(module) }
}

final class DawnBindGroupLayoutObject {
    let layout: WGPUBindGroupLayout
    init(layout: WGPUBindGroupLayout) { self.layout = layout }
    deinit { wgpuBindGroupLayoutRelease(layout) }
}

final class DawnPipelineLayoutObject {
    let layout: WGPUPipelineLayout
    init(layout: WGPUPipelineLayout) { self.layout = layout }
    deinit { wgpuPipelineLayoutRelease(layout) }
}

final class DawnBindGroupObject {
    let group: WGPUBindGroup
    init(group: WGPUBindGroup) { self.group = group }
    deinit { wgpuBindGroupRelease(group) }
}

final class DawnRenderPipelineObject {
    let pipeline: WGPURenderPipeline
    init(pipeline: WGPURenderPipeline) { self.pipeline = pipeline }
    deinit { wgpuRenderPipelineRelease(pipeline) }
}

final class DawnComputePipelineObject {
    let pipeline: WGPUComputePipeline
    init(pipeline: WGPUComputePipeline) { self.pipeline = pipeline }
    deinit { wgpuComputePipelineRelease(pipeline) }
}

final class DawnQuerySetObject {
    let querySet: WGPUQuerySet
    let type: LynxWebGPUCore.WGPUQueryType
    let count: Int

    init(querySet: WGPUQuerySet, type: LynxWebGPUCore.WGPUQueryType, count: Int) {
        self.querySet = querySet
        self.type = type
        self.count = count
    }

    deinit { wgpuQuerySetRelease(querySet) }
}

final class DawnRenderBundleObject {
    let bundle: WGPURenderBundle
    init(bundle: WGPURenderBundle) { self.bundle = bundle }
    deinit { wgpuRenderBundleRelease(bundle) }
}

/// 캔버스 표면 — 화면(`DawnLayerCanvas`) 또는 오프스크린(`DawnOffscreenCanvas`).
///
/// Metal 런타임의 `WGPUSurface` 프로토콜에 해당하는 자리다 — 표면 추상은 백엔드 내부 사정이고
/// (`docs/ARCHITECTURE.md` §3-1), 런타임 경계는 "레이어/크기를 넘길 테니 표면 타입은 네가 골라라"다.
protocol DawnCanvas: AnyObject {
    var identifier: String { get }
    var size: CGSize { get }
    var format: LynxWebGPUCore.WGPUTextureFormat { get }
    /// 드로어블 풀이 있어 **밀릴 수 있는** 표면인가 — in-flight 회계(`WGPUFrameCoordinator`) 대상.
    var pacesFrames: Bool { get }
    func configure(device: WGPUDevice, format: LynxWebGPUCore.WGPUTextureFormat) throws
    func updateSize(_ size: CGSize, device: WGPUDevice)
    /// 이번 프레임의 텍스처. `owned`면 +1 참조를 호출자가 소유하고(표면 경로 — 매 프레임 새로 나옴),
    /// 아니면 캔버스 소유 참조를 빌린 것이다(오프스크린 — 래퍼가 retain해야 한다).
    func acquireTexture(device: WGPUDevice) throws -> (texture: WGPUTexture, owned: Bool)
    /// 프레임을 화면에 올린다. 오프스크린은 no-op.
    func present()
}

/// 오프스크린 캔버스 — 적합성 스위트의 픽셀 통로 (`attachOffscreenCanvas`/`readCanvasPixels`).
///
/// Metal 런타임의 `WGPUOffscreenSurface`와 같은 역할이다: `configure`가 백킹 텍스처를 만들고,
/// `getCurrentTexture`가 그 텍스처를 프레임 스코프 핸들로 내준다.
final class DawnOffscreenCanvas: DawnCanvas {
    let identifier: String
    private(set) var size: CGSize
    private(set) var format: LynxWebGPUCore.WGPUTextureFormat = .rgba8unorm
    private(set) var texture: WGPUTexture?
    var pacesFrames: Bool { false }

    init(identifier: String, size: CGSize) {
        self.identifier = identifier
        self.size = size
    }

    deinit { releaseTexture() }

    func configure(device: WGPUDevice, format: LynxWebGPUCore.WGPUTextureFormat) throws {
        self.format = format
        try remakeTexture(device: device)
    }

    /// 크기 변경 — 이미 configure됐다면 백킹 텍스처를 새 크기로 다시 만든다
    /// (`resize-canvas` 검사의 계약, 오프스크린 Metal 표면과 같은 규칙).
    func updateSize(_ newSize: CGSize, device: WGPUDevice) {
        guard newSize != size else { return }
        size = newSize
        guard texture != nil else { return }
        try? remakeTexture(device: device)
    }

    func acquireTexture(device: WGPUDevice) throws -> (texture: WGPUTexture, owned: Bool) {
        guard let texture else {
            throw WGPUError.validation("캔버스 '\(identifier)'이(가) 아직 configure되지 않았다")
        }
        return (texture, false)
    }

    func present() {}

    private func remakeTexture(device: WGPUDevice) throws {
        let dimensions = try dawnTextureDimensions(size, "오프스크린 캔버스 '\(identifier)'")
        releaseTexture()
        var descriptor = WGPUTextureDescriptor()
        descriptor.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc
            | WGPUTextureUsage_TextureBinding
        descriptor.dimension = WGPUTextureDimension_2D
        descriptor.size = WebGPU.WGPUExtent3D(
            width: dimensions.width, height: dimensions.height, depthOrArrayLayers: 1
        )
        descriptor.format = try DawnEnum.textureFormat(format)
        descriptor.mipLevelCount = 1
        descriptor.sampleCount = 1
        guard let texture = wgpuDeviceCreateTexture(device, &descriptor) else {
            throw WGPUError.outOfMemory("오프스크린 캔버스 텍스처 생성 실패")
        }
        self.texture = texture
    }

    private func releaseTexture() {
        if let texture { wgpuTextureRelease(texture) }
        texture = nil
    }
}

/// 화면 캔버스 — `<webgpu-canvas>`의 `CAMetalLayer`를 Dawn 표면(`WGPUSurfaceSourceMetalLayer`)으로
/// 감싼다. **엘리먼트·브리지는 무변경**이다: 레이어만 넘어오고, 표면 생성·설정·present는 전부
/// 런타임 몫이다 (검토 문서 §3-1이 약속한 그 절단면).
///
/// 스레딩: 생성은 메인(attachCanvas 계약), `updateSize`는 메인, `acquireTexture`/`present`는
/// JS 스레드다. 크기·설정 상태는 락으로 감싸고, **실제 wgpuSurfaceConfigure는 acquire 직전에
/// JS 스레드에서 lazy로** 한다 — 메인↔JS 양쪽에서 Dawn 표면을 만지지 않게 하는 배치다.
final class DawnLayerCanvas: DawnCanvas {
    let identifier: String
    private let layer: CAMetalLayer
    private var surface: WGPUSurface?
    private let lock = NSLock()
    private var desiredSize: CGSize
    private var desiredFormat: LynxWebGPUCore.WGPUTextureFormat = .bgra8unorm
    private var configuredSize: CGSize = .zero
    private var needsConfigure = true
    var pacesFrames: Bool { true }

    init(identifier: String, layer: CAMetalLayer, instance: WGPUInstance) {
        self.identifier = identifier
        self.layer = layer
        self.desiredSize = layer.drawableSize
        var source = WGPUSurfaceSourceMetalLayer()
        source.chain.sType = WGPUSType_SurfaceSourceMetalLayer
        source.layer = Unmanaged.passUnretained(layer).toOpaque()
        surface = withUnsafeMutablePointer(to: &source) { sourcePointer in
            var descriptor = WGPUSurfaceDescriptor()
            descriptor.nextInChain = UnsafeMutableRawPointer(sourcePointer)
                .assumingMemoryBound(to: WGPUChainedStruct.self)
            return wgpuInstanceCreateSurface(instance, &descriptor)
        }
    }

    deinit {
        if let surface {
            wgpuSurfaceUnconfigure(surface)
            wgpuSurfaceRelease(surface)
        }
    }

    var size: CGSize {
        lock.lock()
        defer { lock.unlock() }
        return desiredSize
    }

    var format: LynxWebGPUCore.WGPUTextureFormat {
        lock.lock()
        defer { lock.unlock() }
        return desiredFormat
    }

    func configure(device: WGPUDevice, format: LynxWebGPUCore.WGPUTextureFormat) throws {
        lock.lock()
        desiredFormat = format
        needsConfigure = true
        lock.unlock()
    }

    func updateSize(_ newSize: CGSize, device: WGPUDevice) {
        lock.lock()
        if newSize != desiredSize {
            desiredSize = newSize
            needsConfigure = true
        }
        lock.unlock()
    }

    func acquireTexture(device: WGPUDevice) throws -> (texture: WGPUTexture, owned: Bool) {
        guard let surface else {
            throw WGPUError.backend("캔버스 '\(identifier)'의 Dawn 표면 생성이 실패했었다")
        }
        try reconfigureIfNeeded(surface: surface, device: device)

        var surfaceTexture = WGPUSurfaceTexture()
        wgpuSurfaceGetCurrentTexture(surface, &surfaceTexture)
        let usable = surfaceTexture.status == WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal
            || surfaceTexture.status == WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal
        guard usable, let texture = surfaceTexture.texture else {
            throw WGPUError.validation(
                "캔버스 '\(identifier)'의 드로어블을 얻지 못했다 (status \(surfaceTexture.status))"
            )
        }
        return (texture, true)   // 표면 텍스처는 매 프레임 호출자 소유 +1로 나온다
    }

    func present() {
        if let surface { _ = wgpuSurfacePresent(surface) }
    }

    private func reconfigureIfNeeded(surface: WGPUSurface, device: WGPUDevice) throws {
        lock.lock()
        let size = desiredSize
        let format = desiredFormat
        let needed = needsConfigure
        lock.unlock()
        guard needed else { return }
        // 0·NaN·상한 초과는 트랩이 아니라 validation이다 — 레이아웃 전이면 아직 크기가 없다.
        let dimensions = try dawnTextureDimensions(size, "캔버스 '\(identifier)'")
        var configuration = WGPUSurfaceConfiguration()
        configuration.device = device
        configuration.format = try DawnEnum.textureFormat(format)
        configuration.usage = WGPUTextureUsage_RenderAttachment
        configuration.width = dimensions.width
        configuration.height = dimensions.height
        configuration.alphaMode = WGPUCompositeAlphaMode_Auto
        configuration.presentMode = WGPUPresentMode_Fifo
        wgpuSurfaceConfigure(surface, &configuration)
        lock.lock()
        configuredSize = size
        needsConfigure = false
        lock.unlock()
    }
}
