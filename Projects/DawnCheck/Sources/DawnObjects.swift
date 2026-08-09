import Foundation
import CoreGraphics
import QuartzCore
import WebGPU
import LynxWebGPUCore

// Swift wrappers around Dawn C handles.
//
// `WGPUObjectRegistry` (Core) stores `AnyObject`, so C's opaque pointers are wrapped in classes.
// The wrapper owns a +1 reference and releases it in `deinit` — a lifetime model where leaving the
// registry (destroy/reset) kills the Dawn object with it. A handle type mismatch becoming a validation
// error with a path attached is the registry contract as it stands too (the `handle-type-mismatch` check).

final class DawnBufferObject {
    let buffer: WGPUBuffer
    let size: Int
    let usage: LynxWebGPUCore.WGPUBufferUsage
    /// Whether it is actually mapped at the Dawn level (mappedAtCreation and the direct map path).
    /// The wire contract's mapping state is managed by the engine (`WGPUEngineBuffer.isMapped`).
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

    /// - Parameter retain: false when wrapping an already-owned reference (a creation return value),
    ///   true when sharing someone else's reference (a canvas texture — the canvas keeps using it).
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

/// A canvas surface — either on screen (`DawnLayerCanvas`) or offscreen (`DawnOffscreenCanvas`).
///
/// This is the counterpart of the Metal runtime's `WGPUSurface` protocol — the surface abstraction is
/// a backend's own business (`docs/ARCHITECTURE.md` §3-1), and the runtime boundary says "I hand you a layer and a size; you pick the surface type".
protocol DawnCanvas: AnyObject {
    var identifier: String { get }
    var size: CGSize { get }
    var format: LynxWebGPUCore.WGPUTextureFormat { get }
    /// Whether it is a surface with a drawable pool that **can fall behind** — subject to in-flight accounting (`WGPUFrameCoordinator`).
    var pacesFrames: Bool { get }
    func configure(device: WGPUDevice, format: LynxWebGPUCore.WGPUTextureFormat) throws
    func updateSize(_ size: CGSize, device: WGPUDevice)
    /// This frame's texture. When `owned`, the caller owns a +1 reference (the surface path — a new one each frame);
    /// otherwise it is a borrow of the canvas-owned reference (offscreen — the wrapper must retain).
    func acquireTexture(device: WGPUDevice) throws -> (texture: WGPUTexture, owned: Bool)
    /// Puts the frame on screen. Offscreen is a no-op.
    func present()
}

/// The offscreen canvas — the conformance suite's pixel channel (`attachOffscreenCanvas`/`readCanvasPixels`).
///
/// It plays the same role as the Metal runtime's `WGPUOffscreenSurface`: `configure` creates the backing
/// texture, and `getCurrentTexture` hands that texture out as a frame-scoped handle.
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

    /// A size change — if it is already configured, the backing texture is recreated at the new size
    /// (the `resize-canvas` check's contract, the same rule as the offscreen Metal surface).
    func updateSize(_ newSize: CGSize, device: WGPUDevice) {
        guard newSize != size else { return }
        size = newSize
        guard texture != nil else { return }
        try? remakeTexture(device: device)
    }

    func acquireTexture(device: WGPUDevice) throws -> (texture: WGPUTexture, owned: Bool) {
        guard let texture else {
            throw WGPUError.validation("canvas '\(identifier)' is not configured yet")
        }
        return (texture, false)
    }

    func present() {}

    private func remakeTexture(device: WGPUDevice) throws {
        let dimensions = try dawnTextureDimensions(size, "offscreen canvas '\(identifier)'")
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
            throw WGPUError.outOfMemory("failed to create the offscreen canvas texture")
        }
        self.texture = texture
    }

    private func releaseTexture() {
        if let texture { wgpuTextureRelease(texture) }
        texture = nil
    }
}

/// The screen canvas — it wraps `<webgpu-canvas>`'s `CAMetalLayer` in a Dawn surface
/// (`WGPUSurfaceSourceMetalLayer`). **The element and the bridge are unchanged**: only the layer comes
/// across, and surface creation, configuration and present are all the runtime's job (the very cut line the review document §3-1 promised).
///
/// Threading: creation is on main (the attachCanvas contract), `updateSize` is on main, and
/// `acquireTexture`/`present` are on the JS thread. Size and configuration state are wrapped in a lock,
/// and **the actual wgpuSurfaceConfigure happens lazily on the JS thread just before acquire** — an arrangement that keeps the Dawn surface from being touched on both main and JS.
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
            throw WGPUError.backend("creating the Dawn surface for canvas '\(identifier)' had failed")
        }
        try reconfigureIfNeeded(surface: surface, device: device)

        var surfaceTexture = WGPUSurfaceTexture()
        wgpuSurfaceGetCurrentTexture(surface, &surfaceTexture)
        let usable = surfaceTexture.status == WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal
            || surfaceTexture.status == WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal
        guard usable, let texture = surfaceTexture.texture else {
            throw WGPUError.validation(
                "could not obtain a drawable for canvas '\(identifier)' (status \(surfaceTexture.status))"
            )
        }
        return (texture, true)   // a surface texture comes out caller-owned +1 every frame
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
        // Zero, NaN or over the cap is validation, not a trap — before layout there is no size yet.
        let dimensions = try dawnTextureDimensions(size, "canvas '\(identifier)'")
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
