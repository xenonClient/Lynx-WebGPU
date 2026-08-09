import Foundation
import CoreGraphics
import Metal
import QuartzCore
import LynxWebGPUCore

/// The display target for one frame.
public protocol WGPUDrawable: AnyObject {
    var texture: MTLTexture { get }
    /// Puts it on screen when the command buffer is committed. An offscreen surface does nothing.
    func present(with commandBuffer: MTLCommandBuffer)
}

/// The surface a `GPUCanvasContext` draws into — a screen (CAMetalLayer) or an offscreen texture.
///
/// The host registers it with `LynxWebGPUContext.registerSurface(_:)`, and JS names it by the `canvas` string id.
public protocol WGPUSurface: AnyObject {
    /// The id JS uses in `context.configure({ canvas: "main" })`.
    var identifier: String { get }
    /// Size in pixels. Returned so JS can compute viewport and projection matrices.
    var pixelSize: CGSize { get }
    var configuredFormat: WGPUTextureFormat { get }

    func configure(_ configuration: WGPUCanvasConfiguration, device: MTLDevice) throws
    /// The drawable to draw this frame. Nil when the surface has no size yet, or drawables are exhausted.
    func nextDrawable() -> WGPUDrawable?

    /// The drawable size in pixels changed (`WebGPURuntime.resizeCanvas` — main thread).
    /// A screen surface updates the layer's `drawableSize`; an offscreen one resizes its backing texture.
    func updateDrawableSize(_ size: CGSize)

    /// Whether the surface has a drawable pool and **can fall behind**.
    ///
    /// The in-flight accounting itself lives outside the backend (`WGPUFrameCoordinator`) — the same
    /// policy is needed whether you use Dawn or Metal. All a surface answers is "am I subject to it?".
    /// An offscreen surface has no pool, so false.
    var pacesFrames: Bool { get }
}

public extension WGPUSurface {
    var pacesFrames: Bool { false }
    /// No-op by default — a fixed-size surface (a test double) has nothing to react to.
    func updateDrawableSize(_ size: CGSize) {}
}

// MARK: - Screen surface (CAMetalLayer)

final class WGPUMetalLayerDrawable: WGPUDrawable {
    private let drawable: CAMetalDrawable
    var texture: MTLTexture { drawable.texture }

    init(_ drawable: CAMetalDrawable) { self.drawable = drawable }

    func present(with commandBuffer: MTLCommandBuffer) {
        commandBuffer.present(drawable)
    }
}

/// A `CAMetalLayer`-backed surface. The `<webgpu-canvas>` element creates and registers it.
///
/// Threading: setting layer properties (`drawableSize`, `pixelFormat`) happens on the **main thread**,
/// while `nextDrawable()` happens on the **JS thread** interpreting commands. The size is read from a
/// lock-wrapped cache the main thread updates, keeping the render thread away from layer properties.
public final class WGPUMetalLayerSurface: WGPUSurface {
    public let identifier: String
    public let layer: CAMetalLayer

    /// When the drawable pool drains, `nextDrawable()` stalls the JS thread — so it is paced.
    /// The counting is done by `WGPUFrameCoordinator`.
    public var pacesFrames: Bool { true }

    private var cachedSize: CGSize = .zero
    private var format: WGPUTextureFormat = .bgra8unorm
    private let lock = NSLock()

    public init(identifier: String, layer: CAMetalLayer) {
        self.identifier = identifier
        self.layer = layer
        // The layer's initial properties are runtime policy — the element only hands the layer over
        // (`WebGPURuntime.attachCanvas`). Because applying configure is asynchronous (see configure
        // below), the defaults are set before the first frame — in the common case of
        // getPreferredCanvasFormat() (= bgra8unorm) they match from the very first frame. attachCanvas
        // is a main-thread contract, so synchronous setup is safe.
        layer.pixelFormat = .bgra8Unorm
        layer.framebufferOnly = true
    }

    public var pixelSize: CGSize {
        lock.lock()
        defer { lock.unlock() }
        return cachedSize
    }

    public var configuredFormat: WGPUTextureFormat {
        lock.lock()
        defer { lock.unlock() }
        return format
    }

    /// Called from the main thread when layout changes.
    public func updateDrawableSize(_ size: CGSize) {
        layer.drawableSize = size
        lock.lock()
        cachedSize = size
        lock.unlock()
    }

    public func configure(_ configuration: WGPUCanvasConfiguration, device: MTLDevice) throws {
        let pixelFormat = try WGPUMetalMapping.pixelFormat(configuration.format)
        lock.lock()
        format = configuration.format
        let size = layer.drawableSize
        cachedSize = size
        lock.unlock()

        // CAMetalLayer configuration is main-thread only. A `main.sync` from the JS thread deadlocks
        // the moment main is waiting on the Lynx runtime, so it is handed over **asynchronously**.
        // The requested format therefore applies from the next frame — `getCurrentTexture` reports the
        // actual drawable texture's format rather than the cached configuration (with bgra8unorm as the
        // default they usually agree from the first frame anyway).
        let apply = { [layer] in
            layer.device = device
            layer.pixelFormat = pixelFormat
            layer.isOpaque = configuration.alphaMode == .opaque
            layer.framebufferOnly = !configuration.usage.contains(.copySrc)

            // EDR — with `extended`, values above 1.0 go out as headroom brighter than SDR white.
            // The color space must move to extended **linear** at the same time. Setting only one of
            // the two clips the values or applies gamma twice. (The shader must emit linear values
            // with no sRGB encoding, and the format must be `rgba16float` to hold above 1.0 for it to actually brighten.)
            let extended = configuration.toneMappingMode == .extended
            layer.wantsExtendedDynamicRangeContent = extended
            let space: CFString
            switch (configuration.colorSpace, extended) {
            case (.srgb, false): space = CGColorSpace.sRGB
            case (.srgb, true): space = CGColorSpace.extendedLinearSRGB
            case (.displayP3, false): space = CGColorSpace.displayP3
            case (.displayP3, true): space = CGColorSpace.extendedLinearDisplayP3
            }
            layer.colorspace = CGColorSpace(name: space)
        }
        if Thread.isMainThread {
            apply()
        } else {
            DispatchQueue.main.async(execute: apply)
        }
    }

    public func nextDrawable() -> WGPUDrawable? {
        guard pixelSize.width > 0, pixelSize.height > 0 else { return nil }
        guard let drawable = layer.nextDrawable() else { return nil }
        return WGPUMetalLayerDrawable(drawable)
    }

}

// MARK: - Offscreen surface (test harness)

final class WGPUOffscreenDrawable: WGPUDrawable {
    let texture: MTLTexture
    init(texture: MTLTexture) { self.texture = texture }
    func present(with commandBuffer: MTLCommandBuffer) {}
}

/// A surface that draws into a texture with no screen.
///
/// It exists to **verify GPU results as pixel values** rather than by eye — instead of a simulator
/// screenshot, the render result is read back and asserted (`docs/TESTING.md` §4).
public final class WGPUOffscreenSurface: WGPUSurface {
    public let identifier: String
    public private(set) var texture: MTLTexture?
    public private(set) var size: CGSize
    private var format: WGPUTextureFormat = .rgba8unorm
    private let device: MTLDevice

    public init(identifier: String = "offscreen", size: CGSize, device: MTLDevice) {
        self.identifier = identifier
        self.size = size
        self.device = device
    }

    public var pixelSize: CGSize { size }
    public var configuredFormat: WGPUTextureFormat { format }

    public func configure(_ configuration: WGPUCanvasConfiguration, device: MTLDevice) throws {
        format = configuration.format
        try remakeTexture()
    }

    /// When the size changes the backing texture is rebuilt at the new size — the same meaning as a
    /// screen surface's `drawableSize` update (`canvasInfo`, `getCurrentTexture` and `readCanvasPixels`
    /// see the new size immediately). Before configure it only remembers the size — configure builds the texture.
    public func updateDrawableSize(_ size: CGSize) {
        guard size != self.size else { return }
        self.size = size
        guard texture != nil else { return }
        try? remakeTexture()
    }

    private func remakeTexture() throws {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: try WGPUMetalMapping.pixelFormat(format),
            width: Int(size.width),
            height: Int(size.height),
            mipmapped: false
        )
        descriptor.usage = [.renderTarget, .shaderRead]
        descriptor.storageMode = .private
        guard let texture = device.makeTexture(descriptor: descriptor) else {
            throw WGPUError.outOfMemory("offscreen surface texture creation failed")
        }
        texture.label = "webgpu.offscreen.\(identifier)"
        self.texture = texture
    }

    public func nextDrawable() -> WGPUDrawable? {
        guard let texture else { return nil }
        return WGPUOffscreenDrawable(texture: texture)
    }

    /// Reads the render result **in the format the surface was configured with**. GPU work must have finished before calling.
    ///
    /// It used to return only `Data`, assuming 4 bytes per pixel. On an `rgba16float` surface that
    /// yields bytes wrong in both length and interpretation **with no error**, so it now returns a
    /// `WGPUPixelReadback` carrying format, size and row stride together. One value comes out via
    /// `readback.rgba(x:y:)`.
    ///
    /// - Throws: `WGPUError.validation` if it has not been `configure`d yet, or the surface is a
    ///           depth/stencil format. A Metal blit cannot copy depth/stencil as one block without
    ///           naming an aspect, and in a mixed format such as `depth32float-stencil8` the bytes per
    ///           pixel do not exist as one contiguous block at all.
    public func readPixels(queue: MTLCommandQueue) throws -> WGPUPixelReadback {
        guard let texture else {
            throw WGPUError.validation("the surface has not been configured yet")
        }
        guard !format.isDepthOrStencil else {
            throw WGPUError.validation(
                "a \(format.rawValue) surface cannot be read with readPixels — depth/stencil needs a per-aspect copy"
            )
        }
        let bytesPerRow = texture.width * format.bytesPerPixel
        let length = bytesPerRow * texture.height
        guard let staging = device.makeBuffer(length: length, options: .storageModeShared),
              let commandBuffer = queue.makeCommandBuffer(),
              let blit = commandBuffer.makeBlitCommandEncoder() else {
            throw WGPUError.backend("blit creation for pixel readback failed")
        }
        blit.copy(
            from: texture,
            sourceSlice: 0,
            sourceLevel: 0,
            sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
            sourceSize: MTLSize(width: texture.width, height: texture.height, depth: 1),
            to: staging,
            destinationOffset: 0,
            destinationBytesPerRow: bytesPerRow,
            destinationBytesPerImage: length
        )
        blit.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        return WGPUPixelReadback(
            data: Data(bytes: staging.contents(), count: length),
            format: format,
            width: texture.width,
            height: texture.height,
            bytesPerRow: bytesPerRow
        )
    }
}
