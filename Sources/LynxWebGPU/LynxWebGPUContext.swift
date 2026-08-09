import Foundation
import CoreGraphics
import Metal
import QuartzCore
import LynxWebGPUCore
import LynxWebGPUShader

/// The **direct Metal implementation** of `WebGPURuntime` — what this package ships by default.
///
/// In substance it is `WGPUMetalBackend` (encoding) mounted on `WGPUBackendEngine` (Core —
/// orchestration). This type builds and binds that pair and exposes the Metal-specific surface a host
/// uses (device, queue, registering a custom `WGPUSurface`). A host app can create this object and
/// hand it to the Lynx bridge (`LynxWebGPUHost(runtime:)`), or run the command stream straight from
/// Swift without Lynx at all (which is how the test harness uses it).
///
/// Swapping in a different implementation is covered in `WebGPURuntime`; swapping only the backend is
/// covered in `WGPUBackend`.
///
/// **Threading** — serialization is the engine's responsibility (see `WGPUBackendEngine`). Surface
/// registration and removal are safe from the main thread or any other (the engine's registry lock).
public final class LynxWebGPUContext: WebGPURuntime {
    public let device: MTLDevice
    public let queue: MTLCommandQueue

    private let backend: WGPUMetalBackend
    private let engine: WGPUBackendEngine<WGPUMetalBackend>

    /// In-flight frame accounting — **backend-independent policy**, so it lives in Core (see
    /// `WGPUFrameCoordinator`). Exposed so a host driving the frame ticker its own way can inspect it.
    public var frameCoordinator: WGPUFrameCoordinator { engine.frameCoordinator }

    public init(device: MTLDevice? = nil, frameCoordinator: WGPUFrameCoordinator = WGPUFrameCoordinator()) throws {
        guard let resolved = device ?? MTLCreateSystemDefaultDevice() else {
            throw WGPUError.backend("cannot create a Metal device (check simulator/device support)")
        }
        guard let queue = resolved.makeCommandQueue() else {
            throw WGPUError.backend("MTLCommandQueue creation failed")
        }
        self.device = resolved
        self.queue = queue
        self.backend = WGPUMetalBackend(device: resolved, queue: queue)
        self.engine = WGPUBackendEngine(backend: backend, frameCoordinator: frameCoordinator)
        WGPULog.device.info("WebGPU context started — \(resolved.name, privacy: .public)")
    }

    // MARK: - Canvas (WebGPURuntime)

    /// Attaches the `CAMetalLayer` of a `<webgpu-canvas>` as a surface.
    ///
    /// The point is that **the runtime** picks the surface type — the bridge only hands over the layer,
    /// so swapping the runtime leaves the element code untouched (see `WebGPURuntime.attachCanvas`).
    public func attachCanvas(identifier: String, layer: CAMetalLayer) {
        engine.attachCanvas(identifier: identifier, layer: layer)
    }

    public func attachOffscreenCanvas(identifier: String, size: CGSize) throws {
        try engine.attachOffscreenCanvas(identifier: identifier, size: size)
    }

    public func resizeCanvas(identifier: String, drawableSize: CGSize) {
        engine.resizeCanvas(identifier: identifier, drawableSize: drawableSize)
    }

    public func detachCanvas(identifier: String) {
        engine.detachCanvas(identifier: identifier)
    }

    public func canvasInfo(identifier: String) -> [String: Any] {
        engine.canvasInfo(identifier: identifier)
    }

    public func readCanvasPixels(identifier: String) throws -> WGPUPixelReadback {
        try engine.readCanvasPixels(identifier: identifier)
    }

    // MARK: - Surface registration (Metal-specific surfaces)

    /// Registers a custom `WGPUSurface` implementation (a test double, say) directly.
    public func registerSurface(_ surface: WGPUSurface) {
        engine.registerSurface(surface, identifier: surface.identifier, pacesFrames: surface.pacesFrames)
    }

    public func unregisterSurface(identifier: String) {
        engine.detachCanvas(identifier: identifier)
    }

    public func surface(for identifier: String) -> WGPUSurface? {
        engine.surface(for: identifier)
    }

    public var registeredSurfaceIdentifiers: [String] {
        engine.registeredSurfaceIdentifiers
    }

    /// Whether every registered surface can take a new frame — `frameCoordinator` does the accounting.
    public var isReadyForNextFrame: Bool { engine.isReadyForNextFrame }

    public func processEvents() {
        engine.processEvents()
    }

    // MARK: - Command execution

    /// Runs one command stream.
    ///
    /// - Parameter payload: `{"commands": [ {op: …}, … ]}`
    /// - Returns: `{"ok": Bool, "errors": [...], "canvases": {...}}` — passed straight back to JS.
    public func execute(_ payload: [String: Any]) -> [String: Any] {
        engine.execute(payload)
    }

    /// Reads buffer contents (corresponding to `GPUBuffer.mapAsync` + `getMappedRange`).
    public func readBuffer(
        handle: Int,
        offset: Int = 0,
        size: Int? = nil,
        completion: @escaping ([String: Any]) -> Void
    ) {
        engine.readBuffer(handle: handle, offset: offset, size: size, completion: completion)
    }

    /// Decodes an encoded image and registers it as the object standing in for `ImageBitmap` (JS `createImageBitmap`).
    public func decodeImage(
        handle: Int,
        data: Data?,
        name: String?,
        options: WGPUImageDecodeOptions,
        provider: WGPUAssetProvider?,
        completion: @escaping ([String: Any]) -> Void
    ) {
        engine.decodeImage(
            handle: handle, data: data, name: name,
            options: options, provider: provider, completion: completion
        )
    }

    /// `GPUShaderModule.getCompilationInfo()` — that module's compilation diagnostics.
    public func shaderCompilationInfo(handle: Int) -> [String: Any] {
        engine.shaderCompilationInfo(handle: handle)
    }

    /// The adapter info and limits `navigator.gpu.requestAdapter()` returns (`WGPUMetalBackend`).
    public func adapterInfo() -> [String: Any] {
        engine.adapterInfo()
    }

    /// Discards every GPU object (leaving the page, and so on).
    public func reset() {
        engine.reset()
    }

    public var liveObjectCount: Int { engine.liveObjectCount }

    /// Test observation hook — the upload staging pool.
    var stagingPool: WGPUStagingPool { backend.stagingPool }
}
