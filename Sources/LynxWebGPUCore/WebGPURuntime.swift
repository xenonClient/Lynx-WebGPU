import Foundation
import CoreGraphics
import QuartzCore

/// `createImageBitmap()` decode options (a subset of the spec's `ImageBitmapOptions`).
///
/// It holds values only, so it is backend-independent — which is why it lives here rather than
/// with the decoder (`WGPUImageDecoder`, ImageIO). `WGPUImageDecoder.Options` aliases this type.
public struct WGPUImageDecodeOptions {
    /// Flips top to bottom. Web images have a top-left origin, and depending on the texture
    /// coordinate system a flip is sometimes needed (`copyExternalImageToTexture`'s `flipY`).
    public var flipY: Bool
    /// Whether to premultiply alpha into the color. The spec's `premultiplyAlpha`
    /// `'premultiply'`/`'none'`. `'default'` is source-defined; the ImageIO path treats
    /// unpremultiplied as the baseline.
    public var premultiplyAlpha: Bool
    /// Target size (`resizeWidth`/`resizeHeight`). nil keeps the original size.
    public var resize: (width: Int, height: Int)?

    public init(flipY: Bool = false, premultiplyAlpha: Bool = false,
                resize: (width: Int, height: Int)? = nil) {
        self.flipY = flipY
        self.premultiplyAlpha = premultiplyAlpha
        self.resize = resize
    }
}

/// The **far side** of the command stream — whatever receives commands and drives the GPU.
///
/// ## Why a protocol
///
/// The command stream (`docs/COMMAND-STREAM.md`) is pure data. An array of `{op: …}` goes in and
/// `{ok, errors, canvases, errorScopes}` comes out — what runs in between is not part of the
/// contract. That is what makes the implementation swappable, and this protocol is that seam.
///
/// The implementation this package ships by default is `LynxWebGPU`'s `LynxWebGPUContext` (Metal
/// directly). Another implementation — a runtime on top of [Dawn](https://github.com/google/dawn),
/// say — is built and injected **from outside this package**, for the same reason the Lynx SDK is
/// not pulled in here (`docs/LYNX-INTEGRATION.md` §1): the **app** decides version, distribution
/// and binary size.
///
/// ```swift
/// let host = LynxWebGPUHost(runtime: try LynxWebGPUContext())   // default (Metal)
/// let host = LynxWebGPUHost(runtime: try DawnWebGPURuntime())   // an app-supplied implementation
/// ```
///
/// The bridge (`LynxWebGPUBridge`) sees only this protocol. **The JS bundle is untouched either
/// way** — preserving that property is the reason this boundary exists.
///
/// ## Threading and lifetime contract
///
/// What an implementation must honour — things that live only in prose and no compiler catches, so
/// they are pinned down here (the same content as `docs/COMMAND-STREAM.md` §5-1).
///
/// - `execute` is called on Lynx's **JS background thread**. The implementation owns serialization.
/// - `attachCanvas`, `attachOffscreenCanvas` and `resizeCanvas` come from the **main thread**
///   (UI layout). That is what makes setting layer properties synchronously inside attachCanvas safe.
/// - `detachCanvas` can come from **any thread** — the element's deinit calls it. Protect the
///   surface registry with a lock (it can be entered concurrently with execute).
/// - `reset` comes from the main thread (page exit, hot reload) and can race with execute.
/// - The completion callbacks of `readBuffer` and `decodeImage` may be called **from any thread,
///   even synchronously** — for already-finished work, arriving immediately on the calling thread
///   is not a contract violation. Callers are written to handle that.
/// - Applying the command stream's `configureCanvas` to the layer **may be asynchronous**
///   (CAMetalLayer properties are main-thread only, so the JS thread must hand them over —
///   `main.sync` deadlocks). The first frame may go out with the previous settings, so
///   `getCurrentTexture` must report **the actual drawable's format**, not a cached one.
/// - `isReadyForNextFrame` and `processEvents` are called every tick on the main thread (the frame
///   ticker) — they must be cheap and non-blocking. **They are called concurrently with `execute`**
///   — if the backend API is not thread-safe (Dawn), the implementation must serialize them under
///   the same lock as execute. Skip that and it blows up not as a validation error but as an
///   internal backend assertion (process death) — the conformance check `pump-concurrency` judges
///   this contract. The way to honour both requirements (non-blocking and serialized) is to **only
///   try the lock**: if it is held, skip the pump for that tick. Waiting instead ties the main
///   thread to a whole batch encode and the UI hitches. Why skipping is safe is explained in the
///   `processEvents` docs below.
///
/// ## Asynchronous error delivery
///
/// Errors caught inside a batch come back in that batch's `errors` (or `errorScopes` for scoped
/// ones). **Failures that surface after the batch** (GPU execution failure, Dawn's uncaptured
/// error) have no callback channel — they are collected and **ride out in the next batch result's
/// `errors`.** `WGPUDeferredErrorQueue` is that place.
public protocol WebGPURuntime: AnyObject {

    // MARK: - Command stream

    /// Runs one batch.
    ///
    /// - Parameter payload: `{"commands": [{op: …}, …], "present": Bool}`.
    ///   `present: false` means an internal submit from the **middle** of a frame — commit, but
    ///   defer the drawable present and frame-scoped handle expiry to the real frame submit to come.
    /// - Returns: `{"ok", "commandCount", "objects"}` always, plus `{"errors", "canvases",
    ///   "errorScopes"}` only when non-empty. It goes straight back to JS — the shape is fixed by
    ///   `docs/COMMAND-STREAM.md` §2, and assembly must go through `WGPUBatchResult` (so key
    ///   spelling and omission rules are not re-derived per backend).
    func execute(_ payload: [String: Any]) -> [String: Any]

    // MARK: - Queries

    /// Adapter info, limits and features used by `navigator.gpu.requestAdapter()`.
    /// Keys must use **the spec spelling exactly** — web libraries read them by those names.
    func adapterInfo() -> [String: Any]

    /// `GPUShaderModule.getCompilationInfo()`.
    func shaderCompilationInfo(handle: Int) -> [String: Any]

    /// The canvas's current pixel size and format (`{ok, width, height, format}`).
    func canvasInfo(identifier: String) -> [String: Any]

    // MARK: - Asynchronous

    /// `GPUBuffer.mapAsync` + `getMappedRange` — returns the contents once prior GPU work finishes.
    func readBuffer(handle: Int, offset: Int, size: Int?, completion: @escaping ([String: Any]) -> Void)

    /// `createImageBitmap()` — decodes an image and registers it under a **JS-issued handle**.
    ///
    /// - Parameter data: image bytes. When nil, `name` is resolved through `provider`.
    func decodeImage(
        handle: Int,
        data: Data?,
        name: String?,
        options: WGPUImageDecodeOptions,
        provider: WGPUAssetProvider?,
        completion: @escaping ([String: Any]) -> Void
    )

    // MARK: - Canvas

    /// Attaches an on-screen surface. JS names it with `configure({canvas: identifier})`.
    ///
    /// **`CAMetalLayer` is the common denominator of both backends** — on Apple platforms Dawn also
    /// takes the same layer through `WGPUSurfaceSourceMetalLayer`. That is why the `<webgpu-canvas>`
    /// element's code is identical whichever runtime is in use.
    ///
    /// The element **only hands over** the layer — initial properties such as `pixelFormat` are the
    /// runtime's call. This call arrives on the main thread (see above), so synchronous setup is safe.
    func attachCanvas(identifier: String, layer: CAMetalLayer)

    /// Attaches a surface that draws without a screen.
    ///
    /// This is the conformance suite's **channel for reading render results as pixels** (paired with
    /// `readCanvasPixels`). Two runtimes must be mechanically checkable for drawing the same picture
    /// in an environment with no display.
    func attachOffscreenCanvas(identifier: String, size: CGSize) throws

    /// Updates the drawable size in pixels. Called from the **main thread** when layout changes.
    func resizeCanvas(identifier: String, drawableSize: CGSize)

    func detachCanvas(identifier: String)

    /// Reads an offscreen surface's pixels **in the configured format**. GPU work must have finished.
    func readCanvasPixels(identifier: String) throws -> WGPUPixelReadback

    // MARK: - Frames

    /// Whether every registered surface can take a new frame.
    ///
    /// The frame ticker consults this and skips a tick when saturated — if JS builds a frame while
    /// the GPU is behind, acquiring a drawable stalls **the entire JS thread**.
    var isReadyForNextFrame: Bool { get }

    /// Asynchronous completion pump — the seat for backends that require explicit event processing
    /// (Dawn's `wgpuInstanceProcessEvents` goes here).
    ///
    /// The frame ticker calls it every display period **even when skipping the tick** — on a runtime
    /// whose completions come from the pump, clearing saturation depends on the pump itself, and
    /// `mapAsync`-style completions must not starve while JS is idle. The conformance harness also
    /// calls it periodically while waiting on a callback.
    ///
    /// Never depend on this call for correctness — it narrows the latency bound while a frame loop
    /// runs. Completions must still arrive in a configuration with no ticker (an app that never
    /// starts the frame loop), so a runtime that needs pumping must have its own waiting mechanism
    /// (a dedicated thread, spontaneous callbacks).
    func processEvents()

    // MARK: - Lifetime

    /// Discards every GPU object (page exit, hot reload).
    func reset()
}

public extension WebGPURuntime {
    /// No-op by default — a backend whose completions arrive on their own (Metal's completion
    /// handler) has nothing to pump. (It is paired with the protocol requirement: living only in the
    /// extension would bind protocol-typed calls here and never reach an implementation's pump.)
    func processEvents() {}

    /// Convenience overload — reads the whole buffer.
    func readBuffer(handle: Int, completion: @escaping ([String: Any]) -> Void) {
        readBuffer(handle: handle, offset: 0, size: nil, completion: completion)
    }

    /// Convenience overload — passes a command array straight through (test harnesses, native-only use).
    @discardableResult
    func execute(commands: [[String: Any]], present: Bool = true) -> [String: Any] {
        execute(["commands": commands, "present": present])
    }
}
