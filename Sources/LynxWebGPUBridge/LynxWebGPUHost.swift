#if canImport(Lynx)
import Foundation
import UIKit
import QuartzCore
import Lynx
import LynxWebGPUCore

/// The WebGPU runtime attached to one LynxView.
///
/// It is the shared seam between the native module (`NativeModules.WebGPU`) and the
/// `<webgpu-canvas>` element. Lynx builds custom UI with `[[cls alloc] init]` directly, so
/// constructor injection is impossible — instead the host registers itself on the LynxView (the UI
/// tree's rootView) and the element retrieves it (`LynxUIContext.rootView`, weak on both sides).
///
/// **The app supplies the runtime.** This bridge knows only the `WebGPURuntime` protocol and does not
/// import the Metal engine (`LynxWebGPU`) — the same reason this package does not pull in the Lynx
/// SDK (`docs/LYNX-INTEGRATION.md` §1). So swapping to another backend such as Dawn leaves **both the
/// bridge and the JS bundle untouched.**
///
/// ```swift
/// import LynxWebGPU                                            // only when using the default engine
/// let host = LynxWebGPUHost(runtime: try LynxWebGPUContext())
/// ```
public final class LynxWebGPUHost: NSObject {
    public let runtime: WebGPURuntime
    private weak var lynxView: LynxView?
    private let ticker = WebGPUFrameTicker()

    /// Where the name from JS's `loadAsset(name)` is resolved into bytes. Swap it and the app decides
    /// the resolution rules and the access scope (see `WGPUAssetProvider`).
    ///
    /// The default allows any path — **narrow it whenever the bundle (JS) cannot be trusted**:
    /// ```swift
    /// host.assetProvider = WGPUFileAssetProvider(
    ///     allowedRoots: [FileManager.default.temporaryDirectory]
    /// )
    /// ```
    public var assetProvider: WGPUAssetProvider = WGPUFileAssetProvider()

    public init(runtime: WebGPURuntime) {
        self.runtime = runtime
        super.init()
    }

    /// Call once after the LynxView is created. Needed for frame events and `<webgpu-canvas>` wiring.
    public func attach(to lynxView: LynxView) {
        self.lynxView = lynxView
        LynxWebGPUHostRegistry.register(self, for: lynxView)
        ticker.onFrame = { [weak self] timestamp, deltaSeconds in
            guard let self else { return }
            // The pump goes **before** the readiness gate — on a runtime whose completion notices come
            // from the pump (Dawn's processEvents), putting it after the gate means saturation never clears.
            self.runtime.processEvents()
            // When the GPU is behind by the in-flight limit, skip this tick. Sending the event here
            // would have JS build a frame and stall **the entire JS thread** at nextDrawable() —
            // dropping the frame is better. Once a completion returns it resumes from the next tick.
            guard self.runtime.isReadyForNextFrame else { return }
            self.lynxView?.sendGlobalEvent("webgpu:frame", withParams: [[
                "timestamp": timestamp * 1000,
                "delta": deltaSeconds * 1000,
            ]])
        }
    }

    /// Call when leaving the page — stops the display link and discards GPU objects.
    public func detach() {
        ticker.stop()
        runtime.reset()
        lynxView = nil
    }

    // MARK: - Frame loop

    /// Sends the `webgpu:frame` global event in step with the display refresh.
    ///
    /// Driving frames from JS's `setInterval` drifts out of step with refresh and frames bunch up or
    /// get dropped. A CADisplayLink drives them far more evenly (`docs/JS-AUTHORING.md` §4).
    ///
    /// **The normal path is JS** — when a bundle calls `startFrameLoop(handler)`, the shim routes
    /// through `NativeModules.WebGPU.startFrameLoop` to here. `attach(to:)` only wires the tick
    /// callback and **does not start the loop** (so the display link does not spin on a page that
    /// never draws frames — `docs/LYNX-INTEGRATION.md` §3).
    ///
    /// It is left open for the host to call directly to support **driving frames from outside JS**
    /// (native code building the command stream itself, or diagnosing the JS path). Using both paths
    /// does not double the ticks — there is one link, and `start` replaces the existing one.
    public func startFrameLoop(preferredFramesPerSecond: Int = 60) {
        ticker.start(preferredFramesPerSecond: preferredFramesPerSecond)
    }

    /// Stops the frame loop. `detach()` already calls it, so leaving a page needs no separate call.
    public func stopFrameLoop() {
        ticker.stop()
    }

    // MARK: - Canvas registration

    /// `<webgpu-canvas>` hands over **only the layer** — the runtime picks the surface type.
    /// That is what leaves the element code unchanged when the backend is swapped.
    func attachCanvas(identifier: String, layer: CAMetalLayer) {
        runtime.attachCanvas(identifier: identifier, layer: layer)
    }

    func resizeCanvas(identifier: String, drawableSize: CGSize) {
        runtime.resizeCanvas(identifier: identifier, drawableSize: drawableSize)
    }

    func detachCanvas(identifier: String) {
        runtime.detachCanvas(identifier: identifier)
    }
}

/// rootView → host mapping. The entry disappears with the view.
enum LynxWebGPUHostRegistry {
    private static let table = NSMapTable<UIView, LynxWebGPUHost>.weakToWeakObjects()
    private static let lock = NSLock()

    static func register(_ host: LynxWebGPUHost, for rootView: UIView) {
        lock.lock()
        table.setObject(host, forKey: rootView)
        lock.unlock()
    }

    static func host(for rootView: UIView?) -> LynxWebGPUHost? {
        guard let rootView else { return nil }
        lock.lock()
        defer { lock.unlock() }
        return table.object(forKey: rootView)
    }
}

/// What this package registers with Lynx.
public enum LynxWebGPU {
    /// Registers the native module and the custom element with LynxConfig.
    ///
    /// ```swift
    /// let host = LynxWebGPUHost(runtime: try LynxWebGPUContext())   // the app picks the runtime
    /// let lynxView = LynxView { builder in
    ///     let config = LynxConfig(provider: provider)
    ///     LynxWebGPU.register(in: config, host: host)
    ///     builder.config = config
    /// }
    /// host.attach(to: lynxView)   // needed for global events and canvas wiring
    /// ```
    public static func register(in config: LynxConfig, host: LynxWebGPUHost) {
        config.register(WebGPUNativeModule.self, param: host)
        config.registerUI(WebGPUCanvasUI.self, withName: elementName)
    }

    /// The tag name used from JS.
    public static let elementName = "webgpu-canvas"
    /// The module name used from JS (`NativeModules.WebGPU`).
    public static let moduleName = WebGPUNativeModule.name
}
#endif
