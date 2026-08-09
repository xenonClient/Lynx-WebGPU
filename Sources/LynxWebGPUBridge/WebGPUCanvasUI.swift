#if canImport(Lynx)
import Foundation
import UIKit
import QuartzCore
import Lynx
import LynxWebGPUCore

/// A view whose backing layer is a `CAMetalLayer`.
///
/// Overriding `layerClass` makes the view itself the swapchain with no sublayer — one layer fewer
/// also means one compositing step fewer.
public final class WebGPUCanvasView: UIView {
    public override class var layerClass: AnyClass { CAMetalLayer.self }

    var metalLayer: CAMetalLayer { layer as! CAMetalLayer }
    /// Reports that the drawable size in pixels changed.
    var onDrawableSizeChange: ((CGSize) -> Void)?
    /// CSS px → pixel scale. Uses the screen scale when unspecified.
    var pixelRatioOverride: CGFloat?
    /// Whether this view is transparent to UIKit hit testing (the `passthrough-touches` prop).
    var passthroughTouches = false

    private var lastReportedSize: CGSize = .zero

    public override init(frame: CGRect) {
        super.init(frame: frame)
        isOpaque = true
        // The layer's initial properties, such as pixel format, are not touched here — that is
        // **runtime policy** (`WebGPURuntime.attachCanvas`). Only by handing over the layer alone does
        // this code stay genuinely unchanged when the backend is swapped.
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }

    var pixelRatio: CGFloat {
        pixelRatioOverride ?? window?.screen.scale ?? traitCollection.displayScale
    }

    /// In passthrough mode it is invisible to UIKit hit testing — gesture recognizers on native views
    /// **beneath** the canvas (a `<scroll-view>`'s UIScrollView, say) receive the touch.
    ///
    /// Lynx events (`bindtouchstart` and the rest) are unaffected: Lynx's touch recognizer is attached
    /// to **the rootView (LynxView)**, not to individual views (`LynxEventHandler.attachContainerView`),
    /// and the target is decided by Lynx's own hitTest (`LynxEventTarget`) rather than UIKit.
    public override func hitTest(_ point: CGPoint, with event: UIEvent?) -> UIView? {
        passthroughTouches ? nil : super.hitTest(point, with: event)
    }

    public override func layoutSubviews() {
        super.layoutSubviews()
        let scale = pixelRatio
        let size = CGSize(
            width: (bounds.width * scale).rounded(),
            height: (bounds.height * scale).rounded()
        )
        guard size.width > 0, size.height > 0, size != lastReportedSize else { return }
        lastReportedSize = size
        onDrawableSizeChange?(size)
    }
}

/// `<webgpu-canvas>` — the screen surface WebGPU draws into.
///
/// props: `canvas-id` (the name JS uses in `configure({canvas})`, required), `pixel-ratio` (force a scale)
/// events: `bindcanvasresize` (detail: width/height/pixelRatio — when the pixel size changes)
/// UI methods: `getInfo()` → `{width, height, pixelRatio}`
///
/// Registration and removal follow **prop lifetime**, not view lifetime — it registers when
/// `canvas-id` is settled and removes when that changes or the element goes away.
public final class WebGPUCanvasUI: LynxUI<WebGPUCanvasView> {
    private var canvasIdentifier: String?
    private var pendingPixelRatio: CGFloat?
    private var propsDirty = false

    public override func createView() -> WebGPUCanvasView? {
        let view = WebGPUCanvasView()
        view.onDrawableSizeChange = { [weak self] size in
            guard let self else { return }
            if let canvasIdentifier = self.canvasIdentifier {
                self.host?.resizeCanvas(identifier: canvasIdentifier, drawableSize: size)
            }
            self.emit("canvasresize", detail: [
                "width": Int(size.width),
                "height": Int(size.height),
                "pixelRatio": Double(view.pixelRatio),
            ])
        }
        return view
    }

    deinit {
        // deinit can be on any thread, but the removal itself is protected by the runtime's lock.
        if let canvasIdentifier {
            host?.detachCanvas(identifier: canvasIdentifier)
        }
    }

    private var host: LynxWebGPUHost? {
        LynxWebGPUHostRegistry.host(for: context?.rootView)
    }

    // MARK: - Props

    @objc(__lynx_prop_config__webgpuCanvasId)
    public static func propConfigCanvasId() -> [String] { ["canvas-id", "setCanvasId", "NSString*"] }

    @objc(setCanvasId:requestReset:)
    public func setCanvasId(_ value: NSString?, requestReset: Bool) {
        let next = requestReset ? nil : (value as String?)
        guard next != canvasIdentifier else { return }
        if let canvasIdentifier { host?.detachCanvas(identifier: canvasIdentifier) }
        canvasIdentifier = next
        propsDirty = true
    }

    @objc(__lynx_prop_config__webgpuCanvasPixelRatio)
    public static func propConfigPixelRatio() -> [String] { ["pixel-ratio", "setPixelRatio", "CGFloat"] }

    @objc(setPixelRatio:requestReset:)
    public func setPixelRatio(_ value: CGFloat, requestReset: Bool) {
        pendingPixelRatio = requestReset || value <= 0 ? nil : value
        propsDirty = true
    }

    @objc(__lynx_prop_config__webgpuCanvasPassthroughTouches)
    public static func propConfigPassthroughTouches() -> [String] {
        ["passthrough-touches", "setPassthroughTouches", "BOOL"]
    }

    /// UIKit touch passthrough (off by default — the canvas covers what is beneath, as on the web).
    ///
    /// Turning it on lets native gestures **behind** the canvas through (a sibling `<scroll-view>`'s
    /// scrolling, say). The canvas's own Lynx events keep arriving — if the passed-through gesture
    /// wins, Lynx sends `touchcancel`, following the same contention rules as any other element.
    @objc(setPassthroughTouches:requestReset:)
    public func setPassthroughTouches(_ value: Bool, requestReset: Bool) {
        view().passthroughTouches = requestReset ? false : value
    }

    public override func propsDidUpdate() {
        super.propsDidUpdate()
        guard propsDirty else { return }
        propsDirty = false

        let canvasView = view()
        canvasView.pixelRatioOverride = pendingPixelRatio
        canvasView.setNeedsLayout()

        guard let canvasIdentifier, let host else { return }
        // Hand over only the layer — the runtime picks the surface type (`WebGPURuntime.attachCanvas`).
        host.attachCanvas(identifier: canvasIdentifier, layer: canvasView.metalLayer)
        host.resizeCanvas(identifier: canvasIdentifier, drawableSize: CGSize(
            width: (canvasView.bounds.width * canvasView.pixelRatio).rounded(),
            height: (canvasView.bounds.height * canvasView.pixelRatio).rounded()
        ))
        WGPULog.canvas.info("<webgpu-canvas> registered — \(canvasIdentifier, privacy: .public)")
    }

    // MARK: - UI methods

    @objc(__lynx_ui_method_config__webgpuCanvasGetInfo)
    public static func uiMethodConfigGetInfo() -> String { "getInfo" }

    /// Returns the current drawable size (a fallback for when `bindcanvasresize` was missed).
    @objc(getInfo:withResult:)
    public func getInfo(_ params: [AnyHashable: Any]?, withResult callback: LynxUIMethodCallbackBlock?) {
        let canvasView = view()
        let scale = canvasView.pixelRatio
        callback?(Int32(kUIMethodSuccess.rawValue), [
            "canvasId": canvasIdentifier ?? "",
            "width": Int((canvasView.bounds.width * scale).rounded()),
            "height": Int((canvasView.bounds.height * scale).rounded()),
            "pixelRatio": Double(scale),
        ])
    }

    // MARK: - Events

    private func emit(_ name: String, detail: [String: Any]) {
        guard let emitter = context?.eventEmitter else { return }
        emitter.send(LynxCustomEvent(name: name, targetSign: sign, params: detail))
    }
}
#endif
