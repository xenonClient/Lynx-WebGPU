#if canImport(Lynx)
import Foundation
import UIKit
import QuartzCore
import Lynx
import LynxWebGPUCore

/// `CAMetalLayer`를 백킹 레이어로 쓰는 뷰.
///
/// `layerClass`를 바꿔 서브레이어 없이 뷰 자체가 스왑체인이 되게 한다 — 레이어 하나가 줄면
/// 합성 단계도 한 번 줄어든다.
public final class WebGPUCanvasView: UIView {
    public override class var layerClass: AnyClass { CAMetalLayer.self }

    var metalLayer: CAMetalLayer { layer as! CAMetalLayer }
    /// 드로어블 크기(픽셀)가 바뀌면 알린다.
    var onDrawableSizeChange: ((CGSize) -> Void)?
    /// CSS px → 픽셀 배율. 지정하지 않으면 화면 배율을 쓴다.
    var pixelRatioOverride: CGFloat?
    /// UIKit 히트 테스트에서 이 뷰를 투명하게 만들지 (`passthrough-touches` prop).
    var passthroughTouches = false

    private var lastReportedSize: CGSize = .zero

    public override init(frame: CGRect) {
        super.init(frame: frame)
        isOpaque = true
        // 레이어의 픽셀 포맷 등 초기 속성은 여기서 건드리지 않는다 — 그건 **런타임 정책**이다
        // (`WebGPURuntime.attachCanvas`). 엘리먼트는 레이어를 넘기기만 해야 백엔드를
        // 갈아끼워도 이 코드가 정말로 무변경이 된다.
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }

    var pixelRatio: CGFloat {
        pixelRatioOverride ?? window?.screen.scale ?? traitCollection.displayScale
    }

    /// 통과 모드에서는 UIKit 히트 테스트에 잡히지 않는다 — 캔버스 **아래** 네이티브 뷰
    /// (`<scroll-view>`의 UIScrollView 등)에 붙은 제스처 인식기가 터치를 받는다.
    ///
    /// Lynx 이벤트(`bindtouchstart` 등)는 영향을 받지 않는다: Lynx의 터치 인식기는
    /// 개별 뷰가 아니라 **rootView(LynxView)에 붙어 있고**(`LynxEventHandler.attachContainerView`),
    /// 타깃 결정도 UIKit이 아니라 Lynx 자체 hitTest(`LynxEventTarget`)가 한다.
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

/// `<webgpu-canvas>` — WebGPU가 그리는 화면 표면.
///
/// props: `canvas-id`(JS가 `configure({canvas})`에서 쓰는 이름, 필수) `pixel-ratio`(배율 강제)
/// events: `bindcanvasresize`(detail: width/height/pixelRatio — 픽셀 크기가 바뀔 때)
/// UI 메서드: `getInfo()` → `{width, height, pixelRatio}`
///
/// 등록/해제는 뷰 수명이 아니라 **prop 수명**을 따른다 — `canvas-id`가 정해지는 시점에 등록하고,
/// 바뀌거나 엘리먼트가 사라질 때 해제한다.
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
        // deinit은 임의 스레드일 수 있으나, 해제 자체는 런타임의 락으로 보호된다.
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

    /// UIKit 터치 통과 (기본 꺼짐 — 웹처럼 캔버스가 아래를 가린다).
    ///
    /// 켜면 캔버스 **뒤**의 네이티브 제스처(형제 `<scroll-view>`의 스크롤 등)가 통과한다.
    /// 캔버스 자신의 Lynx 이벤트는 계속 온다 — 통과한 제스처가 이기면 Lynx가
    /// `touchcancel`을 보내는, 다른 엘리먼트와 같은 경쟁 규칙을 따른다.
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
        // 레이어만 넘긴다 — 표면 타입은 런타임이 고른다 (`WebGPURuntime.attachCanvas`).
        host.attachCanvas(identifier: canvasIdentifier, layer: canvasView.metalLayer)
        host.resizeCanvas(identifier: canvasIdentifier, drawableSize: CGSize(
            width: (canvasView.bounds.width * canvasView.pixelRatio).rounded(),
            height: (canvasView.bounds.height * canvasView.pixelRatio).rounded()
        ))
        WGPULog.canvas.info("<webgpu-canvas> 등록 — \(canvasIdentifier, privacy: .public)")
    }

    // MARK: - UI 메서드

    @objc(__lynx_ui_method_config__webgpuCanvasGetInfo)
    public static func uiMethodConfigGetInfo() -> String { "getInfo" }

    /// 현재 드로어블 크기를 돌려준다 (`bindcanvasresize`를 놓쳤을 때의 폴백).
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

    // MARK: - 이벤트

    private func emit(_ name: String, detail: [String: Any]) {
        guard let emitter = context?.eventEmitter else { return }
        emitter.send(LynxCustomEvent(name: name, targetSign: sign, params: detail))
    }
}
#endif
