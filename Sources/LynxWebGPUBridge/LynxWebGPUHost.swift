#if canImport(Lynx)
import Foundation
import UIKit
import QuartzCore
import Lynx
import LynxWebGPUCore

/// LynxView 하나에 붙는 WebGPU 런타임.
///
/// 네이티브 모듈(`NativeModules.WebGPU`)과 `<webgpu-canvas>` 엘리먼트가 공유하는 접점이다.
/// Lynx는 커스텀 UI를 `[[cls alloc] init]`으로 직접 만들기 때문에 생성자 주입을 할 수 없다 —
/// 대신 호스트가 LynxView(= UI 트리의 rootView)에 자신을 등록해 두고 엘리먼트가 되찾는다
/// (`LynxUIContext.rootView`, 양쪽 모두 약한 참조).
///
/// **런타임은 앱이 넣는다.** 이 브리지는 `WebGPURuntime` 프로토콜만 알고, Metal 엔진
/// (`LynxWebGPU`)을 import하지 않는다 — Lynx SDK를 이 패키지가 가져오지 않는 것과 같은
/// 이유다 (`docs/LYNX-INTEGRATION.md` §1). 그래서 Dawn 같은 다른 백엔드로 갈아끼울 때
/// **브리지도 JS 번들도 손대지 않는다.**
///
/// ```swift
/// import LynxWebGPU                                            // 기본 엔진을 쓸 때만
/// let host = LynxWebGPUHost(runtime: try LynxWebGPUContext())
/// ```
public final class LynxWebGPUHost: NSObject {
    public let runtime: WebGPURuntime
    private weak var lynxView: LynxView?
    private let ticker = WebGPUFrameTicker()

    /// JS `loadAsset(name)`의 이름을 바이트로 해석하는 곳. 갈아끼워서 해석 규칙과
    /// 접근 범위를 앱이 정한다 (`WGPUAssetProvider` 문서 참고).
    ///
    /// 기본은 전체 경로 허용이다 — **번들(JS)을 신뢰할 수 없으면 반드시 좁힐 것**:
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

    /// LynxView가 만들어진 뒤 1회 호출한다. 프레임 이벤트 전송과 `<webgpu-canvas>` 연결에 필요하다.
    public func attach(to lynxView: LynxView) {
        self.lynxView = lynxView
        LynxWebGPUHostRegistry.register(self, for: lynxView)
        ticker.onFrame = { [weak self] timestamp, deltaSeconds in
            guard let self else { return }
            // 펌프는 준비 게이트 **앞**이다 — 완료 통지가 펌프에서 나오는 런타임(Dawn의
            // processEvents)이라면, 게이트 뒤에 두는 순간 포화가 영영 안 풀린다.
            self.runtime.processEvents()
            // GPU가 in-flight 한도만큼 밀려 있으면 이 틱을 건너뛴다. 여기서 이벤트를 보내면
            // JS가 프레임을 만들다 nextDrawable()에서 **JS 스레드 전체가** 서기 때문이다 —
            // 프레임을 거르는 쪽이 낫다. 완료가 돌아오면 다음 틱부터 재개된다.
            guard self.runtime.isReadyForNextFrame else { return }
            self.lynxView?.sendGlobalEvent("webgpu:frame", withParams: [[
                "timestamp": timestamp * 1000,
                "delta": deltaSeconds * 1000,
            ]])
        }
    }

    /// 페이지를 떠날 때 호출한다 — 디스플레이 링크를 멈추고 GPU 객체를 버린다.
    public func detach() {
        ticker.stop()
        runtime.reset()
        lynxView = nil
    }

    // MARK: - 프레임 루프

    /// 화면 갱신 주기에 맞춰 `webgpu:frame` 전역 이벤트를 보낸다.
    ///
    /// JS의 `setInterval`로 프레임을 돌리면 화면 갱신과 어긋나 프레임이 뭉치거나 버려진다.
    /// CADisplayLink로 몰아 주는 편이 훨씬 고르다 (`docs/JS-AUTHORING.md` §4).
    ///
    /// **정상 경로는 JS다** — 번들이 `startFrameLoop(handler)`를 부르면 shim이
    /// `NativeModules.WebGPU.startFrameLoop`를 거쳐 여기로 온다. `attach(to:)`는 틱 콜백을
    /// 배선만 하고 **루프를 시작하지 않는다** (프레임을 쓰지 않는 페이지에서 디스플레이 링크가
    /// 헛도는 것을 막기 위해서다 — `docs/LYNX-INTEGRATION.md` §3).
    ///
    /// 호스트가 직접 부를 수 있게 열어 둔 것은 **JS 밖에서 프레임을 모는 구성**을 위해서다
    /// (네이티브가 커맨드 스트림을 직접 만드는 경우, 또는 JS 경로를 진단할 때). 두 경로를
    /// 함께 쓰면 틱이 겹치지 않는다 — 링크는 하나이고 `start`는 기존 것을 갈아 끼운다.
    public func startFrameLoop(preferredFramesPerSecond: Int = 60) {
        ticker.start(preferredFramesPerSecond: preferredFramesPerSecond)
    }

    /// 프레임 루프를 멈춘다. `detach()`가 이미 부르므로 페이지 이탈에서는 따로 부를 필요가 없다.
    public func stopFrameLoop() {
        ticker.stop()
    }

    // MARK: - 캔버스 등록

    /// `<webgpu-canvas>`는 **레이어만** 넘긴다 — 표면 타입은 런타임이 고른다.
    /// 그래서 백엔드를 갈아끼워도 엘리먼트 코드가 그대로다.
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

/// rootView → 호스트 매핑. 뷰가 사라지면 항목도 사라진다.
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

/// 이 패키지가 Lynx에 등록하는 것들.
public enum LynxWebGPU {
    /// 네이티브 모듈과 커스텀 엘리먼트를 LynxConfig에 등록한다.
    ///
    /// ```swift
    /// let host = LynxWebGPUHost(runtime: try LynxWebGPUContext())   // 런타임은 앱이 고른다
    /// let lynxView = LynxView { builder in
    ///     let config = LynxConfig(provider: provider)
    ///     LynxWebGPU.register(in: config, host: host)
    ///     builder.config = config
    /// }
    /// host.attach(to: lynxView)   // 전역 이벤트/캔버스 연결에 필요
    /// ```
    public static func register(in config: LynxConfig, host: LynxWebGPUHost) {
        config.register(WebGPUNativeModule.self, param: host)
        config.registerUI(WebGPUCanvasUI.self, withName: elementName)
    }

    /// JS에서 쓰는 태그 이름.
    public static let elementName = "webgpu-canvas"
    /// JS에서 쓰는 모듈 이름 (`NativeModules.WebGPU`).
    public static let moduleName = WebGPUNativeModule.name
}
#endif
