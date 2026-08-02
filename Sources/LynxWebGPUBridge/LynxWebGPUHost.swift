#if canImport(Lynx)
import Foundation
import UIKit
import Lynx
import LynxWebGPUCore
import LynxWebGPU

/// LynxView 하나에 붙는 WebGPU 런타임.
///
/// 네이티브 모듈(`NativeModules.WebGPU`)과 `<webgpu-canvas>` 엘리먼트가 공유하는 접점이다.
/// Lynx는 커스텀 UI를 `[[cls alloc] init]`으로 직접 만들기 때문에 생성자 주입을 할 수 없다 —
/// 대신 호스트가 LynxView(= UI 트리의 rootView)에 자신을 등록해 두고 엘리먼트가 되찾는다
/// (`LynxUIContext.rootView`, 양쪽 모두 약한 참조).
public final class LynxWebGPUHost: NSObject {
    public let context: LynxWebGPUContext
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

    public init(context: LynxWebGPUContext) {
        self.context = context
        super.init()
    }

    public convenience init(device: MTLDevice? = nil) throws {
        self.init(context: try LynxWebGPUContext(device: device))
    }

    /// LynxView가 만들어진 뒤 1회 호출한다. 프레임 이벤트 전송과 `<webgpu-canvas>` 연결에 필요하다.
    public func attach(to lynxView: LynxView) {
        self.lynxView = lynxView
        LynxWebGPUHostRegistry.register(self, for: lynxView)
        ticker.onFrame = { [weak self] timestamp, deltaSeconds in
            guard let self else { return }
            // GPU가 in-flight 한도만큼 밀려 있으면 이 틱을 건너뛴다. 여기서 이벤트를 보내면
            // JS가 프레임을 만들다 nextDrawable()에서 **JS 스레드 전체가** 서기 때문이다 —
            // 프레임을 거르는 쪽이 낫다. 완료가 돌아오면 다음 틱부터 재개된다.
            guard self.context.isReadyForNextFrame else { return }
            self.lynxView?.sendGlobalEvent("webgpu:frame", withParams: [[
                "timestamp": timestamp * 1000,
                "delta": deltaSeconds * 1000,
            ]])
        }
    }

    /// 페이지를 떠날 때 호출한다 — 디스플레이 링크를 멈추고 GPU 객체를 버린다.
    public func detach() {
        ticker.stop()
        context.reset()
        lynxView = nil
    }

    // MARK: - 프레임 루프

    /// 화면 갱신 주기에 맞춰 `webgpu:frame` 전역 이벤트를 보낸다.
    ///
    /// JS의 `setInterval`로 프레임을 돌리면 화면 갱신과 어긋나 프레임이 뭉치거나 버려진다.
    /// CADisplayLink로 몰아 주는 편이 훨씬 고르다 (`docs/JS-AUTHORING.md` §4).
    func startFrameLoop(preferredFramesPerSecond: Int) {
        ticker.start(preferredFramesPerSecond: preferredFramesPerSecond)
    }

    func stopFrameLoop() {
        ticker.stop()
    }

    // MARK: - 캔버스 등록

    func registerCanvas(_ surface: WGPUSurface) {
        context.registerSurface(surface)
    }

    func unregisterCanvas(identifier: String) {
        context.unregisterSurface(identifier: identifier)
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
    /// let host = try LynxWebGPUHost()
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
