#if canImport(Lynx)
import Foundation
import Lynx
import LynxWebGPUCore
import LynxWebGPU

/// JS의 `NativeModules.WebGPU` — WebGPU 커맨드 스트림 입구.
///
/// **스레딩** — Lynx는 모듈 메서드를 JS 백그라운드 스레드에서 호출한다. 여기서는 그 스레드를
/// 그대로 쓴다. Metal 인코딩은 메인 스레드를 요구하지 않으므로, 메인으로 넘기면 UI 작업과
/// 경쟁하며 프레임만 늦어진다. (UIKit을 건드리는 브리지라면 반대로 메인으로 넘겨야 하지만,
/// 여기서 하는 일은 커맨드 해석과 Metal 인코딩뿐이다.)
///
/// `execute`는 **동기 반환**이다. 프레임당 커맨드 배열 하나를 넘기고 결과를 바로 받는 구조라
/// 콜백 왕복이 필요 없다.
@objcMembers
public final class WebGPUNativeModule: NSObject, LynxModule {
    public static var name: String { "WebGPU" }

    public static var methodLookup: [String: String] {
        [
            "execute": NSStringFromSelector(#selector(execute(_:))),
            "adapterInfo": NSStringFromSelector(#selector(adapterInfo)),
            "shaderCompilationInfo": NSStringFromSelector(#selector(shaderCompilationInfo(_:))),
            "canvasInfo": NSStringFromSelector(#selector(canvasInfo(_:))),
            "readBuffer": NSStringFromSelector(#selector(readBuffer(_:callback:))),
            "loadAsset": NSStringFromSelector(#selector(loadAsset(_:callback:))),
            "startFrameLoop": NSStringFromSelector(#selector(startFrameLoop(_:))),
            "stopFrameLoop": NSStringFromSelector(#selector(stopFrameLoop)),
            "reset": NSStringFromSelector(#selector(reset)),
        ]
    }

    /// `LynxConfig.register(_:param:)`으로 주입되는 호스트.
    private weak var host: LynxWebGPUHost?

    public init(param: Any) {
        host = param as? LynxWebGPUHost
        super.init()
    }

    public override init() {
        super.init()
    }

    // MARK: - 커맨드 실행

    /// 한 프레임 분량의 커맨드 스트림을 실행한다.
    ///
    /// - Parameter payload: `{"commands": [{op: …}, …]}`
    /// - Returns: `{"ok": Bool, "errors": [...], "canvases": {...}}`
    public func execute(_ payload: [String: Any]) -> [String: Any] {
        guard let host else { return Self.unavailable }
        return host.context.execute(payload)
    }

    /// `navigator.gpu.requestAdapter()`가 쓰는 디바이스 정보/한계값.
    public func adapterInfo() -> [String: Any] {
        guard let host else { return Self.unavailable }
        return host.context.adapterInfo()
    }

    /// `GPUShaderModule.getCompilationInfo()` — 그 모듈의 컴파일 진단.
    public func shaderCompilationInfo(_ params: [String: Any]) -> [String: Any] {
        guard let host else { return Self.unavailable }
        guard let handle = params["module"] as? Int else {
            return ["ok": false, "errors": [WGPUError.validation("module 핸들이 필요하다").payload]]
        }
        return host.context.shaderCompilationInfo(handle: handle)
    }

    /// 캔버스의 현재 픽셀 크기. 뷰포트·투영행렬 계산에 쓴다.
    public func canvasInfo(_ params: [String: Any]) -> [String: Any] {
        guard let host else { return Self.unavailable }
        guard let identifier = params["canvas"] as? String else {
            return ["ok": false, "errors": [WGPUError.validation("canvas 이름이 필요하다").payload]]
        }
        guard let surface = host.context.surface(for: identifier) else {
            return [
                "ok": false,
                "errors": [WGPUError.validation(
                    "캔버스 '\(identifier)'이(가) 없다 (등록된 것: "
                        + "\(host.context.registeredSurfaceIdentifiers.joined(separator: ", ")))"
                ).payload],
            ]
        }
        return [
            "ok": true,
            "width": Int(surface.pixelSize.width),
            "height": Int(surface.pixelSize.height),
            "format": surface.configuredFormat.rawValue,
        ]
    }

    /// 버퍼 내용을 읽는다 (`GPUBuffer.mapAsync` 대응). GPU 완료를 기다리므로 콜백형이다.
    public func readBuffer(_ params: [String: Any], callback: @escaping LynxCallbackBlock) {
        guard let host else {
            callback(Self.unavailable)
            return
        }
        guard let handle = (params["buffer"] as? NSNumber)?.intValue else {
            callback(["ok": false, "errors": [WGPUError.validation("buffer 핸들이 필요하다").payload]])
            return
        }
        host.context.readBuffer(
            handle: handle,
            offset: (params["offset"] as? NSNumber)?.intValue ?? 0,
            size: (params["size"] as? NSNumber)?.intValue,
            completion: callback
        )
    }

    // MARK: - 애셋

    /// 애셋을 `ArrayBuffer`로 읽는다. 브라우저의 `fetch()`가 하던 역할을 최소한으로 대신한다.
    ///
    /// 이름 해석은 전적으로 `host.assetProvider`에 맡긴다 — 기본 공급자는 등록된 메모리
    /// 데이터·파일 경로·번들 상대 이름을 받고, 앱이 공급자를 갈아끼워 규칙과 접근 범위를
    /// 정할 수 있다 (`WGPUAssetProvider` 참고).
    ///
    /// - Parameter params: `{"name": String}`
    /// - Returns: 콜백으로 `{"ok": true, "data": ArrayBuffer, "byteLength": Int}`.
    public func loadAsset(_ params: [String: Any], callback: @escaping LynxCallbackBlock) {
        guard let host else {
            callback(Self.unavailable)
            return
        }
        WGPUAssetLoading.load(params, provider: host.assetProvider, callback: callback)
    }

    // MARK: - 프레임 루프

    /// `webgpu:frame` 전역 이벤트를 화면 갱신 주기에 맞춰 보내기 시작한다.
    public func startFrameLoop(_ params: [String: Any]) -> [String: Any] {
        guard let host else { return Self.unavailable }
        host.startFrameLoop(preferredFramesPerSecond: (params["fps"] as? NSNumber)?.intValue ?? 60)
        return ["ok": true]
    }

    public func stopFrameLoop() -> [String: Any] {
        guard let host else { return Self.unavailable }
        host.stopFrameLoop()
        return ["ok": true]
    }

    /// 모든 GPU 객체를 버린다 (페이지 이탈/핫 리로드).
    public func reset() -> [String: Any] {
        guard let host else { return Self.unavailable }
        host.context.reset()
        return ["ok": true]
    }

    private static let unavailable: [String: Any] = [
        "ok": false,
        "errors": [WGPUError.backend(
            "WebGPU 호스트가 연결되지 않았다 — LynxWebGPU.register(in:host:)와 host.attach(to:)를 확인할 것"
        ).payload],
    ]
}
#endif
