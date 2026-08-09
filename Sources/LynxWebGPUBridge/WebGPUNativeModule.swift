#if canImport(Lynx)
import Foundation
import Lynx
import LynxWebGPUCore

/// JS's `NativeModules.WebGPU` — the entrance to the WebGPU command stream.
///
/// **Threading** — Lynx calls module methods on a JS background thread, and we stay on that thread.
/// Metal encoding does not require the main thread, so hopping to main would only contend with UI
/// work and delay frames. (A bridge touching UIKit would have to hop the other way, but all this one
/// does is interpret commands and encode Metal.)
///
/// `execute` **returns synchronously**. One command array per frame goes in and the result comes
/// straight back, so no callback round trip is needed.
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
            "decodeImage": NSStringFromSelector(#selector(decodeImage(_:callback:))),
            "startFrameLoop": NSStringFromSelector(#selector(startFrameLoop(_:))),
            "stopFrameLoop": NSStringFromSelector(#selector(stopFrameLoop)),
            "reset": NSStringFromSelector(#selector(reset)),
        ]
    }

    /// The host injected through `LynxConfig.register(_:param:)`.
    private weak var host: LynxWebGPUHost?

    public init(param: Any) {
        host = param as? LynxWebGPUHost
        super.init()
    }

    public override init() {
        super.init()
    }

    // MARK: - Command execution

    /// Runs one frame's worth of the command stream.
    ///
    /// - Parameter payload: `{"commands": [{op: …}, …]}`
    /// - Returns: `{"ok": Bool, "errors": [...], "canvases": {...}}`
    public func execute(_ payload: [String: Any]) -> [String: Any] {
        guard let host else { return Self.unavailable }
        return host.runtime.execute(payload)
    }

    /// The device info and limits `navigator.gpu.requestAdapter()` uses.
    public func adapterInfo() -> [String: Any] {
        guard let host else { return Self.unavailable }
        return host.runtime.adapterInfo()
    }

    /// `GPUShaderModule.getCompilationInfo()` — that module's compilation diagnostics.
    public func shaderCompilationInfo(_ params: [String: Any]) -> [String: Any] {
        guard let host else { return Self.unavailable }
        guard let handle = params["module"] as? Int else {
            return ["ok": false, "errors": [WGPUError.validation("a module handle is required").payload]]
        }
        return host.runtime.shaderCompilationInfo(handle: handle)
    }

    /// The canvas's current pixel size. Used to compute viewport and projection matrices.
    public func canvasInfo(_ params: [String: Any]) -> [String: Any] {
        guard let host else { return Self.unavailable }
        guard let identifier = params["canvas"] as? String else {
            return ["ok": false, "errors": [WGPUError.validation("a canvas name is required").payload]]
        }
        return host.runtime.canvasInfo(identifier: identifier)
    }

    /// Reads buffer contents (corresponding to `GPUBuffer.mapAsync`). It waits on GPU completion, hence the callback form.
    public func readBuffer(_ params: [String: Any], callback: @escaping LynxCallbackBlock) {
        guard let host else {
            callback(Self.unavailable)
            return
        }
        guard let handle = (params["buffer"] as? NSNumber)?.intValue else {
            callback(["ok": false, "errors": [WGPUError.validation("a buffer handle is required").payload]])
            return
        }
        host.runtime.readBuffer(
            handle: handle,
            offset: (params["offset"] as? NSNumber)?.intValue ?? 0,
            size: (params["size"] as? NSNumber)?.intValue,
            completion: callback
        )
    }

    // MARK: - Assets

    /// Reads an asset as an `ArrayBuffer`. A minimal stand-in for what the browser's `fetch()` did.
    ///
    /// Name resolution is left entirely to `host.assetProvider` — the default provider accepts
    /// registered in-memory data, file paths and bundle-relative names, and an app can swap the
    /// provider to decide the rules and the access scope (see `WGPUAssetProvider`).
    ///
    /// - Parameter params: `{"name": String}`
    /// - Returns: through the callback, `{"ok": true, "data": ArrayBuffer, "byteLength": Int}`.
    public func loadAsset(_ params: [String: Any], callback: @escaping LynxCallbackBlock) {
        guard let host else {
            callback(Self.unavailable)
            return
        }
        WGPUAssetLoading.load(params, provider: host.assetProvider, callback: callback)
    }

    /// Decodes an encoded image and registers it as the object standing in for `ImageBitmap` (JS `createImageBitmap`).
    ///
    /// This is what the browser's `createImageBitmap()` did. Lynx has neither `<img>` nor
    /// `ImageBitmap`, so ImageIO stands in — far faster than unpacking PNG/JPEG/HEIC by hand in JS,
    /// and the pixels stay native, **never crossing the bridge at all.**
    ///
    /// - Parameter params: `{"id": Int, "data"?: ArrayBuffer, "name"?: String,
    ///   "flipY"?: Bool, "premultiplyAlpha"?: Bool, "resizeWidth"?: Int, "resizeHeight"?: Int}`
    /// - Returns: through the callback, `{"ok": true, "width": Int, "height": Int}`.
    public func decodeImage(_ params: [String: Any], callback: @escaping LynxCallbackBlock) {
        guard let host else {
            callback(Self.unavailable)
            return
        }
        guard let handle = (params["id"] as? NSNumber)?.intValue else {
            callback(["ok": false, "errors": [WGPUError.validation("createImageBitmap requires an id").payload]])
            return
        }
        var resize: (width: Int, height: Int)?
        if let width = (params["resizeWidth"] as? NSNumber)?.intValue,
           let height = (params["resizeHeight"] as? NSNumber)?.intValue {
            resize = (width, height)
        }
        host.runtime.decodeImage(
            handle: handle,
            data: WGPUValueReader(params).data("data"),
            name: params["name"] as? String,
            options: WGPUImageDecodeOptions(
                flipY: (params["flipY"] as? NSNumber)?.boolValue ?? false,
                premultiplyAlpha: (params["premultiplyAlpha"] as? NSNumber)?.boolValue ?? false,
                resize: resize
            ),
            provider: host.assetProvider,
            completion: callback
        )
    }

    // MARK: - Frame loop

    /// Starts sending the `webgpu:frame` global event in step with the display refresh.
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

    /// Discards every GPU object (page exit, hot reload).
    public func reset() -> [String: Any] {
        guard let host else { return Self.unavailable }
        host.runtime.reset()
        return ["ok": true]
    }

    private static let unavailable: [String: Any] = [
        "ok": false,
        "errors": [WGPUError.backend(
            "the WebGPU host is not connected — check LynxWebGPU.register(in:host:) and host.attach(to:)"
        ).payload],
    ]
}
#endif
