import Foundation

/// What JS's `importSharedTexture(device, {key, url, resizeWidth?, resizeHeight?, label?})` asks for.
///
/// A shared texture is decoded and uploaded **once per (device, key)** and handed to every context
/// on that device that asks for the key — LynxViews each run their own context and JS runtime, so an
/// image used by several pages would otherwise sit in GPU memory once per page. Handles are issued per
/// JS runtime and mean nothing across views, so the key is a string the caller chooses.
///
/// **The key is the caller's promise about the pixels.** The same key must always mean the same image
/// at the same size — a second request under a key already in use gets the first request's texture,
/// whatever its `name` and size say. Put everything that changes the pixels (the source's identity and
/// version, the resize) into the key.
public struct WGPUSharedTextureRequest: Equatable {
    /// The cache key — see the type's documentation for the contract.
    public var key: String
    /// The image, resolved through the context's `WGPUAssetProvider` like `loadAsset` (a registered
    /// name, a file path or `file://` URL, or a bundle-relative path). JS sends it as `url`.
    public var name: String
    /// The requested size. Both given: exactly that size. One given: the other follows the source's
    /// aspect ratio, rounded up. Neither: the source's size. See `targetSize(sourceWidth:sourceHeight:)`.
    public var resizeWidth: Int?
    public var resizeHeight: Int?
    /// The Metal label of the texture, for GPU captures — the first request's label wins.
    public var label: String?

    public init(key: String, name: String, resizeWidth: Int? = nil, resizeHeight: Int? = nil, label: String? = nil) {
        self.key = key
        self.name = name
        self.resizeWidth = resizeWidth
        self.resizeHeight = resizeHeight
        self.label = label
    }

    /// Reads the JS payload's `{key, url, resizeWidth?, resizeHeight?, label?}`.
    public init(from reader: WGPUValueReader) throws {
        key = try reader.requiredString("key")
        guard !key.isEmpty else {
            throw WGPUError.validation("importSharedTexture needs a non-empty key", path: reader.fieldPath("key"))
        }
        name = try reader.requiredString("url")
        guard !name.isEmpty else {
            throw WGPUError.validation("importSharedTexture needs a non-empty url", path: reader.fieldPath("url"))
        }
        resizeWidth = try reader.optionalInt("resizeWidth")
        resizeHeight = try reader.optionalInt("resizeHeight")
        for (field, value) in [("resizeWidth", resizeWidth), ("resizeHeight", resizeHeight)] {
            if let value, value < 1 {
                throw WGPUError.validation(
                    "importSharedTexture \(field) must be at least 1 (got \(value))", path: reader.fieldPath(field)
                )
            }
        }
        label = reader.optionalString("label")
    }

    /// Whether `other` asks for the same pixels — the same source at the same requested size. Two
    /// requests under one key that differ here break the key contract (the cache warns in debug builds).
    public func describesSamePixels(as other: WGPUSharedTextureRequest) -> Bool {
        name == other.name && resizeWidth == other.resizeWidth && resizeHeight == other.resizeHeight
    }

    /// The pixel size to produce from a source of `sourceWidth`×`sourceHeight`.
    ///
    /// Both sizes given: that size, aspect ratio or not (keeping the ratio is the caller's call). One
    /// given: the other is derived from the source's aspect ratio and **rounded up** — the HTML
    /// `createImageBitmap` rule ("cropped to the source rectangle with formatting"), so a browser
    /// running the same request gets the same size. Neither: the source's size.
    ///
    /// The caller bounds a requested axis by its texture limit **before** calling, which keeps a derived
    /// axis (at most the source's side × that limit) inside `Int`, and checks the result against the same
    /// limit after — `WGPUSharedTextureUpload` does both.
    public func targetSize(sourceWidth: Int, sourceHeight: Int) -> (width: Int, height: Int) {
        switch (resizeWidth, resizeHeight) {
        case let (width?, height?):
            return (width, height)
        case let (width?, nil):
            let height = (Double(sourceHeight) * Double(width) / Double(max(sourceWidth, 1))).rounded(.up)
            return (width, max(Int(height), 1))
        case let (nil, height?):
            let width = (Double(sourceWidth) * Double(height) / Double(max(sourceHeight, 1))).rounded(.up)
            return (max(Int(width), 1), height)
        case (nil, nil):
            return (sourceWidth, sourceHeight)
        }
    }
}

/// Hands the bridge's `importSharedTexture` request to the runtime and guarantees **exactly one**
/// callback — a JS promise left unsettled stops the page without an error.
///
/// Lives here, free of Lynx, for the same reason as `WGPUAssetLoading`: the bridge is iOS-only and
/// never reaches macOS tests, so the payload parsing is verified here (`SharedTextureRequestTests`).
public enum WGPUSharedTextureImport {
    /// - Parameter params: `{"id": Int, "key": String, "url": String, "resizeWidth"?: Int,
    ///   "resizeHeight"?: Int, "label"?: String}` — exactly what the shim sent.
    /// - Parameter callback: `{"ok": true, "width": Int, "height": Int, "format": String}` or
    ///   `{"ok": false, "errors": [...]}`.
    public static func start(
        _ params: [String: Any],
        runtime: WebGPURuntime,
        provider: WGPUAssetProvider?,
        callback: @escaping ([String: Any]) -> Void
    ) {
        // Host payloads are materialized before reading (`docs/COMMAND-STREAM.md` §5-1-1).
        let reader = WGPUValueReader(WGPUPayload.materialize(params))
        let handle: Int
        let request: WGPUSharedTextureRequest
        do {
            handle = try reader.requiredInt("id")
            request = try WGPUSharedTextureRequest(from: reader)
        } catch let error as WGPUError {
            callback(["ok": false, "errors": [error.payload]])
            return
        } catch {
            callback(["ok": false, "errors": [WGPUError.validation("\(error)").payload]])
            return
        }
        runtime.importSharedTexture(handle: handle, request: request, provider: provider, completion: callback)
    }
}
