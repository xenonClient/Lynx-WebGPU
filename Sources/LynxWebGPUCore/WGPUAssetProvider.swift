import Foundation

/// Where an asset requested by JS's `loadAsset(name)` is turned into bytes.
///
/// Just as Lynx hands bundle loading to the host through `LynxTemplateProvider`, asset resolution
/// is swappable too — put another implementation in `LynxWebGPUHost.assetProvider` and the app
/// decides the name-resolution rules and the access scope (a narrower directory, a handle-based
/// registry, and so on).
///
/// The default is `WGPUFileAssetProvider()`, which **allows any path**. An app that cannot trust
/// its bundle (one downloaded from a server, say) must narrow `allowedRoots` or swap in its own
/// implementation. Bytes JS has read belong to JS — it can send them anywhere.
public protocol WGPUAssetProvider: AnyObject {
    /// Turns `name` into bytes.
    ///
    /// Called on the JS thread — a slow source such as a file must **move to a background queue
    /// itself**. The completion callback may come from any thread (Lynx returns it to JS).
    func loadAsset(named name: String, completion: @escaping (Result<Data, WGPUError>) -> Void)
}

/// Delegates the bridge's `loadAsset` request to the provider and shapes the result as a JS payload.
///
/// Why it lives here, free of any Lynx dependency: the bridge (`WebGPUNativeModule`) is iOS-only,
/// so macOS tests never reach it — **the contract "swapping the provider changes the scope" is
/// verified here** (`AssetProviderTests`).
public enum WGPUAssetLoading {
    /// - Parameter params: `{"name": String}` — exactly what JS's `loadAsset(name)` sent.
    /// - Parameter callback: `{"ok": true, "data": Data, "byteLength": Int}` or
    ///   `{"ok": false, "errors": [...]}`. Lynx converts `Data` into an `ArrayBuffer`.
    public static func load(
        _ params: [String: Any],
        provider: WGPUAssetProvider,
        callback: @escaping ([String: Any]) -> Void
    ) {
        guard let name = params["name"] as? String else {
            callback(["ok": false, "errors": [WGPUError.validation("asset name is required").payload]])
            return
        }
        provider.loadAsset(named: name) { result in
            switch result {
            case .success(let data):
                // Same convention as `readBuffer` — pass `Data` straight through and Lynx turns it
                // into an `ArrayBuffer`.
                callback(["ok": true, "data": data, "byteLength": data.count])
            case .failure(let error):
                callback(["ok": false, "errors": [error.payload]])
            }
        }
    }
}

/// The default asset provider — resolves files and registered in-memory data under one name.
///
/// Resolution order:
/// 1. A name registered with `register(_:for:)` — the channel for things that arrive **as `Data`
///    rather than a file**, such as an image picker.
/// 2. A `file://` URL or an absolute path starting with `/` — a file URL from a picker or download,
///    used as-is.
/// 3. Anything else — a bundle-relative path (`"hdr-sample.bin"`, `"LUTs/neutral.cube"`).
///
/// Access scope: **everything is allowed by default**. Supplying `allowedRoots` restricts case 2
/// (file paths) to those directories — the comparison uses symlink-resolved real paths, so `..` or
/// a link cannot escape. Cases 1 (registered) and 3 (bundle) are host-controlled and unrestricted.
public final class WGPUFileAssetProvider: WGPUAssetProvider {
    private let bundle: Bundle
    private let allowedRoots: [URL]?
    private let lock = NSLock()
    private var registered: [String: Data] = [:]

    /// - Parameters:
    ///   - bundle: where bundle-relative names are looked up. Defaults to `Bundle.main` — pass an
    ///     SPM library's `Bundle.module` to serve resources shipped alongside it.
    ///   - allowedRoots: directories file-path access is confined to. `nil` allows everything.
    public init(bundle: Bundle = .main, allowedRoots: [URL]? = nil) {
        self.bundle = bundle
        // Normalize once here instead of resolving on every comparison.
        self.allowedRoots = allowedRoots?.map { $0.standardizedFileURL.resolvingSymlinksInPath() }
    }

    // MARK: - Registering in-memory data (for the host app)

    /// Binds in-memory bytes to a name. JS calling `loadAsset` with that name receives them.
    public func register(_ data: Data, for name: String) {
        lock.lock()
        registered[name] = data
        lock.unlock()
    }

    public func unregister(_ name: String) {
        lock.lock()
        registered[name] = nil
        lock.unlock()
    }

    // MARK: - Resolution

    public func loadAsset(named name: String, completion: @escaping (Result<Data, WGPUError>) -> Void) {
        guard !name.isEmpty else {
            completion(.failure(.validation("asset name is required")))
            return
        }

        lock.lock()
        let registeredData = registered[name]
        lock.unlock()
        if let registeredData {
            // Already in memory, so there is no reason to hand it to a queue.
            completion(.success(registeredData))
            return
        }

        if let url = Self.fileURL(from: name) {
            guard isAllowed(url) else {
                completion(.failure(.validation("path is outside the allowed directories: \(name)")))
                return
            }
            readInBackground(url, describedAs: name, completion: completion)
            return
        }

        guard let url = bundledURL(named: name) else {
            completion(.failure(.validation("'\(name)' is not in the bundle")))
            return
        }
        readInBackground(url, describedAs: name, completion: completion)
    }

    // MARK: - File paths

    /// Turns a name into a URL when it denotes a file path. Otherwise `nil` — treated as a bundle name.
    private static func fileURL(from name: String) -> URL? {
        if name.hasPrefix("file://") { return URL(string: name) }
        if name.hasPrefix("/") { return URL(fileURLWithPath: name) }
        return nil
    }

    private func isAllowed(_ url: URL) -> Bool {
        guard let allowedRoots else { return true }
        // Reconciles link-split spellings such as the picker's `/var/...` versus the real `/private/var/...`.
        let resolved = url.standardizedFileURL.resolvingSymlinksInPath().path
        return allowedRoots.contains { root in
            resolved == root.path || resolved.hasPrefix(root.path.hasSuffix("/") ? root.path : root.path + "/")
        }
    }

    // MARK: - Bundle names

    private func bundledURL(named name: String) -> URL? {
        let components = name.split(separator: "/")
        // There is no reason to climb out of a bundle — `..` and hidden names count as tampering.
        guard !components.isEmpty,
              !components.contains(where: { $0 == ".." || $0.hasPrefix(".") })
        else { return nil }

        let file = String(components[components.count - 1])
        let base = (file as NSString).deletingPathExtension
        let ext = (file as NSString).pathExtension
        let subdirectory = components.dropLast().joined(separator: "/")
        return bundle.url(
            forResource: base,
            withExtension: ext.isEmpty ? nil : ext,
            subdirectory: subdirectory.isEmpty ? nil : subdirectory
        )
    }

    // MARK: - Reading

    private func readInBackground(
        _ url: URL,
        describedAs description: String,
        completion: @escaping (Result<Data, WGPUError>) -> Void
    ) {
        // A file can run to several MB, so we never read it synchronously on the JS thread.
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                completion(.success(try Data(contentsOf: url, options: .mappedIfSafe)))
            } catch {
                completion(.failure(.backend("could not read asset '\(description)': \(error)")))
            }
        }
    }
}
