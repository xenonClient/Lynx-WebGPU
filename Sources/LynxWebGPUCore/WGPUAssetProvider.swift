import Foundation

/// JS의 `loadAsset(name)`이 요청한 애셋을 바이트로 바꿔 주는 곳.
///
/// Lynx가 번들 로딩을 `LynxTemplateProvider`로 호스트에 맡기듯, 애셋 해석도 호스트가
/// 갈아끼울 수 있다 — `LynxWebGPUHost.assetProvider`에 다른 구현을 넣으면 이름 해석
/// 규칙과 접근 범위를 앱이 정한다 (예: 허용 디렉토리 축소, 핸들 기반 레지스트리).
///
/// 기본값은 `WGPUFileAssetProvider()` — **전체 경로를 허용**한다. 번들(JS)을 신뢰할 수
/// 없는 앱(서버에서 내려받는 번들 등)은 반드시 `allowedRoots`를 좁히거나 자체 구현으로
/// 바꿀 것. JS가 읽은 바이트는 JS의 것이 된다 — 밖으로 보낼 수도 있다.
public protocol WGPUAssetProvider: AnyObject {
    /// `name`을 바이트로 바꾼다.
    ///
    /// JS 스레드에서 불린다 — 파일처럼 느린 소스는 **직접 백그라운드 큐로 넘길 것**.
    /// 완료 콜백은 아무 스레드에서 불러도 된다 (Lynx가 JS로 되돌린다).
    func loadAsset(named name: String, completion: @escaping (Result<Data, WGPUError>) -> Void)
}

/// 브리지의 `loadAsset` 요청을 공급자에 위임하고 결과를 JS 페이로드로 바꾼다.
///
/// Lynx 의존이 없는 이 위치에 두는 이유: 브리지(`WebGPUNativeModule`)는 iOS 전용이라
/// macOS 테스트가 닿지 않는다 — **"공급자를 갈아끼우면 스코프가 바뀐다"는 계약은
/// 여기서 검증한다** (`AssetProviderTests`).
public enum WGPUAssetLoading {
    /// - Parameter params: `{"name": String}` — JS `loadAsset(name)`이 보낸 그대로.
    /// - Parameter callback: `{"ok": true, "data": Data, "byteLength": Int}` 또는
    ///   `{"ok": false, "errors": [...]}`. `Data`는 Lynx가 `ArrayBuffer`로 바꿔 준다.
    public static func load(
        _ params: [String: Any],
        provider: WGPUAssetProvider,
        callback: @escaping ([String: Any]) -> Void
    ) {
        guard let name = params["name"] as? String else {
            callback(["ok": false, "errors": [WGPUError.validation("애셋 name이 필요하다").payload]])
            return
        }
        provider.loadAsset(named: name) { result in
            switch result {
            case .success(let data):
                // `readBuffer`와 같은 규약 — `Data`를 그대로 실으면 Lynx가 `ArrayBuffer`로 바꿔 준다.
                callback(["ok": true, "data": data, "byteLength": data.count])
            case .failure(let error):
                callback(["ok": false, "errors": [error.payload]])
            }
        }
    }
}

/// 기본 애셋 공급자 — 파일과 등록된 메모리 데이터를 이름 하나로 해석한다.
///
/// 해석 순서:
/// 1. `register(_:for:)`로 등록된 이름 — 이미지 피커처럼 **파일이 아니라 `Data`로 오는 것**의 통로.
/// 2. `file://` URL 또는 `/`로 시작하는 절대 경로 — 피커·다운로드가 준 파일 URL을 그대로 쓴다.
/// 3. 그 외 — 번들 상대 경로 (`"hdr-sample.bin"`, `"LUTs/neutral.cube"`).
///
/// 접근 범위: 기본은 **전체 허용**이다. `allowedRoots`를 주면 2번(파일 경로)이 그 디렉토리들
/// 아래로 제한된다 — 심볼릭 링크를 푼 실제 경로로 비교하므로 `..`나 링크로 벗어날 수 없다.
/// 1번(등록)과 3번(번들)은 호스트가 내용을 통제하므로 제한 대상이 아니다.
public final class WGPUFileAssetProvider: WGPUAssetProvider {
    private let bundle: Bundle
    private let allowedRoots: [URL]?
    private let lock = NSLock()
    private var registered: [String: Data] = [:]

    /// - Parameters:
    ///   - bundle: 번들 상대 이름을 찾을 곳. 기본은 `Bundle.main` — SPM 라이브러리로
    ///     동봉한 리소스를 내주려면 그쪽의 `Bundle.module`을 넘긴다.
    ///   - allowedRoots: 파일 경로 접근을 허용할 디렉토리 목록. `nil`이면 전체 허용.
    public init(bundle: Bundle = .main, allowedRoots: [URL]? = nil) {
        self.bundle = bundle
        // 비교 시점마다 푸는 대신 여기서 한 번 정규화해 둔다.
        self.allowedRoots = allowedRoots?.map { $0.standardizedFileURL.resolvingSymlinksInPath() }
    }

    // MARK: - 메모리 데이터 등록 (호스트 앱용)

    /// 메모리에 있는 바이트를 이름에 묶는다. JS가 같은 이름으로 `loadAsset`을 부르면 이것을 받는다.
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

    // MARK: - 해석

    public func loadAsset(named name: String, completion: @escaping (Result<Data, WGPUError>) -> Void) {
        guard !name.isEmpty else {
            completion(.failure(.validation("애셋 name이 필요하다")))
            return
        }

        lock.lock()
        let registeredData = registered[name]
        lock.unlock()
        if let registeredData {
            // 이미 메모리에 있으므로 큐로 넘길 이유가 없다.
            completion(.success(registeredData))
            return
        }

        if let url = Self.fileURL(from: name) {
            guard isAllowed(url) else {
                completion(.failure(.validation("허용된 디렉토리 밖의 경로다: \(name)")))
                return
            }
            readInBackground(url, describedAs: name, completion: completion)
            return
        }

        guard let url = bundledURL(named: name) else {
            completion(.failure(.validation("번들에 '\(name)'이(가) 없다")))
            return
        }
        readInBackground(url, describedAs: name, completion: completion)
    }

    // MARK: - 파일 경로

    /// 이름이 파일 경로를 뜻하면 URL로 바꾼다. 아니면 `nil` — 번들 이름으로 본다.
    private static func fileURL(from name: String) -> URL? {
        if name.hasPrefix("file://") { return URL(string: name) }
        if name.hasPrefix("/") { return URL(fileURLWithPath: name) }
        return nil
    }

    private func isAllowed(_ url: URL) -> Bool {
        guard let allowedRoots else { return true }
        // 피커가 주는 `/var/...`와 실제 `/private/var/...`처럼 링크로 갈라진 표기를 맞춘다.
        let resolved = url.standardizedFileURL.resolvingSymlinksInPath().path
        return allowedRoots.contains { root in
            resolved == root.path || resolved.hasPrefix(root.path.hasSuffix("/") ? root.path : root.path + "/")
        }
    }

    // MARK: - 번들 이름

    private func bundledURL(named name: String) -> URL? {
        let components = name.split(separator: "/")
        // 번들 안에서는 위로 올라갈 이유가 없다 — `..`과 숨김 이름은 조작으로 본다.
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

    // MARK: - 읽기

    private func readInBackground(
        _ url: URL,
        describedAs description: String,
        completion: @escaping (Result<Data, WGPUError>) -> Void
    ) {
        // 파일이 수 MB에 이를 수 있어 JS 스레드에서 동기로 읽지 않는다.
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                completion(.success(try Data(contentsOf: url, options: .mappedIfSafe)))
            } catch {
                completion(.failure(.backend("애셋 '\(description)'을(를) 읽지 못했다: \(error)")))
            }
        }
    }
}
