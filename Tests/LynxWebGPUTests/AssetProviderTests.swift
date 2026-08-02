import XCTest
import LynxWebGPUCore
@testable import LynxWebGPU

/// 기본 애셋 공급자의 계약 — 등록 이름 → 파일 경로 → 번들 순으로 해석하고,
/// `allowedRoots`가 있으면 파일 경로 접근이 그 아래로 제한된다.
final class AssetProviderTests: XCTestCase {
    private var directory: URL!

    override func setUpWithError() throws {
        directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("AssetProviderTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: directory)
    }

    private func write(_ bytes: [UInt8], to name: String) throws -> URL {
        let url = directory.appendingPathComponent(name)
        try Data(bytes).write(to: url)
        return url
    }

    private func load(_ provider: WGPUFileAssetProvider, _ name: String) -> Result<Data, WGPUError> {
        let expectation = expectation(description: "loadAsset(\(name))")
        var received: Result<Data, WGPUError>!
        provider.loadAsset(named: name) { result in
            received = result
            expectation.fulfill()
        }
        wait(for: [expectation], timeout: 5)
        return received
    }

    // MARK: - 등록된 메모리 데이터

    func test_등록한_데이터를_이름으로_돌려받는다() {
        let provider = WGPUFileAssetProvider()
        provider.register(Data([1, 2, 3]), for: "picked-image")

        XCTAssertEqual(try load(provider, "picked-image").get(), Data([1, 2, 3]))
    }

    func test_등록_해제하면_더는_해석되지_않는다() {
        let provider = WGPUFileAssetProvider(bundle: Bundle(url: directory)!)
        provider.register(Data([1]), for: "gone")
        provider.unregister("gone")

        XCTAssertThrowsError(try load(provider, "gone").get())
    }

    func test_등록_이름이_파일보다_먼저다() throws {
        let url = try write([9, 9], to: "clash.bin")
        let provider = WGPUFileAssetProvider()
        provider.register(Data([1]), for: url.path)

        XCTAssertEqual(try load(provider, url.path).get(), Data([1]))
    }

    // MARK: - 파일 경로 (기본: 전체 허용)

    func test_절대_경로를_읽는다() throws {
        let url = try write([10, 20, 30], to: "path.bin")

        XCTAssertEqual(try load(WGPUFileAssetProvider(), url.path).get(), Data([10, 20, 30]))
    }

    func test_file_URL을_읽는다() throws {
        let url = try write([7], to: "url.bin")

        XCTAssertEqual(try load(WGPUFileAssetProvider(), url.absoluteString).get(), Data([7]))
    }

    func test_없는_파일은_backend_오류다() {
        let result = load(WGPUFileAssetProvider(), directory.appendingPathComponent("absent.bin").path)

        guard case .failure(let error) = result else { return XCTFail("성공하면 안 된다") }
        XCTAssertEqual(error.kind, .backend)
    }

    // MARK: - allowedRoots 제한

    func test_허용_디렉토리_안은_통과한다() throws {
        let url = try write([1], to: "inside.bin")
        let provider = WGPUFileAssetProvider(allowedRoots: [directory])

        XCTAssertEqual(try load(provider, url.path).get(), Data([1]))
    }

    func test_허용_디렉토리_밖은_validation_오류다() throws {
        let provider = WGPUFileAssetProvider(allowedRoots: [directory.appendingPathComponent("sub")])
        let url = try write([1], to: "outside.bin")

        guard case .failure(let error) = load(provider, url.path) else { return XCTFail("성공하면 안 된다") }
        XCTAssertEqual(error.kind, .validation)
    }

    func test_상위로_거슬러_올라가는_경로는_막힌다() throws {
        _ = try write([1], to: "secret.bin")
        let sub = directory.appendingPathComponent("sub", isDirectory: true)
        try FileManager.default.createDirectory(at: sub, withIntermediateDirectories: true)
        let provider = WGPUFileAssetProvider(allowedRoots: [sub])

        // `sub/../secret.bin`은 표기상 sub 아래지만 실제로는 밖이다.
        let sneaky = sub.appendingPathComponent("../secret.bin").path
        guard case .failure(let error) = load(provider, sneaky) else { return XCTFail("성공하면 안 된다") }
        XCTAssertEqual(error.kind, .validation)
    }

    func test_이름이_접두사만_같은_형제_디렉토리는_막힌다() throws {
        // 허용 루트가 `…/sub`일 때 `…/subevil/x`가 새어 나가면 안 된다.
        let sub = directory.appendingPathComponent("sub", isDirectory: true)
        let evil = directory.appendingPathComponent("subevil", isDirectory: true)
        try FileManager.default.createDirectory(at: sub, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: evil, withIntermediateDirectories: true)
        try Data([1]).write(to: evil.appendingPathComponent("x.bin"))
        let provider = WGPUFileAssetProvider(allowedRoots: [sub])

        guard case .failure = load(provider, evil.appendingPathComponent("x.bin").path) else {
            return XCTFail("성공하면 안 된다")
        }
    }

    // MARK: - 번들 상대 이름

    func test_번들에서_이름으로_찾는다() throws {
        _ = try write([5, 6], to: "asset.bin")
        let provider = WGPUFileAssetProvider(bundle: Bundle(url: directory)!)

        XCTAssertEqual(try load(provider, "asset.bin").get(), Data([5, 6]))
    }

    func test_번들_하위_디렉토리도_찾는다() throws {
        let sub = directory.appendingPathComponent("LUTs", isDirectory: true)
        try FileManager.default.createDirectory(at: sub, withIntermediateDirectories: true)
        try Data([3]).write(to: sub.appendingPathComponent("neutral.cube"))
        let provider = WGPUFileAssetProvider(bundle: Bundle(url: directory)!)

        XCTAssertEqual(try load(provider, "LUTs/neutral.cube").get(), Data([3]))
    }

    func test_번들_이름에_상위_경로가_섞이면_validation_오류다() {
        let provider = WGPUFileAssetProvider(bundle: Bundle(url: directory)!)

        guard case .failure(let error) = load(provider, "LUTs/../secret.bin") else {
            return XCTFail("성공하면 안 된다")
        }
        XCTAssertEqual(error.kind, .validation)
    }

    func test_빈_이름은_validation_오류다() {
        guard case .failure(let error) = load(WGPUFileAssetProvider(), "") else {
            return XCTFail("성공하면 안 된다")
        }
        XCTAssertEqual(error.kind, .validation)
    }

    func test_기본_공급자는_https_URL을_해석하지_않는다() {
        // 네트워크는 기본 스코프 밖이다 — 번들 이름으로 떨어져 "없다"로 끝나야 한다.
        guard case .failure(let error) = load(WGPUFileAssetProvider(), "https://example.com/a.bin") else {
            return XCTFail("성공하면 안 된다")
        }
        XCTAssertEqual(error.kind, .validation)
    }
}

/// 공급자를 갈아끼우면 브리지 경로(`WGPUAssetLoading`)의 스코프가 실제로 바뀐다는 계약.
///
/// 예시 공급자는 기본과 정반대다 — https URL만 받고 파일 경로는 거부한다.
final class AssetProviderSwapTests: XCTestCase {

    /// https URL만 허용하는 공급자. 네트워크 대신 canned 데이터를 준다 —
    /// 검증 대상은 전송이 아니라 **스코프 규칙이 공급자를 따라간다**는 것이다.
    private final class HTTPSOnlyProvider: WGPUAssetProvider {
        var served: [String: Data] = [:]

        func loadAsset(named name: String, completion: @escaping (Result<Data, WGPUError>) -> Void) {
            guard name.hasPrefix("https://") else {
                completion(.failure(.validation("https URL만 허용한다: \(name)")))
                return
            }
            guard let data = served[name] else {
                completion(.failure(.backend("가져오지 못했다: \(name)")))
                return
            }
            completion(.success(data))
        }
    }

    private func load(
        _ provider: WGPUAssetProvider, _ params: [String: Any]
    ) -> [String: Any] {
        let expectation = expectation(description: "load")
        var received: [String: Any]!
        WGPUAssetLoading.load(params, provider: provider) { payload in
            received = payload
            expectation.fulfill()
        }
        wait(for: [expectation], timeout: 5)
        return received
    }

    private func firstError(_ payload: [String: Any]) -> [String: Any]? {
        (payload["errors"] as? [[String: Any]])?.first
    }

    func test_교체한_공급자는_https_URL을_허용한다() {
        let provider = HTTPSOnlyProvider()
        provider.served["https://example.com/lut.cube"] = Data([1, 2])

        let payload = load(provider, ["name": "https://example.com/lut.cube"])

        XCTAssertEqual(payload["ok"] as? Bool, true)
        XCTAssertEqual(payload["data"] as? Data, Data([1, 2]))
        XCTAssertEqual(payload["byteLength"] as? Int, 2)
    }

    func test_교체한_공급자는_기본이_허용하던_파일_경로를_차단한다() throws {
        // 실존하는 파일이라도 공급자가 거부하면 못 읽는다 — 스코프는 공급자의 것이다.
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("swap-\(UUID().uuidString).bin")
        try Data([9]).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        XCTAssertEqual(try WGPUFileAssetProvider().loadSync(url.path), Data([9]))  // 기본은 통과
        let payload = load(HTTPSOnlyProvider(), ["name": url.path])                // 교체 후 차단

        XCTAssertEqual(payload["ok"] as? Bool, false)
        XCTAssertEqual(firstError(payload)?["kind"] as? String, "validation")
    }

    func test_공급자의_오류_분류가_JS_페이로드까지_그대로_간다() {
        let payload = load(HTTPSOnlyProvider(), ["name": "https://example.com/absent.bin"])

        XCTAssertEqual(payload["ok"] as? Bool, false)
        XCTAssertEqual(firstError(payload)?["kind"] as? String, "backend")
    }

    func test_name이_없으면_공급자까지_가지_않고_validation_오류다() {
        let payload = load(HTTPSOnlyProvider(), [:])

        XCTAssertEqual(payload["ok"] as? Bool, false)
        XCTAssertEqual(firstError(payload)?["kind"] as? String, "validation")
    }
}

private extension WGPUFileAssetProvider {
    /// 테스트 편의 — 동기로 결과를 꺼낸다.
    func loadSync(_ name: String) throws -> Data {
        let semaphore = DispatchSemaphore(value: 0)
        var received: Result<Data, WGPUError>!
        loadAsset(named: name) { result in
            received = result
            semaphore.signal()
        }
        semaphore.wait()
        return try received.get()
    }
}
