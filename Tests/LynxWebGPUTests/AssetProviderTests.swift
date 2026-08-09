import XCTest
import LynxWebGPUCore
@testable import LynxWebGPU

/// The default asset provider's contract — it resolves registered name → file path → bundle, in that
/// order, and with `allowedRoots` set, file path access is confined beneath them.
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

    // MARK: - Registered in-memory data

    func test_registeredDataComesBackByName() {
        let provider = WGPUFileAssetProvider()
        provider.register(Data([1, 2, 3]), for: "picked-image")

        XCTAssertEqual(try load(provider, "picked-image").get(), Data([1, 2, 3]))
    }

    func test_afterUnregisteringItNoLongerResolves() {
        let provider = WGPUFileAssetProvider(bundle: Bundle(url: directory)!)
        provider.register(Data([1]), for: "gone")
        provider.unregister("gone")

        XCTAssertThrowsError(try load(provider, "gone").get())
    }

    func test_aRegisteredNameTakesPrecedenceOverAFile() throws {
        let url = try write([9, 9], to: "clash.bin")
        let provider = WGPUFileAssetProvider()
        provider.register(Data([1]), for: url.path)

        XCTAssertEqual(try load(provider, url.path).get(), Data([1]))
    }

    // MARK: - File paths (default: everything allowed)

    func test_readsAnAbsolutePath() throws {
        let url = try write([10, 20, 30], to: "path.bin")

        XCTAssertEqual(try load(WGPUFileAssetProvider(), url.path).get(), Data([10, 20, 30]))
    }

    func test_readsAFileURL() throws {
        let url = try write([7], to: "url.bin")

        XCTAssertEqual(try load(WGPUFileAssetProvider(), url.absoluteString).get(), Data([7]))
    }

    func test_aMissingFileIsABackendError() {
        let result = load(WGPUFileAssetProvider(), directory.appendingPathComponent("absent.bin").path)

        guard case .failure(let error) = result else { return XCTFail("it must not succeed") }
        XCTAssertEqual(error.kind, .backend)
    }

    // MARK: - allowedRoots restriction

    func test_insideTheAllowedDirectoriesPasses() throws {
        let url = try write([1], to: "inside.bin")
        let provider = WGPUFileAssetProvider(allowedRoots: [directory])

        XCTAssertEqual(try load(provider, url.path).get(), Data([1]))
    }

    func test_outsideTheAllowedDirectoriesIsAValidationError() throws {
        let provider = WGPUFileAssetProvider(allowedRoots: [directory.appendingPathComponent("sub")])
        let url = try write([1], to: "outside.bin")

        guard case .failure(let error) = load(provider, url.path) else { return XCTFail("it must not succeed") }
        XCTAssertEqual(error.kind, .validation)
    }

    func test_aPathClimbingToAParentIsBlocked() throws {
        _ = try write([1], to: "secret.bin")
        let sub = directory.appendingPathComponent("sub", isDirectory: true)
        try FileManager.default.createDirectory(at: sub, withIntermediateDirectories: true)
        let provider = WGPUFileAssetProvider(allowedRoots: [sub])

        // `sub/../secret.bin` reads as being under sub but actually lies outside.
        let sneaky = sub.appendingPathComponent("../secret.bin").path
        guard case .failure(let error) = load(provider, sneaky) else { return XCTFail("it must not succeed") }
        XCTAssertEqual(error.kind, .validation)
    }

    func test_aSiblingDirectorySharingOnlyThePrefixIsBlocked() throws {
        // With the allowed root at `…/sub`, `…/subevil/x` must not leak through.
        let sub = directory.appendingPathComponent("sub", isDirectory: true)
        let evil = directory.appendingPathComponent("subevil", isDirectory: true)
        try FileManager.default.createDirectory(at: sub, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: evil, withIntermediateDirectories: true)
        try Data([1]).write(to: evil.appendingPathComponent("x.bin"))
        let provider = WGPUFileAssetProvider(allowedRoots: [sub])

        guard case .failure = load(provider, evil.appendingPathComponent("x.bin").path) else {
            return XCTFail("it must not succeed")
        }
    }

    // MARK: - Bundle-relative names

    func test_findsByNameInTheBundle() throws {
        _ = try write([5, 6], to: "asset.bin")
        let provider = WGPUFileAssetProvider(bundle: Bundle(url: directory)!)

        XCTAssertEqual(try load(provider, "asset.bin").get(), Data([5, 6]))
    }

    func test_findsBundleSubdirectoriesToo() throws {
        let sub = directory.appendingPathComponent("LUTs", isDirectory: true)
        try FileManager.default.createDirectory(at: sub, withIntermediateDirectories: true)
        try Data([3]).write(to: sub.appendingPathComponent("neutral.cube"))
        let provider = WGPUFileAssetProvider(bundle: Bundle(url: directory)!)

        XCTAssertEqual(try load(provider, "LUTs/neutral.cube").get(), Data([3]))
    }

    func test_aParentPathInABundleNameIsAValidationError() {
        let provider = WGPUFileAssetProvider(bundle: Bundle(url: directory)!)

        guard case .failure(let error) = load(provider, "LUTs/../secret.bin") else {
            return XCTFail("it must not succeed")
        }
        XCTAssertEqual(error.kind, .validation)
    }

    func test_anEmptyNameIsAValidationError() {
        guard case .failure(let error) = load(WGPUFileAssetProvider(), "") else {
            return XCTFail("it must not succeed")
        }
        XCTAssertEqual(error.kind, .validation)
    }

    func test_theDefaultProviderDoesNotResolveHTTPSURLs() {
        // The network is outside the default scope — it falls through as a bundle name and ends as "absent".
        guard case .failure(let error) = load(WGPUFileAssetProvider(), "https://example.com/a.bin") else {
            return XCTFail("it must not succeed")
        }
        XCTAssertEqual(error.kind, .validation)
    }
}

/// The contract that swapping the provider really changes the scope of the bridge path (`WGPUAssetLoading`).
///
/// The example provider is the opposite of the default — it takes only https URLs and refuses file paths.
final class AssetProviderSwapTests: XCTestCase {

    /// A provider allowing only https URLs. It returns canned data instead of hitting the network —
    /// what is under test is not transport but that **the scope rules follow the provider**.
    private final class HTTPSOnlyProvider: WGPUAssetProvider {
        var served: [String: Data] = [:]

        func loadAsset(named name: String, completion: @escaping (Result<Data, WGPUError>) -> Void) {
            guard name.hasPrefix("https://") else {
                completion(.failure(.validation("only https URLs are allowed: \(name)")))
                return
            }
            guard let data = served[name] else {
                completion(.failure(.backend("could not fetch: \(name)")))
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

    func test_aReplacedProviderCanAllowHTTPSURLs() {
        let provider = HTTPSOnlyProvider()
        provider.served["https://example.com/lut.cube"] = Data([1, 2])

        let payload = load(provider, ["name": "https://example.com/lut.cube"])

        XCTAssertEqual(payload["ok"] as? Bool, true)
        XCTAssertEqual(payload["data"] as? Data, Data([1, 2]))
        XCTAssertEqual(payload["byteLength"] as? Int, 2)
    }

    func test_aReplacedProviderCanBlockFilePathsTheDefaultAllowed() throws {
        // Even a file that exists cannot be read once the provider refuses — the scope is the provider's.
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("swap-\(UUID().uuidString).bin")
        try Data([9]).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        XCTAssertEqual(try WGPUFileAssetProvider().loadSync(url.path), Data([9]))  // the default passes
        let payload = load(HTTPSOnlyProvider(), ["name": url.path])                // blocked after the swap

        XCTAssertEqual(payload["ok"] as? Bool, false)
        XCTAssertEqual(firstError(payload)?["kind"] as? String, "validation")
    }

    func test_theProvidersErrorKindReachesTheJSPayloadUnchanged() {
        let payload = load(HTTPSOnlyProvider(), ["name": "https://example.com/absent.bin"])

        XCTAssertEqual(payload["ok"] as? Bool, false)
        XCTAssertEqual(firstError(payload)?["kind"] as? String, "backend")
    }

    func test_withNoNameItIsAValidationErrorBeforeReachingTheProvider() {
        let payload = load(HTTPSOnlyProvider(), [:])

        XCTAssertEqual(payload["ok"] as? Bool, false)
        XCTAssertEqual(firstError(payload)?["kind"] as? String, "validation")
    }
}

private extension WGPUFileAssetProvider {
    /// Test convenience — pulls the result out synchronously.
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
