import XCTest
import Metal
import LynxWebGPU
import LynxWebGPUConformance

/// Runs **the default runtime (Metal) on iOS** through the conformance suite.
///
/// `swift test`'s `ConformanceTests` runs the same suite, but that is **macOS** — a different driver
/// and a different GPU family. Where this library actually ships is iOS, and with no place measuring
/// the default backend there, only the experimental backend (Dawn) would get iOS verification — backwards.
/// It runs **the same suite at the same destination** as `DawnCheck`, so both backends' results sit side by side.
///
///   arch -arm64 xcodebuild -workspace LynxWebGPUDemo.xcworkspace -scheme WebGPUCheck \
///     -destination 'platform=iOS Simulator,name=iPhone 17,OS=26.2' test
final class MetalConformanceTests: XCTestCase {

    func test_theMetalRuntimeRunsTheConformanceSuite() throws {
        let runtime = try LynxWebGPUContext()
        let outcomes = WebGPUConformance.run(on: runtime)
        XCTAssertEqual(
            outcomes.count, WebGPUConformance.checks.count,
            "the check count disagrees — the suite was cut short"
        )
        for outcome in outcomes where outcome.status == .skipped {
            print("conformance skipped — [\(outcome.name)] \(outcome.detail)")
        }
        for outcome in outcomes where outcome.status == .failed {
            XCTFail("[\(outcome.name)] \(outcome.detail)")
        }
        print(WebGPUConformance.summary(outcomes))
    }
}

final class MetalSmokeTests: XCTestCase {

    /// A Metal device comes up on the simulator or hardware and the runtime mounts on it.
    func test_theMetalRuntimeStartsUp() throws {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice(), "no Metal device")
        print("Metal device — \(device.name)")

        let runtime = try LynxWebGPUContext(device: device)
        let info = runtime.adapterInfo()
        XCTAssertEqual(info["ok"] as? Bool ?? true, true, "\(info)")
        let limits = try XCTUnwrap(info["limits"] as? [String: Any], "no limits")
        XCTAssertGreaterThan(limits["maxTextureDimension2D"] as? Int ?? 0, 0)
        print("adapter limits — maxTextureDimension2D: \(limits["maxTextureDimension2D"] ?? "?")")
    }
}
