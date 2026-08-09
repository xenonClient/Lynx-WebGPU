import XCTest
import Metal
import LynxWebGPU
import LynxWebGPUConformance

/// **기본 런타임(Metal)을 iOS에서** 적합성 스위트에 건다.
///
/// `swift test`의 `ConformanceTests`도 같은 스위트를 돌리지만 그건 **macOS**다 — 드라이버도
/// GPU 패밀리도 다르다. 정작 이 라이브러리가 실려 나가는 곳은 iOS인데, 거기서 기본 백엔드를
/// 재는 자리가 없으면 실험 백엔드(Dawn)만 iOS 검증을 받는 뒤집힌 모양이 된다.
/// `DawnCheck`와 **같은 스위트·같은 목적지**로 돌아서, 두 백엔드의 결과를 그대로 나란히 볼 수 있다.
///
///   arch -arm64 xcodebuild -workspace LynxWebGPUDemo.xcworkspace -scheme WebGPUCheck \
///     -destination 'platform=iOS Simulator,name=iPhone 17,OS=26.2' test
final class MetalConformanceTests: XCTestCase {

    func test_Metal런타임이_적합성_스위트를_돈다() throws {
        let runtime = try LynxWebGPUContext()
        let outcomes = WebGPUConformance.run(on: runtime)
        XCTAssertEqual(
            outcomes.count, WebGPUConformance.checks.count,
            "검사 수가 어긋난다 — 스위트가 중간에 끊겼다"
        )
        for outcome in outcomes where outcome.status == .skipped {
            print("적합성 건너뜀 — [\(outcome.name)] \(outcome.detail)")
        }
        for outcome in outcomes where outcome.status == .failed {
            XCTFail("[\(outcome.name)] \(outcome.detail)")
        }
        print(WebGPUConformance.summary(outcomes))
    }
}

final class MetalSmokeTests: XCTestCase {

    /// Metal 디바이스가 시뮬레이터/기기에서 서고, 런타임이 그 위에 올라간다.
    func test_Metal런타임이_기동된다() throws {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice(), "Metal 디바이스가 없다")
        print("Metal 디바이스 — \(device.name)")

        let runtime = try LynxWebGPUContext(device: device)
        let info = runtime.adapterInfo()
        XCTAssertEqual(info["ok"] as? Bool ?? true, true, "\(info)")
        let limits = try XCTUnwrap(info["limits"] as? [String: Any], "limits가 없다")
        XCTAssertGreaterThan(limits["maxTextureDimension2D"] as? Int ?? 0, 0)
        print("어댑터 한계 — maxTextureDimension2D: \(limits["maxTextureDimension2D"] ?? "?")")
    }
}
