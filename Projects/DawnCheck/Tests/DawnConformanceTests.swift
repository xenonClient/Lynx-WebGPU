import XCTest
import WebGPU
import LynxWebGPUConformance

/// Dawn 런타임을 **Metal 런타임과 같은 적합성 스위트**에 건다 — 이것이 이 프로젝트의 목적이다.
/// 28개 검사 전부가 같은 계약(`docs/COMMAND-STREAM.md`)으로 판정된다.
final class DawnConformanceTests: XCTestCase {

    func test_Dawn런타임이_적합성_스위트를_돈다() throws {
        let runtime = try DawnWebGPURuntime()
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

final class DawnSmokeTests: XCTestCase {

    /// XCFramework 링크 → 인스턴스 → 어댑터 → 디바이스가 시뮬레이터에서 선다.
    func test_Dawn이_시뮬레이터에서_기동된다() throws {
        guard let instance = wgpuCreateInstance(nil) else {
            return XCTFail("wgpuCreateInstance가 nil을 돌려줬다")
        }
        defer { wgpuInstanceRelease(instance) }

        let adapter = try DawnBootstrap.requestAdapter(instance: instance)
        defer { wgpuAdapterRelease(adapter) }

        var info = WGPUAdapterInfo()
        XCTAssertEqual(wgpuAdapterGetInfo(adapter, &info), WGPUStatus_Success)
        print("Dawn 어댑터 — vendor: \(String(wgpu: info.vendor)), "
              + "architecture: \(String(wgpu: info.architecture)), "
              + "description: \(String(wgpu: info.description))")

        let device = try DawnBootstrap.requestDevice(
            instance: instance, adapter: adapter, onUncapturedError: { type, message in
                print("uncaptured error [\(type)] \(message)")
            }
        )
        defer { wgpuDeviceRelease(device) }

        var limits = WGPULimits()
        XCTAssertEqual(wgpuDeviceGetLimits(device, &limits), WGPUStatus_Success)
        XCTAssertGreaterThan(limits.maxTextureDimension2D, 0)
        print("Dawn 디바이스 한계 — maxTextureDimension2D: \(limits.maxTextureDimension2D)")
    }
}
