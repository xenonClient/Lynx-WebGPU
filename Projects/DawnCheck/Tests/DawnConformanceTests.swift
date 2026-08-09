import XCTest
import WebGPU
import LynxWebGPUConformance

/// Puts the Dawn runtime through **the same conformance suite as the Metal runtime** — that is this project's purpose.
/// Every check is judged by the same contract (`docs/COMMAND-STREAM.md`).
final class DawnConformanceTests: XCTestCase {

    func test_theDawnRuntimeRunsTheConformanceSuite() throws {
        let runtime = try DawnWebGPURuntime()
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

final class DawnSmokeTests: XCTestCase {

    /// The XCFramework link → instance → adapter → device comes up on the simulator.
    func test_dawnStartsUpOnTheSimulator() throws {
        guard let instance = wgpuCreateInstance(nil) else {
            return XCTFail("wgpuCreateInstance returned nil")
        }
        defer { wgpuInstanceRelease(instance) }

        let adapter = try DawnBootstrap.requestAdapter(instance: instance)
        defer { wgpuAdapterRelease(adapter) }

        var info = WGPUAdapterInfo()
        XCTAssertEqual(wgpuAdapterGetInfo(adapter, &info), WGPUStatus_Success)
        print("Dawn adapter — vendor: \(String(wgpu: info.vendor)), "
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
        print("Dawn device limits — maxTextureDimension2D: \(limits.maxTextureDimension2D)")
    }
}
