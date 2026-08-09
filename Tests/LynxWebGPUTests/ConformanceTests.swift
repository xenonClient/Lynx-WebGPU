import XCTest
import Metal
import CoreGraphics
import QuartzCore
import LynxWebGPUCore
import LynxWebGPUConformance
@testable import LynxWebGPU

/// Runs the default runtime (`LynxWebGPUContext`) through the conformance suite.
///
/// **Not one check here knows anything about Metal** — they use only the command stream and
/// `WebGPURuntime` (`Sources/LynxWebGPUConformance`). So building another runtime lets you point the
/// same file at it and check mechanically that both implementations draw the same picture.
///
/// The rest of this repository's GPU tests split in two:
/// - **Contract** (`CommandInterpreterTests`, `RenderPipelineTests`, `ErrorScopeTests`, `StencilTests`,
///   `QuerySetTests` `RenderBundleTests` `IndirectDrawTests` `CompressedTextureTests`
///   `ExternalImageTests`, `OffscreenReadbackTests`) — command stream level. They carry over to another
///   runtime unchanged, and this suite is **the core of them** moved into a library.
/// - **Metal internals** (`MetalMappingTests`, `StagingPoolTests`, `SurfaceInFlightTests`,
///   `RenderHarnessTests`) — argument table assignment, the staging pool, drawable accounting: things
///   only this backend has. They do not carry over.
final class ConformanceTests: XCTestCase {

    func test_theDefaultRuntimePassesTheConformanceSuite() throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "no Metal device")
        let runtime = try LynxWebGPUContext()

        let outcomes = WebGPUConformance.run(on: runtime)
        XCTAssertFalse(outcomes.isEmpty, "not a single check ran")

        for outcome in outcomes where outcome.status == .failed {
            XCTFail("[\(outcome.name)] \(outcome.detail)")
        }
        // A skipped check is not a failure, but **passing over it silently creates an illusion of coverage.**
        for outcome in outcomes where outcome.status == .skipped {
            print("conformance skipped — [\(outcome.name)] \(outcome.detail)")
        }
        print(WebGPUConformance.summary(outcomes))
    }

    /// The suite **resets the runtime between checks** — objects left by one check would make the next
    /// one's verdict depend on chance.
    func test_runtimeStateIsResetForEveryCheck() throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "no Metal device")
        let runtime = try LynxWebGPUContext()

        runtime.execute(commands: [
            ["op": "createBuffer", "id": 777, "size": 16, "usage": 0x0040],
        ])
        XCTAssertGreaterThan(runtime.liveObjectCount, 0)

        _ = WebGPUConformance.run(on: runtime, only: ["clear-color"])

        // If handle 777 is still alive after the suite has run, the reset did not happen.
        let result = runtime.execute(commands: [
            ["op": "beginRenderPass", "colorAttachments": [["view": 777]]],
        ])
        XCTAssertEqual(result["ok"] as? Bool, false, "a handle from an earlier batch survived the suite")
    }

    /// **Does the suite actually filter?**
    ///
    /// An always-passing conformance suite is useless — it would make "the Dawn runtime scores 19/19"
    /// guarantee nothing. We point it at runtimes that break the contract on purpose and check that failures come out.
    func test_theSuiteCatchesAContractViolation() throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "no Metal device")
        let runtime = try LynxWebGPUContext()

        // (1) a runtime swallowing error scope results
        let swallowing = MisbehavingRuntime(runtime) { result in
            var broken = result
            broken.removeValue(forKey: "errorScopes")
            return broken
        }
        let scopeOutcomes = WebGPUConformance.run(on: swallowing, only: ["error-scope-capture"])
        XCTAssertEqual(scopeOutcomes.first?.status, .failed, "it swallowed the scope result and still passed")

        // (2) a runtime hiding errors and always answering success — the error accumulation contract breaks.
        let lying = MisbehavingRuntime(runtime) { result in
            var broken = result
            broken["ok"] = true
            broken.removeValue(forKey: "errors")
            return broken
        }
        let errorOutcomes = WebGPUConformance.run(on: lying, only: ["error-accumulation"])
        XCTAssertEqual(errorOutcomes.first?.status, .failed, "it hid the errors and still passed")

        // (3) a runtime ignoring present:false and always closing the frame — the mid-frame submit contract breaks.
        let impatient = MisbehavingRuntime(runtime)
        impatient.corruptPayload = { payload in
            var broken = payload
            broken["present"] = true
            return broken
        }
        let frameOutcomes = WebGPUConformance.run(on: impatient, only: ["present-false-preserves-frame"])
        XCTAssertEqual(frameOutcomes.first?.status, .failed, "it closed the frame early and still passed")

        // (4) a runtime dropping the data from the readBuffer response
        let dataless = MisbehavingRuntime(runtime)
        dataless.corruptReadBuffer = { result in
            var broken = result
            broken.removeValue(forKey: "data")
            return broken
        }
        let readOutcomes = WebGPUConformance.run(on: dataless, only: ["read-buffer-contract"])
        XCTAssertEqual(readOutcomes.first?.status, .failed, "it dropped the readBuffer data and still passed")

        // (5) a runtime dropping the line number from diagnostics — the GPUCompilationMessage shape breaks.
        let vague = MisbehavingRuntime(runtime)
        vague.corruptCompilationInfo = { result in
            var broken = result
            if var messages = broken["messages"] as? [[String: Any]] {
                for index in messages.indices { messages[index].removeValue(forKey: "lineNum") }
                broken["messages"] = messages
            }
            return broken
        }
        let infoOutcomes = WebGPUConformance.run(on: vague, only: ["shader-compilation-info"])
        XCTAssertEqual(infoOutcomes.first?.status, .failed, "it dropped a diagnostic key and still passed")

        // (6) a runtime swallowing resize
        let rigid = MisbehavingRuntime(runtime)
        rigid.swallowResize = true
        let resizeOutcomes = WebGPUConformance.run(on: rigid, only: ["resize-canvas"])
        XCTAssertEqual(resizeOutcomes.first?.status, .failed, "it swallowed resize and still passed")

        // (7) a runtime frozen with the readiness signal off
        let stuck = MisbehavingRuntime(runtime)
        stuck.forcedReadiness = false
        let readyOutcomes = WebGPUConformance.run(on: stuck, only: ["frame-readiness"])
        XCTAssertEqual(readyOutcomes.first?.status, .failed, "the readiness signal was false and it still passed")
    }
}

/// Wraps another runtime and **breaks its responses on purpose.** Used only to measure the suite's discrimination.
///
/// That this class compiles at all shows `WebGPURuntime` is implementable from outside the repository —
/// this is exactly the surface a Dawn runtime has to fill.
private final class MisbehavingRuntime: WebGPURuntime {
    private let inner: WebGPURuntime
    private let corrupt: ([String: Any]) -> [String: Any]

    /// Tampers with the payload **before** execute — for frame boundary violations such as forcing present.
    var corruptPayload: (([String: Any]) -> [String: Any])?
    /// Tampers with the readBuffer callback result.
    var corruptReadBuffer: (([String: Any]) -> [String: Any])?
    /// Tampers with the shaderCompilationInfo result.
    var corruptCompilationInfo: (([String: Any]) -> [String: Any])?
    /// Silently swallows resizeCanvas.
    var swallowResize = false
    /// Forces isReadyForNextFrame.
    var forcedReadiness: Bool?

    init(_ inner: WebGPURuntime, corrupt: @escaping ([String: Any]) -> [String: Any] = { $0 }) {
        self.inner = inner
        self.corrupt = corrupt
    }

    func execute(_ payload: [String: Any]) -> [String: Any] {
        corrupt(inner.execute(corruptPayload?(payload) ?? payload))
    }

    func adapterInfo() -> [String: Any] { inner.adapterInfo() }
    func shaderCompilationInfo(handle: Int) -> [String: Any] {
        let result = inner.shaderCompilationInfo(handle: handle)
        return corruptCompilationInfo?(result) ?? result
    }
    func canvasInfo(identifier: String) -> [String: Any] { inner.canvasInfo(identifier: identifier) }

    func readBuffer(handle: Int, offset: Int, size: Int?, completion: @escaping ([String: Any]) -> Void) {
        let corruptReadBuffer = self.corruptReadBuffer
        inner.readBuffer(handle: handle, offset: offset, size: size) { result in
            completion(corruptReadBuffer?(result) ?? result)
        }
    }

    func decodeImage(
        handle: Int, data: Data?, name: String?, options: WGPUImageDecodeOptions,
        provider: WGPUAssetProvider?, completion: @escaping ([String: Any]) -> Void
    ) {
        inner.decodeImage(
            handle: handle, data: data, name: name, options: options,
            provider: provider, completion: completion
        )
    }

    func attachCanvas(identifier: String, layer: CAMetalLayer) {
        inner.attachCanvas(identifier: identifier, layer: layer)
    }
    func attachOffscreenCanvas(identifier: String, size: CGSize) throws {
        try inner.attachOffscreenCanvas(identifier: identifier, size: size)
    }
    func resizeCanvas(identifier: String, drawableSize: CGSize) {
        guard !swallowResize else { return }
        inner.resizeCanvas(identifier: identifier, drawableSize: drawableSize)
    }
    func detachCanvas(identifier: String) { inner.detachCanvas(identifier: identifier) }
    func readCanvasPixels(identifier: String) throws -> WGPUPixelReadback {
        try inner.readCanvasPixels(identifier: identifier)
    }

    var isReadyForNextFrame: Bool { forcedReadiness ?? inner.isReadyForNextFrame }
    func processEvents() { inner.processEvents() }
    func reset() { inner.reset() }
}
