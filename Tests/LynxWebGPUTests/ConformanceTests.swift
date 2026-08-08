import XCTest
import Metal
import CoreGraphics
import QuartzCore
import LynxWebGPUCore
import LynxWebGPUConformance
@testable import LynxWebGPU

/// 기본 런타임(`LynxWebGPUContext`)을 적합성 스위트에 건다.
///
/// **여기서 도는 검사는 하나도 Metal을 모른다** — 커맨드 스트림과 `WebGPURuntime`만 쓴다
/// (`Sources/LynxWebGPUConformance`). 그래서 다른 런타임을 만들면 같은 파일을 그대로
/// 걸어 두 구현이 같은 그림을 그리는지 기계로 확인할 수 있다.
///
/// 이 저장소의 나머지 GPU 테스트는 두 종류로 갈린다:
/// - **계약** (`CommandInterpreterTests` `RenderPipelineTests` `ErrorScopeTests` `StencilTests`
///   `QuerySetTests` `RenderBundleTests` `IndirectDrawTests` `CompressedTextureTests`
///   `ExternalImageTests` `OffscreenReadbackTests`) — 커맨드 스트림 수준. 다른 런타임에도
///   그대로 옮길 수 있다. 이 스위트가 그중 **핵심을 추려** 라이브러리로 옮겨 둔 것이다.
/// - **Metal 내부** (`MetalMappingTests` `StagingPoolTests` `SurfaceInFlightTests`
///   `RenderHarnessTests`) — 인자 테이블 배정·스테이징 풀·드로어블 회계처럼 이 백엔드에만
///   있는 것. 다른 런타임으로 옮겨지지 않는다.
final class ConformanceTests: XCTestCase {

    func test_기본런타임이_적합성_스위트를_통과한다() throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "Metal 디바이스 없음")
        let runtime = try LynxWebGPUContext()

        let outcomes = WebGPUConformance.run(on: runtime)
        XCTAssertFalse(outcomes.isEmpty, "검사가 하나도 돌지 않았다")

        for outcome in outcomes where outcome.status == .failed {
            XCTFail("[\(outcome.name)] \(outcome.detail)")
        }
        // 건너뛴 검사는 실패가 아니지만 **조용히 지나가면 커버리지 착시가 생긴다.**
        for outcome in outcomes where outcome.status == .skipped {
            print("적합성 건너뜀 — [\(outcome.name)] \(outcome.detail)")
        }
        print(WebGPUConformance.summary(outcomes))
    }

    /// 스위트가 **런타임을 검사 사이에 초기화**한다 — 앞 검사의 객체가 남으면 뒤 검사의
    /// 판정이 우연에 기대게 된다.
    func test_검사마다_런타임_상태가_초기화된다() throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "Metal 디바이스 없음")
        let runtime = try LynxWebGPUContext()

        runtime.execute(commands: [
            ["op": "createBuffer", "id": 777, "size": 16, "usage": 0x0040],
        ])
        XCTAssertGreaterThan(runtime.liveObjectCount, 0)

        _ = WebGPUConformance.run(on: runtime, only: ["clear-color"])

        // 스위트가 지나간 뒤 777번이 살아 있으면 초기화가 안 된 것이다.
        let result = runtime.execute(commands: [
            ["op": "beginRenderPass", "colorAttachments": [["view": 777]]],
        ])
        XCTAssertEqual(result["ok"] as? Bool, false, "앞선 배치의 핸들이 스위트 뒤에도 살아 있다")
    }

    /// **스위트가 실제로 걸러 내는가.**
    ///
    /// 항상 통과하는 적합성 스위트는 쓸모가 없다 — "Dawn 런타임이 19/19"라는 문장이 아무것도
    /// 보증하지 못하게 된다. 계약을 일부러 어기는 런타임을 걸어 실패가 나오는지 확인한다.
    func test_계약을_어기면_스위트가_잡아낸다() throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "Metal 디바이스 없음")
        let runtime = try LynxWebGPUContext()

        // (1) 오류 스코프 결과를 삼키는 런타임
        let swallowing = MisbehavingRuntime(runtime) { result in
            var broken = result
            broken.removeValue(forKey: "errorScopes")
            return broken
        }
        let scopeOutcomes = WebGPUConformance.run(on: swallowing, only: ["error-scope-capture"])
        XCTAssertEqual(scopeOutcomes.first?.status, .failed, "스코프 결과를 삼켰는데 통과했다")

        // (2) 오류를 숨겨 항상 성공했다고 답하는 런타임 — 오류 누적 계약이 깨진다.
        let lying = MisbehavingRuntime(runtime) { result in
            var broken = result
            broken["ok"] = true
            broken.removeValue(forKey: "errors")
            return broken
        }
        let errorOutcomes = WebGPUConformance.run(on: lying, only: ["error-accumulation"])
        XCTAssertEqual(errorOutcomes.first?.status, .failed, "오류를 숨겼는데 통과했다")

        // (3) present:false를 무시하고 항상 프레임을 닫는 런타임 — 중간 제출 계약이 깨진다.
        let impatient = MisbehavingRuntime(runtime)
        impatient.corruptPayload = { payload in
            var broken = payload
            broken["present"] = true
            return broken
        }
        let frameOutcomes = WebGPUConformance.run(on: impatient, only: ["present-false-preserves-frame"])
        XCTAssertEqual(frameOutcomes.first?.status, .failed, "프레임을 조기에 닫았는데 통과했다")

        // (4) readBuffer 응답에서 데이터를 빼먹는 런타임
        let dataless = MisbehavingRuntime(runtime)
        dataless.corruptReadBuffer = { result in
            var broken = result
            broken.removeValue(forKey: "data")
            return broken
        }
        let readOutcomes = WebGPUConformance.run(on: dataless, only: ["read-buffer-contract"])
        XCTAssertEqual(readOutcomes.first?.status, .failed, "readBuffer 데이터를 뺐는데 통과했다")

        // (5) 진단에서 줄 번호를 빼는 런타임 — GPUCompilationMessage 모양이 깨진다.
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
        XCTAssertEqual(infoOutcomes.first?.status, .failed, "진단 키를 뺐는데 통과했다")

        // (6) resize를 삼키는 런타임
        let rigid = MisbehavingRuntime(runtime)
        rigid.swallowResize = true
        let resizeOutcomes = WebGPUConformance.run(on: rigid, only: ["resize-canvas"])
        XCTAssertEqual(resizeOutcomes.first?.status, .failed, "resize를 삼켰는데 통과했다")

        // (7) 준비 신호가 꺼진 채 굳은 런타임
        let stuck = MisbehavingRuntime(runtime)
        stuck.forcedReadiness = false
        let readyOutcomes = WebGPUConformance.run(on: stuck, only: ["frame-readiness"])
        XCTAssertEqual(readyOutcomes.first?.status, .failed, "준비 신호가 false인데 통과했다")
    }
}

/// 다른 런타임을 감싸 **응답을 일부러 망가뜨린다.** 스위트의 변별력을 재는 데만 쓴다.
///
/// 이 클래스가 컴파일된다는 것 자체가 `WebGPURuntime`이 저장소 밖에서도 구현 가능한
/// 모양임을 보여 준다 — Dawn 런타임이 채워야 할 자리가 정확히 이만큼이다.
private final class MisbehavingRuntime: WebGPURuntime {
    private let inner: WebGPURuntime
    private let corrupt: ([String: Any]) -> [String: Any]

    /// execute **전에** 페이로드를 변조한다 — present 강제 같은 프레임 경계 위반용.
    var corruptPayload: (([String: Any]) -> [String: Any])?
    /// readBuffer 콜백 결과를 변조한다.
    var corruptReadBuffer: (([String: Any]) -> [String: Any])?
    /// shaderCompilationInfo 결과를 변조한다.
    var corruptCompilationInfo: (([String: Any]) -> [String: Any])?
    /// resizeCanvas를 조용히 삼킨다.
    var swallowResize = false
    /// isReadyForNextFrame을 강제한다.
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
