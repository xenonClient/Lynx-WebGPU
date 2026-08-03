import XCTest
import Metal
import LynxWebGPUCore
@testable import LynxWebGPU

/// `pushErrorScope` / `popErrorScope` — **GPU가 필요 없는 프로토콜 계약**이다.
///
/// 오류를 어디로 보낼지 정하는 것이 전부이므로 렌더 결과와 무관하다. 잘못된 커맨드로
/// 오류를 일부러 내고, 그것이 스코프에 잡히는지 / 결과의 `errors`에서 빠지는지를 본다.
final class ErrorScopeTests: XCTestCase {
    private var harness: RenderHarness!

    override func setUpWithError() throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "Metal 디바이스 없음")
        harness = try XCTUnwrap(RenderHarness.make(width: 8, height: 8))
    }

    override func tearDown() {
        harness = nil
        super.tearDown()
    }

    private func errors(_ result: [String: Any]) -> [[String: Any]] {
        result["errors"] as? [[String: Any]] ?? []
    }

    /// pop 결과 배열 — `NSNull`은 "그 스코프에서는 오류가 없었다"는 뜻이다.
    private func scopes(_ result: [String: Any]) -> [[String: Any]?] {
        (result["errorScopes"] as? [Any] ?? []).map { $0 as? [String: Any] }
    }

    /// validation 오류를 하나 내는 명령 (없는 핸들 참조).
    private let failingCommand: [String: Any] = ["op": "setVertexBuffer", "slot": 0, "buffer": 999]

    // MARK: - 기본 동작

    func test_스코프가_오류를_가로채고_결과에서_뺀다() {
        let result = harness.execute([
            ["op": "pushErrorScope", "filter": "validation"],
            failingCommand,
            ["op": "popErrorScope"],
        ])

        // 스코프가 가져갔으므로 전역으로 새지 않는다 — 그래야 JS의 onError가 중복 보고를 안 한다.
        XCTAssertEqual(errors(result).count, 0, "잡힌 오류는 errors에 실리면 안 된다")
        XCTAssertEqual(result["ok"] as? Bool, true, "처리된 오류는 배치를 실패로 만들지 않는다")
        XCTAssertEqual(scopes(result).first??["kind"] as? String, "validation")
    }

    func test_오류가_없으면_스코프는_비어서_돌아온다() {
        let result = harness.executeExpectingSuccess([
            ["op": "pushErrorScope", "filter": "validation"],
            ["op": "createBuffer", "id": 1, "size": 16, "usage": TestUsage.uniform],
            ["op": "popErrorScope"],
        ])

        XCTAssertEqual(scopes(result).count, 1)
        XCTAssertNil(scopes(result).first ?? nil, "오류가 없으면 null이어야 한다")
    }

    func test_스코프_밖의_오류는_전역으로_간다() {
        let result = harness.execute([
            ["op": "pushErrorScope", "filter": "validation"],
            ["op": "popErrorScope"],
            failingCommand,
        ])

        XCTAssertNil(scopes(result).first ?? nil, "닫힌 스코프는 뒤의 오류를 못 잡는다")
        XCTAssertEqual(errors(result).count, 1)
        XCTAssertEqual(errors(result).first?["kind"] as? String, "validation")
    }

    // MARK: - 필터

    func test_필터가_맞지_않으면_바깥으로_흘려보낸다() {
        let result = harness.execute([
            ["op": "pushErrorScope", "filter": "out-of-memory"],
            failingCommand,   // validation
            ["op": "popErrorScope"],
        ])

        XCTAssertNil(scopes(result).first ?? nil, "out-of-memory 스코프는 validation을 안 잡는다")
        XCTAssertEqual(errors(result).count, 1, "못 잡은 오류는 전역으로 나가야 한다")
    }

    func test_중첩_스코프에서는_안쪽_것만_잡는다() {
        let result = harness.execute([
            ["op": "pushErrorScope", "filter": "validation"],   // 바깥
            ["op": "pushErrorScope", "filter": "validation"],   // 안쪽
            failingCommand,
            ["op": "popErrorScope"],                            // 안쪽이 먼저 닫힌다
            ["op": "popErrorScope"],
        ])

        XCTAssertEqual(scopes(result).count, 2)
        XCTAssertEqual(scopes(result)[0]?["kind"] as? String, "validation", "안쪽이 가져간다")
        XCTAssertNil(scopes(result)[1], "바깥은 못 본다")
        XCTAssertEqual(errors(result).count, 0)
    }

    func test_안쪽_필터가_안_맞으면_바깥_스코프가_잡는다() {
        let result = harness.execute([
            ["op": "pushErrorScope", "filter": "validation"],     // 바깥 — 이쪽이 잡아야 한다
            ["op": "pushErrorScope", "filter": "out-of-memory"],  // 안쪽 — 필터가 안 맞는다
            failingCommand,
            ["op": "popErrorScope"],
            ["op": "popErrorScope"],
        ])

        XCTAssertNil(scopes(result)[0], "안쪽은 필터가 달라 못 잡는다")
        XCTAssertEqual(scopes(result)[1]?["kind"] as? String, "validation", "바깥으로 흘러가 잡힌다")
        XCTAssertEqual(errors(result).count, 0)
    }

    func test_알수없는_기능은_validation_스코프가_잡는다() {
        // `unsupported`는 "명세상 유효하지만 이 구현이 아직 안 하는 것"이다. 브라우저에서
        // 같은 코드는 validation으로 나거나 성공하므로, validation 스코프로 잡아야 대응이 된다.
        let result = harness.execute([
            ["op": "pushErrorScope", "filter": "validation"],
            ["op": "텔레포트"],
            ["op": "popErrorScope"],
        ])

        XCTAssertEqual(scopes(result).first??["kind"] as? String, "unsupported")
        XCTAssertEqual(errors(result).count, 0)
    }

    func test_셰이더_컴파일_실패는_internal_스코프가_잡는다() {
        let result = harness.execute([
            ["op": "pushErrorScope", "filter": "internal"],
            ["op": "createShaderModule", "id": 1, "code": """
             @vertex fn vs() -> @builtin(position) vec4f { return nonexistent(1.0); }
             """],
            ["op": "createRenderPipeline", "id": 2, "layout": "auto",
             "vertex": ["module": 1, "entryPoint": "vs"]],
            ["op": "popErrorScope"],
        ])

        XCTAssertEqual(scopes(result).first??["kind"] as? String, "backend")
        XCTAssertEqual(errors(result).count, 0)
    }

    // MARK: - 수명

    /// WebGPU에서 오류 스코프는 **디바이스 상태**다. `push`와 `pop` 사이에 `submit`이 몇 번이든
    /// 들어갈 수 있으므로, 배치 경계에서 스택이 초기화되면 안 된다.
    func test_스코프는_배치를_넘어_이어진다() {
        harness.executeExpectingSuccess([["op": "pushErrorScope", "filter": "validation"]])

        let middle = harness.execute([failingCommand])
        XCTAssertEqual(errors(middle).count, 0, "다른 배치의 오류도 열린 스코프가 잡는다")

        let closing = harness.execute([["op": "popErrorScope"]])
        XCTAssertEqual(scopes(closing).first??["kind"] as? String, "validation")
    }

    func test_스코프는_처음_잡힌_오류_하나만_돌려준다() {
        // 핸들 번호가 메시지에 드러나는 명령을 쓴다 — 어느 쪽이 잡혔는지 구분하려면 필요하다.
        let bytes = [Float](repeating: 0, count: 4).base64
        let result = harness.execute([
            ["op": "pushErrorScope", "filter": "validation"],
            ["op": "writeBuffer", "buffer": 111, "data": bytes],
            ["op": "writeBuffer", "buffer": 222, "data": bytes],
            ["op": "popErrorScope"],
        ])

        XCTAssertEqual(scopes(result).count, 1, "스코프 하나당 결과 하나")
        XCTAssertTrue(
            ((scopes(result).first??["message"] as? String) ?? "").contains("111"),
            "명세상 처음 잡힌 오류를 돌려준다"
        )
    }

    func test_짝이_없는_pop은_오류지만_결과_자리는_남긴다() {
        let result = harness.execute([["op": "popErrorScope"]])

        XCTAssertEqual(errors(result).first?["kind"] as? String, "validation")
        // 자리가 밀리면 JS가 Promise를 엉뚱한 결과로 풀게 된다 — 개수는 지켜야 한다.
        XCTAssertEqual(scopes(result).count, 1)
        XCTAssertNil(scopes(result).first ?? nil)
    }

    func test_reset은_열려_있던_스코프를_버린다() {
        harness.executeExpectingSuccess([["op": "pushErrorScope", "filter": "validation"]])
        harness.context.reset()

        // 스코프가 남아 있었다면 이 오류를 조용히 삼켰을 것이다.
        let result = harness.execute([failingCommand])
        XCTAssertEqual(errors(result).count, 1, "디바이스를 버리면 스코프도 함께 버려야 한다")
    }
}
