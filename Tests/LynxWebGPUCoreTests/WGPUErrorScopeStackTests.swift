import XCTest
@testable import LynxWebGPUCore

/// 오류 스코프 스택·배치 결과 조립기·지연 실패 큐 — GPU 없이 도는 와이어 정책 검증.
///
/// 같은 규칙을 Metal 해석기 경유로 검증하는 테스트가 `LynxWebGPUTests/ErrorScopeTests`에 있다.
/// 여기는 **Core 타입 단독**의 계약이다 — 외부 런타임이 이 타입만 들고 갔을 때 성립해야 하는 것.
final class WGPUErrorScopeStackTests: XCTestCase {

    // MARK: - 스코프 스택

    func test_poppingAnEmptyStackIsUnmatched() {
        var stack = WGPUErrorScopeStack()
        XCTAssertEqual(stack.pop(), .unmatched)
    }

    func test_aScopeThatCaughtNothingClosesClean() {
        var stack = WGPUErrorScopeStack()
        stack.push(.validation)
        XCTAssertEqual(stack.pop(), .clean)
    }

    func test_aMatchingFilterCatchesAndKeepsItOutOfErrors() {
        var stack = WGPUErrorScopeStack()
        stack.push(.validation)
        let error = WGPUError.validation("잘못된 인자")
        XCTAssertTrue(stack.capture(error))
        XCTAssertEqual(stack.pop(), .captured(error))
    }

    func test_aNonMatchingFilterDoesNotCatch() {
        var stack = WGPUErrorScopeStack()
        stack.push(.outOfMemory)
        XCTAssertFalse(stack.capture(.validation("잘못된 인자")))
        XCTAssertEqual(stack.pop(), .clean)
    }

    func test_innermostScopeCatchesFirst() {
        var stack = WGPUErrorScopeStack()
        stack.push(.validation)   // 바깥
        stack.push(.validation)   // 안쪽
        let error = WGPUError.validation("안쪽이 잡아야 한다")
        XCTAssertTrue(stack.capture(error))
        XCTAssertEqual(stack.pop(), .captured(error))   // 안쪽
        XCTAssertEqual(stack.pop(), .clean)             // 바깥은 비어 있다
    }

    func test_aScopeReturnsOnlyTheFirstErrorItCaught() {
        var stack = WGPUErrorScopeStack()
        stack.push(.validation)
        let first = WGPUError.validation("첫 번째")
        XCTAssertTrue(stack.capture(first))
        XCTAssertTrue(stack.capture(.validation("두 번째")))   // 잡히긴 한다 (전파 안 됨)
        XCTAssertEqual(stack.pop(), .captured(first))
    }

    func test_aPlaceholderCatchesNothingAndOnlyHoldsDepth() {
        var stack = WGPUErrorScopeStack()
        stack.push(.validation)   // 바깥
        stack.push(nil)           // 자리표시자 (필터 파싱 실패)
        let error = WGPUError.validation("바깥이 잡는다")
        XCTAssertTrue(stack.capture(error))
        XCTAssertEqual(stack.pop(), .clean)             // 자리표시자
        XCTAssertEqual(stack.pop(), .captured(error))   // 바깥
    }

    func test_filterMappingSplitsFourErrorKindsIntoThreeFilters() {
        // 근거는 WGPUErrorFilter.captures 문서 — unsupported는 validation이, backend는 internal이.
        var validation = WGPUErrorScopeStack()
        validation.push(.validation)
        XCTAssertTrue(validation.capture(.unsupported("아직 안 되는 기능")))

        var internalScope = WGPUErrorScopeStack()
        internalScope.push(.internal)
        XCTAssertTrue(internalScope.capture(.backend("Metal 내부 실패")))
        XCTAssertFalse(internalScope.capture(.validation("internal은 validation을 못 잡는다")))
    }

    func test_discardAll_뒤에는_잡지도_못하고_pop도_unmatched다() {
        var stack = WGPUErrorScopeStack()
        stack.push(.validation)
        stack.discardAll()
        XCTAssertEqual(stack.depth, 0)
        XCTAssertFalse(stack.capture(.validation("버려진 스코프")))
        XCTAssertEqual(stack.pop(), .unmatched)
    }

    // MARK: - pop 결과 직렬화

    func test_pop_결과의_payload_모양() {
        XCTAssertTrue(WGPUPoppedErrorScope.clean.payload is NSNull)

        let error = WGPUError.validation("잡힌 오류", path: "commands[0]")
        let captured = WGPUPoppedErrorScope.captured(error).payload as? [String: Any]
        XCTAssertEqual(captured?["kind"] as? String, "validation")
        XCTAssertEqual(captured?["message"] as? String, "잡힌 오류")
        XCTAssertEqual(captured?["path"] as? String, "commands[0]")

        let unmatched = WGPUPoppedErrorScope.unmatched.payload as? [String: Bool]
        XCTAssertEqual(unmatched?["rejected"], true)
    }

    // MARK: - 배치 결과 조립

    func test_cleanBatchCarriesOnlyTheThreeRequiredKeys() {
        let payload = WGPUBatchResult(commandCount: 3, liveObjectCount: 7).payload
        XCTAssertEqual(payload["ok"] as? Bool, true)
        XCTAssertEqual(payload["commandCount"] as? Int, 3)
        XCTAssertEqual(payload["objects"] as? Int, 7)
        // 빈 배열은 키 자체를 생략한다 — 생략이 계약이다.
        XCTAssertNil(payload["errors"])
        XCTAssertNil(payload["canvases"])
        XCTAssertNil(payload["errorScopes"])
    }

    func test_withErrorsOkIsFalseAndErrorsAreCarried() {
        let payload = WGPUBatchResult(
            commandCount: 1, liveObjectCount: 0,
            errors: [.validation("깨진 명령", path: "commands[0].draw")]
        ).payload
        XCTAssertEqual(payload["ok"] as? Bool, false)
        let errors = payload["errors"] as? [[String: Any]]
        XCTAssertEqual(errors?.count, 1)
        XCTAssertEqual(errors?.first?["path"] as? String, "commands[0].draw")
    }

    func test_aCaughtErrorLeavesThroughErrorScopesNotErrors() {
        // 해석기 흐름의 축소판: capture가 true면 errors에 안 넣고 pop 결과로만 나간다.
        var stack = WGPUErrorScopeStack()
        stack.push(.validation)
        let error = WGPUError.validation("스코프가 잡았다")
        XCTAssertTrue(stack.capture(error))

        let payload = WGPUBatchResult(
            commandCount: 2, liveObjectCount: 0, poppedScopes: [stack.pop()]
        ).payload
        XCTAssertEqual(payload["ok"] as? Bool, true)   // 잡힌 오류는 ok를 깨지 않는다
        XCTAssertNil(payload["errors"])
        let scopes = payload["errorScopes"] as? [Any]
        XCTAssertEqual(scopes?.count, 1)
        XCTAssertEqual((scopes?.first as? [String: Any])?["message"] as? String, "스코프가 잡았다")
    }

    func test_errorScopes는_pop_순서를_지킨다() {
        let payload = WGPUBatchResult(
            commandCount: 0, liveObjectCount: 0,
            poppedScopes: [.clean, .captured(.validation("두 번째 pop")), .unmatched]
        ).payload
        let scopes = payload["errorScopes"] as? [Any]
        XCTAssertEqual(scopes?.count, 3)
        XCTAssertTrue(scopes?[0] is NSNull)
        XCTAssertEqual((scopes?[1] as? [String: Any])?["message"] as? String, "두 번째 pop")
        XCTAssertEqual((scopes?[2] as? [String: Any])?["rejected"] as? Bool, true)
    }

    func test_canvases는_식별자별_픽셀_크기를_싣는다() {
        let payload = WGPUBatchResult(
            commandCount: 1, liveObjectCount: 0,
            canvases: ["main": WGPUCanvasReport(width: 390, height: 844)]
        ).payload
        let canvases = payload["canvases"] as? [String: [String: Any]]
        XCTAssertEqual(canvases?["main"]?["width"] as? Int, 390)
        XCTAssertEqual(canvases?["main"]?["height"] as? Int, 844)
    }

    func test_aTopLevelFailureCarriesOnlyOkAndErrors() {
        let payload = WGPUBatchResult.failure([.validation("commands 배열이 없다")])
        XCTAssertEqual(payload["ok"] as? Bool, false)
        XCTAssertEqual((payload["errors"] as? [[String: Any]])?.count, 1)
        // 명령을 하나도 안 봤으므로 배치 결과 키가 없어야 한다.
        XCTAssertNil(payload["commandCount"])
        XCTAssertNil(payload["objects"])
    }

    // MARK: - 지연 실패 큐

    func test_theDeferredQueueEmptiesOnDrain() {
        let queue = WGPUDeferredErrorQueue()
        queue.report(.backend("GPU 작업이 실패했다"))
        XCTAssertEqual(queue.drain().count, 1)
        XCTAssertTrue(queue.drain().isEmpty)
    }

    func test_theDeferredQueueLosesNoReportFromAnyThread() {
        let queue = WGPUDeferredErrorQueue()
        DispatchQueue.concurrentPerform(iterations: 64) { index in
            queue.report(.backend("실패 \(index)"))
        }
        XCTAssertEqual(queue.drain().count, 64)
    }
}
