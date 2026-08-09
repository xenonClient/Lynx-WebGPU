import XCTest
@testable import LynxWebGPUCore

/// The error scope stack, batch result assembler and deferred failure queue — wire policy verified with no GPU.
///
/// A test verifying the same rules through the Metal interpreter lives in `LynxWebGPUTests/ErrorScopeTests`.
/// Here it is the contract of **the Core types alone** — what must hold when an external runtime takes only these types.
final class WGPUErrorScopeStackTests: XCTestCase {

    // MARK: - Scope stack

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
        let error = WGPUError.validation("bad argument")
        XCTAssertTrue(stack.capture(error))
        XCTAssertEqual(stack.pop(), .captured(error))
    }

    func test_aNonMatchingFilterDoesNotCatch() {
        var stack = WGPUErrorScopeStack()
        stack.push(.outOfMemory)
        XCTAssertFalse(stack.capture(.validation("bad argument")))
        XCTAssertEqual(stack.pop(), .clean)
    }

    func test_innermostScopeCatchesFirst() {
        var stack = WGPUErrorScopeStack()
        stack.push(.validation)   // outer
        stack.push(.validation)   // inner
        let error = WGPUError.validation("the inner one must catch")
        XCTAssertTrue(stack.capture(error))
        XCTAssertEqual(stack.pop(), .captured(error))   // inner
        XCTAssertEqual(stack.pop(), .clean)             // the outer one is empty
    }

    func test_aScopeReturnsOnlyTheFirstErrorItCaught() {
        var stack = WGPUErrorScopeStack()
        stack.push(.validation)
        let first = WGPUError.validation("first")
        XCTAssertTrue(stack.capture(first))
        XCTAssertTrue(stack.capture(.validation("second")))   // it is caught (but not propagated)
        XCTAssertEqual(stack.pop(), .captured(first))
    }

    func test_aPlaceholderCatchesNothingAndOnlyHoldsDepth() {
        var stack = WGPUErrorScopeStack()
        stack.push(.validation)   // outer
        stack.push(nil)           // placeholder (filter parsing failed)
        let error = WGPUError.validation("the outer one catches")
        XCTAssertTrue(stack.capture(error))
        XCTAssertEqual(stack.pop(), .clean)             // placeholder
        XCTAssertEqual(stack.pop(), .captured(error))   // outer
    }

    func test_filterMappingSplitsFourErrorKindsIntoThreeFilters() {
        // The basis is the WGPUErrorFilter.captures docs — validation takes unsupported, internal takes backend.
        var validation = WGPUErrorScopeStack()
        validation.push(.validation)
        XCTAssertTrue(validation.capture(.unsupported("a feature not done yet")))

        var internalScope = WGPUErrorScopeStack()
        internalScope.push(.internal)
        XCTAssertTrue(internalScope.capture(.backend("internal Metal failure")))
        XCTAssertFalse(internalScope.capture(.validation("internal cannot catch validation")))
    }

    func test_afterDiscardAllNothingIsCaughtAndPopIsUnmatched() {
        var stack = WGPUErrorScopeStack()
        stack.push(.validation)
        stack.discardAll()
        XCTAssertEqual(stack.depth, 0)
        XCTAssertFalse(stack.capture(.validation("a discarded scope")))
        XCTAssertEqual(stack.pop(), .unmatched)
    }

    // MARK: - Serializing the pop result

    func test_thePayloadShapeOfAPopResult() {
        XCTAssertTrue(WGPUPoppedErrorScope.clean.payload is NSNull)

        let error = WGPUError.validation("caught error", path: "commands[0]")
        let captured = WGPUPoppedErrorScope.captured(error).payload as? [String: Any]
        XCTAssertEqual(captured?["kind"] as? String, "validation")
        XCTAssertEqual(captured?["message"] as? String, "caught error")
        XCTAssertEqual(captured?["path"] as? String, "commands[0]")

        let unmatched = WGPUPoppedErrorScope.unmatched.payload as? [String: Bool]
        XCTAssertEqual(unmatched?["rejected"], true)
    }

    // MARK: - Batch result assembly

    func test_cleanBatchCarriesOnlyTheThreeRequiredKeys() {
        let payload = WGPUBatchResult(commandCount: 3, liveObjectCount: 7).payload
        XCTAssertEqual(payload["ok"] as? Bool, true)
        XCTAssertEqual(payload["commandCount"] as? Int, 3)
        XCTAssertEqual(payload["objects"] as? Int, 7)
        // An empty array omits the key entirely — the omission is the contract.
        XCTAssertNil(payload["errors"])
        XCTAssertNil(payload["canvases"])
        XCTAssertNil(payload["errorScopes"])
    }

    func test_withErrorsOkIsFalseAndErrorsAreCarried() {
        let payload = WGPUBatchResult(
            commandCount: 1, liveObjectCount: 0,
            errors: [.validation("broken command", path: "commands[0].draw")]
        ).payload
        XCTAssertEqual(payload["ok"] as? Bool, false)
        let errors = payload["errors"] as? [[String: Any]]
        XCTAssertEqual(errors?.count, 1)
        XCTAssertEqual(errors?.first?["path"] as? String, "commands[0].draw")
    }

    func test_aCaughtErrorLeavesThroughErrorScopesNotErrors() {
        // A miniature of the interpreter flow: when capture is true it stays out of errors and leaves only as a pop result.
        var stack = WGPUErrorScopeStack()
        stack.push(.validation)
        let error = WGPUError.validation("the scope caught it")
        XCTAssertTrue(stack.capture(error))

        let payload = WGPUBatchResult(
            commandCount: 2, liveObjectCount: 0, poppedScopes: [stack.pop()]
        ).payload
        XCTAssertEqual(payload["ok"] as? Bool, true)   // a caught error does not break ok
        XCTAssertNil(payload["errors"])
        let scopes = payload["errorScopes"] as? [Any]
        XCTAssertEqual(scopes?.count, 1)
        XCTAssertEqual((scopes?.first as? [String: Any])?["message"] as? String, "the scope caught it")
    }

    func test_errorScopesKeepsThePopOrder() {
        let payload = WGPUBatchResult(
            commandCount: 0, liveObjectCount: 0,
            poppedScopes: [.clean, .captured(.validation("second pop")), .unmatched]
        ).payload
        let scopes = payload["errorScopes"] as? [Any]
        XCTAssertEqual(scopes?.count, 3)
        XCTAssertTrue(scopes?[0] is NSNull)
        XCTAssertEqual((scopes?[1] as? [String: Any])?["message"] as? String, "second pop")
        XCTAssertEqual((scopes?[2] as? [String: Any])?["rejected"] as? Bool, true)
    }

    func test_canvasesCarriesThePixelSizePerIdentifier() {
        let payload = WGPUBatchResult(
            commandCount: 1, liveObjectCount: 0,
            canvases: ["main": WGPUCanvasReport(width: 390, height: 844)]
        ).payload
        let canvases = payload["canvases"] as? [String: [String: Any]]
        XCTAssertEqual(canvases?["main"]?["width"] as? Int, 390)
        XCTAssertEqual(canvases?["main"]?["height"] as? Int, 844)
    }

    func test_aTopLevelFailureCarriesOnlyOkAndErrors() {
        let payload = WGPUBatchResult.failure([.validation("no commands array")])
        XCTAssertEqual(payload["ok"] as? Bool, false)
        XCTAssertEqual((payload["errors"] as? [[String: Any]])?.count, 1)
        // No command was seen, so there must be no batch result keys.
        XCTAssertNil(payload["commandCount"])
        XCTAssertNil(payload["objects"])
    }

    // MARK: - Deferred failure queue

    func test_theDeferredQueueEmptiesOnDrain() {
        let queue = WGPUDeferredErrorQueue()
        queue.report(.backend("GPU work failed"))
        XCTAssertEqual(queue.drain().count, 1)
        XCTAssertTrue(queue.drain().isEmpty)
    }

    func test_theDeferredQueueLosesNoReportFromAnyThread() {
        let queue = WGPUDeferredErrorQueue()
        DispatchQueue.concurrentPerform(iterations: 64) { index in
            queue.report(.backend("failure \(index)"))
        }
        XCTAssertEqual(queue.drain().count, 64)
    }
}
