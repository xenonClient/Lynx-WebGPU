import XCTest
import Metal
import LynxWebGPUCore
@testable import LynxWebGPU

/// `pushErrorScope` / `popErrorScope` — **a protocol contract needing no GPU**.
///
/// All it decides is where an error goes, so it is unrelated to render results. We raise errors on
/// purpose with bad commands and check whether a scope catches them and whether they drop out of the
/// result's `errors`.
final class ErrorScopeTests: XCTestCase {
    private var harness: RenderHarness!

    override func setUpWithError() throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "no Metal device")
        harness = try XCTUnwrap(RenderHarness.make(width: 8, height: 8))
    }

    override func tearDown() {
        harness = nil
        super.tearDown()
    }

    private func errors(_ result: [String: Any]) -> [[String: Any]] {
        result["errors"] as? [[String: Any]] ?? []
    }

    /// The pop result array — `NSNull` means "that scope caught no error".
    private func scopes(_ result: [String: Any]) -> [[String: Any]?] {
        (result["errorScopes"] as? [Any] ?? []).map { $0 as? [String: Any] }
    }

    /// A command raising exactly one validation error (referencing a missing handle).
    private let failingCommand: [String: Any] = ["op": "setVertexBuffer", "slot": 0, "buffer": 999]

    // MARK: - Basic behaviour

    func test_aScopeInterceptsTheErrorAndRemovesItFromTheResult() {
        let result = harness.execute([
            ["op": "pushErrorScope", "filter": "validation"],
            failingCommand,
            ["op": "popErrorScope"],
        ])

        // The scope took it, so it does not leak globally — that is what stops JS's onError double-reporting.
        XCTAssertEqual(errors(result).count, 0, "a caught error must not appear in errors")
        XCTAssertEqual(result["ok"] as? Bool, true, "a handled error does not make the batch a failure")
        XCTAssertEqual(scopes(result).first??["kind"] as? String, "validation")
    }

    func test_withNoErrorTheScopeComesBackEmpty() {
        let result = harness.executeExpectingSuccess([
            ["op": "pushErrorScope", "filter": "validation"],
            ["op": "createBuffer", "id": 1, "size": 16, "usage": TestUsage.uniform],
            ["op": "popErrorScope"],
        ])

        XCTAssertEqual(scopes(result).count, 1)
        XCTAssertNil(scopes(result).first ?? nil, "with no error it must be null")
    }

    func test_anErrorOutsideAnyScopeGoesGlobal() {
        let result = harness.execute([
            ["op": "pushErrorScope", "filter": "validation"],
            ["op": "popErrorScope"],
            failingCommand,
        ])

        XCTAssertNil(scopes(result).first ?? nil, "a closed scope cannot catch later errors")
        XCTAssertEqual(errors(result).count, 1)
        XCTAssertEqual(errors(result).first?["kind"] as? String, "validation")
    }

    // MARK: - Filters

    func test_aNonMatchingFilterLetsItThroughToTheOutside() {
        let result = harness.execute([
            ["op": "pushErrorScope", "filter": "out-of-memory"],
            failingCommand,   // validation
            ["op": "popErrorScope"],
        ])

        XCTAssertNil(scopes(result).first ?? nil, "an out-of-memory scope does not catch validation")
        XCTAssertEqual(errors(result).count, 1, "an uncaught error must go global")
    }

    func test_withNestedScopesOnlyTheInnerOneCatches() {
        let result = harness.execute([
            ["op": "pushErrorScope", "filter": "validation"],   // outer
            ["op": "pushErrorScope", "filter": "validation"],   // inner
            failingCommand,
            ["op": "popErrorScope"],                            // the inner one closes first
            ["op": "popErrorScope"],
        ])

        XCTAssertEqual(scopes(result).count, 2)
        XCTAssertEqual(scopes(result)[0]?["kind"] as? String, "validation", "the inner one takes it")
        XCTAssertNil(scopes(result)[1], "the outer one never sees it")
        XCTAssertEqual(errors(result).count, 0)
    }

    func test_whenTheInnerFilterDoesNotMatchTheOuterScopeCatches() {
        let result = harness.execute([
            ["op": "pushErrorScope", "filter": "validation"],     // outer — this one must catch
            ["op": "pushErrorScope", "filter": "out-of-memory"],  // inner — the filter does not match
            failingCommand,
            ["op": "popErrorScope"],
            ["op": "popErrorScope"],
        ])

        XCTAssertNil(scopes(result)[0], "the inner one cannot catch, its filter differs")
        XCTAssertEqual(scopes(result)[1]?["kind"] as? String, "validation", "it flows out and is caught")
        XCTAssertEqual(errors(result).count, 0)
    }

    func test_theValidationScopeCatchesAnUnsupportedFeature() {
        // `unsupported` means "valid per spec, but this implementation does not do it yet". The same code
        // in a browser either raises validation or succeeds, so a validation scope must catch it to allow a response.
        let result = harness.execute([
            ["op": "pushErrorScope", "filter": "validation"],
            ["op": "teleport"],
            ["op": "popErrorScope"],
        ])

        XCTAssertEqual(scopes(result).first??["kind"] as? String, "unsupported")
        XCTAssertEqual(errors(result).count, 0)
    }

    func test_theInternalScopeCatchesAShaderCompileFailure() {
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

    /// Exactly the batch the shim's `createRenderPipelineAsync` uses — validation outside, internal inside.
    ///
    /// Pipeline creation fails in two ways. A descriptor problem is `validation` (plus `unsupported`),
    /// while a shader translation or compilation failure is `backend` and only an `internal` filter
    /// catches it. Wrapping in two layers is what makes **either kind land in a scope** — with one layer
    /// the other half leaks globally, the promise resolves as success, and you are left holding an unusable pipeline.
    func test_twoNestedScopesTakeBothPipelineFailures() {
        // (1) shader compile failure → the inner one (internal) takes it and the outer (validation) stays clean.
        let backend = harness.execute([
            ["op": "pushErrorScope", "filter": "validation"],
            ["op": "pushErrorScope", "filter": "internal"],
            ["op": "createShaderModule", "id": 1, "code": """
             @vertex fn vs() -> @builtin(position) vec4f { return nonexistent(1.0); }
             """],
            ["op": "createRenderPipeline", "id": 2, "layout": "auto",
             "vertex": ["module": 1, "entryPoint": "vs"]],
            ["op": "popErrorScope"],
            ["op": "popErrorScope"],
        ])
        XCTAssertEqual(scopes(backend).first??["kind"] as? String, "backend", "the inner one takes it first")
        XCTAssertNil(scopes(backend).last ?? nil, "the outer one must be empty")
        XCTAssertEqual(errors(backend).count, 0, "it must not leak globally")

        // (2) descriptor error → the inner filter does not match, so the outer (validation) takes it.
        let validation = harness.execute([
            ["op": "pushErrorScope", "filter": "validation"],
            ["op": "pushErrorScope", "filter": "internal"],
            ["op": "createRenderPipeline", "id": 3, "layout": "auto",
             "vertex": ["module": 999, "entryPoint": "vs"]],
            ["op": "popErrorScope"],
            ["op": "popErrorScope"],
        ])
        XCTAssertNil(scopes(validation).first ?? nil, "the inner one must be empty")
        XCTAssertEqual(scopes(validation).last??["kind"] as? String, "validation")
        XCTAssertEqual(errors(validation).count, 0)
    }

    // MARK: - Lifetime

    /// In WebGPU an error scope is **device state**. Any number of `submit`s may fall between `push`
    /// and `pop`, so the stack must not reset at a batch boundary.
    func test_scopesSpanBatches() {
        harness.executeExpectingSuccess([["op": "pushErrorScope", "filter": "validation"]])

        let middle = harness.execute([failingCommand])
        XCTAssertEqual(errors(middle).count, 0, "an open scope catches errors from another batch too")

        let closing = harness.execute([["op": "popErrorScope"]])
        XCTAssertEqual(scopes(closing).first??["kind"] as? String, "validation")
    }

    func test_aScopeReturnsOnlyTheFirstErrorItCaught() {
        // Use commands whose handle number shows in the message — needed to tell which one was caught.
        let bytes = [Float](repeating: 0, count: 4).base64
        let result = harness.execute([
            ["op": "pushErrorScope", "filter": "validation"],
            ["op": "writeBuffer", "buffer": 111, "data": bytes],
            ["op": "writeBuffer", "buffer": 222, "data": bytes],
            ["op": "popErrorScope"],
        ])

        XCTAssertEqual(scopes(result).count, 1, "one result per scope")
        XCTAssertTrue(
            ((scopes(result).first??["message"] as? String) ?? "").contains("111"),
            "per spec it returns the first error caught"
        )
    }

    /// The spec defines an unmatched `pop` as **a promise rejection** and **creates no error**.
    /// So we do not route it as a GPUError but carry only "unmatched" in the slot — otherwise an error
    /// the spec never defined mixes into the app's global handler and telemetry, and the app cannot
    /// tell "it was clean (null)" from "it was unmatched".
    func test_anUnmatchedPopReturnsARejectedStateRatherThanAnError() {
        let result = harness.execute([["op": "popErrorScope"]])

        XCTAssertEqual(result["ok"] as? Bool, true, "it must not create a GPUError the spec does not define")
        XCTAssertNil(result["errors"])
        // A shifted slot makes JS resolve the promise with the wrong result — the count has to hold.
        XCTAssertEqual(scopes(result).count, 1)
        XCTAssertEqual((scopes(result).first ?? nil)?["rejected"] as? Bool, true)
    }

    /// The stack depth must hold even when the filter cannot be read.
    /// Otherwise a later `pop` takes **the outer scope**, and the value the app believes is the inner
    /// region's is really the outer region's — a scope opened to diagnose creates a misdiagnosis.
    func test_anUnreadableFilterStillKeepsTheScopeStackDepth() {
        let result = harness.execute([
            ["op": "pushErrorScope", "filter": "validation"],   // outer A
            ["op": "pushErrorScope", "filter": "Validation"],   // a typo — pushed as a placeholder
            failingCommand,                                     // the placeholder catches nothing
            ["op": "popErrorScope"],                            // the inner one (the placeholder)
            ["op": "popErrorScope"],                            // outer A
        ])

        XCTAssertEqual(scopes(result).count, 2, "two pops means two results")
        XCTAssertNil(scopes(result)[0], "the placeholder catches nothing")
        XCTAssertNotNil(
            scopes(result)[1]?["message"],
            "the outer scope must catch the error — a shifted stack would leave this empty"
        )
    }

    func test_resetDiscardsOpenScopes() {
        harness.executeExpectingSuccess([["op": "pushErrorScope", "filter": "validation"]])
        harness.runtime.reset()

        // Had a scope remained, it would have swallowed this error silently.
        let result = harness.execute([failingCommand])
        XCTAssertEqual(errors(result).count, 1, "discarding the device must discard its scopes too")
    }
}
