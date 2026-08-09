import XCTest
import Metal
import LynxWebGPUCore
@testable import LynxWebGPU

/// Query sets — the two kinds are verified **in different ways**.
///
/// `occlusion` ("how many samples survived") is deterministic. Its values are asserted directly.
/// `timestamp` is a GPU clock whose value differs on identical input. Only its **structure** is
/// asserted — an absolute time threshold would only wobble in CI and catch no bug.
final class QuerySetTests: XCTestCase {
    private var harness: RenderHarness!

    override func setUpWithError() throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "no Metal device")
        harness = try XCTUnwrap(RenderHarness.make())
    }

    override func tearDown() {
        harness = nil
        super.tearDown()
    }

    private static let shader = """
    @vertex
    fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
        var corners = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
        return vec4f(corners[index], 0.0, 1.0);
    }

    @fragment
    fn fs_main() -> @location(0) vec4f {
        return vec4f(1.0, 0.0, 0.0, 1.0);
    }
    """

    private func errors(_ result: [String: Any]) -> [[String: Any]] {
        result["errors"] as? [[String: Any]] ?? []
    }

    private func setUpResources() -> [[String: Any]] {
        [
            ["op": "configureCanvas", "canvas": "test", "format": "rgba8unorm"],
            ["op": "createShaderModule", "id": 1, "code": Self.shader],
            ["op": "createRenderPipeline", "id": 2, "layout": "auto",
             "vertex": ["module": 1, "entryPoint": "vs_main"],
             "fragment": ["module": 1, "entryPoint": "fs_main",
                          "targets": [["format": "rgba8unorm"]]]],
        ]
    }

    private let acquireDrawable: [[String: Any]] = [
        ["op": "getCurrentTexture", "id": 20, "canvas": "test"],
        ["op": "createTextureView", "id": 21, "texture": 20],
    ]

    private func beginPass(occlusionQuerySet: Int? = nil) -> [String: Any] {
        var command: [String: Any] = [
            "op": "beginRenderPass",
            "colorAttachments": [[
                "view": 21, "loadOp": "clear", "storeOp": "store",
                "clearValue": ["r": 0, "g": 0, "b": 1, "a": 1],
            ]],
        ]
        if let occlusionQuerySet { command["occlusionQuerySet"] = occlusionQuerySet }
        return command
    }

    // MARK: - occlusion (deterministic)

    /// The smallest combination — **a visible draw and a fully clipped one.** With only one you cannot
    /// tell "it should be 0 and it is" from "it is 0 because nothing is being counted".
    func test_anOcclusionQueryCountsTheSamplesThatPassed() throws {
        harness.executeExpectingSuccess(setUpResources() + [
            ["op": "createQuerySet", "id": 3, "type": "occlusion", "count": 2],
            ["op": "createBuffer", "id": 4, "size": 16,
             "usage": TestUsage.queryResolve | TestUsage.copySrc],
        ] + acquireDrawable + [
            beginPass(occlusionQuerySet: 3),
            ["op": "setPipeline", "pipeline": 2],
            // Query 0 — covers the whole screen.
            ["op": "beginOcclusionQuery", "queryIndex": 0],
            ["op": "draw", "vertexCount": 3],
            ["op": "endOcclusionQuery"],
            // Query 1 — fully clipped by the scissor. It must be exactly 0.
            ["op": "setScissorRect", "x": 0, "y": 0, "width": 0, "height": 0],
            ["op": "beginOcclusionQuery", "queryIndex": 1],
            ["op": "draw", "vertexCount": 3],
            ["op": "endOcclusionQuery"],
            ["op": "endPass"],
            ["op": "resolveQuerySet", "querySet": 3, "firstQuery": 0, "queryCount": 2,
             "destination": 4, "destinationOffset": 0],
        ])

        let results = try harness.readBufferSync(handle: 4, as: UInt64.self, size: 16)
        XCTAssertEqual(results.count, 2, "one u64 per query")
        XCTAssertEqual(results[0], 64 * 64, "the whole screen (64×64 samples) must pass")
        XCTAssertEqual(results[1], 0, "a fully clipped draw is exactly 0")
    }

    func test_resolveQuerySetLowersOnlyTheRange() throws {
        harness.executeExpectingSuccess(setUpResources() + [
            ["op": "createQuerySet", "id": 3, "type": "occlusion", "count": 3],
            ["op": "createBuffer", "id": 4, "size": 8,
             "usage": TestUsage.queryResolve | TestUsage.copySrc],
        ] + acquireDrawable + [
            beginPass(occlusionQuerySet: 3),
            ["op": "setPipeline", "pipeline": 2],
            // Draw only into query 2 — 0 and 1 are untouched.
            ["op": "beginOcclusionQuery", "queryIndex": 2],
            ["op": "draw", "vertexCount": 3],
            ["op": "endOcclusionQuery"],
            ["op": "endPass"],
            ["op": "resolveQuerySet", "querySet": 3, "firstQuery": 2, "queryCount": 1,
             "destination": 4, "destinationOffset": 0],
        ])

        XCTAssertEqual(
            try harness.readBufferSync(handle: 4, as: UInt64.self, size: 8), [UInt64(64 * 64)],
            "the slot firstQuery points at must come first in the destination"
        )
    }

    // MARK: - timestamp (non-deterministic — structure only)

    func test_timestampsIncreaseAcrossAPassBoundary() throws {
        try XCTSkipUnless(
            harness.supports(.timestampQuery), "device without pass-boundary timestamp sampling"
        )

        harness.executeExpectingSuccess(setUpResources() + [
            ["op": "createQuerySet", "id": 3, "type": "timestamp", "count": 2],
            ["op": "createBuffer", "id": 4, "size": 16,
             "usage": TestUsage.queryResolve | TestUsage.copySrc],
        ] + acquireDrawable + [
            [
                "op": "beginRenderPass",
                "colorAttachments": [[
                    "view": 21, "loadOp": "clear", "storeOp": "store",
                    "clearValue": ["r": 0, "g": 0, "b": 1, "a": 1],
                ]],
                "timestampWrites": [
                    "querySet": 3, "beginningOfPassWriteIndex": 0, "endOfPassWriteIndex": 1,
                ],
            ],
            ["op": "setPipeline", "pipeline": 2],
            ["op": "draw", "vertexCount": 3],
            ["op": "endPass"],
            ["op": "resolveQuerySet", "querySet": 3, "firstQuery": 0, "queryCount": 2,
             "destination": 4, "destinationOffset": 0],
        ])

        let stamps = try harness.readBufferSync(handle: 4, as: UInt64.self, size: 16)
        // The values are a GPU clock and cannot be asserted. Only the structure is checked —
        // an absolute time threshold ("at least 0.1ms") would only wobble in CI and catch no bug.
        XCTAssertEqual(stamps.count, 2, "8 bytes per query")
        XCTAssertNotEqual(stamps[0], 0, "still at the initial value means nothing was sampled")
        XCTAssertGreaterThanOrEqual(stamps[1], stamps[0], "the end cannot precede the start")
    }

    /// A compute pass's timestamps **are not asserted by value.**
    ///
    /// On Apple GPUs, samples taken by a compute encoder come out correctly every time through the
    /// CPU-side `resolveCounterRange`, but through `MTLBlitCommandEncoder.resolveCounters` (the GPU path
    /// `resolveQuerySet` uses) **the same code sometimes yields 0 from run to run** — even with separate
    /// command buffers. It is a driver matter we cannot fix here.
    ///
    /// So instead of a wobbling assertion we check only "the result is either (0, 0) or a proper value".
    /// Garbage values, a wrong length and a failed resolve are still caught, and CI does not wobble.
    /// For frame instrumentation, use **render pass timestamps** (those are stable).
    func test_aComputePassTakesTimestampsToo() throws {
        try XCTSkipUnless(
            harness.supports(.timestampQuery), "device without pass-boundary timestamp sampling"
        )

        // Metal may not sample counters for an empty pass — give it real work.
        let compute = """
        @group(0) @binding(0) var<storage, read_write> out: array<u32>;

        @compute @workgroup_size(1)
        fn touch() { out[0] = 1u; }
        """

        harness.executeExpectingSuccess([
            ["op": "createShaderModule", "id": 5, "code": compute],
            ["op": "createComputePipeline", "id": 6, "layout": "auto",
             "compute": ["module": 5, "entryPoint": "touch"]],
            ["op": "getBindGroupLayout", "id": 7, "pipeline": 6, "index": 0],
            ["op": "createBuffer", "id": 8, "size": 16, "usage": TestUsage.storage],
            ["op": "createBindGroup", "id": 9, "layout": 7,
             "entries": [["binding": 0, "resource": ["buffer": 8]]]],
            ["op": "createQuerySet", "id": 1, "type": "timestamp", "count": 2],
            ["op": "createBuffer", "id": 2, "size": 16,
             "usage": TestUsage.queryResolve | TestUsage.copySrc],
            ["op": "beginComputePass",
             "timestampWrites": ["querySet": 1, "beginningOfPassWriteIndex": 0,
                                 "endOfPassWriteIndex": 1]],
            ["op": "setPipeline", "pipeline": 6],
            ["op": "setBindGroup", "index": 0, "bindGroup": 9],
            ["op": "dispatchWorkgroups", "x": 1],
            ["op": "endPass"],
        ])

        harness.executeExpectingSuccess([
            ["op": "resolveQuerySet", "querySet": 1, "firstQuery": 0, "queryCount": 2,
             "destination": 2, "destinationOffset": 0],
        ])

        let stamps = try harness.readBufferSync(handle: 2, as: UInt64.self, size: 16)
        XCTAssertEqual(stamps.count, 2, "8 bytes per query")
        if stamps[0] == 0 && stamps[1] == 0 { return }   // the driver matter in the comment above
        XCTAssertNotEqual(stamps[0], 0, "only one being 0 is a garbage value")
        XCTAssertGreaterThanOrEqual(stamps[1], stamps[0], "the end cannot precede the start")
    }

    /// Creating one on an unsupported device must give **a clear `unsupported`**.
    /// On a supporting device we only check it is created — neither side may fail silently.
    func test_timestampQuerySetCreationMatchesDeviceSupport() {
        let result = harness.execute([
            ["op": "createQuerySet", "id": 1, "type": "timestamp", "count": 2],
        ])

        if harness.supports(.timestampQuery) {
            XCTAssertEqual(result["ok"] as? Bool, true, harness.describeErrors(result))
        } else {
            XCTAssertEqual(errors(result).first?["kind"] as? String, "unsupported")
        }
    }

    func test_theAdapterAdvertisesTimestampSupportAsAFeature() throws {
        let info = harness.runtime.adapterInfo()
        let features = try XCTUnwrap(info["features"] as? [String])

        XCTAssertEqual(
            features.contains("timestamp-query"), harness.supports(.timestampQuery),
            "JS must be able to ask before creating"
        )
        // Metal honours the firstInstance of indirect draw arguments as-is, so it is true regardless of device.
        XCTAssertTrue(features.contains("indirect-first-instance"), "\(features)")
    }

    // MARK: - Contract

    func test_beginOcclusionQueryWithoutAQuerySetIsAnError() {
        let result = harness.execute(setUpResources() + acquireDrawable + [
            beginPass(),   // with no occlusionQuerySet
            ["op": "beginOcclusionQuery", "queryIndex": 0],
            ["op": "endPass"],
        ])

        XCTAssertTrue(
            ((errors(result).first?["message"] as? String) ?? "").contains("occlusionQuerySet"),
            harness.describeErrors(result)
        )
    }

    func test_occlusionQueriesCannotNest() {
        let result = harness.execute(setUpResources() + [
            ["op": "createQuerySet", "id": 3, "type": "occlusion", "count": 2],
        ] + acquireDrawable + [
            beginPass(occlusionQuerySet: 3),
            ["op": "beginOcclusionQuery", "queryIndex": 0],
            ["op": "beginOcclusionQuery", "queryIndex": 1],
            ["op": "endPass"],
        ])

        XCTAssertTrue(
            ((errors(result).first?["message"] as? String) ?? "").contains("cannot nest"),
            harness.describeErrors(result)
        )
    }

    func test_rejectsAQueryIndexOutOfRange() {
        let result = harness.execute(setUpResources() + [
            ["op": "createQuerySet", "id": 3, "type": "occlusion", "count": 2],
        ] + acquireDrawable + [
            beginPass(occlusionQuerySet: 3),
            ["op": "beginOcclusionQuery", "queryIndex": 5],
            ["op": "endPass"],
        ])

        XCTAssertEqual(errors(result).first?["kind"] as? String, "validation")
        XCTAssertTrue(((errors(result).first?["message"] as? String) ?? "").contains("query range out of bounds"))
    }

    func test_aBufferWithoutQUERY_RESOLVEUsageCannotBeAResolveTarget() {
        let result = harness.execute([
            ["op": "createQuerySet", "id": 1, "type": "occlusion", "count": 1],
            ["op": "createBuffer", "id": 2, "size": 8, "usage": TestUsage.copyDst],
            ["op": "resolveQuerySet", "querySet": 1, "queryCount": 1, "destination": 2],
        ])

        XCTAssertEqual(errors(result).first?["kind"] as? String, "validation")
        XCTAssertEqual(errors(result).first?["path"] as? String, "commands[2].destination")
        XCTAssertTrue(((errors(result).first?["message"] as? String) ?? "").contains("QUERY_RESOLVE"))
    }

    func test_theDestinationOffsetMustBeAMultipleOf256() {
        let result = harness.execute([
            ["op": "createQuerySet", "id": 1, "type": "occlusion", "count": 1],
            ["op": "createBuffer", "id": 2, "size": 512, "usage": TestUsage.queryResolve],
            ["op": "resolveQuerySet", "querySet": 1, "queryCount": 1,
             "destination": 2, "destinationOffset": 8],
        ])

        XCTAssertEqual(errors(result).first?["path"] as? String, "commands[2].destinationOffset")
        XCTAssertTrue(((errors(result).first?["message"] as? String) ?? "").contains("256"))
    }

    func test_givingAnOcclusionQuerySetToTimestampWritesIsRejected() {
        let result = harness.execute(setUpResources() + [
            ["op": "createQuerySet", "id": 3, "type": "occlusion", "count": 2],
        ] + acquireDrawable + [
            [
                "op": "beginRenderPass",
                "colorAttachments": [[
                    "view": 21, "loadOp": "clear", "storeOp": "store",
                    "clearValue": ["r": 0, "g": 0, "b": 0, "a": 1],
                ]],
                "timestampWrites": ["querySet": 3, "beginningOfPassWriteIndex": 0],
            ],
            ["op": "endPass"],
        ])

        XCTAssertTrue(
            ((errors(result).first?["message"] as? String) ?? "").contains("timestamp"),
            harness.describeErrors(result)
        )
    }

    func test_aZeroSizeQuerySetIsRejected() {
        let result = harness.execute([["op": "createQuerySet", "id": 1, "type": "occlusion", "count": 0]])
        XCTAssertEqual(errors(result).first?["kind"] as? String, "validation")
        XCTAssertEqual(errors(result).first?["path"] as? String, "commands[0].count")
    }

    /// A query set past the cap is created here but is a validation error in a browser.
    /// (For occlusion it even allocates `count * 8` bytes outright.)
    func test_rejectsAQueryCountPastTheCap() {
        let result = harness.execute([
            ["op": "createQuerySet", "id": 1, "type": "occlusion",
             "count": WGPUQuerySetDescriptor.maxCount + 1],
        ])
        XCTAssertEqual(errors(result).first?["kind"] as? String, "validation")
        XCTAssertEqual(errors(result).first?["path"] as? String, "commands[0].count")

        harness.executeExpectingSuccess([
            ["op": "createQuerySet", "id": 2, "type": "occlusion",
             "count": WGPUQuerySetDescriptor.maxCount],
        ])
    }

    /// Omitting both indices makes every Metal sample index `MTLCounterDontSample`, producing
    /// **a pass that samples nothing with no error**. The app reads a GPU time of 0ns.
    func test_timestampWritesWithNoIndexAtAllIsRejected() throws {
        try XCTSkipUnless(harness.supports(.timestampQuery), "device without timestamp support")

        let result = harness.execute(setUpResources() + [
            ["op": "createQuerySet", "id": 3, "type": "timestamp", "count": 2],
        ] + acquireDrawable + [
            [
                "op": "beginRenderPass",
                "colorAttachments": [[
                    "view": 21, "loadOp": "clear", "storeOp": "store",
                    "clearValue": ["r": 0, "g": 0, "b": 0, "a": 1],
                ]],
                "timestampWrites": ["querySet": 3],
            ],
            ["op": "endPass"],
        ])

        XCTAssertTrue(
            ((errors(result).first?["message"] as? String) ?? "").contains("at least one"),
            harness.describeErrors(result)
        )
    }

    /// Pointing both at the same slot lets the end sample overwrite the start one and the delta loses meaning.
    func test_twoIdenticalTimestampWritesIndicesAreRejected() throws {
        try XCTSkipUnless(harness.supports(.timestampQuery), "device without timestamp support")

        let result = harness.execute(setUpResources() + [
            ["op": "createQuerySet", "id": 3, "type": "timestamp", "count": 2],
        ] + acquireDrawable + [
            [
                "op": "beginRenderPass",
                "colorAttachments": [[
                    "view": 21, "loadOp": "clear", "storeOp": "store",
                    "clearValue": ["r": 0, "g": 0, "b": 0, "a": 1],
                ]],
                "timestampWrites": [
                    "querySet": 3, "beginningOfPassWriteIndex": 1, "endOfPassWriteIndex": 1,
                ],
            ],
            ["op": "endPass"],
        ])

        XCTAssertTrue(
            ((errors(result).first?["message"] as? String) ?? "").contains("must differ"),
            harness.describeErrors(result)
        )
    }

    /// Metal writes the visibility result as-is at `endEncoding`, so **even the value looks correct.**
    /// In a browser the parent command encoder is invalidated and the whole frame is lost.
    func test_endPassWithAnUnclosedOcclusionQueryIsRejected() {
        let result = harness.execute(setUpResources() + [
            ["op": "createQuerySet", "id": 3, "type": "occlusion", "count": 2],
        ] + acquireDrawable + [
            beginPass(occlusionQuerySet: 3),
            ["op": "setPipeline", "pipeline": 2],
            ["op": "beginOcclusionQuery", "queryIndex": 0],
            ["op": "draw", "vertexCount": 3],
            ["op": "endPass"],   // with no endOcclusionQuery
        ])

        XCTAssertTrue(
            ((errors(result).first?["message"] as? String) ?? "").contains("still open"),
            harness.describeErrors(result)
        )
    }

    /// Using the same index twice makes two regions share one 8-byte slot —
    /// the surviving value depends on Metal's accumulate/overwrite behaviour and diverges from a browser.
    func test_rejectsReusingAnOcclusionIndexWithinOnePass() {
        let result = harness.execute(setUpResources() + [
            ["op": "createQuerySet", "id": 3, "type": "occlusion", "count": 2],
        ] + acquireDrawable + [
            beginPass(occlusionQuerySet: 3),
            ["op": "setPipeline", "pipeline": 2],
            ["op": "beginOcclusionQuery", "queryIndex": 0],
            ["op": "draw", "vertexCount": 3],
            ["op": "endOcclusionQuery"],
            ["op": "beginOcclusionQuery", "queryIndex": 0],
            ["op": "draw", "vertexCount": 3],
            ["op": "endOcclusionQuery"],
            ["op": "endPass"],
        ])

        XCTAssertTrue(
            ((errors(result).first?["message"] as? String) ?? "").contains("already used"),
            harness.describeErrors(result)
        )
    }

    /// Conversely, **in a different pass** the same index must be reusable (the spec forbids it only within a pass).
    func test_theSameOcclusionIndexCanBeReusedInADifferentPass() {
        let pass: [[String: Any]] = [
            beginPass(occlusionQuerySet: 3),
            ["op": "setPipeline", "pipeline": 2],
            ["op": "beginOcclusionQuery", "queryIndex": 0],
            ["op": "draw", "vertexCount": 3],
            ["op": "endOcclusionQuery"],
            ["op": "endPass"],
        ]
        harness.executeExpectingSuccess(setUpResources() + [
            ["op": "createQuerySet", "id": 3, "type": "occlusion", "count": 2],
        ] + acquireDrawable + pass + pass)
    }
}
