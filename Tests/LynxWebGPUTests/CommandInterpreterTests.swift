import XCTest
import Metal
import LynxWebGPUCore
@testable import LynxWebGPU

/// The command interpreter's contract — **errors are collected and returned without killing the process.**
final class CommandInterpreterTests: XCTestCase {
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

    func test_anUnknownCommandIsReportedAsUnsupported() {
        let result = harness.execute([["op": "teleport"]])

        XCTAssertEqual(result["ok"] as? Bool, false)
        XCTAssertEqual(errors(result).first?["kind"] as? String, "unsupported")
        XCTAssertEqual(errors(result).first?["path"] as? String, "commands[0].op")
    }

    func test_referencingAMissingHandleIsAValidationError() {
        let result = harness.execute([
            ["op": "setVertexBuffer", "slot": 0, "buffer": 999],
        ])

        XCTAssertEqual(result["ok"] as? Bool, false)
        XCTAssertEqual(errors(result).first?["kind"] as? String, "validation")
    }

    func test_laterCommandsKeepRunningAfterAnError() {
        let result = harness.execute([
            ["op": "teleport"],
            ["op": "createBuffer", "id": 1, "size": 16, "usage": TestUsage.uniform],
            ["op": "nonsense"],
        ])

        // Both errors are reported, and the valid command between them still runs.
        XCTAssertEqual(errors(result).count, 2)
        XCTAssertEqual(harness.liveObjects, 1)
    }

    func test_drawingWithNoPassIsAnError() {
        let result = harness.execute([["op": "draw", "vertexCount": 3]])
        XCTAssertEqual(errors(result).first?["kind"] as? String, "validation")
        XCTAssertTrue(
            ((errors(result).first?["message"] as? String) ?? "").contains("beginRenderPass"),
            "it must say what has to happen first"
        )
    }

    func test_anUnregisteredCanvasListsTheRegisteredOnes() {
        let result = harness.execute([["op": "configureCanvas", "canvas": "no-such-canvas"]])
        let message = (errors(result).first?["message"] as? String) ?? ""
        XCTAssertTrue(message.contains("no-such-canvas"))
    }

    func test_aShaderCompileFailureReportsTheGeneratedMSLToo() {
        let result = harness.execute([
            ["op": "createShaderModule", "id": 1, "code": """
             @vertex fn vs() -> @builtin(position) vec4f {
                 return nonexistent_function(1.0);
             }
             """],
            ["op": "createRenderPipeline", "id": 2, "layout": "auto",
             "vertex": ["module": 1, "entryPoint": "vs"]],
        ])

        let message = errors(result).map { $0["message"] as? String ?? "" }.joined(separator: "\n")
        XCTAssertTrue(message.contains("MSL"), "the generated MSL must be in the diagnostic: \(message)")
    }

    func test_drawableTextureHandlesAreReclaimedWhenTheFrameEnds() {
        harness.executeExpectingSuccess([
            ["op": "configureCanvas", "canvas": "test", "format": "rgba8unorm"],
        ])
        let before = harness.liveObjects

        harness.executeExpectingSuccess([
            ["op": "getCurrentTexture", "id": 50, "canvas": "test"],
            ["op": "createTextureView", "id": 51, "texture": 50],
            ["op": "beginRenderPass", "colorAttachments": [[
                "view": 51, "loadOp": "clear", "storeOp": "store",
                "clearValue": ["r": 0, "g": 0, "b": 0, "a": 1],
            ]]],
            ["op": "endPass"],
        ])

        // The swapchain texture and its view are not valid outside the frame (the same rule as a browser).
        XCTAssertEqual(harness.liveObjects, before)
    }

    /// A frame's boundary is **present, not the batch**.
    ///
    /// `popErrorScope` and `mapAsync` submit mid-frame to get a result. Closing the frame scope at the
    /// end of every batch erases the swapchain handles there, and the following `beginRenderPass` breaks
    /// with a "missing handle" — losing the whole frame.
    func test_swapchainHandlesSurviveAMidFrameSubmit() {
        harness.executeExpectingSuccess([
            ["op": "configureCanvas", "canvas": "test", "format": "rgba8unorm"],
        ])
        let before = harness.liveObjects

        // Batch 1: it only acquires a drawable and ends (the situation a mid-frame flush creates).
        harness.executeExpectingSuccess([
            ["op": "getCurrentTexture", "id": 50, "canvas": "test"],
            ["op": "createTextureView", "id": 51, "texture": 50],
        ])
        XCTAssertEqual(
            harness.liveObjects, before + 2,
            "nothing has been presented yet, so the handles must be alive"
        )

        // Batch 2: actually draws with the view obtained in the previous batch.
        harness.executeExpectingSuccess([
            ["op": "beginRenderPass", "colorAttachments": [[
                "view": 51, "loadOp": "clear", "storeOp": "store",
                "clearValue": ["r": 0, "g": 0, "b": 0, "a": 1],
            ]]],
            ["op": "endPass"],
        ])

        XCTAssertEqual(harness.liveObjects, before, "it presented, so they are reclaimed now")
    }

    /// **A batch with no commands presents too.**
    ///
    /// The frame boundary is not `submit()` but **the end of the frame loop callback** (where a browser
    /// presents at the end of the task), so the last thing in a tick is a batch with no commands that
    /// only presents. Passing over it for lack of a command buffer **freezes the screen with nothing said.**
    func test_aBatchWithNoCommandsStillPresentsAndReclaimsFrameHandles() {
        harness.executeExpectingSuccess([
            ["op": "configureCanvas", "canvas": "test", "format": "rgba8unorm"],
        ])
        let before = harness.liveObjects

        // The submits within the tick — present is deferred.
        let midFrame = harness.runtime.execute([
            "commands": [
                ["op": "getCurrentTexture", "id": 60, "canvas": "test"],
                ["op": "createTextureView", "id": 61, "texture": 60],
                ["op": "beginRenderPass", "colorAttachments": [[
                    "view": 61, "loadOp": "clear", "storeOp": "store",
                    "clearValue": ["r": 0, "g": 1, "b": 0, "a": 1],
                ]]],
                ["op": "endPass"],
            ] as [[String: Any]],
            "present": false,
        ])
        XCTAssertEqual(midFrame["ok"] as? Bool, true, harness.describeErrors(midFrame))
        XCTAssertEqual(harness.liveObjects, before + 2, "still mid-frame")

        // The end of the tick — no commands, present only.
        let closing = harness.runtime.execute(["commands": [[String: Any]](), "present": true])
        XCTAssertEqual(closing["ok"] as? Bool, true, harness.describeErrors(closing))
        XCTAssertEqual(
            harness.liveObjects, before,
            "even an empty batch that presented must reclaim the frame-scoped handles"
        )
    }

    /// With no drawable acquired, an empty batch **does nothing** — creating and committing a command
    /// buffer for nothing would spin the in-flight accounting.
    func test_withNoDrawableAnEmptyBatchDoesNothing() {
        harness.executeExpectingSuccess([
            ["op": "configureCanvas", "canvas": "test", "format": "rgba8unorm"],
        ])
        let before = harness.liveObjects

        let result = harness.runtime.execute(["commands": [[String: Any]](), "present": true])

        XCTAssertEqual(result["ok"] as? Bool, true, harness.describeErrors(result))
        XCTAssertEqual(harness.liveObjects, before)
    }

    /// Several submits in one frame **share the drawable view** — three.js's post-processing looks like
    /// this (scene pass → bloom mip chain → output pass). Presenting in between rejects the second pass
    /// with "GPUTextureView does not exist".
    func test_severalSubmitsInOneFrameShareTheDrawableView() {
        harness.executeExpectingSuccess([
            ["op": "configureCanvas", "canvas": "test", "format": "rgba8unorm"],
        ])

        let acquire = harness.runtime.execute([
            "commands": [
                ["op": "getCurrentTexture", "id": 70, "canvas": "test"],
                ["op": "createTextureView", "id": 71, "texture": 70],
                ["op": "beginRenderPass", "colorAttachments": [[
                    "view": 71, "loadOp": "clear", "storeOp": "store",
                    "clearValue": ["r": 1, "g": 0, "b": 0, "a": 1],
                ]]],
                ["op": "endPass"],
            ] as [[String: Any]],
            "present": false,
        ])
        XCTAssertEqual(acquire["ok"] as? Bool, true, harness.describeErrors(acquire))

        // The frame's second submit draws again with **the same view**.
        let second = harness.runtime.execute([
            "commands": [
                ["op": "beginRenderPass", "colorAttachments": [[
                    "view": 71, "loadOp": "clear", "storeOp": "store",
                    "clearValue": ["r": 0, "g": 1, "b": 0, "a": 1],
                ]]],
                ["op": "endPass"],
            ] as [[String: Any]],
            "present": false,
        ])
        XCTAssertEqual(second["ok"] as? Bool, true, harness.describeErrors(second))

        harness.runtime.execute(["commands": [[String: Any]](), "present": true])
    }

    /// The swapchain must survive even when a mid-frame batch **creates a command buffer** (writeBuffer, say).
    ///
    /// Three.js's lazy pipeline creation looks exactly like this — with a drawable acquired, a uniform
    /// writeBuffer plus popErrorScope flushes immediately. The shim marks such internal submits
    /// `present: false`, and the interpreter commits only, deferring present and handle expiry to the real
    /// frame submit. Without that distinction an undrawn drawable is presented and the following output
    /// pass is rejected outright with "GPUTextureView does not exist".
    func test_aPresentFalseBatchKeepsTheDrawableEvenWithACommandBuffer() {
        harness.executeExpectingSuccess([
            ["op": "configureCanvas", "canvas": "test", "format": "rgba8unorm"],
        ])
        let before = harness.liveObjects

        // Batch 1: acquire a drawable + writeBuffer (a blit encoder → a command buffer) — an internal submit.
        let midFrame = harness.runtime.execute([
            "commands": [
                ["op": "getCurrentTexture", "id": 50, "canvas": "test"],
                ["op": "createTextureView", "id": 51, "texture": 50],
                ["op": "createBuffer", "id": 52, "size": 16, "usage": TestUsage.copyDst],
                ["op": "writeBuffer", "buffer": 52, "data": [Float]([1, 2, 3, 4]).base64],
            ] as [[String: Any]],
            "present": false,
        ])
        XCTAssertEqual(midFrame["ok"] as? Bool, true, harness.describeErrors(midFrame))
        XCTAssertEqual(
            harness.liveObjects, before + 3,
            "an internal submit must not expire frame-scoped handles"
        )

        // Batch 2: the real frame submit — draws with the view from the previous batch. Present happens here.
        harness.executeExpectingSuccess([
            ["op": "beginRenderPass", "colorAttachments": [[
                "view": 51, "loadOp": "clear", "storeOp": "store",
                "clearValue": ["r": 0, "g": 1, "b": 0, "a": 1],
            ]]],
            ["op": "endPass"],
        ])
        XCTAssertEqual(
            harness.liveObjects, before + 1,
            "after present only the frame-scoped handles are reclaimed (buffer 52 remains)"
        )
    }

    /// Frames that failed with nothing to submit must not **accumulate**.
    ///
    /// When a validation error hits before the first encoder (a `beginRenderPass` with a view killed by a
    /// resize, say) there is no command buffer and nothing to present. That drawable cannot be released at
    /// the end of the batch — the frame may still be going (the contract of the test just above). Instead,
    /// **when the next frame asks for the same canvas's drawable** the previous frame is confirmed over, and it is reclaimed then.
    ///
    /// Without that reclamation one piles up per frame and an on-screen surface drains its drawable pool in
    /// three frames, after which `nextDrawable()` stalls the JS thread up to a second and then fails forever.
    func test_failedFrameDrawablesDoNotAccumulatePerFrame() {
        harness.executeExpectingSuccess([
            ["op": "configureCanvas", "canvas": "test", "format": "rgba8unorm"],
        ])
        let before = harness.liveObjects

        var counts: [Int] = []
        for frame in 0..<5 {
            let result = harness.execute([
                ["op": "getCurrentTexture", "id": 100 + frame, "canvas": "test"],
                // A missing view — rejected before reaching the backend, so no command buffer is created.
                ["op": "beginRenderPass", "colorAttachments": [["view": 9999]]],
            ])
            XCTAssertEqual(result["ok"] as? Bool, false, "this batch must fail")
            counts.append((result["objects"] as? Int ?? -1) - before)
        }

        XCTAssertEqual(
            counts, [1, 1, 1, 1, 1],
            "only **one frame in progress** may be held (growing per frame is a leak)"
        )
    }

    /// A canvas detached **mid-frame** must hand back its acquisition.
    ///
    /// The reclaim in `getCurrentTexture` only fires when the same canvas asks again — a detached
    /// canvas never does. Without cleanup at detach, the drawable sits in the backend for good (and
    /// the next present would put an undrawn frame on the dead layer), while the swapchain texture
    /// handle pins the registry. Detach is that canvas's last chance to clean up after itself.
    func test_detachingACanvasReclaimsItsPendingFrame() throws {
        try harness.runtime.attachOffscreenCanvas(
            identifier: "doomed", size: CGSize(width: 8, height: 8)
        )
        harness.executeExpectingSuccess([
            ["op": "configureCanvas", "canvas": "doomed", "format": "rgba8unorm"],
        ])
        let before = harness.liveObjects

        // Mid-frame: acquired, nothing presented — and then the element goes away.
        let midFrame = harness.execute([
            ["op": "getCurrentTexture", "id": 80, "canvas": "doomed"],
            ["op": "createTextureView", "id": 81, "texture": 80],
        ], present: false)
        XCTAssertEqual(midFrame["ok"] as? Bool, true, harness.describeErrors(midFrame))
        XCTAssertEqual(harness.liveObjects, before + 2, "mid-frame — the acquisition is held")

        harness.runtime.detachCanvas(identifier: "doomed")

        XCTAssertEqual(
            harness.liveObjects, before,
            "a detached canvas's frame handles must be reclaimed — held on, they leak for good"
        )
    }

    /// Reclaiming one canvas's failed frame must not tear down **another canvas's** frame in flight.
    ///
    /// Frame N: canvas "test" acquires and its frame dies with nothing submitted. Frame N+1: canvas
    /// "other" acquires first (a fresh frame), then "test" asks again — the stale-frame reclaim fires
    /// there. Were the reclaim global rather than per-canvas, it would drop "other"'s just-acquired
    /// drawable and expire its handles, and the render pass right after would be rejected with
    /// "GPUTextureView does not exist".
    func test_reclaimingOneCanvasLeavesAnotherCanvasesFrameAlone() throws {
        try harness.runtime.attachOffscreenCanvas(
            identifier: "other", size: CGSize(width: 8, height: 8)
        )
        harness.executeExpectingSuccess([
            ["op": "configureCanvas", "canvas": "test", "format": "rgba8unorm"],
            ["op": "configureCanvas", "canvas": "other", "format": "rgba8unorm"],
        ])

        // Frame N: "test" acquires and the frame ends with nothing to submit.
        let stale = harness.execute([
            ["op": "getCurrentTexture", "id": 90, "canvas": "test"],
        ], present: false)
        XCTAssertEqual(stale["ok"] as? Bool, true, harness.describeErrors(stale))

        // Frame N+1: "other" first, then "test" (whose reclaim fires), then "other" draws.
        harness.executeExpectingSuccess([
            ["op": "getCurrentTexture", "id": 92, "canvas": "other"],
            ["op": "createTextureView", "id": 93, "texture": 92],
            ["op": "getCurrentTexture", "id": 94, "canvas": "test"],
            ["op": "beginRenderPass", "colorAttachments": [[
                "view": 93, "loadOp": "clear", "storeOp": "store",
                "clearValue": ["r": 1, "g": 0, "b": 0, "a": 1],
            ]]],
            ["op": "endPass"],
        ])
    }

    func test_bufferWriteCopyAndReadHappenInOrder() throws {
        let source: [Float] = [1, 2, 3, 4]
        harness.executeExpectingSuccess([
            ["op": "createBuffer", "id": 1, "size": 16, "usage": TestUsage.copySrc | TestUsage.copyDst],
            ["op": "createBuffer", "id": 2, "size": 16, "usage": TestUsage.copyDst | TestUsage.mapRead],
            ["op": "writeBuffer", "buffer": 1, "data": source.base64],
            ["op": "copyBufferToBuffer", "source": 1, "sourceOffset": 0,
             "destination": 2, "destinationOffset": 0, "size": 16],
        ])

        XCTAssertEqual(try harness.readBufferSync(handle: 2, as: Float.self), source)
    }

    /// Omitting `size` means **the rest of the source** (the spec's `copyBufferToBuffer(src, dst)`).
    ///
    /// The JS shim fills `size` in, but anyone building the command stream directly (using it without Lynx)
    /// must get the same result when they omit it as documented.
    func test_omittingCopyBufferToBufferSizeMeansTheRest() throws {
        let source: [Float] = [1, 2, 3, 4]
        harness.executeExpectingSuccess([
            ["op": "createBuffer", "id": 1, "size": 16, "usage": TestUsage.copySrc | TestUsage.copyDst],
            ["op": "createBuffer", "id": 2, "size": 16, "usage": TestUsage.copyDst | TestUsage.mapRead],
            ["op": "writeBuffer", "buffer": 1, "data": source.base64],
            ["op": "copyBufferToBuffer", "source": 1, "destination": 2],
        ])
        XCTAssertEqual(try harness.readBufferSync(handle: 2, as: Float.self), source)
    }

    func test_copyBufferToBufferSizeIsTheRestAfterSourceOffset() throws {
        harness.executeExpectingSuccess([
            ["op": "createBuffer", "id": 1, "size": 16, "usage": TestUsage.copySrc | TestUsage.copyDst],
            ["op": "createBuffer", "id": 2, "size": 16, "usage": TestUsage.copyDst | TestUsage.mapRead],
            ["op": "writeBuffer", "buffer": 1, "data": [Float]([1, 2, 3, 4]).base64],
        // 8 bytes were skipped, so only the remaining 8 travel.
            ["op": "copyBufferToBuffer", "source": 1, "sourceOffset": 8, "destination": 2],
        ])
        XCTAssertEqual(try harness.readBufferSync(handle: 2, as: Float.self), [3, 4, 0, 0])
    }

    /// A copy past the end **kills the process with a Metal assertion** — it has to be caught as a validation error.
    func test_rejectsACopyBufferToBufferPastTheEnd() throws {
        harness.executeExpectingSuccess([
            ["op": "createBuffer", "id": 1, "size": 16, "usage": TestUsage.copySrc],
            ["op": "createBuffer", "id": 2, "size": 8, "usage": TestUsage.copyDst],
        ])

        for (command, expected) in [
            (["op": "copyBufferToBuffer", "source": 1, "destination": 2, "size": 16], "destination range exceeds"),
            (["op": "copyBufferToBuffer", "source": 1, "sourceOffset": 12, "destination": 2, "size": 8], "source range exceeds"),
            (["op": "copyBufferToBuffer", "source": 1, "destination": 2,
              "destinationOffset": 4, "size": 8], "destination range exceeds"),
            (["op": "copyBufferToBuffer", "source": 1, "destination": 2, "size": -4], "cannot be negative"),
        ] as [([String: Any], String)] {
            let result = harness.execute([command])
            XCTAssertTrue(
                ((errors(result).first?["message"] as? String) ?? "").contains(expected),
                "\(command) passed: \(harness.describeErrors(result))"
            )
        }
    }

    /// A zero-byte copy is a no-op — a Metal blit rejects it, so passing it through would be an error.
    func test_aZeroSizeCopyBufferToBufferIsANoOp() throws {
        harness.executeExpectingSuccess([
            ["op": "createBuffer", "id": 1, "size": 16, "usage": TestUsage.copySrc],
            ["op": "createBuffer", "id": 2, "size": 16, "usage": TestUsage.copyDst],
            ["op": "copyBufferToBuffer", "source": 1, "destination": 2, "size": 0],
        ])
    }

    // MARK: - Buffer mapping state

    /// The spec makes `mapAsync` mark a buffer "unavailable" so it cannot be used in queue work, removing the race.
    /// This implementation reads a `.storageModeShared` buffer without staging, so without this check a
    /// write from the next frame overlaps the same memory while a readback waits on GPU completion.
    func test_aMappedBufferCannotBeUsedInQueueWork() throws {
        harness.executeExpectingSuccess([
            ["op": "createBuffer", "id": 1, "size": 16, "usage": TestUsage.copyDst | TestUsage.mapRead],
            ["op": "createBuffer", "id": 2, "size": 16, "usage": TestUsage.copySrc],
        ])
        _ = try harness.readBufferSync(handle: 1)   // it becomes mapped here

        for command in [
            ["op": "writeBuffer", "buffer": 1, "data": [Float]([1, 2, 3, 4]).base64],
            ["op": "copyBufferToBuffer", "source": 2, "destination": 1, "size": 16],
            ["op": "copyBufferToBuffer", "source": 1, "destination": 2, "size": 16],
        ] as [[String: Any]] {
            let result = harness.execute([command])
            XCTAssertTrue(
                ((errors(result).first?["message"] as? String) ?? "").contains("is mapped"),
                "\(command["op"] ?? "?") passed: \(harness.describeErrors(result))"
            )
        }

        // After unmap it must be writable again.
        harness.executeExpectingSuccess([
            ["op": "unmapBuffer", "buffer": 1],
            ["op": "writeBuffer", "buffer": 1, "data": [Float]([1, 2, 3, 4]).base64],
        ])
    }

    func test_rereadingAnAlreadyMappedBufferIsRejected() throws {
        harness.executeExpectingSuccess([
            ["op": "createBuffer", "id": 1, "size": 16, "usage": TestUsage.copyDst | TestUsage.mapRead],
        ])
        _ = try harness.readBufferSync(handle: 1)

        XCTAssertThrowsError(try harness.readBufferSync(handle: 1), "a second mapping is rejected")
    }

    /// `mapAsync(GPUMapMode.WRITE)` → write into the mapping → `unmap()` must land in the buffer.
    /// The JS mapping is a copy, so the bytes come back in `unmapBuffer`'s `data` — before, they were
    /// silently dropped. Read back through the spec's path (copy into a `MAP_READ` staging buffer).
    func test_aWriteMappingIsWrittenBackAtUnmap() throws {
        harness.executeExpectingSuccess([
            ["op": "createBuffer", "id": 1, "size": 16, "usage": TestUsage.mapWrite | TestUsage.copySrc,
             "data": [Float]([9, 9, 9, 9]).base64],
            ["op": "createBuffer", "id": 2, "size": 16, "usage": TestUsage.copyDst | TestUsage.mapRead],
        ])
        // The mapping starts as the current contents (spec).
        XCTAssertEqual(try harness.readBufferSync(handle: 1, as: Float.self), [9, 9, 9, 9])

        // Still unavailable to queue work until unmap.
        let copy = harness.execute([
            ["op": "copyBufferToBuffer", "source": 1, "destination": 2, "size": 16],
        ])
        XCTAssertTrue(
            ((errors(copy).first?["message"] as? String) ?? "").contains("is mapped"),
            harness.describeErrors(copy)
        )

        harness.executeExpectingSuccess([
            ["op": "unmapBuffer", "buffer": 1, "offset": 0, "data": [Float]([1, 2, 3, 4]).base64],
            ["op": "copyBufferToBuffer", "source": 1, "destination": 2, "size": 16],
        ])
        XCTAssertEqual(try harness.readBufferSync(handle: 2, as: Float.self), [1, 2, 3, 4])
    }

    /// A partial write mapping is addressed in buffer offsets: only its range is written back, the rest survives.
    func test_aPartialWriteMappingWritesBackOnlyItsRange() throws {
        harness.executeExpectingSuccess([
            ["op": "createBuffer", "id": 1, "size": 16, "usage": TestUsage.mapWrite | TestUsage.copySrc,
             "data": [Float]([9, 9, 9, 9]).base64],
            ["op": "createBuffer", "id": 2, "size": 16, "usage": TestUsage.copyDst | TestUsage.mapRead],
        ])
        XCTAssertEqual(try harness.readBufferSync(handle: 1, as: Float.self, offset: 8, size: 8), [9, 9])

        // Outside the mapping — rejected, and the buffer is unmapped anyway (unmap cannot fail in the spec).
        let outside = harness.execute([
            ["op": "unmapBuffer", "buffer": 1, "offset": 0, "data": [Float]([5, 5]).base64],
        ])
        XCTAssertTrue(
            ((errors(outside).first?["message"] as? String) ?? "").contains("outside the mapped range"),
            harness.describeErrors(outside)
        )

        _ = try harness.readBufferSync(handle: 1, offset: 8, size: 8)
        harness.executeExpectingSuccess([
            ["op": "unmapBuffer", "buffer": 1, "offset": 8, "data": [Float]([3, 4]).base64],
            ["op": "copyBufferToBuffer", "source": 1, "destination": 2, "size": 16],
        ])
        XCTAssertEqual(try harness.readBufferSync(handle: 2, as: Float.self), [9, 9, 3, 4])
    }

    /// Data on an unmap needs a write mapping — otherwise `unmapBuffer` would be a back door around
    /// `writeBuffer`'s COPY_DST rule.
    func test_unmapDataWithoutAWriteMappingIsRejected() throws {
        harness.executeExpectingSuccess([
            ["op": "createBuffer", "id": 1, "size": 16, "usage": TestUsage.copyDst | TestUsage.mapRead,
             "data": [Float]([7, 7, 7, 7]).base64],
        ])
        _ = try harness.readBufferSync(handle: 1)   // a read mapping

        let result = harness.execute([
            ["op": "unmapBuffer", "buffer": 1, "data": [Float]([1, 2, 3, 4]).base64],
        ])
        XCTAssertTrue(
            ((errors(result).first?["message"] as? String) ?? "").contains("mapped for writing"),
            harness.describeErrors(result)
        )
        XCTAssertEqual(try harness.readBufferSync(handle: 1, as: Float.self), [7, 7, 7, 7],
                       "the data was not written, and the buffer was unmapped (mappable again)")
    }

    /// The spec allows `MAP_READ` only with `COPY_DST` and `MAP_WRITE` only with `COPY_SRC`.
    /// Metal covers everything with `.storageModeShared`, but unchecked it breaks only in a browser.
    func test_mappingUsageCombinesOnlyWithCopies() {
        for usage in [
            TestUsage.mapRead | TestUsage.queryResolve,
            TestUsage.mapRead | TestUsage.copySrc,
            TestUsage.mapRead | TestUsage.storage,
        ] {
            let result = harness.execute([["op": "createBuffer", "id": 1, "size": 16, "usage": usage]])
            XCTAssertTrue(
                ((errors(result).first?["message"] as? String) ?? "").contains("MAP_READ"),
                "usage \(usage) passed: \(harness.describeErrors(result))"
            )
        }
        harness.executeExpectingSuccess([
            ["op": "createBuffer", "id": 1, "size": 16, "usage": TestUsage.mapRead | TestUsage.copyDst],
            ["op": "createBuffer", "id": 2, "size": 16, "usage": TestUsage.mapRead],
        ])
    }

    func test_anOutOfRangeWriteBufferIsRejected() {
        let result = harness.execute([
            ["op": "createBuffer", "id": 1, "size": 8, "usage": TestUsage.copyDst],
            ["op": "writeBuffer", "buffer": 1, "bufferOffset": 4,
             "data": [Float](repeating: 0, count: 4).base64],
        ])
        XCTAssertTrue(((errors(result).first?["message"] as? String) ?? "").contains("out of range"))
    }

    func test_theExecuteResponseCarriesTheLiveObjectCount() {
        let result = harness.executeExpectingSuccess([
            ["op": "createBuffer", "id": 1, "size": 16, "usage": TestUsage.uniform],
            ["op": "createBuffer", "id": 2, "size": 16, "usage": TestUsage.uniform],
        ])
        XCTAssertEqual(result["objects"] as? Int, 2, "the count for watching missed destroys")

        let afterDestroy = harness.executeExpectingSuccess([["op": "destroy", "id": 1]])
        XCTAssertEqual(afterDestroy["objects"] as? Int, 1)
    }

    func test_resetDiscardsEveryObject() {
        harness.executeExpectingSuccess([
            ["op": "createBuffer", "id": 1, "size": 16, "usage": TestUsage.uniform],
            ["op": "createBuffer", "id": 2, "size": 16, "usage": TestUsage.uniform],
        ])
        XCTAssertEqual(harness.liveObjects, 2)

        harness.runtime.reset()
        XCTAssertEqual(harness.liveObjects, 0)
    }

    // MARK: - Queue write ordering

    /// Queue writes (`writeBuffer`/`writeTexture`) land on the queue timeline **at the call** — before
    /// the batch's encoder work, which is the batch's submit (`WGPUBackend.writeBuffer`). Under the old
    /// scheme they rode the frame command buffer's blit in stream order instead, so a write after a
    /// pass in the same batch ran after it, and a write while a pass was open was rejected outright.
    /// The `bytesPerRow` of a buffer↔texture copy must be **a multiple of 256** (a spec requirement the
    /// engine enforces). So the readback buffer carries padding per row and the tight pixels must be re-extracted.
    private static let copyRowStride = 256

    /// Joins only the valid span of each row out of bytes received at a 256 stride.
    private func packedRows(
        _ bytes: [UInt8], rowBytes: Int, rows: Int, offset: Int = 0
    ) -> [UInt8] {
        (0..<rows).flatMap { row -> [UInt8] in
            let start = offset + row * Self.copyRowStride
            return Array(bytes[start..<(start + rowBytes)])
        }
    }

    /// Batch 1 paints the texture red and is submitted; batch 2 uploads green. The upload is serialized
    /// **after** the earlier submit — not run ahead of it, which would leave red.
    func test_writeTextureRunsAfterARenderPassSubmittedEarlier() throws {
        let green = [UInt8]([0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255])
        harness.executeExpectingSuccess([
            ["op": "createTexture", "id": 1, "size": ["width": 2, "height": 2], "format": "rgba8unorm",
             "usage": TestUsage.renderAttachment | TestUsage.textureCopyDst | TestUsage.textureCopySrc],
            ["op": "createTextureView", "id": 2, "texture": 1],
            ["op": "beginRenderPass", "colorAttachments": [[
                "view": 2, "loadOp": "clear", "storeOp": "store",
                "clearValue": ["r": 1, "g": 0, "b": 0, "a": 1],
            ]]],
            ["op": "endPass"],
        ])
        harness.executeExpectingSuccess([
            // `queue.writeTexture` has no 256 limit — the tight 8B rows go up as they are.
            ["op": "writeTexture", "texture": 1, "data": Data(green).base64EncodedString(),
             "size": ["width": 2, "height": 2], "bytesPerRow": 8],
            ["op": "createBuffer", "id": 3, "size": 512, "usage": TestUsage.copyDst | TestUsage.mapRead],
            ["op": "copyTextureToBuffer",
             "source": ["texture": 1],
             "destination": ["buffer": 3, "bytesPerRow": Self.copyRowStride],
             "copySize": ["width": 2, "height": 2]],
        ])

        let bytes = Array(try harness.readBufferSync(handle: 3))
        XCTAssertEqual(
            packedRows(bytes, rowBytes: 8, rows: 2), green,
            "the upload must land after the earlier submit's render pass"
        )
    }

    /// In **one** batch a render pass paints red and a writeTexture **later in the stream** uploads green.
    /// The batch's encoder work is one submit at the batch's end, so the queue write lands first and
    /// the pass's clear is the final content — the browser and Dawn order of
    /// `writeTexture(…); queue.submit([cb])` with the pass recorded into `cb` before the call.
    func test_writeTextureLandsBeforeTheEncoderWorkOfItsOwnBatch() throws {
        let green = [UInt8]([0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255])
        let red = [UInt8]([255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255])
        harness.executeExpectingSuccess([
            ["op": "createTexture", "id": 1, "size": ["width": 2, "height": 2], "format": "rgba8unorm",
             "usage": TestUsage.renderAttachment | TestUsage.textureCopyDst | TestUsage.textureCopySrc],
            ["op": "createTextureView", "id": 2, "texture": 1],
            ["op": "beginRenderPass", "colorAttachments": [[
                "view": 2, "loadOp": "clear", "storeOp": "store",
                "clearValue": ["r": 1, "g": 0, "b": 0, "a": 1],
            ]]],
            ["op": "endPass"],
            ["op": "writeTexture", "texture": 1, "data": Data(green).base64EncodedString(),
             "size": ["width": 2, "height": 2], "bytesPerRow": 8],
            ["op": "createBuffer", "id": 3, "size": 512, "usage": TestUsage.copyDst | TestUsage.mapRead],
            ["op": "copyTextureToBuffer",
             "source": ["texture": 1],
             "destination": ["buffer": 3, "bytesPerRow": Self.copyRowStride],
             "copySize": ["width": 2, "height": 2]],
        ])

        let bytes = Array(try harness.readBufferSync(handle: 3))
        XCTAssertEqual(
            packedRows(bytes, rowBytes: 8, rows: 2), red,
            "the queue write precedes the batch's command buffer, so the pass's clear wins"
        )
    }

    /// A fragment shader painting the color held in a uniform — for the open-pass write tests below.
    private static let uniformColorShader = """
    struct Tint { color: vec4f };
    @group(0) @binding(0) var<uniform> tint: Tint;

    @vertex
    fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
        var positions = array<vec2f, 3>(
            vec2f(-1.0, -1.0),
            vec2f( 3.0, -1.0),
            vec2f(-1.0,  3.0),
        );
        return vec4f(positions[index], 0.0, 1.0);
    }

    @fragment
    fn fs_main() -> @location(0) vec4f {
        return tint.color;
    }
    """

    /// Resources for a full-screen draw whose color comes from uniform buffer 2 (initially red).
    private func uniformColorSetup() -> [[String: Any]] {
        let red: [Float] = [1, 0, 0, 1]
        return [
            ["op": "configureCanvas", "canvas": "test", "format": "rgba8unorm"],
            ["op": "createShaderModule", "id": 1, "code": Self.uniformColorShader],
            ["op": "createBuffer", "id": 2, "size": 16,
             "usage": TestUsage.uniform | TestUsage.copyDst, "data": red.base64],
            ["op": "createRenderPipeline", "id": 3, "layout": "auto",
             "vertex": ["module": 1, "entryPoint": "vs_main"],
             "fragment": ["module": 1, "entryPoint": "fs_main", "targets": [["format": "rgba8unorm"]]]],
            ["op": "getBindGroupLayout", "id": 4, "pipeline": 3, "index": 0],
            ["op": "createBindGroup", "id": 5, "layout": 4,
             "entries": [["binding": 0, "resource": ["buffer": 2]]]],
            ["op": "getCurrentTexture", "id": 10, "canvas": "test"],
            ["op": "createTextureView", "id": 11, "texture": 10],
            ["op": "beginRenderPass", "colorAttachments": [[
                "view": 11, "loadOp": "clear", "storeOp": "store",
                "clearValue": ["r": 0, "g": 0, "b": 0, "a": 1],
            ]]],
        ]
    }

    /// The everyday per-object loop — `queue.writeBuffer(uniform); pass.setBindGroup(); pass.draw()`
    /// with the pass still recording. A queue operation is legal at any moment; it used to be rejected
    /// as "copy and upload commands cannot be used inside a render/compute pass".
    func test_aWriteBufferWhileARenderPassIsOpenIsAcceptedAndSeenByTheDraw() throws {
        let green: [Float] = [0, 1, 0, 1]
        harness.executeExpectingSuccess(uniformColorSetup() + [
            ["op": "writeBuffer", "buffer": 2, "data": green.base64],
            ["op": "setPipeline", "pipeline": 3],
            ["op": "setBindGroup", "index": 0, "bindGroup": 5],
            ["op": "draw", "vertexCount": 3],
            ["op": "endPass"],
        ])

        try harness.assertPixel(x: 4, y: 4, equals: (0, 255, 0, 255), "the draw sees the written uniform")
    }

    /// A draw **recorded before** the write but submitted after it sees the **new** value — the pass is
    /// still recording, so its command buffer is necessarily submitted after the write (spec queue
    /// timeline; what browsers and Dawn do). Stream order would have painted red.
    func test_aDrawRecordedBeforeAnOpenPassWriteSeesTheNewValue() throws {
        let green: [Float] = [0, 1, 0, 1]
        harness.executeExpectingSuccess(uniformColorSetup() + [
            ["op": "setPipeline", "pipeline": 3],
            ["op": "setBindGroup", "index": 0, "bindGroup": 5],
            ["op": "draw", "vertexCount": 3],
            ["op": "writeBuffer", "buffer": 2, "data": green.base64],
            ["op": "endPass"],
        ])

        try harness.assertPixel(
            x: 4, y: 4, equals: (0, 255, 0, 255), "the write lands before the pass is submitted"
        )
    }

    /// Queue writes keep **call order among themselves** wherever the pass state stood when they were
    /// issued — a write outside a pass and a later one inside a pass, to the same bytes, leave the
    /// later value.
    func test_queueWritesKeepCallOrderAcrossAnOpenPass() throws {
        harness.executeExpectingSuccess([
            ["op": "createBuffer", "id": 1, "size": 16, "usage": TestUsage.copyDst | TestUsage.mapRead],
            ["op": "writeBuffer", "buffer": 1, "data": [Float]([1, 2, 3, 4]).base64],
            ["op": "beginComputePass"],
            ["op": "writeBuffer", "buffer": 1, "data": [Float]([5, 6, 7, 8]).base64],
            ["op": "endPass"],
        ])

        XCTAssertEqual(try harness.readBufferSync(handle: 1, as: Float.self), [5, 6, 7, 8])
    }

    func test_writeTextureUploadsArrayTextureLayersSliceBySlice() throws {
        let red = [UInt8]([255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255])
        let blue = [UInt8]([0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255])
        // One layer occupies 2 rows × 256B = 512B.
        let layerBytes = 2 * Self.copyRowStride
        harness.executeExpectingSuccess([
            ["op": "createTexture", "id": 1,
             "size": ["width": 2, "height": 2, "depthOrArrayLayers": 2], "format": "rgba8unorm",
             "usage": TestUsage.textureCopyDst | TestUsage.textureCopySrc],
            // Two layers at once — data joined at bytesPerImage (16B) intervals.
            ["op": "writeTexture", "texture": 1, "data": Data(red + blue).base64EncodedString(),
             "size": ["width": 2, "height": 2, "depthOrArrayLayers": 2],
             "bytesPerRow": 8, "rowsPerImage": 2],
            ["op": "createBuffer", "id": 2, "size": 2 * layerBytes,
             "usage": TestUsage.copyDst | TestUsage.mapRead],
            ["op": "copyTextureToBuffer",
             "source": ["texture": 1, "origin": ["x": 0, "y": 0, "z": 0]],
             "destination": ["buffer": 2, "bytesPerRow": Self.copyRowStride, "offset": 0],
             "copySize": ["width": 2, "height": 2]],
            ["op": "copyTextureToBuffer",
             "source": ["texture": 1, "origin": ["x": 0, "y": 0, "z": 1]],
             "destination": ["buffer": 2, "bytesPerRow": Self.copyRowStride, "offset": layerBytes],
             "copySize": ["width": 2, "height": 2]],
        ])

        let bytes = Array(try harness.readBufferSync(handle: 2))
        XCTAssertEqual(packedRows(bytes, rowBytes: 8, rows: 2), red, "layer 0")
        XCTAssertEqual(
            packedRows(bytes, rowBytes: 8, rows: 2, offset: layerBytes), blue, "layer 1"
        )
    }

    // MARK: - Indirect draw contract

    /// These three are **the boundary between a crash and an error.** Letting alignment and range reach
    /// Metal has the validation layer kill the process with an assertion, and `INDIRECT` usage has no
    /// concept in Metal so nobody checks it (unchecked here, code that breaks only in a browser ships).
    /// One argument buffer and one pass. Argument validation precedes `setPipeline`, so no pipeline is needed.
    private func indirectSetup(usage: Int, size: Int = 32, compute: Bool = false) -> [[String: Any]] {
        let buffer: [[String: Any]] = [["op": "createBuffer", "id": 1, "size": size, "usage": usage]]
        if compute { return buffer + [["op": "beginComputePass"]] }
        return buffer + [
            ["op": "configureCanvas", "canvas": "test", "format": "rgba8unorm"],
            ["op": "getCurrentTexture", "id": 90, "canvas": "test"],
            ["op": "createTextureView", "id": 91, "texture": 90],
            ["op": "beginRenderPass", "colorAttachments": [[
                "view": 91, "loadOp": "clear", "storeOp": "store",
                "clearValue": ["r": 0, "g": 0, "b": 0, "a": 1],
            ]]],
        ]
    }

    func test_rejectsAnIndirectOffsetThatIsNotAMultipleOf4() {
        let setup = indirectSetup(usage: TestUsage.indirect)
        let result = harness.execute(setup + [
            ["op": "drawIndirect", "indirectBuffer": 1, "indirectOffset": 2],
        ])

        XCTAssertEqual(errors(result).first?["kind"] as? String, "validation")
        XCTAssertEqual(errors(result).first?["path"] as? String, "commands[\(setup.count)].indirectOffset")
    }

    func test_rejectsIndirectArgumentsPastTheBufferEnd() {
        // drawIndexedIndirect reads 20B — offset 16 + 20 > 32.
        let setup = indirectSetup(usage: TestUsage.indirect)
        let result = harness.execute(setup + [
            ["op": "drawIndexedIndirect", "indirectBuffer": 1, "indirectOffset": 16],
        ])

        XCTAssertEqual(errors(result).first?["kind"] as? String, "validation")
        XCTAssertEqual(errors(result).first?["path"] as? String, "commands[\(setup.count)].indirectOffset")
        XCTAssertTrue(((errors(result).first?["message"] as? String) ?? "").contains("exceed the buffer"))
    }

    func test_aBufferWithoutINDIRECTUsageCannotBeUsedForAnIndirectDispatch() {
        let setup = indirectSetup(usage: TestUsage.storage, compute: true)
        let result = harness.execute(setup + [
            ["op": "dispatchWorkgroupsIndirect", "indirectBuffer": 1],
        ])

        XCTAssertEqual(errors(result).first?["kind"] as? String, "validation")
        XCTAssertEqual(errors(result).first?["path"] as? String, "commands[\(setup.count)].indirectBuffer")
        XCTAssertTrue(((errors(result).first?["message"] as? String) ?? "").contains("INDIRECT"))
    }

    func test_anIndirectDrawWithNoPassIsAnError() {
        let result = harness.execute([
            ["op": "createBuffer", "id": 1, "size": 32, "usage": TestUsage.indirect],
            ["op": "drawIndirect", "indirectBuffer": 1],
        ])
        XCTAssertTrue(
            ((errors(result).first?["message"] as? String) ?? "").contains("beginRenderPass"),
            "it must say what has to happen first"
        )
    }

    // MARK: - Shader compilation diagnostics

    /// In the spec **a shader module is created even when compilation fails.** With no handle at all,
    /// every later command breaks with only "does not exist" and **the real cause (the parse failure) vanishes from view.**
    func test_theModuleIsStillCreatedOnAParseFailureAndReturnsTheCause() {
        let result = harness.execute([
            ["op": "createShaderModule", "id": 1, "code": """
             @vertex
             fn vs() -> @builtin(position) vec4f {
                 return vec4f(1.0 1.0, 1.0, 1.0);
             }
             """],
        ])

        // (1) The cause is reported on the spot (down to the line number).
        let first = errors(result).first
        XCTAssertEqual(first?["kind"] as? String, "validation")
        XCTAssertTrue(((first?["message"] as? String) ?? "").contains("parse"))
        XCTAssertEqual(first?["line"] as? Int, 3, "the line number must ride as a number so an editor can jump")

        // (2) The module still exists and returns diagnostics.
        let info = harness.runtime.shaderCompilationInfo(handle: 1)
        XCTAssertEqual(info["ok"] as? Bool, true)
        let messages = try? XCTUnwrap(info["messages"] as? [[String: Any]])
        XCTAssertEqual(messages?.count, 1)
        XCTAssertEqual(messages?.first?["type"] as? String, "error")
        XCTAssertEqual(messages?.first?["lineNum"] as? Int, 3)
    }

    /// Building a pipeline from a broken module must restate **the real cause** — rephrasing it as
    /// "no entry point" sends the user off suspecting the shader name and fixing the wrong thing.
    func test_buildingAPipelineFromABrokenModuleRestatesTheCause() {
        let result = harness.execute([
            ["op": "createShaderModule", "id": 1, "code": "fn broken( {"],
            ["op": "createRenderPipeline", "id": 2, "layout": "auto",
             "vertex": ["module": 1, "entryPoint": "vs"]],
        ])

        let messages = errors(result).compactMap { $0["message"] as? String }
        XCTAssertTrue(
            messages.contains { $0.contains("compil") },
            "the pipeline error does not carry the cause: \(messages)"
        )
    }

    func test_aHealthyModuleHasNoDiagnostics() {
        harness.executeExpectingSuccess([
            ["op": "createShaderModule", "id": 1, "code": """
             @fragment fn fs() -> @location(0) vec4f { return vec4f(1.0); }
             """],
        ])

        let info = harness.runtime.shaderCompilationInfo(handle: 1)
        XCTAssertEqual(info["ok"] as? Bool, true)
        XCTAssertEqual((info["messages"] as? [[String: Any]])?.count, 0)
    }

    func test_diagnosticsForAMissingModuleAreAnError() {
        let info = harness.runtime.shaderCompilationInfo(handle: 999)
        XCTAssertEqual(info["ok"] as? Bool, false)
    }

    // MARK: - Debug markers

    /// They must be accepted both inside and outside a pass — this is where Xcode GPU capture's section names come from.
    func test_acceptsDebugMarkersInsideAndOutsideAPass() {
        let result = harness.execute([
            ["op": "configureCanvas", "canvas": "test", "format": "rgba8unorm"],
            ["op": "createBuffer", "id": 1, "size": 16, "usage": TestUsage.copyDst],
            // Outside a pass — it attaches to the command buffer. (writeBuffer rides the separate
            // queue-write command buffer — not a scope.)
            ["op": "pushDebugGroup", "groupLabel": "upload"],
            ["op": "writeBuffer", "buffer": 1, "data": [Float](repeating: 1, count: 4).base64],
            ["op": "insertDebugMarker", "markerLabel": "marker"],
            ["op": "popDebugGroup"],
            // Inside a pass — it attaches to the render encoder.
            ["op": "getCurrentTexture", "id": 10, "canvas": "test"],
            ["op": "createTextureView", "id": 11, "texture": 10],
            ["op": "beginRenderPass", "colorAttachments": [[
                "view": 11, "loadOp": "clear", "storeOp": "store",
                "clearValue": ["r": 0, "g": 0, "b": 0, "a": 1],
            ]]],
            ["op": "pushDebugGroup", "groupLabel": "main pass"],
            ["op": "insertDebugMarker", "markerLabel": "just before the draw"],
            ["op": "popDebugGroup"],
            ["op": "endPass"],
        ])

        XCTAssertEqual(result["ok"] as? Bool, true, harness.describeErrors(result))
    }

    /// An unmatched `pop` **kills the process with a Metal assertion.** We count the depth, stop it and
    /// report a validation error — dying here would remove any chance to diagnose.
    func test_anUnmatchedPopDebugGroupIsAnErrorNotProcessDeath() {
        let result = harness.execute([
            ["op": "createBuffer", "id": 1, "size": 16, "usage": TestUsage.copyDst],
            ["op": "writeBuffer", "buffer": 1, "data": [Float](repeating: 1, count: 4).base64],
            ["op": "popDebugGroup"],
        ])

        XCTAssertEqual(errors(result).first?["kind"] as? String, "validation")
        XCTAssertTrue(
            ((errors(result).first?["message"] as? String) ?? "").contains("no matching pushDebugGroup"),
            "\(errors(result))"
        )
    }

    /// An unmatched pop **when there is no GPU work yet** must be reported too.
    ///
    /// With no command buffer there is nothing to send the backend, but skipping the report too lets the
    /// most common mistake — popping before anything has been done — pass silently. Counting the pairs is this function's job.
    func test_anUnmatchedPopDebugGroupWithNoWorkIsAlsoAnError() {
        let result = harness.execute([["op": "popDebugGroup"]])

        XCTAssertEqual(errors(result).first?["kind"] as? String, "validation", "\(result)")
        XCTAssertTrue(
            ((errors(result).first?["message"] as? String) ?? "").contains("no matching pushDebugGroup"),
            "\(errors(result))"
        )
    }

    /// Metal dies if a pass ends with one open too — we close it and report.
    func test_aDebugGroupLeftOpenIsClosedAndReported() {
        let result = harness.execute([
            ["op": "configureCanvas", "canvas": "test", "format": "rgba8unorm"],
            ["op": "getCurrentTexture", "id": 10, "canvas": "test"],
            ["op": "createTextureView", "id": 11, "texture": 10],
            ["op": "beginRenderPass", "colorAttachments": [[
                "view": 11, "loadOp": "clear", "storeOp": "store",
                "clearValue": ["r": 0, "g": 0, "b": 0, "a": 1],
            ]]],
            ["op": "pushDebugGroup", "groupLabel": "never closed"],
            ["op": "endPass"],
        ])

        XCTAssertEqual(errors(result).first?["kind"] as? String, "validation")
        XCTAssertTrue(
            ((errors(result).first?["message"] as? String) ?? "").contains("still open"),
            "\(errors(result))"
        )
        // And the process is alive — reaching this assertion is itself the evidence.
    }

    /// A batch of markers only — it must pass without error even with no other work.
    ///
    /// A frame-region marker attaches to the command buffer, so one must be created for the pair to match.
    /// (Without it the push disappears and only the pop remains, becoming "unmatched".)
    func test_aBatchOfOnlyMarkersRaisesNoError() {
        let result = harness.execute([
            ["op": "pushDebugGroup", "groupLabel": "empty frame"],
            ["op": "insertDebugMarker", "markerLabel": "marker"],
            ["op": "popDebugGroup"],
        ])

        XCTAssertEqual(result["ok"] as? Bool, true, harness.describeErrors(result))
    }

    /// An MSL module has no WGSL reflection — its diagnostics must be empty and it must be **valid**.
    /// Handled wrongly it is mistaken for "failed to compile", closing off a perfectly good MSL escape hatch.
    func test_anMSLModuleHasEmptyDiagnosticsAndIsUsable() {
        let result = harness.execute([
            ["op": "createShaderModule", "id": 1, "language": "msl", "code": """
             #include <metal_stdlib>
             using namespace metal;
             vertex float4 vs_main() { return float4(0, 0, 0, 1); }
             """],
        ])

        XCTAssertEqual(result["ok"] as? Bool, true, harness.describeErrors(result))
        let info = harness.runtime.shaderCompilationInfo(handle: 1)
        XCTAssertEqual(info["ok"] as? Bool, true)
        XCTAssertEqual((info["messages"] as? [[String: Any]])?.count, 0)
    }

    // MARK: - clearBuffer

    /// The result must match pushing zeros through `writeBuffer` — the only difference is not crossing the bridge.
    func test_clearBufferZeroesTheRange() throws {
        let filled = [Float](repeating: 7, count: 8)
        // MAP_READ combines only with COPY_DST (a spec rule this implementation enforces).
        harness.executeExpectingSuccess([
            ["op": "createBuffer", "id": 1, "size": 32,
             "usage": TestUsage.copyDst | TestUsage.mapRead, "data": filled.base64],
            // Clear only the first 16 bytes (= 4 elements) — the rest must remain to show the range held.
            ["op": "clearBuffer", "buffer": 1, "offset": 0, "size": 16],
        ])

        let values = try harness.readBufferSync(handle: 1, as: Float.self)
        XCTAssertEqual(values, [0, 0, 0, 0, 7, 7, 7, 7])
    }

    func test_clearBufferWithoutASizeClearsToTheEnd() throws {
        harness.executeExpectingSuccess([
            ["op": "createBuffer", "id": 1, "size": 16,
             "usage": TestUsage.copyDst | TestUsage.mapRead,
             "data": [Float](repeating: 3, count: 4).base64],
            ["op": "clearBuffer", "buffer": 1, "offset": 8],
        ])

        XCTAssertEqual(try harness.readBufferSync(handle: 1, as: Float.self), [3, 3, 0, 0])
    }

    func test_clearBufferValidatesAlignmentUsageAndRange() {
        // (1) Rejected without COPY_DST — Metal just fills, so unchecked it breaks only in a browser.
        let noUsage = harness.execute([
            ["op": "createBuffer", "id": 1, "size": 16, "usage": TestUsage.vertex],
            ["op": "clearBuffer", "buffer": 1],
        ])
        XCTAssertTrue(
            ((errors(noUsage).first?["message"] as? String) ?? "").contains("COPY_DST"),
            "\(errors(noUsage))"
        )

        // (2) Rejected when not a multiple of 4 (a spec rule).
        let misaligned = harness.execute([
            ["op": "createBuffer", "id": 2, "size": 16, "usage": TestUsage.copyDst],
            ["op": "clearBuffer", "buffer": 2, "offset": 2, "size": 4],
        ])
        XCTAssertTrue(((errors(misaligned).first?["message"] as? String) ?? "").contains("multiples of 4"))

        // (3) Rejected past the end.
        let overflow = harness.execute([
            ["op": "createBuffer", "id": 3, "size": 16, "usage": TestUsage.copyDst],
            ["op": "clearBuffer", "buffer": 3, "offset": 8, "size": 16],
        ])
        XCTAssertTrue(((errors(overflow).first?["message"] as? String) ?? "").contains("exceeds the buffer"))
    }

    // MARK: - Entry point resolution (the spec's "get the entry point")

    /// A shader with one vertex entry (`mainVS`) and three fragment ones (`main_2d`, …) — the same shape as three.js's mipmap shader.
    private static let multiEntryShader = """
    @vertex fn mainVS() -> @builtin(position) vec4f { return vec4f(0.0, 0.0, 0.0, 1.0); }
    @fragment fn main_2d() -> @location(0) vec4f { return vec4f(1.0, 0.0, 0.0, 1.0); }
    @fragment fn main_cube() -> @location(0) vec4f { return vec4f(0.0, 1.0, 0.0, 1.0); }
    """

    /// `entryPoint` is **not required** in the spec. Omitted, it uses the stage's only entry point.
    /// Guessing `"main"` rejects entire shaders named otherwise — three.js's mipmap generation broke
    /// exactly that way (`mainVS` was present while it looked for `main`).
    func test_omittingEntryPointUsesTheStagesOnlyEntryPoint() {
        let result = harness.execute([
            ["op": "createShaderModule", "id": 1, "code": Self.multiEntryShader],
            // Vertex omitted (it must resolve to the only mainVS); fragment names one of the three.
            ["op": "createRenderPipeline", "id": 2, "layout": "auto",
             "vertex": ["module": 1],
             "fragment": ["module": 1, "entryPoint": "main_2d",
                          "targets": [["format": "rgba8unorm"]]]],
        ])

        XCTAssertEqual(result["ok"] as? Bool, true, harness.describeErrors(result))
    }

    func test_withMoreThanOneCandidateItRefusesToChoose() {
        let result = harness.execute([
            ["op": "createShaderModule", "id": 1, "code": Self.multiEntryShader],
            // With two fragment entry points, omitting it leaves nothing to choose — it must not quietly pick one.
            ["op": "createRenderPipeline", "id": 2, "layout": "auto",
             "vertex": ["module": 1],
             "fragment": ["module": 1, "targets": [["format": "rgba8unorm"]]]],
        ])

        XCTAssertEqual(errors(result).first?["kind"] as? String, "validation")
        let message = (errors(result).first?["message"] as? String) ?? ""
        XCTAssertTrue(message.contains("main_2d"), "it must list the candidate names: \(message)")
        XCTAssertTrue(message.contains("main_cube"))
    }

    func test_computeCanOmitEntryPointToo() {
        let result = harness.execute([
            ["op": "createShaderModule", "id": 1, "code": """
             @compute @workgroup_size(4) fn onlyKernel() {}
             """],
            ["op": "createComputePipeline", "id": 2, "layout": "auto", "compute": ["module": 1]],
        ])

        XCTAssertEqual(result["ok"] as? Bool, true, harness.describeErrors(result))
    }

    func test_rejectsAnEntryPointFromAnotherStage() {
        let result = harness.execute([
            ["op": "createShaderModule", "id": 1, "code": Self.multiEntryShader],
            // A vertex entry point was given in the fragment position.
            ["op": "createRenderPipeline", "id": 2, "layout": "auto",
             "vertex": ["module": 1, "entryPoint": "mainVS"],
             "fragment": ["module": 1, "entryPoint": "mainVS",
                          "targets": [["format": "rgba8unorm"]]]],
        ])

        XCTAssertEqual(errors(result).first?["kind"] as? String, "validation")
        XCTAssertTrue(
            ((errors(result).first?["message"] as? String) ?? "").contains("vertex"),
            "it must say which stage: \(errors(result))"
        )
    }

    /// On a device without indirect argument support it must be **an error rather than process death**.
    ///
    /// Metal ends the app with `MTLValidateFeatureSupport … failed assertion` — the iOS simulator
    /// (Apple family 2) falls exactly here. Real devices are A12 (family 5) or newer and support it, but
    /// dying while developing on the simulator leaves no reason behind.
    func test_aDeviceWithoutIndirectArgumentsErrorsRatherThanDies() throws {
        try XCTSkipIf(
            harness.supports(.indirectArguments),
            "this device supports indirect arguments — the rejection path is only visible on an unsupported one"
        )
        let result = harness.execute([
            ["op": "createBuffer", "id": 1, "size": 32, "usage": TestUsage.indirect],
            ["op": "configureCanvas", "canvas": "test", "format": "rgba8unorm"],
            ["op": "getCurrentTexture", "id": 2, "canvas": "test"],
            ["op": "createTextureView", "id": 3, "texture": 2],
            ["op": "beginRenderPass", "colorAttachments": [[
                "view": 3, "loadOp": "clear", "storeOp": "store",
                "clearValue": ["r": 0, "g": 0, "b": 0, "a": 1],
            ]]],
            ["op": "drawIndirect", "indirectBuffer": 1],
            ["op": "endPass"],
        ])

        XCTAssertEqual(errors(result).first?["kind"] as? String, "unsupported")
        XCTAssertTrue(
            ((errors(result).first?["message"] as? String) ?? "").contains("simulator"),
            "it must say where and why it cannot: \(errors(result))"
        )
    }

    /// The other direction — a supporting device advertises `indirect-first-instance` and one that cannot hides it.
    /// Advertising it and then refusing the first call betrays the app that checked before using it.
    func test_indirectFeatureAdvertisementMatchesDeviceCapability() throws {
        let features = try XCTUnwrap(harness.runtime.adapterInfo()["features"] as? [String])
        XCTAssertEqual(
            features.contains("indirect-first-instance"),
            harness.supports(.indirectArguments),
            "the advertisement and the actual capability disagree"
        )
    }

    func test_anIndirectDrawWithNoPipelineIsRejectedWithItsOpNameInTheMessage() {
        // This guard once sat after `applyDrawState()` and was unreachable — the generic message ("setPipeline
        // is required before draw") went out instead, leaving the user unable to tell which op was at fault.
        let renderSetup = indirectSetup(usage: TestUsage.indirect)
        let renderResult = harness.execute(renderSetup + [
            ["op": "drawIndirect", "indirectBuffer": 1],
        ])
        XCTAssertEqual(errors(renderResult).first?["kind"] as? String, "validation")
        XCTAssertTrue(
            ((errors(renderResult).first?["message"] as? String) ?? "")
                .contains("setPipeline is required before drawIndirect"),
            "the message must carry the op name (drawIndirect): \(errors(renderResult))"
        )

        let computeSetup = indirectSetup(usage: TestUsage.indirect, compute: true)
        let computeResult = harness.execute(computeSetup + [
            ["op": "dispatchWorkgroupsIndirect", "indirectBuffer": 1],
        ])
        XCTAssertTrue(
            ((errors(computeResult).first?["message"] as? String) ?? "")
                .contains("setPipeline is required before dispatchWorkgroupsIndirect"),
            "the message must carry the op name (dispatchWorkgroupsIndirect): \(errors(computeResult))"
        )
    }

    func test_adapterInfoReportsTheLimits() {
        let info = harness.runtime.adapterInfo()

        XCTAssertEqual(info["ok"] as? Bool, true)
        XCTAssertEqual(info["backend"] as? String, "metal")
        XCTAssertEqual(info["preferredCanvasFormat"] as? String, "bgra8unorm")
        let limits = try? XCTUnwrap(info["limits"] as? [String: Any])
        XCTAssertNotNil(limits?["maxVertexBuffers"])
    }

    /// The spec's `GPUAdapterInfo` — the standard names web code reads when branching on GPU kind.
    func test_adapterInfoCarriesTheSpecGPUAdapterInfo() throws {
        let info = try XCTUnwrap(harness.runtime.adapterInfo()["info"] as? [String: Any])

        XCTAssertEqual(info["vendor"] as? String, "apple")
        XCTAssertFalse((info["description"] as? String ?? "").isEmpty, "there must be a device name")
        // Where Metal has no family query the value is **an empty string** (a spec rule) — we do not invent one.
        XCTAssertEqual(info["device"] as? String, "")
        XCTAssertEqual(info["isFallbackAdapter"] as? Bool, false)
        // We do not advertise the subgroups feature, so it is 0.
        XCTAssertEqual(info["subgroupMinSize"] as? Int, 0)
        // architecture reports only as far as we learn — unknown is an empty string, but never nil.
        XCTAssertNotNil(info["architecture"] as? String)
    }

    /// The **keys of limits must use the spec spelling.** Web libraries read them by those names to set
    /// budgets, so naming them our own way makes them see `undefined` and assume wrongly (absent though present).
    func test_limitsCarriesEverySpecName() throws {
        let limits = try XCTUnwrap(harness.runtime.adapterInfo()["limits"] as? [String: Any])

        // Every entry of the spec's `GPUSupportedLimits` (webgpu-md §3.6.2).
        let required = [
            "maxTextureDimension1D", "maxTextureDimension2D", "maxTextureDimension3D",
            "maxTextureArrayLayers", "maxBindGroups", "maxBindGroupsPlusVertexBuffers",
            "maxBindingsPerBindGroup", "maxDynamicUniformBuffersPerPipelineLayout",
            "maxDynamicStorageBuffersPerPipelineLayout", "maxSampledTexturesPerShaderStage",
            "maxSamplersPerShaderStage", "maxStorageBuffersPerShaderStage",
            "maxStorageTexturesPerShaderStage", "maxUniformBuffersPerShaderStage",
            "maxUniformBufferBindingSize", "maxStorageBufferBindingSize",
            "minUniformBufferOffsetAlignment", "minStorageBufferOffsetAlignment",
            "maxVertexBuffers", "maxBufferSize", "maxVertexAttributes", "maxVertexBufferArrayStride",
            "maxInterStageShaderVariables", "maxColorAttachments", "maxColorAttachmentBytesPerSample",
            "maxComputeWorkgroupStorageSize", "maxComputeInvocationsPerWorkgroup",
            "maxComputeWorkgroupSizeX", "maxComputeWorkgroupSizeY", "maxComputeWorkgroupSizeZ",
            "maxComputeWorkgroupsPerDimension",
        ]
        for key in required {
            XCTAssertNotNil(limits[key], "the spec limit '\(key)' is missing")
            XCTAssertGreaterThan((limits[key] as? Int) ?? 0, 0, "'\(key)' is 0")
        }
    }

    // MARK: - minBindingSize

    /// Binding 0 needs 64 bytes (a uniform mat4x4f struct), binding 1 one 16-byte element.
    private let minimumSizeShader = """
    struct Params { m: mat4x4f }
    struct Item { a: vec4f }
    @group(0) @binding(0) var<uniform> params: Params;
    @group(0) @binding(1) var<storage, read_write> items: array<Item>;

    @compute @workgroup_size(1)
    fn main() { items[0].a = params.m[0]; }
    """

    private func minimumSizeLayout(min0: Int) -> [String: Any] {
        ["op": "createBindGroupLayout", "id": 2, "entries": [
            ["binding": 0, "visibility": 4, "buffer": ["type": "uniform", "minBindingSize": min0]],
            ["binding": 1, "visibility": 4, "buffer": ["type": "storage"]],
        ]]
    }

    /// createBindGroup checks the bound range against the layout's minBindingSize — and an auto
    /// layout carries the shader's minimum, so an undersized group fails right there.
    func test_aBindGroupSmallerThanTheLayoutMinBindingSizeIsRejected() {
        let explicit = harness.execute([
            ["op": "createBuffer", "id": 1, "size": 32, "usage": TestUsage.uniform | TestUsage.storage],
            minimumSizeLayout(min0: 64),
            ["op": "createBindGroup", "id": 3, "layout": 2, "entries": [
                ["binding": 0, "resource": ["buffer": 1]],
                ["binding": 1, "resource": ["buffer": 1]],
            ]],
        ])
        XCTAssertEqual(errors(explicit).first?["path"] as? String, "commands[2].entries[0]")
        XCTAssertTrue(((errors(explicit).first?["message"] as? String) ?? "").contains("minBindingSize 64"))

        let auto = harness.execute([
            ["op": "createShaderModule", "id": 10, "code": minimumSizeShader],
            ["op": "createComputePipeline", "id": 11, "layout": "auto", "compute": ["module": 10]],
            ["op": "getBindGroupLayout", "id": 12, "pipeline": 11, "index": 0],
            ["op": "createBindGroup", "id": 13, "layout": 12, "entries": [
                ["binding": 0, "resource": ["buffer": 1]],
                ["binding": 1, "resource": ["buffer": 1]],
            ]],
        ])
        XCTAssertEqual(errors(auto).count, 1, "\(errors(auto))")
        XCTAssertTrue(((errors(auto).first?["message"] as? String) ?? "").contains("minBindingSize 64"))
    }

    /// An explicit layout whose non-zero minBindingSize is below the shader's is rejected at pipeline creation.
    func test_aPipelineLayoutMinBindingSizeBelowTheShaderIsRejected() {
        let result = harness.execute([
            ["op": "createShaderModule", "id": 1, "code": minimumSizeShader],
            minimumSizeLayout(min0: 32),
            ["op": "createPipelineLayout", "id": 3, "bindGroupLayouts": [2]],
            ["op": "createComputePipeline", "id": 4, "layout": 3, "compute": ["module": 1]],
        ])
        XCTAssertEqual(errors(result).first?["path"] as? String, "commands[3].layout")
        XCTAssertTrue(((errors(result).first?["message"] as? String) ?? "").contains("minimum binding size 64"))
    }

    /// A `minBindingSize: 0` entry is checked at the dispatch, against the pipeline's shader — after
    /// the dynamic offset when the entry has no explicit size.
    func test_aZeroMinBindingSizeEntryIsCheckedAtDispatch() {
        let setup: [[String: Any]] = [
            ["op": "createShaderModule", "id": 1, "code": minimumSizeShader],
            minimumSizeLayout(min0: 0),
            ["op": "createPipelineLayout", "id": 3, "bindGroupLayouts": [2]],
            ["op": "createComputePipeline", "id": 4, "layout": 3, "compute": ["module": 1]],
            ["op": "createBuffer", "id": 5, "size": 256, "usage": TestUsage.uniform],
            ["op": "createBuffer", "id": 8, "size": 256, "usage": TestUsage.storage],
            ["op": "createBindGroup", "id": 6, "layout": 2, "entries": [
                ["binding": 0, "resource": ["buffer": 5, "size": 16]],
                ["binding": 1, "resource": ["buffer": 8]],
            ]],
            ["op": "createBindGroup", "id": 7, "layout": 2, "entries": [
                ["binding": 0, "resource": ["buffer": 5, "size": 64]],
                ["binding": 1, "resource": ["buffer": 8, "size": 16]],
            ]],
        ]
        func dispatch(_ group: Int) -> [[String: Any]] {
            [
                ["op": "beginComputePass"],
                ["op": "setPipeline", "pipeline": 4],
                ["op": "setBindGroup", "index": 0, "bindGroup": group],
                ["op": "dispatchWorkgroups", "x": 1],
                ["op": "endPass"],
            ]
        }
        let rejected = harness.execute(setup + dispatch(6))
        XCTAssertEqual(errors(rejected).count, 1, "\(errors(rejected))")
        XCTAssertEqual(errors(rejected).first?["path"] as? String, "commands[\(setup.count + 3)].dispatchWorkgroups")
        XCTAssertTrue(((errors(rejected).first?["message"] as? String) ?? "").contains("needs at least 64"))

        let accepted = harness.execute(dispatch(7))
        XCTAssertTrue(errors(accepted).isEmpty, "\(errors(accepted))")
    }

    // MARK: - Dynamic offsets

    /// setBindGroup checks its dynamic offsets — unchecked, Metal's setBuffer bound past the buffer's end.
    func test_setBindGroupRejectsBadDynamicOffsets() {
        let setup: [[String: Any]] = [
            ["op": "createShaderModule", "id": 1, "code": """
            @group(0) @binding(0) var<uniform> u: vec4f;
            @group(0) @binding(1) var<storage, read_write> o: array<vec4f>;
            @compute @workgroup_size(1) fn main() { o[0] = u; }
            """],
            ["op": "createBindGroupLayout", "id": 2, "entries": [
                ["binding": 0, "visibility": 4, "buffer": ["type": "uniform", "hasDynamicOffset": true]],
                ["binding": 1, "visibility": 4, "buffer": ["type": "storage"]],
            ]],
            ["op": "createPipelineLayout", "id": 3, "bindGroupLayouts": [2]],
            ["op": "createComputePipeline", "id": 4, "layout": 3, "compute": ["module": 1]],
            ["op": "createBuffer", "id": 5, "size": 256, "usage": TestUsage.uniform],
            ["op": "createBuffer", "id": 6, "size": 16, "usage": TestUsage.storage],
            ["op": "createBindGroup", "id": 7, "layout": 2, "entries": [
                ["binding": 0, "resource": ["buffer": 5, "size": 64]],
                ["binding": 1, "resource": ["buffer": 6]],
            ]],
        ]
        func dispatch(_ offsets: [Int]) -> [[String: Any]] {
            [
                ["op": "beginComputePass"],
                ["op": "setPipeline", "pipeline": 4],
                ["op": "setBindGroup", "index": 0, "bindGroup": 7, "dynamicOffsets": offsets],
                ["op": "dispatchWorkgroups", "x": 1],
                ["op": "endPass"],
            ]
        }
        XCTAssertTrue(errors(harness.execute(setup)).isEmpty)

        let pastTheEnd = harness.execute(dispatch([512]))
        XCTAssertEqual(errors(pastTheEnd).first?["path"] as? String, "commands[2].dynamicOffsets[0]")
        XCTAssertTrue(((errors(pastTheEnd).first?["message"] as? String) ?? "").contains("exceeds the buffer"))

        let unaligned = harness.execute(dispatch([4]))
        XCTAssertTrue(((errors(unaligned).first?["message"] as? String) ?? "").contains("minUniformBufferOffsetAlignment"))

        let missing = harness.execute(dispatch([]))
        XCTAssertEqual(errors(missing).first?["path"] as? String, "commands[2].dynamicOffsets")

        XCTAssertFalse(errors(harness.execute(dispatch([256]))).isEmpty, "256 + 64 bytes run past a 256-byte buffer")
        XCTAssertTrue(errors(harness.execute(dispatch([0]))).isEmpty)
    }

    /// The spec fixes each limit's **default (the minimum guarantee)**. Reporting lower makes code that
    /// works in a browser rejected only here, with the app unable to tell why.
    func test_limitsAreNotBelowTheSpecDefaults() throws {
        let limits = try XCTUnwrap(harness.runtime.adapterInfo()["limits"] as? [String: Any])

        let minimums: [String: Int] = [
            "maxTextureDimension1D": 8192, "maxTextureDimension2D": 8192,
            "maxTextureDimension3D": 2048, "maxTextureArrayLayers": 256,
            "maxBindGroups": 4, "maxBindingsPerBindGroup": 1000,
            "maxSampledTexturesPerShaderStage": 16, "maxSamplersPerShaderStage": 16,
            "maxUniformBufferBindingSize": 65536, "maxBufferSize": 268435456,
            "maxVertexBuffers": 8, "maxVertexAttributes": 16, "maxVertexBufferArrayStride": 2048,
            "maxColorAttachments": 8, "maxComputeInvocationsPerWorkgroup": 256,
            "maxComputeWorkgroupSizeX": 256, "maxComputeWorkgroupSizeY": 256,
            "maxComputeWorkgroupSizeZ": 64, "maxComputeWorkgroupsPerDimension": 65535,
        ]
        for (key, minimum) in minimums.sorted(by: { $0.key < $1.key }) {
            let value = (limits[key] as? Int) ?? 0
            XCTAssertGreaterThanOrEqual(value, minimum, "'\(key)' \(value) < the spec default \(minimum)")
        }

        // Alignment is **looser the smaller it is** — reporting larger than the spec default rejects an
        // offset that works in a browser. So this side is checked as an upper bound.
        XCTAssertLessThanOrEqual((limits["minUniformBufferOffsetAlignment"] as? Int) ?? 0, 256)
        XCTAssertLessThanOrEqual((limits["minStorageBufferOffsetAlignment"] as? Int) ?? 0, 256)
    }

    // MARK: - Texel copy arithmetic and regions

    /// `enforceRange` lets any 53-bit integer through, and the byte arithmetic multiplied two of them:
    /// a 2^52 stride × 4096 rows is 2^64 — an `Int` overflow **trap**, not an error. Every value the
    /// arithmetic touches is now bounded to 32 bits first, and the products are overflow-checked.
    func test_aHugeStrideInWriteTextureIsAValidationErrorNotATrap() {
        let result = harness.execute([
            ["op": "createTexture", "id": 1, "size": ["width": 2, "height": 2, "depthOrArrayLayers": 2],
             "format": "rgba8unorm", "usage": TestUsage.textureCopyDst],
            // Past 32 bits — rejected before anything is multiplied.
            ["op": "writeTexture", "texture": 1, "data": Data(count: 16).base64EncodedString(),
             "size": ["width": 1, "height": 1], "bytesPerRow": 1 << 52, "rowsPerImage": 4096],
            // Inside 32 bits each, but their product (bytesPerImage) is not — overflow-checked.
            ["op": "writeTexture", "texture": 1, "data": Data(count: 16).base64EncodedString(),
             "size": ["width": 1, "height": 1, "depthOrArrayLayers": 2],
             "bytesPerRow": Int(UInt32.max), "rowsPerImage": Int(UInt32.max)],
            ["op": "writeTexture", "texture": 1, "data": Data(count: 16).base64EncodedString(),
             "size": ["width": 1, "height": 1], "bytesPerRow": -4],
        ])
        XCTAssertEqual(result["ok"] as? Bool, false)
        XCTAssertEqual(errors(result).count, 3, harness.describeErrors(result))
        for error in errors(result) { XCTAssertEqual(error["kind"] as? String, "validation") }
    }

    /// `copyTextureToTexture` used to reach the blit encoder with **no** validation — a region past
    /// the texture, a negative origin, a mip level that does not exist or a slice beyond the layers
    /// are all Metal assertions. They are the spec's validation errors.
    func test_copyTextureToTextureOutsideTheTextureIsAValidationError() {
        let result = harness.execute([
            ["op": "createTexture", "id": 1, "size": ["width": 4, "height": 4], "format": "rgba8unorm",
             "usage": TestUsage.textureCopySrc | TestUsage.textureCopyDst],
            ["op": "createTexture", "id": 2, "size": ["width": 4, "height": 4], "format": "rgba8unorm",
             "usage": TestUsage.textureCopySrc | TestUsage.textureCopyDst],
            ["op": "createTexture", "id": 3, "size": ["width": 4, "height": 4], "format": "r8unorm",
             "usage": TestUsage.textureCopySrc | TestUsage.textureCopyDst],
            ["op": "copyTextureToTexture", "source": ["texture": 1, "origin": ["x": 3]],
             "destination": ["texture": 2], "copySize": ["width": 2, "height": 2]],
            ["op": "copyTextureToTexture", "source": ["texture": 1, "origin": ["x": -1]],
             "destination": ["texture": 2], "copySize": ["width": 1, "height": 1]],
            ["op": "copyTextureToTexture", "source": ["texture": 1, "mipLevel": 1],
             "destination": ["texture": 2], "copySize": ["width": 1, "height": 1]],
            ["op": "copyTextureToTexture", "source": ["texture": 1],
             "destination": ["texture": 2, "origin": ["z": 1]], "copySize": ["width": 1, "height": 1]],
            ["op": "copyTextureToTexture", "source": ["texture": 1],
             "destination": ["texture": 3], "copySize": ["width": 1, "height": 1]],
            // A valid copy between the two still runs.
            ["op": "copyTextureToTexture", "source": ["texture": 1, "origin": ["x": 2, "y": 2]],
             "destination": ["texture": 2], "copySize": ["width": 2, "height": 2]],
        ])
        XCTAssertEqual(errors(result).count, 5, harness.describeErrors(result))
        for error in errors(result) { XCTAssertEqual(error["kind"] as? String, "validation") }
    }

    /// The buffer↔texture copies checked only block alignment and the 256-byte stride. A region past
    /// the mip level, a mip level that does not exist, a buffer too small for the rows, or a negative
    /// offset went to Metal as they were.
    func test_bufferTextureCopiesPastTheTextureOrBufferAreValidationErrors() {
        let result = harness.execute([
            ["op": "createTexture", "id": 1, "size": ["width": 4, "height": 4], "format": "rgba8unorm",
             "usage": TestUsage.textureCopySrc | TestUsage.textureCopyDst],
            ["op": "createBuffer", "id": 2, "size": 64, "usage": TestUsage.copySrc | TestUsage.copyDst],
            ["op": "copyTextureToBuffer", "source": ["texture": 1, "origin": ["x": 3]],
             "destination": ["buffer": 2, "bytesPerRow": 256], "copySize": ["width": 2, "height": 1]],
            ["op": "copyBufferToTexture", "source": ["buffer": 2, "bytesPerRow": 256],
             "destination": ["texture": 1, "mipLevel": 1], "copySize": ["width": 1, "height": 1]],
            // 4 rows at a 256 B stride need 3 × 256 + 16 = 784 B; the buffer has 64.
            ["op": "copyBufferToTexture", "source": ["buffer": 2, "bytesPerRow": 256],
             "destination": ["texture": 1], "copySize": ["width": 4, "height": 4]],
            ["op": "copyTextureToBuffer", "source": ["texture": 1],
             "destination": ["buffer": 2, "offset": -4], "copySize": ["width": 1, "height": 1]],
            // A single texel at offset 60 fits exactly.
            ["op": "copyTextureToBuffer", "source": ["texture": 1],
             "destination": ["buffer": 2, "offset": 60], "copySize": ["width": 1, "height": 1]],
        ])
        XCTAssertEqual(errors(result).count, 4, harness.describeErrors(result))
        for error in errors(result) { XCTAssertEqual(error["kind"] as? String, "validation") }
    }

    // MARK: - Frame boundary with bookkeeping commands

    /// The tick's closing flush carries **whatever was recorded after the frame's last `submit()`** —
    /// a `destroy()` of a temporary, next frame's bind group. None of it makes GPU work, and the
    /// closing batch used to present only when it had *no* commands at all, so the frame the
    /// mid-frame submit had already drawn was never put out and the next acquisition threw it away.
    func test_aClosingBatchWithOnlyBookkeepingCommandsStillPresentsTheDrawnFrame() {
        harness.executeExpectingSuccess([
            ["op": "configureCanvas", "canvas": "test", "format": "rgba8unorm"],
        ])
        let before = harness.liveObjects

        let midFrame = harness.execute([
            ["op": "getCurrentTexture", "id": 60, "canvas": "test"],
            ["op": "createTextureView", "id": 61, "texture": 60],
            ["op": "beginRenderPass", "colorAttachments": [[
                "view": 61, "loadOp": "clear", "storeOp": "store",
                "clearValue": ["r": 0, "g": 1, "b": 0, "a": 1],
            ]]],
            ["op": "endPass"],
            ["op": "createBuffer", "id": 62, "size": 16, "usage": TestUsage.uniform],
        ], present: false)
        XCTAssertEqual(midFrame["ok"] as? Bool, true, harness.describeErrors(midFrame))
        XCTAssertEqual(harness.liveObjects, before + 3, "still mid-frame")

        // The end of the tick — one bookkeeping command, no GPU work of its own.
        let closing = harness.execute([["op": "destroy", "id": 62]], present: true)
        XCTAssertEqual(closing["ok"] as? Bool, true, harness.describeErrors(closing))
        XCTAssertEqual(
            harness.liveObjects, before,
            "the frame-scoped handles go only with a present — the drawn frame must have been put out"
        )
    }

    /// The other side of the rule: a present batch whose drawable was **never drawn on** (acquired,
    /// then a validation error before the first encoder) must not go out as a blank frame. It is
    /// held, and reclaimed by the canvas's next acquisition.
    func test_aClosingBatchDoesNotPresentAFrameThatWasNeverDrawn() {
        harness.executeExpectingSuccess([
            ["op": "configureCanvas", "canvas": "test", "format": "rgba8unorm"],
        ])
        let before = harness.liveObjects

        let result = harness.execute([
            ["op": "getCurrentTexture", "id": 60, "canvas": "test"],
            ["op": "createBuffer", "id": 62, "size": 16, "usage": TestUsage.uniform],
        ], present: true)
        XCTAssertEqual(result["ok"] as? Bool, true, harness.describeErrors(result))
        XCTAssertEqual(harness.liveObjects, before + 2, "the undrawn frame is held, not presented")

        // The next acquisition reclaims it.
        harness.executeExpectingSuccess([["op": "getCurrentTexture", "id": 63, "canvas": "test"]])
        XCTAssertEqual(harness.liveObjects, before + 2, "60 reclaimed, 63 acquired, 62 still alive")
    }
}
