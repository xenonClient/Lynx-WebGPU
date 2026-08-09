import XCTest
import Metal
import LynxWebGPUCore
@testable import LynxWebGPU

/// Render bundles — the contract is **"the same result as direct encoding"**, so equivalence is the verification.
///
/// This implementation has no Metal counterpart object and replays a stored command list instead. That
/// narrows the room for "a bundle behaving differently", leaving **state isolation and reuse** as the
/// places most likely to go wrong.
final class RenderBundleTests: XCTestCase {
    private var harness: RenderHarness!

    override func setUpWithError() throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "no Metal device")
        harness = try XCTUnwrap(RenderHarness.make())
    }

    override func tearDown() {
        harness = nil
        super.tearDown()
    }

    /// Covers the whole screen with a uniform color.
    private static let shader = """
    struct Tint { color: vec4f };
    @group(0) @binding(0) var<uniform> tint: Tint;

    @vertex
    fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
        var corners = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
        return vec4f(corners[index], 0.0, 1.0);
    }

    @fragment
    fn fs_main() -> @location(0) vec4f {
        return tint.color;
    }
    """

    private let red = (r: 255, g: 0, b: 0, a: 255)
    private let green = (r: 0, g: 255, b: 0, a: 255)

    private func errors(_ result: [String: Any]) -> [[String: Any]] {
        result["errors"] as? [[String: Any]] ?? []
    }

    /// One pipeline plus two bind groups, red and green (handles 6 and 7).
    private func setUpResources() -> [[String: Any]] {
        [
            ["op": "configureCanvas", "canvas": "test", "format": "rgba8unorm"],
            ["op": "createShaderModule", "id": 1, "code": Self.shader],
            ["op": "createRenderPipeline", "id": 2, "layout": "auto",
             "vertex": ["module": 1, "entryPoint": "vs_main"],
             "fragment": ["module": 1, "entryPoint": "fs_main",
                          "targets": [["format": "rgba8unorm"]]]],
            ["op": "createBuffer", "id": 3, "size": 16, "usage": TestUsage.uniform,
             "data": [Float]([1, 0, 0, 1]).base64],
            ["op": "createBuffer", "id": 4, "size": 16, "usage": TestUsage.uniform,
             "data": [Float]([0, 1, 0, 1]).base64],
            ["op": "getBindGroupLayout", "id": 5, "pipeline": 2, "index": 0],
            ["op": "createBindGroup", "id": 6, "layout": 5,
             "entries": [["binding": 0, "resource": ["buffer": 3]]]],
            ["op": "createBindGroup", "id": 7, "layout": 5,
             "entries": [["binding": 0, "resource": ["buffer": 4]]]],
        ]
    }

    private let acquireDrawable: [[String: Any]] = [
        ["op": "getCurrentTexture", "id": 20, "canvas": "test"],
        ["op": "createTextureView", "id": 21, "texture": 20],
    ]

    private let beginPass: [String: Any] = [
        "op": "beginRenderPass",
        "colorAttachments": [[
            "view": 21, "loadOp": "clear", "storeOp": "store",
            "clearValue": ["r": 0, "g": 0, "b": 1, "a": 1],
        ]],
    ]

    /// The three draw lines covering the screen in the `bindGroup` color — the body direct encoding and the bundle share.
    private func fullScreenDraw(bindGroup: Int) -> [[String: Any]] {
        [
            ["op": "setPipeline", "pipeline": 2],
            ["op": "setBindGroup", "index": 0, "bindGroup": bindGroup],
            ["op": "draw", "vertexCount": 3],
        ]
    }

    private func createBundle(id: Int, commands: [[String: Any]]) -> [String: Any] {
        ["op": "createRenderBundle", "id": id, "colorFormats": ["rgba8unorm"], "commands": commands]
    }

    // MARK: - Equivalence

    func test_aBundleProducesTheSameFrameAsDirectEncoding() throws {
        harness.executeExpectingSuccess(setUpResources() + [
            createBundle(id: 10, commands: fullScreenDraw(bindGroup: 6)),
        ])

        harness.executeExpectingSuccess(acquireDrawable + [beginPass]
            + fullScreenDraw(bindGroup: 6) + [["op": "endPass"]])
        try harness.assertPixel(x: 32, y: 32, equals: red, "whether direct encoding really drew")
        let direct = try harness.frameBytes()

        harness.executeExpectingSuccess(acquireDrawable + [
            beginPass,
            ["op": "executeBundles", "bundles": [10]],
            ["op": "endPass"],
        ])

        try harness.assertFrameEquals(direct, "a bundle's contract is the same result as direct encoding")
    }

    /// Reuse is why bundles exist — running one must not be a data structure that spoils.
    func test_runningOneBundleTwoFramesInARowGivesTheSameResult() throws {
        harness.executeExpectingSuccess(setUpResources() + [
            createBundle(id: 10, commands: fullScreenDraw(bindGroup: 6)),
        ])

        let frame: [[String: Any]] = acquireDrawable + [
            beginPass,
            ["op": "executeBundles", "bundles": [10]],
            ["op": "endPass"],
        ]

        harness.executeExpectingSuccess(frame)
        try harness.assertPixel(x: 32, y: 32, equals: red)
        let first = try harness.frameBytes()

        harness.executeExpectingSuccess(frame)
        try harness.assertFrameEquals(first, "the second run must match too")
    }

    func test_severalBundlesRunInTheOrderGiven() throws {
        harness.executeExpectingSuccess(setUpResources() + [
            createBundle(id: 10, commands: fullScreenDraw(bindGroup: 6)),   // red
            createBundle(id: 11, commands: fullScreenDraw(bindGroup: 7)),   // green
        ])

        harness.executeExpectingSuccess(acquireDrawable + [
            beginPass,
            ["op": "executeBundles", "bundles": [10, 11]],
            ["op": "endPass"],
        ])
        try harness.assertPixel(x: 32, y: 32, equals: green, "the later bundle draws on top")

        harness.executeExpectingSuccess(acquireDrawable + [
            beginPass,
            ["op": "executeBundles", "bundles": [11, 10]],
            ["op": "endPass"],
        ])
        try harness.assertPixel(x: 32, y: 32, equals: red, "reversing the order reverses the result")
    }

    /// Compatibility validation must finish before execution and **over the whole list** — one
    /// incompatible bundle stops even the compatible ones before it. The contract exists to avoid
    /// leaving a half-drawn frame, and merging the validation and execution loops breaks it quietly,
    /// so it is pinned by pixels.
    func test_anIncompatibleBundleStopsTheCompatibleOnesBeforeItToo() throws {
        let setUp = setUpResources() + [
            createBundle(id: 10, commands: fullScreenDraw(bindGroup: 6)),   // compatible (red)
            ["op": "createRenderBundle", "id": 12, "colorFormats": ["bgra8unorm"],
             "commands": fullScreenDraw(bindGroup: 7)],                     // format mismatch with the pass
        ]
        let result = harness.execute(setUp + acquireDrawable + [
            beginPass,
            ["op": "executeBundles", "bundles": [10, 12]],
            ["op": "endPass"],
        ])

        XCTAssertEqual(errors(result).first?["kind"] as? String, "validation")
        // The compatible bundle 10 (red) must not draw either — the center must stay the clear color.
        try harness.assertPixel(x: 32, y: 32, equals: (0, 0, 255, 255), "the earlier compatible bundle ran")
    }

    /// The spec's `GPURenderBundleEncoder` includes `GPUDebugCommandsMixin` — markers can go in a bundle.
    ///
    /// Omitting them from the allow-list **rejects the whole bundle over one marker**, and the user is
    /// unlikely to guess the marker was the cause (they suspect the draw and fix the wrong thing).
    func test_aBundleCanContainDebugMarkers() throws {
        harness.executeExpectingSuccess(setUpResources() + [
            createBundle(id: 10, commands:
                [["op": "pushDebugGroup", "groupLabel": "bundle section"]]
                + fullScreenDraw(bindGroup: 6)
                + [["op": "insertDebugMarker", "markerLabel": "after the draw"],
                   ["op": "popDebugGroup"]]
            ),
        ])

        harness.executeExpectingSuccess(acquireDrawable + [
            beginPass,
            ["op": "executeBundles", "bundles": [10]],
            ["op": "endPass"],
        ])
        // Markers do not change the picture — the bundle must draw as usual.
        try harness.assertPixel(x: 32, y: 32, equals: red, "it must draw even with markers mixed in")
    }

    // MARK: - State isolation

    /// The spec states that executing a bundle **invalidates** pass state rather than **restoring** it.
    /// So drawing afterwards must start again from `setPipeline` — otherwise it draws with the bindings
    /// the bundle left, and code a browser rejects runs only here.
    func test_afterBundleExecutionThePipelineMustBeSetAgain() {
        let result = harness.execute(setUpResources() + [
            createBundle(id: 10, commands: fullScreenDraw(bindGroup: 6)),
        ] + acquireDrawable + [
            beginPass,
            ["op": "executeBundles", "bundles": [10]],
            ["op": "draw", "vertexCount": 3],   // with no setPipeline
            ["op": "endPass"],
        ])

        XCTAssertTrue(
            ((errors(result).first?["message"] as? String) ?? "").contains("setPipeline"),
            "it must not inherit the state the bundle left: \(harness.describeErrors(result))"
        )
    }

    /// The other direction — a bundle does not inherit a pipeline the pass already set.
    func test_aBundleDoesNotInheritThePassPipeline() {
        let result = harness.execute(setUpResources() + [
            // A bundle holding only a draw, with no setPipeline.
            createBundle(id: 10, commands: [
                ["op": "setBindGroup", "index": 0, "bindGroup": 6],
                ["op": "draw", "vertexCount": 3],
            ]),
        ] + acquireDrawable + [
            beginPass,
            ["op": "setPipeline", "pipeline": 2],   // set in advance on the pass side
            ["op": "executeBundles", "bundles": [10]],
            ["op": "endPass"],
        ])

        XCTAssertTrue(
            ((errors(result).first?["message"] as? String) ?? "").contains("setPipeline"),
            "it must be set again inside the bundle: \(harness.describeErrors(result))"
        )
    }

    /// The other direction — a bundle does not inherit bind groups the pass already set either.
    ///
    /// Unlike the pipeline, bind groups **remain on the Metal encoder**, so clearing the shadow state
    /// alone does not isolate them. Catching it requires checking that every group the layout requires is bound.
    func test_aBundleDoesNotInheritThePassBindGroups() {
        let result = harness.execute(setUpResources() + [
            createBundle(id: 10, commands: [
                ["op": "setPipeline", "pipeline": 2],   // with no setBindGroup
                ["op": "draw", "vertexCount": 3],
            ]),
        ] + acquireDrawable + [
            beginPass,
            ["op": "setPipeline", "pipeline": 2],
            ["op": "setBindGroup", "index": 0, "bindGroup": 6],   // set in advance on the pass side
            ["op": "executeBundles", "bundles": [10]],
            ["op": "endPass"],
        ])

        XCTAssertTrue(
            ((errors(result).first?["message"] as? String) ?? "").contains("@group(0)"),
            "it must be bound again inside the bundle: \(harness.describeErrors(result))"
        )
    }

    /// Setting only `setPipeline` again after a bundle and drawing would use the bind group the bundle left.
    func test_afterBundleExecutionBindGroupsMustBeSetAgain() {
        let result = harness.execute(setUpResources() + [
            createBundle(id: 10, commands: fullScreenDraw(bindGroup: 6)),
        ] + acquireDrawable + [
            beginPass,
            ["op": "executeBundles", "bundles": [10]],
            ["op": "setPipeline", "pipeline": 2],   // with no setBindGroup
            ["op": "draw", "vertexCount": 3],
            ["op": "endPass"],
        ])

        XCTAssertTrue(
            ((errors(result).first?["message"] as? String) ?? "").contains("@group(0)"),
            harness.describeErrors(result)
        )
    }

    // MARK: - State isolation (vertex buffers)

    /// A pipeline that really reads a vertex buffer — seeing isolation needs a side that requires the slot.
    private static let vertexPullingShader = """
    @vertex
    fn vs_main(@location(0) position: vec2f) -> @builtin(position) vec4f {
        return vec4f(position, 0.0, 1.0);
    }

    @fragment
    fn fs_main() -> @location(0) vec4f {
        return vec4f(1.0, 0.0, 0.0, 1.0);
    }
    """

    /// Shader (40), pipeline (41) and vertex buffer (42).
    private func setUpVertexPulling() -> [[String: Any]] {
        [
            ["op": "configureCanvas", "canvas": "test", "format": "rgba8unorm"],
            ["op": "createShaderModule", "id": 40, "code": Self.vertexPullingShader],
            ["op": "createRenderPipeline", "id": 41, "layout": "auto",
             "vertex": ["module": 40, "entryPoint": "vs_main",
                        "buffers": [[
                            "arrayStride": 8,
                            "attributes": [["shaderLocation": 0, "offset": 0, "format": "float32x2"]],
                        ]]],
             "fragment": ["module": 40, "entryPoint": "fs_main",
                          "targets": [["format": "rgba8unorm"]]]],
            ["op": "createBuffer", "id": 42, "size": 24, "usage": TestUsage.vertex,
             "data": [Float]([-1, -1, 3, -1, -1, 3]).base64],
        ]
    }

    private func vertexBundle(id: Int, commands: [[String: Any]]) -> [String: Any] {
        ["op": "createRenderBundle", "id": id, "colorFormats": ["rgba8unorm"], "commands": commands]
    }

    /// The baseline — a bundle carrying its own vertex buffer draws normally.
    func test_aBundleCarryingItsVertexBufferDrawsNormally() throws {
        harness.executeExpectingSuccess(setUpVertexPulling() + [
            vertexBundle(id: 43, commands: [
                ["op": "setPipeline", "pipeline": 41],
                ["op": "setVertexBuffer", "slot": 0, "buffer": 42],
                ["op": "draw", "vertexCount": 3],
            ]),
        ] + acquireDrawable + [
            beginPass,
            ["op": "executeBundles", "bundles": [43]],
            ["op": "endPass"],
        ])

        try harness.assertPixel(x: 32, y: 32, equals: red)
    }

    /// The case where only the pass set the vertex buffer — the bundle does not inherit it, so it must be rejected.
    func test_aBundleDoesNotInheritThePassVertexBuffers() {
        let result = harness.execute(setUpVertexPulling() + [
            vertexBundle(id: 43, commands: [
                ["op": "setPipeline", "pipeline": 41],   // with no setVertexBuffer
                ["op": "draw", "vertexCount": 3],
            ]),
        ] + acquireDrawable + [
            beginPass,
            ["op": "setVertexBuffer", "slot": 0, "buffer": 42],   // set in advance on the pass side
            ["op": "executeBundles", "bundles": [43]],
            ["op": "endPass"],
        ])

        XCTAssertTrue(
            ((errors(result).first?["message"] as? String) ?? "").contains("vertex buffer slot 0"),
            harness.describeErrors(result)
        )
    }

    /// The other direction — a vertex buffer the bundle set is invalidated once it finishes.
    func test_afterBundleExecutionVertexBuffersMustBeSetAgain() {
        let result = harness.execute(setUpVertexPulling() + [
            vertexBundle(id: 43, commands: [
                ["op": "setPipeline", "pipeline": 41],
                ["op": "setVertexBuffer", "slot": 0, "buffer": 42],
                ["op": "draw", "vertexCount": 3],
            ]),
        ] + acquireDrawable + [
            beginPass,
            ["op": "executeBundles", "bundles": [43]],
            ["op": "setPipeline", "pipeline": 41],   // with no setVertexBuffer
            ["op": "draw", "vertexCount": 3],
            ["op": "endPass"],
        ])

        XCTAssertTrue(
            ((errors(result).first?["message"] as? String) ?? "").contains("vertex buffer slot 0"),
            harness.describeErrors(result)
        )
    }

    /// The index buffer is invalidated at bundle boundaries too — the place the docs record as "pipeline,
    /// bind groups, vertex buffers and **the index buffer**, four things". Left behind, a following
    /// `drawIndexed` **really draws** with the index the bundle left and diverges from a browser.
    func test_afterBundleExecutionTheIndexBufferMustBeSetAgain() {
        let result = harness.execute(setUpVertexPulling() + [
            ["op": "createBuffer", "id": 44, "size": 6, "usage": TestUsage.index,
             "data": Data([0, 0, 1, 0, 2, 0]).base64EncodedString()],
            vertexBundle(id: 43, commands: [
                ["op": "setPipeline", "pipeline": 41],
                ["op": "setVertexBuffer", "slot": 0, "buffer": 42],
                ["op": "setIndexBuffer", "buffer": 44, "format": "uint16"],
                ["op": "drawIndexed", "indexCount": 3],
            ]),
        ] + acquireDrawable + [
            beginPass,
            ["op": "executeBundles", "bundles": [43]],
            ["op": "setPipeline", "pipeline": 41],
            ["op": "setVertexBuffer", "slot": 0, "buffer": 42],
            ["op": "drawIndexed", "indexCount": 3],   // with no setIndexBuffer
            ["op": "endPass"],
        ])

        XCTAssertTrue(
            ((errors(result).first?["message"] as? String) ?? "").contains("setIndexBuffer is required"),
            harness.describeErrors(result)
        )
    }

    /// The other direction — a bundle does not inherit an index buffer bound by the pass.
    func test_aBundleDoesNotInheritThePassIndexBuffer() {
        let result = harness.execute(setUpVertexPulling() + [
            ["op": "createBuffer", "id": 44, "size": 6, "usage": TestUsage.index,
             "data": Data([0, 0, 1, 0, 2, 0]).base64EncodedString()],
            vertexBundle(id: 43, commands: [
                ["op": "setPipeline", "pipeline": 41],
                ["op": "setVertexBuffer", "slot": 0, "buffer": 42],
                ["op": "drawIndexed", "indexCount": 3],   // never bound inside the bundle
            ]),
        ] + acquireDrawable + [
            beginPass,
            ["op": "setIndexBuffer", "buffer": 44, "format": "uint16"],
            ["op": "executeBundles", "bundles": [43]],
            ["op": "endPass"],
        ])

        XCTAssertTrue(
            ((errors(result).first?["message"] as? String) ?? "").contains("setIndexBuffer is required"),
            harness.describeErrors(result)
        )
    }

    /// Regression — resetting the pass bindings must not be skipped even when a bundle command fails.
    ///
    /// The interpreter's contract is "errors accumulate without killing the frame", so execution continues
    /// from the next command. Omitting the reset there makes the following draw **really draw** with the
    /// pipeline the bundle left — code a browser rejects produces stray pixels only here.
    func test_passBindingsResetEvenWhenABundleErrorsMidExecution() {
        let result = harness.execute(setUpResources() + [
            // The second command's bind group handle does not exist — it fails mid-execution.
            createBundle(id: 10, commands: [
                ["op": "setPipeline", "pipeline": 2],
                ["op": "setBindGroup", "index": 0, "bindGroup": 999],
            ]),
        ] + acquireDrawable + [
            beginPass,
            ["op": "executeBundles", "bundles": [10]],
            ["op": "draw", "vertexCount": 3],   // with no setPipeline
            ["op": "endPass"],
        ])

        let messages = errors(result).compactMap { $0["message"] as? String }
        XCTAssertEqual(
            messages.count, 2,
            "it must be 1 bundle failure plus 1 rejected draw: \(harness.describeErrors(result))"
        )
        XCTAssertTrue(
            (messages.last ?? "").contains("setPipeline"),
            "it must not draw holding the pipeline the bundle left: \(harness.describeErrors(result))"
        )
    }

    /// Regression — deriving the pass `sampleCount` from color attachments alone leaves it at 1 in an
    /// MSAA pass with depth but no color (a shadow map or depth prepass). A bundle correctly declaring
    /// `sampleCount: 4` per spec is then falsely rejected.
    func test_inAColorlessMSAADepthPassSampleCountComesFromTheDepthView() throws {
        try XCTSkipUnless(harness.context!.device.supportsTextureSampleCount(4), "4x MSAA unsupported")

        let result = harness.execute([
            ["op": "createTexture", "id": 2, "size": ["width": 64, "height": 64],
             "format": "depth32float", "sampleCount": 4, "usage": TestUsage.renderAttachment],
            ["op": "createTextureView", "id": 3, "texture": 2],
            ["op": "createRenderBundle", "id": 10, "colorFormats": [String?](),
             "depthStencilFormat": "depth32float", "sampleCount": 4, "commands": [[String: Any]]()],
            ["op": "beginRenderPass",
             "colorAttachments": [[String: Any]](),
             "depthStencilAttachment": [
                "view": 3, "depthClearValue": 1.0, "depthLoadOp": "clear", "depthStoreOp": "store",
             ]],
            ["op": "executeBundles", "bundles": [10]],
            ["op": "endPass"],
        ])

        XCTAssertEqual(result["ok"] as? Bool, true, harness.describeErrors(result))
    }

    // MARK: - Contract

    func test_rejectsABundleWhoseColorFormatDiffersFromThePass() {
        let result = harness.execute(setUpResources() + [
            ["op": "createRenderBundle", "id": 10, "colorFormats": ["bgra8unorm"],
             "commands": fullScreenDraw(bindGroup: 6)],
        ] + acquireDrawable + [
            beginPass,   // the pass is rgba8unorm
            ["op": "executeBundles", "bundles": [10]],
            ["op": "endPass"],
        ])

        XCTAssertEqual(errors(result).first?["kind"] as? String, "validation")
        let message = (errors(result).first?["message"] as? String) ?? ""
        XCTAssertTrue(message.contains("colorFormats"), message)
        XCTAssertTrue(message.contains("bgra8unorm") && message.contains("rgba8unorm"), message)
    }

    func test_rejectsABundleWhoseAttachmentCountDiffersFromThePass() {
        let result = harness.execute(setUpResources() + [
            ["op": "createRenderBundle", "id": 10, "colorFormats": ["rgba8unorm", "rgba8unorm"],
             "commands": fullScreenDraw(bindGroup: 6)],
        ] + acquireDrawable + [
            beginPass,
            ["op": "executeBundles", "bundles": [10]],
            ["op": "endPass"],
        ])

        XCTAssertTrue(
            ((errors(result).first?["message"] as? String) ?? "").contains("attachment count"),
            harness.describeErrors(result)
        )
    }

    /// The spec's layout equality **ignores trailing nulls.** Without trimming, a valid combination is rejected.
    func test_trailingNullsInBundleColorFormatsAreIgnored() {
        let result = harness.execute(setUpResources() + [
            ["op": "createRenderBundle", "id": 10,
             // A `null` sent by JS arrives as NSNull across the bridge.
             "colorFormats": ["rgba8unorm", NSNull()] as [Any],
             "commands": fullScreenDraw(bindGroup: 6)],
        ] + acquireDrawable + [
            beginPass,   // one color attachment
            ["op": "executeBundles", "bundles": [10]],
            ["op": "endPass"],
        ])

        XCTAssertEqual(result["ok"] as? Bool, true, harness.describeErrors(result))
    }

    /// A bundle with no attachment at all is rejected **at creation**.
    /// It would eventually fail in `makeRenderCommandEncoder` anyway, but then the error lands somewhere unrelated.
    func test_aBundleWithNoAttachmentIsRejectedAtCreation() {
        let result = harness.execute([
            ["op": "createRenderBundle", "id": 10, "colorFormats": [String?](),
             "commands": [[String: Any]]()],
        ])

        XCTAssertEqual(errors(result).first?["kind"] as? String, "validation")
        XCTAssertTrue(
            ((errors(result).first?["message"] as? String) ?? "").contains("attachment"),
            harness.describeErrors(result)
        )
        XCTAssertEqual(harness.liveObjects, 0, "a rejected bundle must not be registered")
    }

    func test_rejectsABundleWhoseDepthFormatDiffersFromThePass() {
        let result = harness.execute(setUpResources() + [
            ["op": "createRenderBundle", "id": 10, "colorFormats": ["rgba8unorm"],
             "depthStencilFormat": "depth32float", "commands": fullScreenDraw(bindGroup: 6)],
        ] + acquireDrawable + [
            beginPass,   // there is no depth attachment
            ["op": "executeBundles", "bundles": [10]],
            ["op": "endPass"],
        ])

        XCTAssertTrue(
            ((errors(result).first?["message"] as? String) ?? "").contains("depthStencilFormat"),
            harness.describeErrors(result)
        )
    }

    /// The commands the spec forbids in a bundle. They must be rejected **when the bundle is created** —
    /// deferring to execution time surfaces them only frames later.
    func test_aForbiddenCommandInsideABundleIsRejectedAtCreation() {
        for forbidden in [
            ["op": "setViewport", "x": 0, "y": 0, "width": 8, "height": 8],
            ["op": "setScissorRect", "x": 0, "y": 0, "width": 8, "height": 8],
            ["op": "setStencilReference", "reference": 1],
            ["op": "endPass"],
            ["op": "executeBundles", "bundles": [99]],
            ["op": "copyBufferToBuffer", "source": 1, "destination": 2, "size": 4],
            // Occlusion queries belong to the render **pass** — the spec's bundle encoder has none at all.
            ["op": "beginOcclusionQuery", "queryIndex": 0],
            ["op": "endOcclusionQuery"],
            // The same goes for compute and write commands.
            ["op": "dispatchWorkgroups", "x": 1],
            ["op": "writeBuffer", "buffer": 1, "data": [0, 0, 0, 0]],
            ["op": "clearBuffer", "buffer": 1],
        ] as [[String: Any]] {
            let result = harness.execute([createBundle(id: 10, commands: [forbidden])])
            XCTAssertEqual(
                errors(result).first?["kind"] as? String, "validation",
                "'\(forbidden["op"] ?? "?")' must not be allowed in a bundle"
            )
            XCTAssertEqual(harness.liveObjects, 0, "a rejected bundle must not be registered")
        }
    }

    /// A read-only pass takes only bundles that declared "I do not write either".
    /// The reverse (a read-only bundle in a writable pass) is fine, so only one direction is checked.
    func test_aDepthReadOnlyPassOnlyExecutesReadOnlyBundles() {
        let depthTarget: [[String: Any]] = [
            ["op": "createTexture", "id": 50, "size": ["width": 64, "height": 64],
             "format": "depth32float", "usage": TestUsage.renderAttachment],
            ["op": "createTextureView", "id": 51, "texture": 50],
        ]
        let readOnlyPass: [String: Any] = [
            "op": "beginRenderPass",
            "colorAttachments": [[
                "view": 21, "loadOp": "clear", "storeOp": "store",
                "clearValue": ["r": 0, "g": 0, "b": 1, "a": 1],
            ]],
            "depthStencilAttachment": ["view": 51, "depthReadOnly": true],
        ]
        func bundle(id: Int, readOnly: Bool) -> [String: Any] {
            var command: [String: Any] = [
                "op": "createRenderBundle", "id": id, "colorFormats": ["rgba8unorm"],
                "depthStencilFormat": "depth32float", "commands": [[String: Any]](),
            ]
            if readOnly { command["depthReadOnly"] = true }
            return command
        }

        let rejected = harness.execute(setUpResources() + depthTarget + [
            bundle(id: 52, readOnly: false),
        ] + acquireDrawable + [
            readOnlyPass,
            ["op": "executeBundles", "bundles": [52]],
            ["op": "endPass"],
        ])
        XCTAssertTrue(
            ((errors(rejected).first?["message"] as? String) ?? "").contains("depthReadOnly"),
            harness.describeErrors(rejected)
        )

        harness.executeExpectingSuccess(setUpResources() + depthTarget + [
            bundle(id: 53, readOnly: true),
        ] + acquireDrawable + [
            readOnlyPass,
            ["op": "executeBundles", "bundles": [53]],
            ["op": "endPass"],
        ])
    }

    func test_executeBundlesWithNoPassIsAnError() {
        let result = harness.execute(setUpResources() + [
            createBundle(id: 10, commands: fullScreenDraw(bindGroup: 6)),
            ["op": "executeBundles", "bundles": [10]],
        ])

        XCTAssertTrue(
            ((errors(result).first?["message"] as? String) ?? "").contains("beginRenderPass"),
            harness.describeErrors(result)
        )
    }

    func test_aMissingBundleHandleIsAValidationError() {
        let result = harness.execute(setUpResources() + acquireDrawable + [
            beginPass,
            ["op": "executeBundles", "bundles": [999]],
            ["op": "endPass"],
        ])

        XCTAssertEqual(errors(result).first?["kind"] as? String, "validation")
        XCTAssertTrue(((errors(result).first?["message"] as? String) ?? "").contains("GPURenderBundle"))
    }
}
