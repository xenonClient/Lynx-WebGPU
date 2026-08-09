import XCTest
import Metal
import LynxWebGPUCore
@testable import LynxWebGPU

/// Stencil testing — asserted **indirectly through color**.
///
/// `readPixels` cannot read a depth/stencil surface (`docs/TESTING.md` §4-1). But what stencil decides
/// is "which fragment survives", so the **color of the surviving fragment is itself the evidence** of
/// stencil behaviour. Not being able to read the stencil buffer directly is no limit on verification.
final class StencilTests: XCTestCase {
    private var harness: RenderHarness!

    override func setUpWithError() throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "no Metal device")
        harness = try XCTUnwrap(RenderHarness.make())
    }

    override func tearDown() {
        harness = nil
        super.tearDown()
    }

    /// One screen-covering triangle. Color and depth come from a uniform, reducing the need for many pipelines.
    private static let shader = """
    struct Fill {
        color: vec4f,
        depth: f32,
    };
    @group(0) @binding(0) var<uniform> fill: Fill;

    @vertex
    fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
        var corners = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
        return vec4f(corners[index], fill.depth, 1.0);
    }

    @fragment
    fn fs_main() -> @location(0) vec4f {
        return fill.color;
    }
    """

    /// One set of `Fill` uniforms — vec4f(16B) + f32, which is 32B under the alignment rules.
    private func fill(red: Float, green: Float, blue: Float, depth: Float = 0) -> String {
        [red, green, blue, 1, depth, 0, 0, 0].base64
    }

    private let red = (r: 255, g: 0, b: 0, a: 255)
    private let clearBlue = (r: 0, g: 0, b: 255, a: 255)

    private func errors(_ result: [String: Any]) -> [[String: Any]] {
        result["errors"] as? [[String: Any]] ?? []
    }

    /// Uploads two color uniforms (red/green) and the shader. Pipelines differ per test, so they are not here.
    private func makeCommonResources() -> [[String: Any]] {
        [
            ["op": "configureCanvas", "canvas": "test", "format": "rgba8unorm"],
            ["op": "createShaderModule", "id": 1, "code": Self.shader],
            ["op": "createBuffer", "id": 4, "size": 32, "usage": TestUsage.uniform,
             "data": fill(red: 1, green: 0, blue: 0)],
            ["op": "createBuffer", "id": 5, "size": 32, "usage": TestUsage.uniform,
             "data": fill(red: 0, green: 1, blue: 0)],
        ]
    }

    /// One pipeline differing only in stencil state. With `colorWriteMask: 0` it writes stencil only and leaves color.
    private func pipeline(
        id: Int,
        format: String = "stencil8",
        stencil: [String: Any],
        writesColor: Bool = true,
        depthCompare: String = "always",
        depthWrite: Bool = false
    ) -> [String: Any] {
        var target: [String: Any] = ["format": "rgba8unorm"]
        if !writesColor { target["writeMask"] = 0 }
        var depthStencil: [String: Any] = [
            "format": format, "depthCompare": depthCompare, "depthWriteEnabled": depthWrite,
        ]
        depthStencil.merge(stencil) { _, new in new }
        return [
            "op": "createRenderPipeline", "id": id, "layout": "auto",
            "vertex": ["module": 1, "entryPoint": "vs_main"],
            "fragment": ["module": 1, "entryPoint": "fs_main", "targets": [target]],
            "depthStencil": depthStencil,
        ]
    }

    /// Puts the same state on the front and back faces (the test geometry faces one way, so splitting is pointless).
    private func bothFaces(_ state: [String: Any]) -> [String: Any] {
        ["stencilFront": state, "stencilBack": state]
    }

    private func beginPass(stencilView: Int) -> [String: Any] {
        [
            "op": "beginRenderPass",
            "colorAttachments": [[
                "view": 21, "loadOp": "clear", "storeOp": "store",
                "clearValue": ["r": 0, "g": 0, "b": 1, "a": 1],
            ]],
            "depthStencilAttachment": [
                "view": stencilView,
                "depthClearValue": 1.0, "depthLoadOp": "clear", "depthStoreOp": "store",
                "stencilClearValue": 0, "stencilLoadOp": "clear", "stencilStoreOp": "store",
            ],
        ]
    }

    private let acquireDrawable: [[String: Any]] = [
        ["op": "getCurrentTexture", "id": 20, "canvas": "test"],
        ["op": "createTextureView", "id": 21, "texture": 20],
    ]

    /// A depth/stencil pass opened read-only. The `readOnly` side cannot be given load/store ops.
    private func readOnlyPass(
        view: Int, depthReadOnly: Bool = false, stencilReadOnly: Bool = false
    ) -> [String: Any] {
        var attachment: [String: Any] = ["view": view]
        if depthReadOnly {
            attachment["depthReadOnly"] = true
        } else {
            attachment["depthClearValue"] = 1.0
            attachment["depthLoadOp"] = "clear"
            attachment["depthStoreOp"] = "store"
        }
        if stencilReadOnly {
            attachment["stencilReadOnly"] = true
        } else {
            attachment["stencilClearValue"] = 0
            attachment["stencilLoadOp"] = "clear"
            attachment["stencilStoreOp"] = "store"
        }
        return [
            "op": "beginRenderPass",
            "colorAttachments": [[
                "view": 21, "loadOp": "clear", "storeOp": "store",
                "clearValue": ["r": 0, "g": 0, "b": 1, "a": 1],
            ]],
            "depthStencilAttachment": attachment,
        ]
    }

    // MARK: - Masking

    func test_theStencilMaskClipsTheFollowingDraw() throws {
        harness.executeExpectingSuccess(makeCommonResources() + [
            ["op": "createTexture", "id": 2, "size": ["width": 64, "height": 64],
             "format": "stencil8", "usage": TestUsage.renderAttachment],
            ["op": "createTextureView", "id": 3, "texture": 2],
            // Marking — sets the stencil to ref where it drew, leaving color alone.
            pipeline(id: 6, stencil: bothFaces(["compare": "always", "passOp": "replace"]),
                     writesColor: false),
            // Painting — only where the stencil equals ref.
            pipeline(id: 7, stencil: bothFaces(["compare": "equal"])),
            ["op": "getBindGroupLayout", "id": 8, "pipeline": 6, "index": 0],
            ["op": "createBindGroup", "id": 9, "layout": 8,
             "entries": [["binding": 0, "resource": ["buffer": 4]]]],
            ["op": "createBindGroup", "id": 10, "layout": 8,
             "entries": [["binding": 0, "resource": ["buffer": 5]]]],
        ] + acquireDrawable + [
            beginPass(stencilView: 3),
            ["op": "setStencilReference", "reference": 1],
            ["op": "setPipeline", "pipeline": 6],
            ["op": "setBindGroup", "index": 0, "bindGroup": 10],   // green — writeMask 0, so it must not show
            ["op": "setScissorRect", "x": 0, "y": 0, "width": 32, "height": 64],
            ["op": "draw", "vertexCount": 3],
            ["op": "setScissorRect", "x": 0, "y": 0, "width": 64, "height": 64],
            ["op": "setPipeline", "pipeline": 7],
            ["op": "setBindGroup", "index": 0, "bindGroup": 9],    // red
            ["op": "draw", "vertexCount": 3],
            ["op": "endPass"],
        ])

        try harness.assertPixel(x: 16, y: 32, equals: red, "the region the stencil passed")
        try harness.assertPixel(x: 48, y: 32, equals: clearBlue, "stencil failed → the clear color")
    }

    /// Builds the same picture with a scissor and compares whole frames.
    ///
    /// Two points alone cannot rule out "clipped for some reason other than stencil". Only when the
    /// boundary pixels match too can we say the stencil clipped exactly that region.
    func test_theStencilMaskResultMatchesAScissorOverTheSameRegion() throws {
        harness.executeExpectingSuccess(makeCommonResources() + [
            ["op": "createTexture", "id": 2, "size": ["width": 64, "height": 64],
             "format": "stencil8", "usage": TestUsage.renderAttachment],
            ["op": "createTextureView", "id": 3, "texture": 2],
            pipeline(id: 6, stencil: bothFaces(["compare": "always", "passOp": "replace"]),
                     writesColor: false),
            pipeline(id: 7, stencil: bothFaces(["compare": "equal"])),
            // The control — a pipeline with a scissor and no stencil.
            ["op": "createRenderPipeline", "id": 30, "layout": "auto",
             "vertex": ["module": 1, "entryPoint": "vs_main"],
             "fragment": ["module": 1, "entryPoint": "fs_main",
                          "targets": [["format": "rgba8unorm"]]]],
            ["op": "getBindGroupLayout", "id": 8, "pipeline": 6, "index": 0],
            ["op": "createBindGroup", "id": 9, "layout": 8,
             "entries": [["binding": 0, "resource": ["buffer": 4]]]],
        ])

        // The baseline — the left half red, by scissor.
        harness.executeExpectingSuccess(acquireDrawable + [
            ["op": "beginRenderPass", "colorAttachments": [[
                "view": 21, "loadOp": "clear", "storeOp": "store",
                "clearValue": ["r": 0, "g": 0, "b": 1, "a": 1],
            ]]],
            ["op": "setPipeline", "pipeline": 30],
            ["op": "setBindGroup", "index": 0, "bindGroup": 9],
            ["op": "setScissorRect", "x": 0, "y": 0, "width": 32, "height": 64],
            ["op": "draw", "vertexCount": 3],
            ["op": "endPass"],
        ])
        let scissored = try harness.frameBytes()

        // The same picture, by stencil mask.
        harness.executeExpectingSuccess(acquireDrawable + [
            beginPass(stencilView: 3),
            ["op": "setStencilReference", "reference": 1],
            ["op": "setPipeline", "pipeline": 6],
            ["op": "setBindGroup", "index": 0, "bindGroup": 9],
            ["op": "setScissorRect", "x": 0, "y": 0, "width": 32, "height": 64],
            ["op": "draw", "vertexCount": 3],
            ["op": "setScissorRect", "x": 0, "y": 0, "width": 64, "height": 64],
            ["op": "setPipeline", "pipeline": 7],
            ["op": "draw", "vertexCount": 3],
            ["op": "endPass"],
        ])

        try harness.assertFrameEquals(scissored, "the stencil must leave exactly the region the scissor did")
    }

    // MARK: - Reference value and masks

    func test_setStencilReferenceAffectsBothWritesAndComparisons() throws {
        harness.executeExpectingSuccess(makeCommonResources() + [
            ["op": "createTexture", "id": 2, "size": ["width": 64, "height": 64],
             "format": "stencil8", "usage": TestUsage.renderAttachment],
            ["op": "createTextureView", "id": 3, "texture": 2],
            pipeline(id: 6, stencil: bothFaces(["compare": "always", "passOp": "replace"]),
                     writesColor: false),
            pipeline(id: 7, stencil: bothFaces(["compare": "equal"])),
            ["op": "getBindGroupLayout", "id": 8, "pipeline": 6, "index": 0],
            ["op": "createBindGroup", "id": 9, "layout": 8,
             "entries": [["binding": 0, "resource": ["buffer": 4]]]],
        ] + acquireDrawable + [
            beginPass(stencilView: 3),
            ["op": "setPipeline", "pipeline": 6],
            ["op": "setBindGroup", "index": 0, "bindGroup": 9],
            // Writes 1 on the left and 2 on the right — the value replace writes is the current reference.
            ["op": "setStencilReference", "reference": 1],
            ["op": "setScissorRect", "x": 0, "y": 0, "width": 32, "height": 64],
            ["op": "draw", "vertexCount": 3],
            ["op": "setStencilReference", "reference": 2],
            ["op": "setScissorRect", "x": 32, "y": 0, "width": 32, "height": 64],
            ["op": "draw", "vertexCount": 3],
            // Comparing with reference 2 → only the right passes.
            ["op": "setScissorRect", "x": 0, "y": 0, "width": 64, "height": 64],
            ["op": "setPipeline", "pipeline": 7],
            ["op": "draw", "vertexCount": 3],
            ["op": "endPass"],
        ])

        try harness.assertPixel(x: 48, y: 32, equals: red, "the region written with reference 2")
        try harness.assertPixel(x: 16, y: 32, equals: clearBlue, "the region written with reference 1 — different from 2")
    }

    func test_aStencilWriteMaskOfZeroLeavesTheStencilUnchanged() throws {
        harness.executeExpectingSuccess(makeCommonResources() + [
            ["op": "createTexture", "id": 2, "size": ["width": 64, "height": 64],
             "format": "stencil8", "usage": TestUsage.renderAttachment],
            ["op": "createTextureView", "id": 3, "texture": 2],
            // Marking the left — with writeMask 0, replace cannot change a single bit.
            pipeline(id: 6,
                     stencil: bothFaces(["compare": "always", "passOp": "replace"])
                        .merging(["stencilWriteMask": 0]) { _, new in new },
                     writesColor: false),
            // Marking the right — the same state with the mask at its default.
            pipeline(id: 7, stencil: bothFaces(["compare": "always", "passOp": "replace"]),
                     writesColor: false),
            pipeline(id: 8, stencil: bothFaces(["compare": "equal"])),
            ["op": "getBindGroupLayout", "id": 9, "pipeline": 6, "index": 0],
            ["op": "createBindGroup", "id": 10, "layout": 9,
             "entries": [["binding": 0, "resource": ["buffer": 4]]]],
        ] + acquireDrawable + [
            beginPass(stencilView: 3),
            ["op": "setStencilReference", "reference": 1],
            ["op": "setBindGroup", "index": 0, "bindGroup": 10],
            ["op": "setPipeline", "pipeline": 6],
            ["op": "setScissorRect", "x": 0, "y": 0, "width": 32, "height": 64],
            ["op": "draw", "vertexCount": 3],
            ["op": "setPipeline", "pipeline": 7],
            ["op": "setScissorRect", "x": 32, "y": 0, "width": 32, "height": 64],
            ["op": "draw", "vertexCount": 3],
            ["op": "setScissorRect", "x": 0, "y": 0, "width": 64, "height": 64],
            ["op": "setPipeline", "pipeline": 8],
            ["op": "draw", "vertexCount": 3],
            ["op": "endPass"],
        ])

        try harness.assertPixel(x: 16, y: 32, equals: clearBlue, "writeMask 0 — the stencil stays 0")
        try harness.assertPixel(x: 48, y: 32, equals: red, "the default mask — the stencil becomes 1")
    }

    func test_stencilReadMaskHidesBitsBeforeTheComparison() throws {
        harness.executeExpectingSuccess(makeCommonResources() + [
            ["op": "createTexture", "id": 2, "size": ["width": 64, "height": 64],
             "format": "stencil8", "usage": TestUsage.renderAttachment],
            ["op": "createTextureView", "id": 3, "texture": 2],
            pipeline(id: 6, stencil: bothFaces(["compare": "always", "passOp": "replace"]),
                     writesColor: false),
            // Comparing with no mask — the stored 3 differs from reference 1.
            pipeline(id: 7, stencil: bothFaces(["compare": "equal"])),
            // Comparing on the low bit only — (1 & 1) == (3 & 1), so it passes.
            pipeline(id: 8,
                     stencil: bothFaces(["compare": "equal"])
                        .merging(["stencilReadMask": 0x1]) { _, new in new }),
            ["op": "getBindGroupLayout", "id": 9, "pipeline": 6, "index": 0],
            ["op": "createBindGroup", "id": 10, "layout": 9,
             "entries": [["binding": 0, "resource": ["buffer": 4]]]],
        ] + acquireDrawable + [
            beginPass(stencilView: 3),
            ["op": "setBindGroup", "index": 0, "bindGroup": 10],
            // Stencil 3 across the whole screen.
            ["op": "setStencilReference", "reference": 3],
            ["op": "setPipeline", "pipeline": 6],
            ["op": "draw", "vertexCount": 3],
            ["op": "setStencilReference", "reference": 1],
            ["op": "setPipeline", "pipeline": 7],
            ["op": "setScissorRect", "x": 0, "y": 0, "width": 32, "height": 64],
            ["op": "draw", "vertexCount": 3],
            ["op": "setPipeline", "pipeline": 8],
            ["op": "setScissorRect", "x": 32, "y": 0, "width": 32, "height": 64],
            ["op": "draw", "vertexCount": 3],
            ["op": "endPass"],
        ])

        try harness.assertPixel(x: 16, y: 32, equals: clearBlue, "with no mask 1 != 3 → fails")
        try harness.assertPixel(x: 48, y: 32, equals: red, "with readMask 0x1, 1 == 3 → passes")
    }

    // MARK: - Combined with depth

    /// The path shadow volumes use — marking the stencil only where a fragment **lost** the depth test.
    /// Swapping `passOp` (both passed) and `depthFailOp` (only stencil passed) shows up right here.
    func test_depthFailOpMarksOnlyWhereDepthWasLost() throws {
        harness.executeExpectingSuccess(makeCommonResources() + [
            ["op": "createTexture", "id": 2, "size": ["width": 64, "height": 64],
             "format": "depth24plus-stencil8", "usage": TestUsage.renderAttachment],
            ["op": "createTextureView", "id": 3, "texture": 2],
            // Leave the near face (0.2) in the depth buffer.
            ["op": "createBuffer", "id": 11, "size": 32, "usage": TestUsage.uniform,
             "data": fill(red: 0, green: 1, blue: 0, depth: 0.2)],
            // The far face (0.8) — it loses the depth test.
            ["op": "createBuffer", "id": 12, "size": 32, "usage": TestUsage.uniform,
             "data": fill(red: 0, green: 1, blue: 0, depth: 0.8)],
            pipeline(id: 6, format: "depth24plus-stencil8", stencil: [:],
                     writesColor: false, depthCompare: "less", depthWrite: true),
            pipeline(id: 7, format: "depth24plus-stencil8",
                     stencil: bothFaces(["compare": "always", "depthFailOp": "replace"]),
                     writesColor: false, depthCompare: "less"),
            pipeline(id: 8, format: "depth24plus-stencil8",
                     stencil: bothFaces(["compare": "equal"])),
            ["op": "getBindGroupLayout", "id": 9, "pipeline": 6, "index": 0],
            ["op": "createBindGroup", "id": 13, "layout": 9,
             "entries": [["binding": 0, "resource": ["buffer": 11]]]],
            ["op": "createBindGroup", "id": 14, "layout": 9,
             "entries": [["binding": 0, "resource": ["buffer": 12]]]],
            ["op": "createBindGroup", "id": 15, "layout": 9,
             "entries": [["binding": 0, "resource": ["buffer": 4]]]],
        ] + acquireDrawable + [
            beginPass(stencilView: 3),
            ["op": "setStencilReference", "reference": 1],
            // 1) Lay down depth 0.2.
            ["op": "setPipeline", "pipeline": 6],
            ["op": "setBindGroup", "index": 0, "bindGroup": 13],
            ["op": "draw", "vertexCount": 3],
            // 2) Depth 0.8 — the stencil test passes, the depth test loses, so depthFailOp runs.
            //    Drawn on the left half only, putting "marked" and "unmarked" in one screen.
            ["op": "setPipeline", "pipeline": 7],
            ["op": "setBindGroup", "index": 0, "bindGroup": 14],
            ["op": "setScissorRect", "x": 0, "y": 0, "width": 32, "height": 64],
            ["op": "draw", "vertexCount": 3],
            // 3) Red only where it was marked.
            ["op": "setScissorRect", "x": 0, "y": 0, "width": 64, "height": 64],
            ["op": "setPipeline", "pipeline": 8],
            ["op": "setBindGroup", "index": 0, "bindGroup": 15],
            ["op": "draw", "vertexCount": 3],
            ["op": "endPass"],
        ])

        try harness.assertPixel(x: 16, y: 32, equals: red, "the region that lost on depth — depthFailOp marked it")
        try harness.assertPixel(x: 48, y: 32, equals: clearBlue, "the region that never took the depth test")
    }

    // MARK: - Regression

    /// Regression — `depthAttachmentPixelFormat` used to be set unconditionally, so a depthless
    /// `stencil8` pipeline demanded a depth attachment that did not exist and failed to be created at all.
    func test_aStencil8OnlyFormatPipelineIsCreated() {
        harness.executeExpectingSuccess(makeCommonResources() + [
            pipeline(id: 6, stencil: bothFaces(["compare": "equal", "passOp": "replace"])),
        ])
    }

    // MARK: - Contract

    /// Attaching stencil state to a format with no stencil aspect makes Metal **ignore it silently**.
    /// You would then debug "why isn't stencil masking working" with not one error, so it is stopped here.
    func test_rejectsStencilStateOnADepthFormatWithoutStencil() {
        for format in ["depth32float", "depth24plus", "depth16unorm"] {
            let result = harness.execute(makeCommonResources() + [
                pipeline(id: 6, format: format,
                         stencil: bothFaces(["compare": "equal", "passOp": "replace"])),
            ])
            let message = (result["errors"] as? [[String: Any]])?.first?["message"] as? String ?? ""
            XCTAssertTrue(
                message.contains("stencil aspect"),
                "\(format) plus a non-default stencilFront must be rejected — got: \(message)"
            )
        }
    }

    // MARK: - Read-only attachments

    /// `depthReadOnly: true` declares "this pass does not write depth". Metal simply writes anyway, so
    /// unchecked here the depth buffer marked read-only really is modified.
    func test_aDepthReadOnlyPassRejectsAPipelineThatWritesDepth() {
        let result = harness.execute(makeCommonResources() + [
            ["op": "createTexture", "id": 2, "size": ["width": 64, "height": 64],
             "format": "depth24plus-stencil8", "usage": TestUsage.renderAttachment],
            ["op": "createTextureView", "id": 3, "texture": 2],
            pipeline(id: 6, format: "depth24plus-stencil8", stencil: [:],
                     depthCompare: "less", depthWrite: true),
        ] + acquireDrawable + [
            readOnlyPass(view: 3, depthReadOnly: true),
            ["op": "setPipeline", "pipeline": 6],
            ["op": "endPass"],
        ])

        XCTAssertTrue(
            ((errors(result).first?["message"] as? String) ?? "").contains("depthReadOnly"),
            harness.describeErrors(result)
        )
    }

    /// A pipeline that only reads depth must pass through the same pass unchanged.
    func test_aDepthReadOnlyPassAcceptsAReadOnlyPipeline() {
        harness.executeExpectingSuccess(makeCommonResources() + [
            ["op": "createTexture", "id": 2, "size": ["width": 64, "height": 64],
             "format": "depth24plus-stencil8", "usage": TestUsage.renderAttachment],
            ["op": "createTextureView", "id": 3, "texture": 2],
            pipeline(id: 6, format: "depth24plus-stencil8", stencil: [:],
                     depthCompare: "less", depthWrite: false),
            ["op": "getBindGroupLayout", "id": 8, "pipeline": 6, "index": 0],
            ["op": "createBindGroup", "id": 9, "layout": 8,
             "entries": [["binding": 0, "resource": ["buffer": 4]]]],
        ] + acquireDrawable + [
            readOnlyPass(view: 3, depthReadOnly: true),
            ["op": "setPipeline", "pipeline": 6],
            ["op": "setBindGroup", "index": 0, "bindGroup": 9],
            ["op": "draw", "vertexCount": 3],
            ["op": "endPass"],
        ])
    }

    func test_aStencilReadOnlyPassRejectsAPipelineThatWritesStencil() {
        let result = harness.execute(makeCommonResources() + [
            ["op": "createTexture", "id": 2, "size": ["width": 64, "height": 64],
             "format": "depth24plus-stencil8", "usage": TestUsage.renderAttachment],
            ["op": "createTextureView", "id": 3, "texture": 2],
            pipeline(id: 6, format: "depth24plus-stencil8",
                     stencil: bothFaces(["compare": "always", "passOp": "replace"])),
        ] + acquireDrawable + [
            readOnlyPass(view: 3, stencilReadOnly: true),
            ["op": "setPipeline", "pipeline": 6],
            ["op": "endPass"],
        ])

        XCTAssertTrue(
            ((errors(result).first?["message"] as? String) ?? "").contains("stencilReadOnly"),
            harness.describeErrors(result)
        )
    }

    /// Stencil state that only compares is "reading" and must pass.
    func test_aStencilReadOnlyPassAcceptsAComparisonOnlyPipeline() {
        harness.executeExpectingSuccess(makeCommonResources() + [
            ["op": "createTexture", "id": 2, "size": ["width": 64, "height": 64],
             "format": "depth24plus-stencil8", "usage": TestUsage.renderAttachment],
            ["op": "createTextureView", "id": 3, "texture": 2],
            pipeline(id: 6, format: "depth24plus-stencil8", stencil: bothFaces(["compare": "equal"])),
        ] + acquireDrawable + [
            readOnlyPass(view: 3, stencilReadOnly: true),
            ["op": "setPipeline", "pipeline": 6],
            ["op": "endPass"],
        ])
    }

    /// Giving `readOnly` together with load/store ops is contradictory and the spec forbids it.
    func test_depthReadOnlyTogetherWithLoadOpIsRejected() {
        let result = harness.execute(makeCommonResources() + [
            ["op": "createTexture", "id": 2, "size": ["width": 64, "height": 64],
             "format": "depth24plus-stencil8", "usage": TestUsage.renderAttachment],
            ["op": "createTextureView", "id": 3, "texture": 2],
        ] + acquireDrawable + [
            [
                "op": "beginRenderPass",
                "colorAttachments": [[
                    "view": 21, "loadOp": "clear", "storeOp": "store",
                    "clearValue": ["r": 0, "g": 0, "b": 1, "a": 1],
                ]],
                "depthStencilAttachment": [
                    "view": 3, "depthReadOnly": true,
                    "depthLoadOp": "clear", "depthStoreOp": "store",
                ],
            ],
            ["op": "endPass"],
        ])

        XCTAssertTrue(
            ((errors(result).first?["message"] as? String) ?? "").contains("cannot be combined"),
            harness.describeErrors(result)
        )
    }

    /// `GPUStencilValue` is `u32` and the WebIDL conversion is modulo — a negative wraps rather than traps.
    /// With a non-truncating initializer this single line would kill the process.
    func test_aNegativeStencilReferenceDoesNotKillTheProcess() {
        harness.executeExpectingSuccess(makeCommonResources() + [
            ["op": "createTexture", "id": 2, "size": ["width": 64, "height": 64],
             "format": "stencil8", "usage": TestUsage.renderAttachment],
            ["op": "createTextureView", "id": 3, "texture": 2],
            pipeline(id: 6, stencil: bothFaces(["compare": "always", "passOp": "replace"])),
        ] + acquireDrawable + [
            [
                "op": "beginRenderPass",
                "colorAttachments": [[
                    "view": 21, "loadOp": "clear", "storeOp": "store",
                    "clearValue": ["r": 0, "g": 0, "b": 1, "a": 1],
                ]],
                "depthStencilAttachment": [
                    "view": 3,
                    "stencilClearValue": -1, "stencilLoadOp": "clear", "stencilStoreOp": "store",
                ],
            ],
            ["op": "setStencilReference", "reference": -1],
            ["op": "endPass"],
        ])
    }

    /// Conversely, with no stencil state given, a depth-only format must pass straight through.
    func test_aDepthOnlyFormatPassesWhenStencilStateIsDefault() {
        harness.executeExpectingSuccess(makeCommonResources() + [
            pipeline(id: 6, format: "depth32float", stencil: [:], depthCompare: "less", depthWrite: true),
        ])
    }
}
