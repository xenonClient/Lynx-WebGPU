import Foundation
import LynxWebGPUCore

/// The checks that make up the suite.
///
/// Each check looks at **one place in the contract**. Adding a check here is the last step of
/// `docs/COMMAND-STREAM.md` §7 when you add a new op.
public extension WebGPUConformance {

    /// A full-screen triangle — no vertex buffer, driven by `vertex_index` alone.
    /// The fragment output is a fixed color, which keeps the pixel assertions simple.
    static let fullscreenShader = """
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
        return vec4f(1.0, 0.0, 0.0, 1.0);
    }
    """

    static var checks: [Check] {
        [
            clearColor,
            solidDraw,
            vertexBuffer,
            uniformBindGroup,
            indexBufferEquivalence,
            computeStorageReadback,
            copyBufferToBuffer,
            clearBuffer,
            textureUploadAndSample,
            copyTextureToBuffer,
            alphaBlending,
            depthTest,
            renderBundleEquivalence,
            indirectDrawEquivalence,
            errorAccumulation,
            errorScopeCapture,
            handleTypeMismatch,
            adapterLimitsSpelling,
            canvasInfoReporting,
            // Frame lifetime and callback APIs (LifecycleChecks.swift)
            presentFalsePreservesFrame,
            presentExpiresFrame,
            emptyPresentClosesFrame,
            readBufferContract,
            resizeCanvasReflects,
            shaderCompilationInfoShape,
            mslOptional,
            decodeImageUpload,
            frameReadiness,
            pumpConcurrency,
        ]
    }

    // MARK: - Render basics

    /// The clear color comes out as given — the shortest path through opening and closing a render pass.
    static var clearColor: Check {
        Check("clear-color") { harness in
            try harness.executeExpectingSuccess(
                harness.canvasSetup + [
                    ["op": "beginRenderPass", "colorAttachments": [[
                        "view": 11, "loadOp": "clear", "storeOp": "store",
                        "clearValue": ["r": 0.25, "g": 0.5, "b": 0.75, "a": 1.0],
                    ]]],
                    ["op": "endPass"],
                ]
            )
            try harness.expectPixel(x: 32, y: 32, equals: (64, 128, 191, 255))
        }
    }

    /// WGSL → pipeline → draw actually changes pixels.
    static var solidDraw: Check {
        Check("solid-draw") { harness in
            try harness.executeExpectingSuccess(
                harness.canvasSetup + harness.fullscreenPipeline(module: 1, pipeline: 2) + [
                    ["op": "beginRenderPass", "colorAttachments": [[
                        "view": 11, "loadOp": "clear", "storeOp": "store",
                        "clearValue": ["r": 0, "g": 0, "b": 1, "a": 1],
                    ]]],
                    ["op": "setPipeline", "pipeline": 2],
                    ["op": "draw", "vertexCount": 3],
                    ["op": "endPass"],
                ]
            )
            try harness.expectPixel(x: 32, y: 32, equals: (255, 0, 0, 255), "draw result")
        }
    }

    /// A vertex buffer's position and color reach rasterization (inside and outside differ).
    static var vertexBuffer: Check {
        Check("vertex-buffer") { harness in
            // Interleaved position (x,y) + color (r,g,b) — stride 20B
            let vertices: [Float] = [
                -0.5, -0.5, 1, 0, 0,
                 0.5, -0.5, 1, 0, 0,
                 0.0,  0.5, 1, 0, 0,
            ]
            try harness.executeExpectingSuccess(
                harness.canvasSetup + [
                    ["op": "createShaderModule", "id": 1, "code": Self.vertexColorShader],
                    ["op": "createBuffer", "id": 2,
                     "usage": WebGPUUsage.vertex | WebGPUUsage.copyDst,
                     "data": vertices.conformanceBase64],
                    ["op": "createRenderPipeline", "id": 3, "layout": "auto",
                     "vertex": ["module": 1, "entryPoint": "vs_main", "buffers": [[
                        "arrayStride": 20,
                        "attributes": [
                            ["format": "float32x2", "offset": 0, "shaderLocation": 0],
                            ["format": "float32x3", "offset": 8, "shaderLocation": 1],
                        ],
                     ]]],
                     "fragment": ["module": 1, "entryPoint": "fs_main",
                                  "targets": [["format": "rgba8unorm"]]]],
                    ["op": "beginRenderPass", "colorAttachments": [[
                        "view": 11, "loadOp": "clear", "storeOp": "store",
                        "clearValue": ["r": 0, "g": 0, "b": 1, "a": 1],
                    ]]],
                    ["op": "setPipeline", "pipeline": 3],
                    ["op": "setVertexBuffer", "slot": 0, "buffer": 2],
                    ["op": "draw", "vertexCount": 3],
                    ["op": "endPass"],
                ]
            )
            try harness.expectPixel(x: 32, y: 32, equals: (255, 0, 0, 255), "inside the triangle")
            try harness.expectPixel(x: 1, y: 1, equals: (0, 0, 255, 255), "outside the triangle (clear color)")
        }
    }

    /// A uniform buffer reaches the fragment output — checks that bind group assignment matches the shader.
    static var uniformBindGroup: Check {
        Check("uniform-bind-group") { harness in
            let tint: [Float] = [0, 1, 0, 1]
            try harness.executeExpectingSuccess(
                harness.canvasSetup + [
                    ["op": "createShaderModule", "id": 1, "code": Self.tintShader],
                    ["op": "createBuffer", "id": 2, "size": 16,
                     "usage": WebGPUUsage.uniform | WebGPUUsage.copyDst,
                     "data": tint.conformanceBase64],
                    ["op": "createRenderPipeline", "id": 3, "layout": "auto",
                     "vertex": ["module": 1, "entryPoint": "vs_main"],
                     "fragment": ["module": 1, "entryPoint": "fs_main",
                                  "targets": [["format": "rgba8unorm"]]]],
                    ["op": "getBindGroupLayout", "id": 4, "pipeline": 3, "index": 0],
                    ["op": "createBindGroup", "id": 5, "layout": 4,
                     "entries": [["binding": 0, "resource": ["buffer": 2]]]],
                    ["op": "beginRenderPass", "colorAttachments": [[
                        "view": 11, "loadOp": "clear", "storeOp": "store",
                        "clearValue": ["r": 0, "g": 0, "b": 0, "a": 1],
                    ]]],
                    ["op": "setPipeline", "pipeline": 3],
                    ["op": "setBindGroup", "index": 0, "bindGroup": 5],
                    ["op": "draw", "vertexCount": 3],
                    ["op": "endPass"],
                ]
            )
            try harness.expectPixel(x: 32, y: 32, equals: (0, 255, 0, 255), "uniform color")
        }
    }

    /// `drawIndexed` draws the same triangle as a direct draw, **pixel for pixel**.
    static var indexBufferEquivalence: Check {
        Check("index-buffer-equivalence") { harness in
            // Four quad corners. The direct path lists six; the indexed path uses four plus six indices.
            let corners: [Float] = [-0.5, -0.5, 0.5, -0.5, -0.5, 0.5, 0.5, 0.5]
            let expanded: [Float] = [
                -0.5, -0.5, 0.5, -0.5, -0.5, 0.5,
                -0.5,  0.5, 0.5, -0.5,  0.5, 0.5,
            ]
            let indices: [UInt16] = [0, 1, 2, 2, 1, 3]

            try harness.executeExpectingSuccess(
                harness.canvasSetup + [
                    ["op": "createShaderModule", "id": 1, "code": Self.positionOnlyShader],
                    ["op": "createBuffer", "id": 2, "usage": WebGPUUsage.vertex,
                     "data": expanded.conformanceBase64],
                    ["op": "createBuffer", "id": 3, "usage": WebGPUUsage.vertex,
                     "data": corners.conformanceBase64],
                    ["op": "createBuffer", "id": 4, "usage": WebGPUUsage.index,
                     "data": indices.conformanceBase64],
                    ["op": "createRenderPipeline", "id": 5, "layout": "auto",
                     "vertex": ["module": 1, "entryPoint": "vs_main", "buffers": [[
                        "arrayStride": 8,
                        "attributes": [["format": "float32x2", "offset": 0, "shaderLocation": 0]],
                     ]]],
                     "fragment": ["module": 1, "entryPoint": "fs_main",
                                  "targets": [["format": "rgba8unorm"]]]],
                ] + harness.quadPass(pipeline: 5, vertexBuffer: 2, drawing: ["op": "draw", "vertexCount": 6])
            )
            let direct = try harness.frameBytes()

            try harness.executeExpectingSuccess(
                harness.canvasSetup + harness.quadPass(
                    pipeline: 5, vertexBuffer: 3,
                    before: [["op": "setIndexBuffer", "buffer": 4, "format": "uint16"]],
                    drawing: ["op": "drawIndexed", "indexCount": 6]
                )
            )
            try harness.expectFrameEquals(direct, "drawIndexed produced a different picture than the direct draw")
        }
    }

    // MARK: - Compute and buffers

    /// A compute pass writes into a storage buffer and the values can be read back.
    static var computeStorageReadback: Check {
        Check("compute-storage-readback") { harness in
            let shader = """
            @group(0) @binding(0) var<storage, read_write> data: array<u32>;

            @compute @workgroup_size(4)
            fn main(@builtin(global_invocation_id) id: vec3u) {
                data[id.x] = id.x * 2u;
            }
            """
            // The spec forbids combining `MAP_READ` with usages other than `COPY_DST` — a storage
            // buffer cannot be mapped directly, so it is copied into a readback buffer first (the same shape as on the web).
            try harness.executeExpectingSuccess([
                ["op": "createShaderModule", "id": 1, "code": shader],
                ["op": "createBuffer", "id": 2, "size": 16,
                 "usage": WebGPUUsage.storage | WebGPUUsage.copySrc],
                ["op": "createBuffer", "id": 6, "size": 16,
                 "usage": WebGPUUsage.copyDst | WebGPUUsage.mapRead],
                ["op": "createComputePipeline", "id": 3, "layout": "auto",
                 "compute": ["module": 1, "entryPoint": "main"]],
                ["op": "getBindGroupLayout", "id": 4, "pipeline": 3, "index": 0],
                ["op": "createBindGroup", "id": 5, "layout": 4,
                 "entries": [["binding": 0, "resource": ["buffer": 2]]]],
                ["op": "beginComputePass"],
                ["op": "setPipeline", "pipeline": 3],
                ["op": "setBindGroup", "index": 0, "bindGroup": 5],
                ["op": "dispatchWorkgroups", "x": 1],
                ["op": "endPass"],
                ["op": "copyBufferToBuffer", "source": 2, "destination": 6, "size": 16],
            ])
            let values = try harness.readBufferSync(handle: 6, as: UInt32.self)
            guard values == [0, 2, 4, 6] else {
                throw ConformanceFailure("compute result was \(values) — expected [0, 2, 4, 6]")
            }
        }
    }

    /// `copyBufferToBuffer` moves bytes across unchanged.
    static var copyBufferToBuffer: Check {
        Check("copy-buffer-to-buffer") { harness in
            let source: [Float] = [1, 2, 3, 4]
            try harness.executeExpectingSuccess([
                ["op": "createBuffer", "id": 1,
                 "usage": WebGPUUsage.copySrc | WebGPUUsage.copyDst,
                 "data": source.conformanceBase64],
                ["op": "createBuffer", "id": 2, "size": 16,
                 "usage": WebGPUUsage.copyDst | WebGPUUsage.mapRead],
                ["op": "copyBufferToBuffer", "source": 1, "destination": 2, "size": 16],
            ])
            let copied = try harness.readBufferSync(handle: 2, as: Float.self)
            guard copied == source else {
                throw ConformanceFailure("copy result was \(copied) — expected \(source)")
            }
        }
    }

    /// `clearBuffer` zeroes a range (the path that never ships a zero array from the CPU).
    static var clearBuffer: Check {
        Check("clear-buffer") { harness in
            let filled: [Float] = [7, 7, 7, 7]
            try harness.executeExpectingSuccess([
                ["op": "createBuffer", "id": 1,
                 "usage": WebGPUUsage.copyDst | WebGPUUsage.mapRead,
                 "data": filled.conformanceBase64],
                ["op": "clearBuffer", "buffer": 1, "offset": 0, "size": 8],
            ])
            let values = try harness.readBufferSync(handle: 1, as: Float.self)
            guard values == [0, 0, 7, 7] else {
                throw ConformanceFailure("clearBuffer result was \(values) — expected [0, 0, 7, 7]")
            }
        }
    }

    // MARK: - Textures

    /// A shader samples pixels uploaded with `writeTexture`.
    static var textureUploadAndSample: Check {
        Check("texture-upload-and-sample") { harness in
            let shader = """
            @group(0) @binding(0) var source: texture_2d<f32>;
            @group(0) @binding(1) var linearSampler: sampler;

            @vertex
            fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
                var positions = array<vec2f, 3>(
                    vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0),
                );
                return vec4f(positions[index], 0.0, 1.0);
            }

            @fragment
            fn fs_main() -> @location(0) vec4f {
                return textureSample(source, linearSampler, vec2f(0.25, 0.25));
            }
            """
            // 2×2 RGBA8 — the top-left is red and the shader samples that spot.
            let pixels: [UInt8] = [
                255, 0, 0, 255, 0, 255, 0, 255,
                0, 0, 255, 255, 255, 255, 0, 255,
            ]
            try harness.executeExpectingSuccess(
                harness.canvasSetup + [
                    ["op": "createShaderModule", "id": 1, "code": shader],
                    ["op": "createTexture", "id": 2, "size": ["width": 2, "height": 2],
                     "format": "rgba8unorm",
                     "usage": WebGPUUsage.textureBinding | WebGPUUsage.textureCopyDst],
                    ["op": "writeTexture", "texture": 2, "data": pixels.conformanceBase64,
                     "bytesPerRow": 8, "size": ["width": 2, "height": 2]],
                    ["op": "createTextureView", "id": 3, "texture": 2],
                    ["op": "createSampler", "id": 4],
                    ["op": "createRenderPipeline", "id": 5, "layout": "auto",
                     "vertex": ["module": 1, "entryPoint": "vs_main"],
                     "fragment": ["module": 1, "entryPoint": "fs_main",
                                  "targets": [["format": "rgba8unorm"]]]],
                    ["op": "getBindGroupLayout", "id": 6, "pipeline": 5, "index": 0],
                    ["op": "createBindGroup", "id": 7, "layout": 6, "entries": [
                        ["binding": 0, "resource": ["textureView": 3]],
                        ["binding": 1, "resource": ["sampler": 4]],
                    ]],
                    ["op": "beginRenderPass", "colorAttachments": [[
                        "view": 11, "loadOp": "clear", "storeOp": "store",
                        "clearValue": ["r": 0, "g": 0, "b": 0, "a": 1],
                    ]]],
                    ["op": "setPipeline", "pipeline": 5],
                    ["op": "setBindGroup", "index": 0, "bindGroup": 7],
                    ["op": "draw", "vertexCount": 3],
                    ["op": "endPass"],
                ]
            )
            try harness.expectPixel(x: 32, y: 32, equals: (255, 0, 0, 255), "top-left texel")
        }
    }

    /// Downloads a render target into a buffer (`copyTextureToBuffer`).
    ///
    /// The width is 64 so the row stride is 256B — the alignment the spec requires.
    static var copyTextureToBuffer: Check {
        Check("copy-texture-to-buffer") { harness in
            try harness.executeExpectingSuccess([
                ["op": "createTexture", "id": 1, "size": ["width": 64, "height": 4],
                 "format": "rgba8unorm",
                 "usage": WebGPUUsage.renderAttachment | WebGPUUsage.textureCopySrc],
                ["op": "createTextureView", "id": 2, "texture": 1],
                ["op": "createBuffer", "id": 3, "size": 1024,
                 "usage": WebGPUUsage.copyDst | WebGPUUsage.mapRead],
                ["op": "beginRenderPass", "colorAttachments": [[
                    "view": 2, "loadOp": "clear", "storeOp": "store",
                    "clearValue": ["r": 1, "g": 0, "b": 0, "a": 1],
                ]]],
                ["op": "endPass"],
                ["op": "copyTextureToBuffer",
                 "source": ["texture": 1],
                 "destination": ["buffer": 3, "bytesPerRow": 256],
                 "copySize": ["width": 64, "height": 4]],
            ])
            let bytes = try harness.readBufferSync(handle: 3, as: UInt8.self)
            guard bytes.count == 1024 else {
                throw ConformanceFailure("readback length was \(bytes.count)B — expected 1024B")
            }
            guard Array(bytes.prefix(4)) == [255, 0, 0, 255],
                  Array(bytes.suffix(4)) == [255, 0, 0, 255] else {
                throw ConformanceFailure(
                    "copied pixels differ — first \(Array(bytes.prefix(4))), last \(Array(bytes.suffix(4)))"
                )
            }
        }
    }

    // MARK: - Pipeline state

    /// The alpha blending formula is applied.
    static var alphaBlending: Check {
        Check("alpha-blending") { harness in
            let shader = Self.fullscreenShader.replacingOccurrences(
                of: "return vec4f(1.0, 0.0, 0.0, 1.0);",
                with: "return vec4f(1.0, 0.0, 0.0, 0.5);"
            )
            try harness.executeExpectingSuccess(
                harness.canvasSetup + [
                    ["op": "createShaderModule", "id": 1, "code": shader],
                    ["op": "createRenderPipeline", "id": 2, "layout": "auto",
                     "vertex": ["module": 1, "entryPoint": "vs_main"],
                     "fragment": ["module": 1, "entryPoint": "fs_main", "targets": [[
                        "format": "rgba8unorm",
                        "blend": [
                            "color": ["srcFactor": "src-alpha", "dstFactor": "one-minus-src-alpha"],
                            "alpha": ["srcFactor": "one", "dstFactor": "one-minus-src-alpha"],
                        ],
                     ]]]],
                    ["op": "beginRenderPass", "colorAttachments": [[
                        "view": 11, "loadOp": "clear", "storeOp": "store",
                        "clearValue": ["r": 0, "g": 0, "b": 1, "a": 1],
                    ]]],
                    ["op": "setPipeline", "pipeline": 2],
                    ["op": "draw", "vertexCount": 3],
                    ["op": "endPass"],
                ]
            )
            // 0.5·red + 0.5·blue
            try harness.expectPixel(x: 32, y: 32, equals: (128, 0, 128, 255), tolerance: 3)
        }
    }

    /// The depth test hides the triangle behind.
    static var depthTest: Check {
        Check("depth-test") { harness in
            let near = Self.depthShader(z: 0.2, color: "vec4f(1.0, 0.0, 0.0, 1.0)")
            let far = Self.depthShader(z: 0.8, color: "vec4f(0.0, 1.0, 0.0, 1.0)")
            try harness.executeExpectingSuccess(
                harness.canvasSetup + [
                    ["op": "createTexture", "id": 20, "size": ["width": 64, "height": 64],
                     "format": "depth32float", "usage": WebGPUUsage.renderAttachment],
                    ["op": "createTextureView", "id": 21, "texture": 20],
                    ["op": "createShaderModule", "id": 1, "code": near],
                    ["op": "createShaderModule", "id": 2, "code": far],
                    ["op": "createRenderPipeline", "id": 3, "layout": "auto",
                     "vertex": ["module": 1, "entryPoint": "vs_main"],
                     "fragment": ["module": 1, "entryPoint": "fs_main",
                                  "targets": [["format": "rgba8unorm"]]],
                     "depthStencil": ["format": "depth32float", "depthWriteEnabled": true,
                                      "depthCompare": "less"]],
                    ["op": "createRenderPipeline", "id": 4, "layout": "auto",
                     "vertex": ["module": 2, "entryPoint": "vs_main"],
                     "fragment": ["module": 2, "entryPoint": "fs_main",
                                  "targets": [["format": "rgba8unorm"]]],
                     "depthStencil": ["format": "depth32float", "depthWriteEnabled": true,
                                      "depthCompare": "less"]],
                    ["op": "beginRenderPass",
                     "colorAttachments": [[
                        "view": 11, "loadOp": "clear", "storeOp": "store",
                        "clearValue": ["r": 0, "g": 0, "b": 0, "a": 1],
                     ]],
                     "depthStencilAttachment": [
                        "view": 21, "depthLoadOp": "clear", "depthStoreOp": "store",
                        "depthClearValue": 1.0,
                     ]],
                    ["op": "setPipeline", "pipeline": 3],
                    ["op": "draw", "vertexCount": 3],
                    ["op": "setPipeline", "pipeline": 4],
                    ["op": "draw", "vertexCount": 3],
                    ["op": "endPass"],
                ]
            )
            try harness.expectPixel(x: 32, y: 32, equals: (255, 0, 0, 255), "the front triangle must remain")
        }
    }

    /// A render bundle produces **the same picture as direct encoding** — the bundle contract itself.
    static var renderBundleEquivalence: Check {
        Check("render-bundle-equivalence") { harness in
            let setup: [[String: Any]] = [
                ["op": "createShaderModule", "id": 1, "code": Self.fullscreenShader],
                ["op": "createRenderPipeline", "id": 2, "layout": "auto",
                 "vertex": ["module": 1, "entryPoint": "vs_main"],
                 "fragment": ["module": 1, "entryPoint": "fs_main",
                              "targets": [["format": "rgba8unorm"]]]],
            ]
            try harness.executeExpectingSuccess(
                harness.canvasSetup + setup + [
                    ["op": "beginRenderPass", "colorAttachments": [[
                        "view": 11, "loadOp": "clear", "storeOp": "store",
                        "clearValue": ["r": 0, "g": 0, "b": 1, "a": 1],
                    ]]],
                    ["op": "setPipeline", "pipeline": 2],
                    ["op": "draw", "vertexCount": 3],
                    ["op": "endPass"],
                ]
            )
            let direct = try harness.frameBytes()

            try harness.executeExpectingSuccess(
                harness.canvasSetup + [
                    ["op": "createRenderBundle", "id": 3,
                     "colorFormats": ["rgba8unorm"],
                     "commands": [
                        ["op": "setPipeline", "pipeline": 2],
                        ["op": "draw", "vertexCount": 3],
                     ]],
                    ["op": "beginRenderPass", "colorAttachments": [[
                        "view": 11, "loadOp": "clear", "storeOp": "store",
                        "clearValue": ["r": 0, "g": 0, "b": 1, "a": 1],
                    ]]],
                    ["op": "executeBundles", "bundles": [3]],
                    ["op": "endPass"],
                ]
            )
            try harness.expectFrameEquals(direct, "bundle execution produced a different picture than direct encoding")
        }
    }

    /// An indirect draw produces **the same picture** as a direct draw.
    ///
    /// Runs only where the device supports indirect arguments — the iOS simulator drops out here.
    static var indirectDrawEquivalence: Check {
        Check("indirect-draw-equivalence", requiresFeature: "indirect-first-instance") { harness in
            // vertexCount, instanceCount, firstVertex, firstInstance — four u32s
            let arguments: [UInt32] = [3, 1, 0, 0]
            let argumentBytes = arguments.withUnsafeBufferPointer {
                Data(buffer: $0).base64EncodedString()
            }
            let setup: [[String: Any]] = [
                ["op": "createShaderModule", "id": 1, "code": Self.fullscreenShader],
                ["op": "createRenderPipeline", "id": 2, "layout": "auto",
                 "vertex": ["module": 1, "entryPoint": "vs_main"],
                 "fragment": ["module": 1, "entryPoint": "fs_main",
                              "targets": [["format": "rgba8unorm"]]]],
                ["op": "createBuffer", "id": 3,
                 "usage": WebGPUUsage.indirect | WebGPUUsage.copyDst, "data": argumentBytes],
            ]
            try harness.executeExpectingSuccess(
                harness.canvasSetup + setup + [
                    ["op": "beginRenderPass", "colorAttachments": [[
                        "view": 11, "loadOp": "clear", "storeOp": "store",
                        "clearValue": ["r": 0, "g": 0, "b": 1, "a": 1],
                    ]]],
                    ["op": "setPipeline", "pipeline": 2],
                    ["op": "draw", "vertexCount": 3],
                    ["op": "endPass"],
                ]
            )
            let direct = try harness.frameBytes()

            try harness.executeExpectingSuccess(
                harness.canvasSetup + [
                    ["op": "beginRenderPass", "colorAttachments": [[
                        "view": 11, "loadOp": "clear", "storeOp": "store",
                        "clearValue": ["r": 0, "g": 0, "b": 1, "a": 1],
                    ]]],
                    ["op": "setPipeline", "pipeline": 2],
                    ["op": "drawIndirect", "indirectBuffer": 3, "indirectOffset": 0],
                    ["op": "endPass"],
                ]
            )
            try harness.expectFrameEquals(direct, "the indirect draw produced a different picture than the direct draw")
        }
    }

    // MARK: - Error contract

    /// **One error does not kill the batch** — the results of earlier commands remain.
    static var errorAccumulation: Check {
        Check("error-accumulation") { harness in
            let result = harness.execute(
                harness.canvasSetup + [
                    ["op": "beginRenderPass", "colorAttachments": [[
                        "view": 11, "loadOp": "clear", "storeOp": "store",
                        "clearValue": ["r": 0, "g": 1, "b": 0, "a": 1],
                    ]]],
                    ["op": "endPass"],
                    // setPipeline outside a pass — not valid.
                    ["op": "setPipeline", "pipeline": 999],
                ]
            )
            guard (result["ok"] as? Bool) == false else {
                throw ConformanceFailure("an invalid command was present but ok: true came back")
            }
            guard let errors = result["errors"] as? [[String: Any]], !errors.isEmpty else {
                throw ConformanceFailure("the error list is empty")
            }
            guard errors.contains(where: { ($0["path"] as? String)?.hasPrefix("commands[") == true }) else {
                throw ConformanceFailure("the error carries no command stream path — \(errors)")
            }
            // The clear before the error must survive.
            try harness.expectPixel(x: 32, y: 32, equals: (0, 255, 0, 255), "the clear before the error")
        }
    }

    /// An error scope **intercepts** the error — it is not carried in the batch result.
    static var errorScopeCapture: Check {
        Check("error-scope-capture") { harness in
            let result = harness.execute([
                ["op": "pushErrorScope", "filter": "validation"],
                ["op": "setPipeline", "pipeline": 1],
                ["op": "popErrorScope"],
            ])
            guard (result["ok"] as? Bool) == true else {
                throw ConformanceFailure(
                    "an error caught by a scope was also carried in the batch result — \(ConformanceHarness.describeErrors(result))"
                )
            }
            guard let scopes = result["errorScopes"] as? [Any], scopes.count == 1 else {
                throw ConformanceFailure("errorScopes was \(result["errorScopes"] ?? "absent") — expected 1")
            }
            guard let captured = scopes[0] as? [String: Any],
                  (captured["kind"] as? String) == "validation" else {
                throw ConformanceFailure("the scope did not catch the validation error — \(scopes[0])")
            }
        }
    }

    /// A handle type mismatch is **a validation error, not a crash** (WebGPU is a safe API).
    static var handleTypeMismatch: Check {
        Check("handle-type-mismatch") { harness in
            let errors = try harness.executeExpectingFailure([
                ["op": "createBuffer", "id": 1, "size": 16, "usage": WebGPUUsage.uniform],
                // Put a buffer handle where a texture view belongs.
                ["op": "beginRenderPass", "colorAttachments": [["view": 1]]],
            ])
            guard errors.contains(where: { ($0["kind"] as? String) == "validation" }) else {
                throw ConformanceFailure("not a validation error — \(errors)")
            }
        }
    }

    // MARK: - Query APIs

    /// `adapter.limits` comes out **with the spec spelling exactly**.
    ///
    /// Web libraries read these names to set their own budgets — naming them our own way makes them
    /// see `undefined` and build wrong assumptions.
    static var adapterLimitsSpelling: Check {
        Check("adapter-limits-spelling") { harness in
            let info = harness.runtime.adapterInfo()
            guard let limits = info["limits"] as? [String: Any] else {
                throw ConformanceFailure("adapterInfo has no limits")
            }
            let required = [
                "maxTextureDimension2D", "maxBindGroups", "maxBufferSize",
                "maxUniformBufferBindingSize", "maxStorageBufferBindingSize",
                "minUniformBufferOffsetAlignment", "minStorageBufferOffsetAlignment",
                "maxVertexBuffers", "maxVertexAttributes", "maxColorAttachments",
                "maxComputeWorkgroupSizeX", "maxComputeWorkgroupsPerDimension",
            ]
            let missing = required.filter { limits[$0] == nil }
            guard missing.isEmpty else {
                throw ConformanceFailure("spec entries missing from limits — \(missing.joined(separator: ", "))")
            }
            guard (info["preferredCanvasFormat"] as? String) != nil else {
                throw ConformanceFailure("preferredCanvasFormat is absent")
            }
            guard let adapter = info["info"] as? [String: Any],
                  adapter["vendor"] is String, adapter["architecture"] is String,
                  adapter["description"] is String else {
                throw ConformanceFailure("not the shape of GPUAdapterInfo — \(info["info"] ?? "absent")")
            }
        }
    }

    /// `canvasInfo` reports the configured size and format.
    static var canvasInfoReporting: Check {
        Check("canvas-info") { harness in
            try harness.executeExpectingSuccess([
                ["op": "configureCanvas", "canvas": harness.canvas, "format": "rgba8unorm"],
            ])
            let info = harness.runtime.canvasInfo(identifier: harness.canvas)
            guard (info["ok"] as? Bool) == true else {
                throw ConformanceFailure("canvasInfo failed — \(ConformanceHarness.describeErrors(info))")
            }
            guard (info["width"] as? Int) == harness.width,
                  (info["height"] as? Int) == harness.height else {
                throw ConformanceFailure(
                    "size was \(info["width"] ?? "?")×\(info["height"] ?? "?") — "
                        + "expected \(harness.width)×\(harness.height)"
                )
            }
            guard (info["format"] as? String) == "rgba8unorm" else {
                throw ConformanceFailure("format was \(info["format"] ?? "absent") — expected rgba8unorm")
            }
            // A missing canvas must answer with an error (not a crash).
            guard (harness.runtime.canvasInfo(identifier: "no-such-canvas")["ok"] as? Bool) == false else {
                throw ConformanceFailure("asked about a missing canvas but got ok: true")
            }
        }
    }

    // MARK: - Shader fragments

    static let vertexColorShader = """
    struct VertexOutput {
        @builtin(position) position: vec4f,
        @location(0) color: vec3f,
    };

    @vertex
    fn vs_main(@location(0) position: vec2f, @location(1) color: vec3f) -> VertexOutput {
        var out: VertexOutput;
        out.position = vec4f(position, 0.0, 1.0);
        out.color = color;
        return out;
    }

    @fragment
    fn fs_main(in: VertexOutput) -> @location(0) vec4f {
        return vec4f(in.color, 1.0);
    }
    """

    static let positionOnlyShader = """
    @vertex
    fn vs_main(@location(0) position: vec2f) -> @builtin(position) vec4f {
        return vec4f(position, 0.0, 1.0);
    }

    @fragment
    fn fs_main() -> @location(0) vec4f {
        return vec4f(1.0, 1.0, 0.0, 1.0);
    }
    """

    static let tintShader = """
    struct Tint { color: vec4f };
    @group(0) @binding(0) var<uniform> tint: Tint;

    @vertex
    fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
        var positions = array<vec2f, 3>(
            vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0),
        );
        return vec4f(positions[index], 0.0, 1.0);
    }

    @fragment
    fn fs_main() -> @location(0) vec4f {
        return tint.color;
    }
    """

    static func depthShader(z: Float, color: String) -> String {
        """
        @vertex
        fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
            var positions = array<vec2f, 3>(
                vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0),
            );
            return vec4f(positions[index], \(z), 1.0);
        }

        @fragment
        fn fs_main() -> @location(0) vec4f {
            return \(color);
        }
        """
    }
}

// MARK: - Stream fragments

extension ConformanceHarness {
    /// Configures the canvas and obtains this batch's drawable texture (10) and view (11).
    ///
    /// Drawable handles are **frame-scoped**, so they must be obtained again in every batch
    /// (`docs/COMMAND-STREAM.md` §4-5).
    var canvasSetup: [[String: Any]] {
        [
            ["op": "configureCanvas", "canvas": canvas, "format": "rgba8unorm"],
            ["op": "getCurrentTexture", "id": 10, "canvas": canvas],
            ["op": "createTextureView", "id": 11, "texture": 10],
        ]
    }

    /// A shader plus pipeline drawing one full-screen triangle.
    func fullscreenPipeline(module: Int, pipeline: Int) -> [[String: Any]] {
        [
            ["op": "createShaderModule", "id": module, "code": WebGPUConformance.fullscreenShader],
            ["op": "createRenderPipeline", "id": pipeline, "layout": "auto",
             "vertex": ["module": module, "entryPoint": "vs_main"],
             "fragment": ["module": module, "entryPoint": "fs_main",
                          "targets": [["format": "rgba8unorm"]]]],
        ]
    }

    /// A render pass drawing one quad (shared by the direct and indexed paths).
    func quadPass(
        pipeline: Int,
        vertexBuffer: Int,
        before: [[String: Any]] = [],
        drawing draw: [String: Any]
    ) -> [[String: Any]] {
        [
            ["op": "beginRenderPass", "colorAttachments": [[
                "view": 11, "loadOp": "clear", "storeOp": "store",
                "clearValue": ["r": 0, "g": 0, "b": 1, "a": 1],
            ]]],
            ["op": "setPipeline", "pipeline": pipeline],
            ["op": "setVertexBuffer", "slot": 0, "buffer": vertexBuffer],
        ] + before + [draw, ["op": "endPass"]]
    }
}
