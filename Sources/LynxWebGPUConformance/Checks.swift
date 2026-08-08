import Foundation
import LynxWebGPUCore

/// 스위트에 담긴 검사들.
///
/// 각 검사는 **계약의 한 자리**를 본다. 새 op을 더하면 여기 검사를 하나 더하는 것이
/// `docs/COMMAND-STREAM.md` §7의 마지막 단계다.
public extension WebGPUConformance {

    /// 화면 전체를 덮는 삼각형 — 정점 버퍼 없이 `vertex_index`만으로.
    /// 프래그먼트 출력이 색 하나로 고정되므로 픽셀 단언이 단순해진다.
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
            // 프레임 수명·콜백 API (LifecycleChecks.swift)
            presentFalsePreservesFrame,
            presentExpiresFrame,
            emptyPresentClosesFrame,
            readBufferContract,
            resizeCanvasReflects,
            shaderCompilationInfoShape,
            mslOptional,
            decodeImageUpload,
            frameReadiness,
        ]
    }

    // MARK: - 렌더 기초

    /// 클리어 색이 그대로 나온다 — 렌더 패스가 열리고 닫히는 가장 짧은 경로.
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

    /// WGSL → 파이프라인 → 드로우가 실제로 픽셀을 바꾼다.
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
            try harness.expectPixel(x: 32, y: 32, equals: (255, 0, 0, 255), "드로우 결과")
        }
    }

    /// 정점 버퍼의 위치·색이 래스터화에 반영된다 (안/밖이 갈린다).
    static var vertexBuffer: Check {
        Check("vertex-buffer") { harness in
            // 위치(x,y) + 색(r,g,b) 인터리브 — stride 20B
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
            try harness.expectPixel(x: 32, y: 32, equals: (255, 0, 0, 255), "삼각형 내부")
            try harness.expectPixel(x: 1, y: 1, equals: (0, 0, 255, 255), "삼각형 외부(클리어색)")
        }
    }

    /// 유니폼 버퍼가 프래그먼트 출력에 닿는다 — 바인드 그룹 배정이 셰이더와 맞는지 본다.
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
            try harness.expectPixel(x: 32, y: 32, equals: (0, 255, 0, 255), "유니폼 색")
        }
    }

    /// `drawIndexed`가 같은 삼각형을 직접 드로우와 **같은 픽셀로** 그린다.
    static var indexBufferEquivalence: Check {
        Check("index-buffer-equivalence") { harness in
            // 사각형 네 꼭짓점. 직접 경로는 6개를 늘어놓고, 인덱스 경로는 4개 + 인덱스 6개다.
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
            try harness.expectFrameEquals(direct, "drawIndexed가 직접 드로우와 다른 그림을 냈다")
        }
    }

    // MARK: - 컴퓨트 · 버퍼

    /// 컴퓨트 패스가 스토리지 버퍼에 쓰고, 그 값을 되읽을 수 있다.
    static var computeStorageReadback: Check {
        Check("compute-storage-readback") { harness in
            let shader = """
            @group(0) @binding(0) var<storage, read_write> data: array<u32>;

            @compute @workgroup_size(4)
            fn main(@builtin(global_invocation_id) id: vec3u) {
                data[id.x] = id.x * 2u;
            }
            """
            // 명세는 `MAP_READ`를 `COPY_DST` 외의 usage와 함께 쓰지 못하게 한다 — 스토리지
            // 버퍼를 직접 매핑할 수 없으므로 리드백 버퍼로 한 번 복사한다 (웹에서도 같은 모양이다).
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
                throw ConformanceFailure("컴퓨트 결과가 \(values) — 기대 [0, 2, 4, 6]")
            }
        }
    }

    /// `copyBufferToBuffer`가 바이트를 그대로 옮긴다.
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
                throw ConformanceFailure("복사 결과가 \(copied) — 기대 \(source)")
            }
        }
    }

    /// `clearBuffer`가 구간을 0으로 만든다 (CPU에서 0 배열을 실어 보내지 않는 경로).
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
                throw ConformanceFailure("clearBuffer 결과가 \(values) — 기대 [0, 0, 7, 7]")
            }
        }
    }

    // MARK: - 텍스처

    /// `writeTexture`로 올린 픽셀을 셰이더가 샘플한다.
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
            // 2×2 RGBA8 — 좌상단이 빨강이고, 셰이더가 그 자리를 샘플한다.
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
            try harness.expectPixel(x: 32, y: 32, equals: (255, 0, 0, 255), "좌상단 텍셀")
        }
    }

    /// 렌더 타깃을 버퍼로 내려 받는다 (`copyTextureToBuffer`).
    ///
    /// 폭을 64로 잡아 행 간격이 256B가 되게 한다 — 명세가 요구하는 정렬이다.
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
                throw ConformanceFailure("리드백 길이가 \(bytes.count)B — 기대 1024B")
            }
            guard Array(bytes.prefix(4)) == [255, 0, 0, 255],
                  Array(bytes.suffix(4)) == [255, 0, 0, 255] else {
                throw ConformanceFailure(
                    "복사된 픽셀이 다르다 — 첫 \(Array(bytes.prefix(4))), 끝 \(Array(bytes.suffix(4)))"
                )
            }
        }
    }

    // MARK: - 파이프라인 상태

    /// 알파 블렌딩 공식이 적용된다.
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
            // 0.5·빨강 + 0.5·파랑
            try harness.expectPixel(x: 32, y: 32, equals: (128, 0, 128, 255), tolerance: 3)
        }
    }

    /// 깊이 테스트가 뒤에 있는 삼각형을 가린다.
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
            try harness.expectPixel(x: 32, y: 32, equals: (255, 0, 0, 255), "앞의 삼각형이 남아야 한다")
        }
    }

    /// 렌더 번들이 **직접 인코딩과 같은 그림**을 낸다 — 번들의 계약 그 자체다.
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
            try harness.expectFrameEquals(direct, "번들 실행이 직접 인코딩과 다른 그림을 냈다")
        }
    }

    /// 간접 드로우가 직접 드로우와 **같은 그림**을 낸다.
    ///
    /// 기기가 간접 인자를 지원할 때만 돈다 — iOS 시뮬레이터는 여기서 빠진다.
    static var indirectDrawEquivalence: Check {
        Check("indirect-draw-equivalence", requiresFeature: "indirect-first-instance") { harness in
            // vertexCount, instanceCount, firstVertex, firstInstance — u32 4개
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
            try harness.expectFrameEquals(direct, "간접 드로우가 직접 드로우와 다른 그림을 냈다")
        }
    }

    // MARK: - 오류 계약

    /// **오류 하나가 배치를 죽이지 않는다** — 앞선 명령의 결과는 그대로 남는다.
    static var errorAccumulation: Check {
        Check("error-accumulation") { harness in
            let result = harness.execute(
                harness.canvasSetup + [
                    ["op": "beginRenderPass", "colorAttachments": [[
                        "view": 11, "loadOp": "clear", "storeOp": "store",
                        "clearValue": ["r": 0, "g": 1, "b": 0, "a": 1],
                    ]]],
                    ["op": "endPass"],
                    // 패스 밖의 setPipeline — 유효하지 않다.
                    ["op": "setPipeline", "pipeline": 999],
                ]
            )
            guard (result["ok"] as? Bool) == false else {
                throw ConformanceFailure("잘못된 명령이 있는데 ok: true로 돌아왔다")
            }
            guard let errors = result["errors"] as? [[String: Any]], !errors.isEmpty else {
                throw ConformanceFailure("오류 목록이 비어 있다")
            }
            guard errors.contains(where: { ($0["path"] as? String)?.hasPrefix("commands[") == true }) else {
                throw ConformanceFailure("오류에 커맨드 스트림 경로가 붙지 않았다 — \(errors)")
            }
            // 오류 앞의 클리어는 살아 있어야 한다.
            try harness.expectPixel(x: 32, y: 32, equals: (0, 255, 0, 255), "오류 앞의 클리어")
        }
    }

    /// 오류 스코프가 오류를 **가로챈다** — 배치 결과에는 실리지 않는다.
    static var errorScopeCapture: Check {
        Check("error-scope-capture") { harness in
            let result = harness.execute([
                ["op": "pushErrorScope", "filter": "validation"],
                ["op": "setPipeline", "pipeline": 1],
                ["op": "popErrorScope"],
            ])
            guard (result["ok"] as? Bool) == true else {
                throw ConformanceFailure(
                    "스코프가 잡은 오류가 배치 결과에도 실렸다 — \(ConformanceHarness.describeErrors(result))"
                )
            }
            guard let scopes = result["errorScopes"] as? [Any], scopes.count == 1 else {
                throw ConformanceFailure("errorScopes가 \(result["errorScopes"] ?? "없음") — 1개를 기대")
            }
            guard let captured = scopes[0] as? [String: Any],
                  (captured["kind"] as? String) == "validation" else {
                throw ConformanceFailure("스코프가 validation 오류를 잡지 못했다 — \(scopes[0])")
            }
        }
    }

    /// 핸들 타입이 어긋나면 **크래시가 아니라 검증 오류**다 (WebGPU는 안전한 API다).
    static var handleTypeMismatch: Check {
        Check("handle-type-mismatch") { harness in
            let errors = try harness.executeExpectingFailure([
                ["op": "createBuffer", "id": 1, "size": 16, "usage": WebGPUUsage.uniform],
                // 버퍼 핸들을 텍스처 뷰 자리에 넣는다.
                ["op": "beginRenderPass", "colorAttachments": [["view": 1]]],
            ])
            guard errors.contains(where: { ($0["kind"] as? String) == "validation" }) else {
                throw ConformanceFailure("validation 오류가 아니다 — \(errors)")
            }
        }
    }

    // MARK: - 조회 API

    /// `adapter.limits`가 **명세 철자 그대로** 나온다.
    ///
    /// 웹 라이브러리가 이 이름으로 읽고 자기 예산을 정한다 — 우리 식으로 이름을 지으면
    /// `undefined`를 보고 잘못된 가정을 세운다.
    static var adapterLimitsSpelling: Check {
        Check("adapter-limits-spelling") { harness in
            let info = harness.runtime.adapterInfo()
            guard let limits = info["limits"] as? [String: Any] else {
                throw ConformanceFailure("adapterInfo에 limits가 없다")
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
                throw ConformanceFailure("limits에 없는 명세 항목 — \(missing.joined(separator: ", "))")
            }
            guard (info["preferredCanvasFormat"] as? String) != nil else {
                throw ConformanceFailure("preferredCanvasFormat이 없다")
            }
            guard let adapter = info["info"] as? [String: Any],
                  adapter["vendor"] is String, adapter["architecture"] is String,
                  adapter["description"] is String else {
                throw ConformanceFailure("GPUAdapterInfo 모양이 아니다 — \(info["info"] ?? "없음")")
            }
        }
    }

    /// `canvasInfo`가 설정한 크기·포맷을 보고한다.
    static var canvasInfoReporting: Check {
        Check("canvas-info") { harness in
            try harness.executeExpectingSuccess([
                ["op": "configureCanvas", "canvas": harness.canvas, "format": "rgba8unorm"],
            ])
            let info = harness.runtime.canvasInfo(identifier: harness.canvas)
            guard (info["ok"] as? Bool) == true else {
                throw ConformanceFailure("canvasInfo 실패 — \(ConformanceHarness.describeErrors(info))")
            }
            guard (info["width"] as? Int) == harness.width,
                  (info["height"] as? Int) == harness.height else {
                throw ConformanceFailure(
                    "크기가 \(info["width"] ?? "?")×\(info["height"] ?? "?") — "
                        + "기대 \(harness.width)×\(harness.height)"
                )
            }
            guard (info["format"] as? String) == "rgba8unorm" else {
                throw ConformanceFailure("포맷이 \(info["format"] ?? "없음") — 기대 rgba8unorm")
            }
            // 없는 캔버스는 오류로 답해야 한다 (크래시가 아니라).
            guard (harness.runtime.canvasInfo(identifier: "없는-캔버스")["ok"] as? Bool) == false else {
                throw ConformanceFailure("없는 캔버스를 물었는데 ok: true로 답했다")
            }
        }
    }

    // MARK: - 셰이더 조각

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

// MARK: - 스트림 조각

extension ConformanceHarness {
    /// 캔버스를 설정하고 이번 배치의 드로어블 텍스처(10)와 뷰(11)를 얻는다.
    ///
    /// 드로어블 핸들은 **프레임 스코프**라 배치마다 다시 얻어야 한다
    /// (`docs/COMMAND-STREAM.md` §4-5).
    var canvasSetup: [[String: Any]] {
        [
            ["op": "configureCanvas", "canvas": canvas, "format": "rgba8unorm"],
            ["op": "getCurrentTexture", "id": 10, "canvas": canvas],
            ["op": "createTextureView", "id": 11, "texture": 10],
        ]
    }

    /// 화면 전체를 덮는 삼각형 하나를 그리는 셰이더 + 파이프라인.
    func fullscreenPipeline(module: Int, pipeline: Int) -> [[String: Any]] {
        [
            ["op": "createShaderModule", "id": module, "code": WebGPUConformance.fullscreenShader],
            ["op": "createRenderPipeline", "id": pipeline, "layout": "auto",
             "vertex": ["module": module, "entryPoint": "vs_main"],
             "fragment": ["module": module, "entryPoint": "fs_main",
                          "targets": [["format": "rgba8unorm"]]]],
        ]
    }

    /// 사각형 하나를 그리는 렌더 패스 (직접/인덱스 경로가 공유한다).
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
