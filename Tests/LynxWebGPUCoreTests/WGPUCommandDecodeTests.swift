import XCTest
@testable import LynxWebGPUCore

/// `WGPUCommand` 디코더 표의 완결성 — GPU 없이 돈다.
///
/// `opName`이 exhaustive switch라 케이스 누락은 컴파일이 잡지만, **디코더 표(문자열 → 케이스)의
/// 누락은 런타임에만 드러난다** (`default`의 unsupported로 샌다). 여기의 픽스처 전수 왕복이
/// 그 구멍을 막는다 — op을 더할 때는 픽스처도 한 줄 더한다 (`docs/COMMAND-STREAM.md` §7).
final class WGPUCommandDecodeTests: XCTestCase {

    /// op 이름 → 디코딩에 성공하는 최소 페이로드 (op 필드는 테스트가 채운다).
    private static let fixtures: [String: [String: Any]] = [
        // 리소스
        "createBuffer": ["id": 1, "size": 16, "usage": 0x0020],
        "writeBuffer": ["buffer": 1, "data": [0, 1, 2, 3]],
        "unmapBuffer": ["buffer": 1],
        "createTexture": [
            "id": 1, "size": ["width": 4, "height": 4], "format": "rgba8unorm", "usage": 0x10,
        ],
        "writeTexture": ["texture": 1, "data": [0, 0, 0, 0], "size": ["width": 1, "height": 1]],
        "copyExternalImageToTexture": [
            "source": ["source": 1], "destination": ["texture": 2],
        ],
        "createTextureView": ["id": 2, "texture": 1],
        "createSampler": ["id": 1],
        "createShaderModule": ["id": 1, "code": "@vertex fn v() {}"],
        "createBindGroupLayout": ["id": 1, "entries": [[String: Any]]()],
        "createPipelineLayout": ["id": 1, "bindGroupLayouts": [Int]()],
        "createBindGroup": ["id": 2, "layout": 1, "entries": [[String: Any]]()],
        "createQuerySet": ["id": 1, "type": "occlusion", "count": 4],
        "createRenderBundle": [
            "id": 1, "commands": [[String: Any]](), "colorFormats": ["bgra8unorm"],
        ],
        "createRenderPipeline": ["id": 2, "vertex": ["module": 1]],
        "createComputePipeline": ["id": 2, "compute": ["module": 1]],
        "getBindGroupLayout": ["id": 3, "pipeline": 2, "index": 0],
        "destroy": ["id": 1],
        // 오류 스코프
        "pushErrorScope": ["filter": "validation"],
        "popErrorScope": [:],
        // 캔버스
        "configureCanvas": ["canvas": "main"],
        "getCurrentTexture": ["id": 1, "canvas": "main"],
        // 렌더 패스
        "beginRenderPass": [
            "colorAttachments": [["view": 1, "loadOp": "clear", "storeOp": "store"]],
        ],
        "setPipeline": ["pipeline": 1],
        "setBindGroup": ["index": 0, "bindGroup": 1],
        "setVertexBuffer": ["slot": 0, "buffer": 1],
        "setIndexBuffer": ["buffer": 1, "format": "uint16"],
        "setViewport": ["width": 1, "height": 1],
        "setScissorRect": ["width": 1, "height": 1],
        "setBlendConstant": [:],
        "setStencilReference": [:],
        "draw": ["vertexCount": 3],
        "drawIndexed": ["indexCount": 3],
        "drawIndirect": ["indirectBuffer": 1],
        "drawIndexedIndirect": ["indirectBuffer": 1],
        "executeBundles": ["bundles": [Int]()],
        "beginOcclusionQuery": ["queryIndex": 0],
        "endOcclusionQuery": [:],
        // 컴퓨트 패스
        "beginComputePass": [:],
        "dispatchWorkgroups": [:],
        "dispatchWorkgroupsIndirect": ["indirectBuffer": 1],
        "endPass": [:],
        // 복사
        "copyBufferToBuffer": ["source": 1, "destination": 2],
        "clearBuffer": ["buffer": 1],
        "copyTextureToBuffer": [
            "source": ["texture": 1], "destination": ["buffer": 2],
            "copySize": ["width": 1, "height": 1],
        ],
        "copyBufferToTexture": [
            "source": ["buffer": 1], "destination": ["texture": 2],
            "copySize": ["width": 1, "height": 1],
        ],
        "copyTextureToTexture": [
            "source": ["texture": 1], "destination": ["texture": 2],
            "copySize": ["width": 1, "height": 1],
        ],
        // 쿼리
        "resolveQuerySet": ["querySet": 1, "destination": 2],
        // 디버그 마커
        "pushDebugGroup": ["groupLabel": "구간"],
        "popDebugGroup": [:],
        "insertDebugMarker": ["markerLabel": "표식"],
    ]

    func test_op_전수가_디코딩되고_opName이_왕복한다() throws {
        // 픽스처 자체의 누락도 잡는다 — 케이스를 더하면 이 수를 함께 올린다.
        XCTAssertEqual(Self.fixtures.count, 51, "op 픽스처 수가 케이스 수와 다르다")

        for (op, fields) in Self.fixtures {
            var payload = fields
            payload["op"] = op
            let decoded: WGPUCommand
            do {
                decoded = try WGPUCommand(from: WGPUValueReader(payload))
            } catch {
                XCTFail("'\(op)' 디코딩 실패: \(error)")
                continue
            }
            XCTAssertEqual(decoded.opName, op, "'\(op)'이(가) 다른 케이스로 디코딩됐다")
        }
    }

    func test_anUnknownOpIsUnsupportedWithAPath() throws {
        // 실전과 같은 리더 경로를 만든다 — execute가 requiredObjects("commands")로 쪼개는 형태.
        let commands = try WGPUValueReader(["commands": [["op": "teleport"]]])
            .requiredObjects("commands")
        XCTAssertThrowsError(try WGPUCommand(from: commands[0])) { error in
            guard let error = error as? WGPUError else { return XCTFail("WGPUError가 아니다") }
            XCTAssertEqual(error.kind, .unsupported)
            XCTAssertEqual(error.path, "commands[0].op")
        }
    }

    func test_pushErrorScope는_필터가_깨져도_케이스가_생긴다() throws {
        let decoded = try WGPUCommand(from: WGPUValueReader([
            "op": "pushErrorScope", "filter": "warp-core",
        ]))
        guard case .pushErrorScope(let filter, let decodeFailure) = decoded else {
            return XCTFail("pushErrorScope 케이스가 아니다: \(decoded.opName)")
        }
        // 자리표시자 + 실패 동반 — 백엔드는 push부터 하고 실패를 던진다 (깊이 유지 계약).
        XCTAssertNil(filter)
        XCTAssertEqual(decodeFailure?.kind, .validation)
    }

    func test_createRenderBundle은_명령_리더를_그대로_운반한다() throws {
        let decoded = try WGPUCommand(from: WGPUValueReader([
            "op": "createRenderBundle", "id": 1,
            "commands": [["op": "draw", "vertexCount": 3]],
            "colorFormats": ["bgra8unorm"],
        ]))
        guard case .createRenderBundle(let command) = decoded else {
            return XCTFail("createRenderBundle 케이스가 아니다")
        }
        // 재생 시점 디코딩 계약 — 저장된 리더를 다시 이 이니셜라이저에 넣으면 명령이 나온다.
        XCTAssertEqual(command.commands.count, 1)
        XCTAssertEqual(try WGPUCommand(from: command.commands[0]).opName, "draw")
    }
}
