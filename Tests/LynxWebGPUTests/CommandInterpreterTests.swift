import XCTest
import Metal
import LynxWebGPUCore
@testable import LynxWebGPU

/// 커맨드 해석기의 계약 — **오류가 나도 프로세스를 죽이지 않고 모아서 돌려준다**.
final class CommandInterpreterTests: XCTestCase {
    private var harness: RenderHarness!

    override func setUpWithError() throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "Metal 디바이스 없음")
        harness = try XCTUnwrap(RenderHarness.make(width: 8, height: 8))
    }

    override func tearDown() {
        harness = nil
        super.tearDown()
    }

    private func errors(_ result: [String: Any]) -> [[String: Any]] {
        result["errors"] as? [[String: Any]] ?? []
    }

    func test_알수없는_명령은_unsupported_오류로_보고된다() {
        let result = harness.execute([["op": "teleport"]])

        XCTAssertEqual(result["ok"] as? Bool, false)
        XCTAssertEqual(errors(result).first?["kind"] as? String, "unsupported")
        XCTAssertEqual(errors(result).first?["path"] as? String, "commands[0].op")
    }

    func test_없는_핸들을_참조하면_validation_오류다() {
        let result = harness.execute([
            ["op": "setVertexBuffer", "slot": 0, "buffer": 999],
        ])

        XCTAssertEqual(result["ok"] as? Bool, false)
        XCTAssertEqual(errors(result).first?["kind"] as? String, "validation")
    }

    func test_오류가_나도_뒤의_명령을_계속_실행한다() {
        let result = harness.execute([
            ["op": "teleport"],
            ["op": "createBuffer", "id": 1, "size": 16, "usage": TestUsage.uniform],
            ["op": "nonsense"],
        ])

        // 오류 2개가 모두 보고되고, 사이의 정상 명령은 실행된다.
        XCTAssertEqual(errors(result).count, 2)
        XCTAssertEqual(harness.context.liveObjectCount, 1)
    }

    func test_패스없이_draw하면_오류다() {
        let result = harness.execute([["op": "draw", "vertexCount": 3]])
        XCTAssertEqual(errors(result).first?["kind"] as? String, "validation")
        XCTAssertTrue(
            ((errors(result).first?["message"] as? String) ?? "").contains("beginRenderPass"),
            "무엇을 먼저 해야 하는지 알려 줘야 한다"
        )
    }

    func test_등록되지_않은_캔버스는_등록된_목록을_알려준다() {
        let result = harness.execute([["op": "configureCanvas", "canvas": "없는캔버스"]])
        let message = (errors(result).first?["message"] as? String) ?? ""
        XCTAssertTrue(message.contains("없는캔버스"))
    }

    func test_shader_컴파일_실패는_생성된_MSL을_함께_보고한다() {
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
        XCTAssertTrue(message.contains("MSL"), "생성된 MSL이 진단에 포함돼야 한다: \(message)")
    }

    func test_드로어블_텍스처_핸들은_프레임이_끝나면_회수된다() {
        harness.executeExpectingSuccess([
            ["op": "configureCanvas", "canvas": "test", "format": "rgba8unorm"],
        ])
        let before = harness.context.liveObjectCount

        harness.executeExpectingSuccess([
            ["op": "getCurrentTexture", "id": 50, "canvas": "test"],
            ["op": "createTextureView", "id": 51, "texture": 50],
            ["op": "beginRenderPass", "colorAttachments": [[
                "view": 51, "loadOp": "clear", "storeOp": "store",
                "clearValue": ["r": 0, "g": 0, "b": 0, "a": 1],
            ]]],
            ["op": "endPass"],
        ])

        // 스왑체인 텍스처와 그 뷰는 프레임 밖에서 유효하지 않다 (브라우저와 같은 규칙).
        XCTAssertEqual(harness.context.liveObjectCount, before)
    }

    func test_버퍼_쓰기와_복사와_읽기가_순서대로_동작한다() throws {
        let source: [Float] = [1, 2, 3, 4]
        harness.executeExpectingSuccess([
            ["op": "createBuffer", "id": 1, "size": 16, "usage": TestUsage.copySrc | TestUsage.copyDst],
            ["op": "createBuffer", "id": 2, "size": 16, "usage": TestUsage.copyDst | TestUsage.mapRead],
            ["op": "writeBuffer", "buffer": 1, "data": source.base64],
            ["op": "copyBufferToBuffer", "source": 1, "sourceOffset": 0,
             "destination": 2, "destinationOffset": 0, "size": 16],
        ])

        let expectation = expectation(description: "readBuffer")
        var output: [Float] = []
        harness.context.readBuffer(handle: 2) { result in
            if let base64 = result["data"] as? String, let data = Data(base64Encoded: base64) {
                output = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            }
            expectation.fulfill()
        }
        wait(for: [expectation], timeout: 10)
        XCTAssertEqual(output, source)
    }

    func test_범위를_벗어난_writeBuffer는_거부된다() {
        let result = harness.execute([
            ["op": "createBuffer", "id": 1, "size": 8, "usage": TestUsage.copyDst],
            ["op": "writeBuffer", "buffer": 1, "bufferOffset": 4,
             "data": [Float](repeating: 0, count: 4).base64],
        ])
        XCTAssertTrue(((errors(result).first?["message"] as? String) ?? "").contains("범위 초과"))
    }

    func test_reset은_모든_객체를_버린다() {
        harness.executeExpectingSuccess([
            ["op": "createBuffer", "id": 1, "size": 16, "usage": TestUsage.uniform],
            ["op": "createBuffer", "id": 2, "size": 16, "usage": TestUsage.uniform],
        ])
        XCTAssertEqual(harness.context.liveObjectCount, 2)

        harness.context.reset()
        XCTAssertEqual(harness.context.liveObjectCount, 0)
    }

    func test_어댑터_정보가_한계값을_보고한다() {
        let info = harness.context.adapterInfo()

        XCTAssertEqual(info["ok"] as? Bool, true)
        XCTAssertEqual(info["backend"] as? String, "metal")
        XCTAssertEqual(info["preferredCanvasFormat"] as? String, "bgra8unorm")
        let limits = try? XCTUnwrap(info["limits"] as? [String: Any])
        XCTAssertNotNil(limits?["maxVertexBuffers"])
    }
}
