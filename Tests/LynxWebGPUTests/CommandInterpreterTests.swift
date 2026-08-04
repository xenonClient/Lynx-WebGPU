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

    /// 프레임의 경계는 **배치가 아니라 present**다.
    ///
    /// `popErrorScope`·`mapAsync`는 결과를 받으려고 프레임 중간에 제출한다. 배치가 끝날 때마다
    /// 프레임 스코프를 닫으면 그 지점에서 스왑체인 핸들이 지워져, 이어지는 `beginRenderPass`가
    /// "없는 핸들"로 깨진다 — 그 프레임이 통째로 날아간다.
    func test_프레임_중간에_제출해도_스왑체인_핸들이_살아_있다() {
        harness.executeExpectingSuccess([
            ["op": "configureCanvas", "canvas": "test", "format": "rgba8unorm"],
        ])
        let before = harness.context.liveObjectCount

        // 배치 ①: 드로어블만 얻고 끝난다 (mid-frame flush가 만드는 상황).
        harness.executeExpectingSuccess([
            ["op": "getCurrentTexture", "id": 50, "canvas": "test"],
            ["op": "createTextureView", "id": 51, "texture": 50],
        ])
        XCTAssertEqual(
            harness.context.liveObjectCount, before + 2,
            "아직 present하지 않았으므로 핸들이 살아 있어야 한다"
        )

        // 배치 ②: 앞 배치에서 얻은 뷰로 실제로 그린다.
        harness.executeExpectingSuccess([
            ["op": "beginRenderPass", "colorAttachments": [[
                "view": 51, "loadOp": "clear", "storeOp": "store",
                "clearValue": ["r": 0, "g": 0, "b": 0, "a": 1],
            ]]],
            ["op": "endPass"],
        ])

        XCTAssertEqual(harness.context.liveObjectCount, before, "present했으니 이제 회수된다")
    }

    /// 프레임 중간 배치가 **커맨드 버퍼를 만들어도**(writeBuffer 등) 스왑체인이 살아남아야 한다.
    ///
    /// Three.js의 지연 파이프라인 생성이 정확히 이 모양이다 — 드로어블을 획득해 둔 채로
    /// 유니폼 writeBuffer + popErrorScope 즉시 flush. shim은 이런 내부 제출에 `present: false`를
    /// 실어 보내고, 해석기는 커밋만 하고 present·핸들 만료를 진짜 프레임 제출까지 미룬다.
    /// 이 구분이 없으면 그리지도 않은 드로어블이 present되고, 뒤따르는 출력 패스가
    /// "GPUTextureView가 존재하지 않는다"로 통째로 거부된다.
    func test_present_false_배치는_커맨드버퍼가_있어도_드로어블을_유지한다() {
        harness.executeExpectingSuccess([
            ["op": "configureCanvas", "canvas": "test", "format": "rgba8unorm"],
        ])
        let before = harness.context.liveObjectCount

        // 배치 ①: 드로어블 획득 + writeBuffer(블릿 인코더 → 커맨드 버퍼 생성) — 내부 제출.
        let midFrame = harness.context.execute([
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
            harness.context.liveObjectCount, before + 3,
            "내부 제출에서는 프레임 스코프 핸들이 만료되면 안 된다"
        )

        // 배치 ②: 진짜 프레임 제출 — 앞 배치에서 얻은 뷰로 그린다. present는 여기서 일어난다.
        harness.executeExpectingSuccess([
            ["op": "beginRenderPass", "colorAttachments": [[
                "view": 51, "loadOp": "clear", "storeOp": "store",
                "clearValue": ["r": 0, "g": 1, "b": 0, "a": 1],
            ]]],
            ["op": "endPass"],
        ])
        XCTAssertEqual(
            harness.context.liveObjectCount, before + 1,
            "present 후에는 프레임 스코프 핸들만 회수된다 (버퍼 52는 남는다)"
        )
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

        XCTAssertEqual(try harness.readBufferSync(handle: 2, as: Float.self), source)
    }

    // MARK: - 버퍼 매핑 상태

    /// 명세는 `mapAsync`가 버퍼를 "unavailable"로 만들어 큐 작업에 못 쓰게 해 경쟁 자체를 없앤다.
    /// 이 구현은 `.storageModeShared` 버퍼를 스테이징 없이 읽으므로, 이 검사가 없으면 리드백이
    /// GPU 완료를 기다리는 동안 다음 프레임의 쓰기가 같은 메모리에 겹친다.
    func test_매핑_중인_버퍼는_큐_작업에_쓸_수_없다() throws {
        harness.executeExpectingSuccess([
            ["op": "createBuffer", "id": 1, "size": 16, "usage": TestUsage.copyDst | TestUsage.mapRead],
            ["op": "createBuffer", "id": 2, "size": 16, "usage": TestUsage.copySrc],
        ])
        _ = try harness.readBufferSync(handle: 1)   // 여기서 매핑된다

        for command in [
            ["op": "writeBuffer", "buffer": 1, "data": [Float]([1, 2, 3, 4]).base64],
            ["op": "copyBufferToBuffer", "source": 2, "destination": 1, "size": 16],
            ["op": "copyBufferToBuffer", "source": 1, "destination": 2, "size": 16],
        ] as [[String: Any]] {
            let result = harness.execute([command])
            XCTAssertTrue(
                ((errors(result).first?["message"] as? String) ?? "").contains("매핑 중인"),
                "\(command["op"] ?? "?")이(가) 통과했다: \(harness.describeErrors(result))"
            )
        }

        // unmap하면 다시 쓸 수 있어야 한다.
        harness.executeExpectingSuccess([
            ["op": "unmapBuffer", "buffer": 1],
            ["op": "writeBuffer", "buffer": 1, "data": [Float]([1, 2, 3, 4]).base64],
        ])
    }

    func test_이미_매핑된_버퍼를_다시_읽으면_거부한다() throws {
        harness.executeExpectingSuccess([
            ["op": "createBuffer", "id": 1, "size": 16, "usage": TestUsage.copyDst | TestUsage.mapRead],
        ])
        _ = try harness.readBufferSync(handle: 1)

        XCTAssertThrowsError(try harness.readBufferSync(handle: 1), "두 번째 매핑은 거부된다")
    }

    /// 명세는 `MAP_READ`를 `COPY_DST`와만, `MAP_WRITE`를 `COPY_SRC`와만 조합하게 한다.
    /// Metal은 `.storageModeShared` 하나로 전부 되지만, 안 막으면 브라우저에서만 깨진다.
    func test_매핑_usage는_복사와만_조합할_수_있다() {
        for usage in [
            TestUsage.mapRead | TestUsage.queryResolve,
            TestUsage.mapRead | TestUsage.copySrc,
            TestUsage.mapRead | TestUsage.storage,
        ] {
            let result = harness.execute([["op": "createBuffer", "id": 1, "size": 16, "usage": usage]])
            XCTAssertTrue(
                ((errors(result).first?["message"] as? String) ?? "").contains("MAP_READ"),
                "usage \(usage)가 통과했다: \(harness.describeErrors(result))"
            )
        }
        harness.executeExpectingSuccess([
            ["op": "createBuffer", "id": 1, "size": 16, "usage": TestUsage.mapRead | TestUsage.copyDst],
            ["op": "createBuffer", "id": 2, "size": 16, "usage": TestUsage.mapRead],
        ])
    }

    func test_범위를_벗어난_writeBuffer는_거부된다() {
        let result = harness.execute([
            ["op": "createBuffer", "id": 1, "size": 8, "usage": TestUsage.copyDst],
            ["op": "writeBuffer", "buffer": 1, "bufferOffset": 4,
             "data": [Float](repeating: 0, count: 4).base64],
        ])
        XCTAssertTrue(((errors(result).first?["message"] as? String) ?? "").contains("범위 초과"))
    }

    func test_실행_응답에_live_객체_수가_실린다() {
        let result = harness.executeExpectingSuccess([
            ["op": "createBuffer", "id": 1, "size": 16, "usage": TestUsage.uniform],
            ["op": "createBuffer", "id": 2, "size": 16, "usage": TestUsage.uniform],
        ])
        XCTAssertEqual(result["objects"] as? Int, 2, "destroy 누락 감시용 카운트")

        let afterDestroy = harness.executeExpectingSuccess([["op": "destroy", "id": 1]])
        XCTAssertEqual(afterDestroy["objects"] as? Int, 1)
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

    // MARK: - writeTexture 큐 순서

    /// 한 배치에서 (1) 렌더 패스가 텍스처를 빨강으로 칠하고 (2) 그 **뒤에** writeTexture가
    /// 초록을 올린다. 스트림 순서대로면 최종 내용은 초록이다 — writeTexture가 자체 커맨드
    /// 버퍼를 먼저 완주시키던 구 방식에서는 빨강이 남는다.
    func test_writeTexture는_같은_배치의_앞선_렌더패스_뒤에_실행된다() throws {
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
            ["op": "writeTexture", "texture": 1, "data": Data(green).base64EncodedString(),
             "size": ["width": 2, "height": 2], "bytesPerRow": 8],
            ["op": "createBuffer", "id": 3, "size": 16, "usage": TestUsage.copyDst | TestUsage.mapRead],
            ["op": "copyTextureToBuffer",
             "source": ["texture": 1], "destination": ["buffer": 3, "bytesPerRow": 8],
             "copySize": ["width": 2, "height": 2]],
        ])

        let bytes = Array(try harness.readBufferSync(handle: 3))
        XCTAssertEqual(bytes, green, "스트림에서 나중에 온 writeTexture가 최종 내용이어야 한다")
    }

    func test_writeTexture가_배열_텍스처_레이어를_슬라이스별로_올린다() throws {
        let red = [UInt8]([255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255])
        let blue = [UInt8]([0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255])
        harness.executeExpectingSuccess([
            ["op": "createTexture", "id": 1,
             "size": ["width": 2, "height": 2, "depthOrArrayLayers": 2], "format": "rgba8unorm",
             "usage": TestUsage.textureCopyDst | TestUsage.textureCopySrc],
            // 레이어 2장을 한 번에 — bytesPerImage(16B) 간격으로 이어 붙인 데이터.
            ["op": "writeTexture", "texture": 1, "data": Data(red + blue).base64EncodedString(),
             "size": ["width": 2, "height": 2, "depthOrArrayLayers": 2],
             "bytesPerRow": 8, "rowsPerImage": 2],
            ["op": "createBuffer", "id": 2, "size": 32, "usage": TestUsage.copyDst | TestUsage.mapRead],
            ["op": "copyTextureToBuffer",
             "source": ["texture": 1, "origin": ["x": 0, "y": 0, "z": 0]],
             "destination": ["buffer": 2, "bytesPerRow": 8, "offset": 0],
             "copySize": ["width": 2, "height": 2]],
            ["op": "copyTextureToBuffer",
             "source": ["texture": 1, "origin": ["x": 0, "y": 0, "z": 1]],
             "destination": ["buffer": 2, "bytesPerRow": 8, "offset": 16],
             "copySize": ["width": 2, "height": 2]],
        ])

        let bytes = Array(try harness.readBufferSync(handle: 2))
        XCTAssertEqual(Array(bytes.prefix(16)), red, "레이어 0")
        XCTAssertEqual(Array(bytes.suffix(16)), blue, "레이어 1")
    }

    // MARK: - 간접 드로우 계약

    /// 이 셋은 **크래시와 오류의 경계**다. 정렬·범위를 Metal까지 흘리면 검증 레이어가 단언으로
    /// 프로세스를 죽이고, `INDIRECT` usage는 Metal에 개념이 없어 아무도 봐 주지 않는다
    /// (여기서 안 막으면 브라우저에서만 깨지는 코드가 나간다).
    /// 인자 버퍼 하나 + 패스 하나. 인자 검증은 `setPipeline`보다 앞서므로 파이프라인은 없어도 된다.
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

    func test_간접_오프셋이_4의_배수가_아니면_거부한다() {
        let setup = indirectSetup(usage: TestUsage.indirect)
        let result = harness.execute(setup + [
            ["op": "drawIndirect", "indirectBuffer": 1, "indirectOffset": 2],
        ])

        XCTAssertEqual(errors(result).first?["kind"] as? String, "validation")
        XCTAssertEqual(errors(result).first?["path"] as? String, "commands[\(setup.count)].indirectOffset")
    }

    func test_간접_인자가_버퍼_범위를_넘으면_거부한다() {
        // drawIndexedIndirect는 20B를 읽는다 — offset 16 + 20 > 32.
        let setup = indirectSetup(usage: TestUsage.indirect)
        let result = harness.execute(setup + [
            ["op": "drawIndexedIndirect", "indirectBuffer": 1, "indirectOffset": 16],
        ])

        XCTAssertEqual(errors(result).first?["kind"] as? String, "validation")
        XCTAssertEqual(errors(result).first?["path"] as? String, "commands[\(setup.count)].indirectOffset")
        XCTAssertTrue(((errors(result).first?["message"] as? String) ?? "").contains("범위"))
    }

    func test_INDIRECT_usage가_없는_버퍼는_간접_디스패치에_못_쓴다() {
        let setup = indirectSetup(usage: TestUsage.storage, compute: true)
        let result = harness.execute(setup + [
            ["op": "dispatchWorkgroupsIndirect", "indirectBuffer": 1],
        ])

        XCTAssertEqual(errors(result).first?["kind"] as? String, "validation")
        XCTAssertEqual(errors(result).first?["path"] as? String, "commands[\(setup.count)].indirectBuffer")
        XCTAssertTrue(((errors(result).first?["message"] as? String) ?? "").contains("INDIRECT"))
    }

    func test_패스없이_간접_드로우하면_오류다() {
        let result = harness.execute([
            ["op": "createBuffer", "id": 1, "size": 32, "usage": TestUsage.indirect],
            ["op": "drawIndirect", "indirectBuffer": 1],
        ])
        XCTAssertTrue(
            ((errors(result).first?["message"] as? String) ?? "").contains("beginRenderPass"),
            "무엇을 먼저 해야 하는지 알려 줘야 한다"
        )
    }

    // MARK: - 셰이더 컴파일 진단

    /// 명세에서 **셰이더 모듈은 컴파일에 실패해도 만들어진다.** 핸들이 아예 없으면 이후 명령이
    /// 전부 "존재하지 않는다"로만 깨져 **진짜 원인(파싱 실패)이 화면에서 사라진다.**
    func test_파싱에_실패해도_모듈은_만들어지고_원인을_돌려준다() {
        let result = harness.execute([
            ["op": "createShaderModule", "id": 1, "code": """
             @vertex
             fn vs() -> @builtin(position) vec4f {
                 return vec4f(1.0 1.0, 1.0, 1.0);
             }
             """],
        ])

        // ① 원인이 그 자리에서 보고된다 (줄 번호까지).
        let first = errors(result).first
        XCTAssertEqual(first?["kind"] as? String, "validation")
        XCTAssertTrue(((first?["message"] as? String) ?? "").contains("파싱 실패"))
        XCTAssertEqual(first?["line"] as? Int, 3, "줄 번호가 숫자로도 실려야 편집기가 점프할 수 있다")

        // ② 그래도 모듈은 있고, 진단을 돌려준다.
        let info = harness.context.shaderCompilationInfo(handle: 1)
        XCTAssertEqual(info["ok"] as? Bool, true)
        let messages = try? XCTUnwrap(info["messages"] as? [[String: Any]])
        XCTAssertEqual(messages?.count, 1)
        XCTAssertEqual(messages?.first?["type"] as? String, "error")
        XCTAssertEqual(messages?.first?["lineNum"] as? Int, 3)
    }

    /// 깨진 모듈로 파이프라인을 만들면 **진짜 원인**을 다시 알려 줘야 한다 —
    /// "진입점이 없다"로 바꿔 말하면 사용자가 셰이더 이름을 의심하며 엉뚱한 곳을 고친다.
    func test_깨진_모듈로_파이프라인을_만들면_원인을_다시_알려_준다() {
        let result = harness.execute([
            ["op": "createShaderModule", "id": 1, "code": "fn broken( {"],
            ["op": "createRenderPipeline", "id": 2, "layout": "auto",
             "vertex": ["module": 1, "entryPoint": "vs"]],
        ])

        let messages = errors(result).compactMap { $0["message"] as? String }
        XCTAssertTrue(
            messages.contains { $0.contains("컴파일에 실패했다") },
            "파이프라인 오류가 원인을 안 담고 있다: \(messages)"
        )
    }

    func test_정상_모듈은_진단이_비어_있다() {
        harness.executeExpectingSuccess([
            ["op": "createShaderModule", "id": 1, "code": """
             @fragment fn fs() -> @location(0) vec4f { return vec4f(1.0); }
             """],
        ])

        let info = harness.context.shaderCompilationInfo(handle: 1)
        XCTAssertEqual(info["ok"] as? Bool, true)
        XCTAssertEqual((info["messages"] as? [[String: Any]])?.count, 0)
    }

    func test_없는_모듈의_진단은_오류다() {
        let info = harness.context.shaderCompilationInfo(handle: 999)
        XCTAssertEqual(info["ok"] as? Bool, false)
    }

    // MARK: - 디버그 마커

    /// 패스 안팎 모두에서 받아야 한다 — Xcode GPU 캡처의 구간 이름이 여기서 나온다.
    func test_디버그_마커를_패스_안팎에서_받는다() {
        let result = harness.execute([
            ["op": "configureCanvas", "canvas": "test", "format": "rgba8unorm"],
            ["op": "createBuffer", "id": 1, "size": 16, "usage": TestUsage.copyDst],
            // 패스 밖 — 커맨드 버퍼에 붙는다. writeBuffer가 blit 인코더를 연다.
            ["op": "pushDebugGroup", "groupLabel": "업로드"],
            ["op": "writeBuffer", "buffer": 1, "data": [Float](repeating: 1, count: 4).base64],
            ["op": "insertDebugMarker", "markerLabel": "표식"],
            ["op": "popDebugGroup"],
            // 패스 안 — 렌더 인코더에 붙는다.
            ["op": "getCurrentTexture", "id": 10, "canvas": "test"],
            ["op": "createTextureView", "id": 11, "texture": 10],
            ["op": "beginRenderPass", "colorAttachments": [[
                "view": 11, "loadOp": "clear", "storeOp": "store",
                "clearValue": ["r": 0, "g": 0, "b": 0, "a": 1],
            ]]],
            ["op": "pushDebugGroup", "groupLabel": "메인 패스"],
            ["op": "insertDebugMarker", "markerLabel": "드로우 직전"],
            ["op": "popDebugGroup"],
            ["op": "endPass"],
        ])

        XCTAssertEqual(result["ok"] as? Bool, true, harness.describeErrors(result))
    }

    /// 짝이 맞지 않는 `pop`은 **Metal이 단언으로 프로세스를 죽인다.** 깊이를 세어 막고
    /// validation 오류로 알린다 — 여기서 죽으면 진단할 기회조차 없다.
    func test_짝이_없는_popDebugGroup은_프로세스를_죽이지_않고_오류다() {
        let result = harness.execute([
            ["op": "createBuffer", "id": 1, "size": 16, "usage": TestUsage.copyDst],
            ["op": "writeBuffer", "buffer": 1, "data": [Float](repeating: 1, count: 4).base64],
            ["op": "popDebugGroup"],
        ])

        XCTAssertEqual(errors(result).first?["kind"] as? String, "validation")
        XCTAssertTrue(
            ((errors(result).first?["message"] as? String) ?? "").contains("짝이 맞는"),
            "\(errors(result))"
        )
    }

    /// 열린 채로 패스가 끝나도 Metal이 죽는다 — 닫아 주고 오류로 알린다.
    func test_열린_채_끝난_디버그_그룹은_닫아_주고_오류로_알린다() {
        let result = harness.execute([
            ["op": "configureCanvas", "canvas": "test", "format": "rgba8unorm"],
            ["op": "getCurrentTexture", "id": 10, "canvas": "test"],
            ["op": "createTextureView", "id": 11, "texture": 10],
            ["op": "beginRenderPass", "colorAttachments": [[
                "view": 11, "loadOp": "clear", "storeOp": "store",
                "clearValue": ["r": 0, "g": 0, "b": 0, "a": 1],
            ]]],
            ["op": "pushDebugGroup", "groupLabel": "안 닫음"],
            ["op": "endPass"],
        ])

        XCTAssertEqual(errors(result).first?["kind"] as? String, "validation")
        XCTAssertTrue(
            ((errors(result).first?["message"] as? String) ?? "").contains("열린 채"),
            "\(errors(result))"
        )
        // 그리고 프로세스가 살아 있다 — 이 단언에 도달한 것 자체가 증거다.
    }

    // MARK: - clearBuffer

    /// `writeBuffer`로 0을 밀어 넣는 것과 결과는 같아야 한다 — 다른 것은 브리지를 안 건넌다는 점뿐이다.
    func test_clearBuffer가_구간을_0으로_채운다() throws {
        let filled = [Float](repeating: 7, count: 8)
        // MAP_READ는 COPY_DST와만 조합할 수 있다 (명세 규칙 — 이 구현이 강제한다).
        harness.executeExpectingSuccess([
            ["op": "createBuffer", "id": 1, "size": 32,
             "usage": TestUsage.copyDst | TestUsage.mapRead, "data": filled.base64],
            // 앞 16바이트(=4개)만 지운다 — 뒤쪽은 그대로여야 구간이 지켜졌음을 안다.
            ["op": "clearBuffer", "buffer": 1, "offset": 0, "size": 16],
        ])

        let values = try harness.readBufferSync(handle: 1, as: Float.self)
        XCTAssertEqual(values, [0, 0, 0, 0, 7, 7, 7, 7])
    }

    func test_clearBuffer는_size를_생략하면_끝까지_지운다() throws {
        harness.executeExpectingSuccess([
            ["op": "createBuffer", "id": 1, "size": 16,
             "usage": TestUsage.copyDst | TestUsage.mapRead,
             "data": [Float](repeating: 3, count: 4).base64],
            ["op": "clearBuffer", "buffer": 1, "offset": 8],
        ])

        XCTAssertEqual(try harness.readBufferSync(handle: 1, as: Float.self), [3, 3, 0, 0])
    }

    func test_clearBuffer의_정렬과_usage와_범위를_검증한다() {
        // ① COPY_DST가 없으면 거부 — Metal은 그냥 채워 주므로 안 막으면 브라우저에서만 깨진다.
        let noUsage = harness.execute([
            ["op": "createBuffer", "id": 1, "size": 16, "usage": TestUsage.vertex],
            ["op": "clearBuffer", "buffer": 1],
        ])
        XCTAssertTrue(
            ((errors(noUsage).first?["message"] as? String) ?? "").contains("COPY_DST"),
            "\(errors(noUsage))"
        )

        // ② 4의 배수가 아니면 거부 (명세 규칙).
        let misaligned = harness.execute([
            ["op": "createBuffer", "id": 2, "size": 16, "usage": TestUsage.copyDst],
            ["op": "clearBuffer", "buffer": 2, "offset": 2, "size": 4],
        ])
        XCTAssertTrue(((errors(misaligned).first?["message"] as? String) ?? "").contains("4의 배수"))

        // ③ 범위를 넘으면 거부.
        let overflow = harness.execute([
            ["op": "createBuffer", "id": 3, "size": 16, "usage": TestUsage.copyDst],
            ["op": "clearBuffer", "buffer": 3, "offset": 8, "size": 16],
        ])
        XCTAssertTrue(((errors(overflow).first?["message"] as? String) ?? "").contains("범위"))
    }

    // MARK: - 진입점 해석 (명세의 "get the entry point")

    /// 셰이더: 정점 하나(`mainVS`) + 프래그먼트 셋(`main_2d` …) — three.js 밉맵 셰이더와 같은 모양.
    private static let multiEntryShader = """
    @vertex fn mainVS() -> @builtin(position) vec4f { return vec4f(0.0, 0.0, 0.0, 1.0); }
    @fragment fn main_2d() -> @location(0) vec4f { return vec4f(1.0, 0.0, 0.0, 1.0); }
    @fragment fn main_cube() -> @location(0) vec4f { return vec4f(0.0, 1.0, 0.0, 1.0); }
    """

    /// `entryPoint`는 명세에서 **필수가 아니다.** 생략하면 그 스테이지의 유일한 진입점을 쓴다.
    /// `"main"`으로 넘겨짚으면 이름이 다른 셰이더가 통째로 거부된다 — three.js 밉맵 생성이
    /// 정확히 그렇게 깨졌다 (`mainVS`가 있는데 `main`을 찾다 실패).
    func test_entryPoint를_생략하면_그_스테이지의_유일한_진입점을_쓴다() {
        let result = harness.execute([
            ["op": "createShaderModule", "id": 1, "code": Self.multiEntryShader],
            // 정점은 생략(유일한 mainVS로 해석돼야 한다), 프래그먼트는 셋 중 하나를 지정한다.
            ["op": "createRenderPipeline", "id": 2, "layout": "auto",
             "vertex": ["module": 1],
             "fragment": ["module": 1, "entryPoint": "main_2d",
                          "targets": [["format": "rgba8unorm"]]]],
        ])

        XCTAssertEqual(result["ok"] as? Bool, true, harness.describeErrors(result))
    }

    func test_후보가_둘_이상이면_고르지_않고_거부한다() {
        let result = harness.execute([
            ["op": "createShaderModule", "id": 1, "code": Self.multiEntryShader],
            // 프래그먼트 진입점이 둘이라 생략하면 고를 수 없다 — 조용히 하나를 집으면 안 된다.
            ["op": "createRenderPipeline", "id": 2, "layout": "auto",
             "vertex": ["module": 1],
             "fragment": ["module": 1, "targets": [["format": "rgba8unorm"]]]],
        ])

        XCTAssertEqual(errors(result).first?["kind"] as? String, "validation")
        let message = (errors(result).first?["message"] as? String) ?? ""
        XCTAssertTrue(message.contains("main_2d"), "후보 이름을 알려 줘야 한다: \(message)")
        XCTAssertTrue(message.contains("main_cube"))
    }

    func test_컴퓨트도_entryPoint를_생략할_수_있다() {
        let result = harness.execute([
            ["op": "createShaderModule", "id": 1, "code": """
             @compute @workgroup_size(4) fn onlyKernel() {}
             """],
            ["op": "createComputePipeline", "id": 2, "layout": "auto", "compute": ["module": 1]],
        ])

        XCTAssertEqual(result["ok"] as? Bool, true, harness.describeErrors(result))
    }

    func test_스테이지가_다른_진입점을_지정하면_거부한다() {
        let result = harness.execute([
            ["op": "createShaderModule", "id": 1, "code": Self.multiEntryShader],
            // 프래그먼트 자리에 정점 진입점을 줬다.
            ["op": "createRenderPipeline", "id": 2, "layout": "auto",
             "vertex": ["module": 1, "entryPoint": "mainVS"],
             "fragment": ["module": 1, "entryPoint": "mainVS",
                          "targets": [["format": "rgba8unorm"]]]],
        ])

        XCTAssertEqual(errors(result).first?["kind"] as? String, "validation")
        XCTAssertTrue(
            ((errors(result).first?["message"] as? String) ?? "").contains("vertex"),
            "어느 스테이지인지 알려 줘야 한다: \(errors(result))"
        )
    }

    func test_파이프라인없는_간접_드로우는_op이름이_든_메시지로_거부한다() {
        // 한때 이 가드가 `applyDrawState()` 뒤에 있어 도달 불가였다 — 일반형 메시지("draw 전에…")가
        // 대신 나가 사용자가 어느 op이 문제인지 알 수 없었다. op 이름이 실제로 나가는지 못 박는다.
        let renderSetup = indirectSetup(usage: TestUsage.indirect)
        let renderResult = harness.execute(renderSetup + [
            ["op": "drawIndirect", "indirectBuffer": 1],
        ])
        XCTAssertEqual(errors(renderResult).first?["kind"] as? String, "validation")
        XCTAssertTrue(
            ((errors(renderResult).first?["message"] as? String) ?? "")
                .contains("drawIndirect 전에 setPipeline"),
            "op 이름(drawIndirect)이 든 메시지여야 한다: \(errors(renderResult))"
        )

        let computeSetup = indirectSetup(usage: TestUsage.indirect, compute: true)
        let computeResult = harness.execute(computeSetup + [
            ["op": "dispatchWorkgroupsIndirect", "indirectBuffer": 1],
        ])
        XCTAssertTrue(
            ((errors(computeResult).first?["message"] as? String) ?? "")
                .contains("dispatchWorkgroupsIndirect 전에 setPipeline"),
            "op 이름(dispatchWorkgroupsIndirect)이 든 메시지여야 한다: \(errors(computeResult))"
        )
    }

    func test_어댑터_정보가_한계값을_보고한다() {
        let info = harness.context.adapterInfo()

        XCTAssertEqual(info["ok"] as? Bool, true)
        XCTAssertEqual(info["backend"] as? String, "metal")
        XCTAssertEqual(info["preferredCanvasFormat"] as? String, "bgra8unorm")
        let limits = try? XCTUnwrap(info["limits"] as? [String: Any])
        XCTAssertNotNil(limits?["maxVertexBuffers"])
    }

    /// limits의 **키는 명세 철자여야 한다.** 웹 라이브러리가 이 이름으로 읽고 예산을 정하므로,
    /// 우리 식으로 지으면 그쪽은 `undefined`를 보고 잘못된 가정을 세운다 (값이 있는데도 없는 것처럼).
    func test_limits는_명세_이름을_전부_싣는다() throws {
        let limits = try XCTUnwrap(harness.context.adapterInfo()["limits"] as? [String: Any])

        // 명세 `GPUSupportedLimits`의 전 항목 (webgpu-md §3.6.2).
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
            XCTAssertNotNil(limits[key], "명세 limit '\(key)'이(가) 빠졌다")
            XCTAssertGreaterThan((limits[key] as? Int) ?? 0, 0, "'\(key)'이(가) 0이다")
        }
    }

    /// 명세는 각 limit의 **기본값(=최소 보장치)**을 정한다. 그보다 낮게 보고하면 브라우저에서
    /// 되는 코드가 여기서만 거부되고, 앱은 이유를 알 수 없다.
    func test_limits는_명세_기본값보다_낮지_않다() throws {
        let limits = try XCTUnwrap(harness.context.adapterInfo()["limits"] as? [String: Any])

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
            XCTAssertGreaterThanOrEqual(value, minimum, "'\(key)' \(value) < 명세 기본값 \(minimum)")
        }

        // 정렬은 **작을수록 느슨하다** — 명세 기본값보다 크게 보고하면 브라우저에서 되는
        // 오프셋이 여기서 거부된다. 그래서 이쪽만 상한으로 본다.
        XCTAssertLessThanOrEqual((limits["minUniformBufferOffsetAlignment"] as? Int) ?? 0, 256)
        XCTAssertLessThanOrEqual((limits["minStorageBufferOffsetAlignment"] as? Int) ?? 0, 256)
    }
}
