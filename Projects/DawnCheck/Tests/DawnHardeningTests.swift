import XCTest
import CoreGraphics
import LynxWebGPUCore

/// 크래시 하드닝 검증 — "잘못된 인자로 프로세스를 죽이지 않는다"(`WGPUError`)가 Dawn 런타임에도
/// 성립하는가. JS는 어떤 정수든 실어 보낼 수 있으므로, 음수·거대값·NaN이 GPU 인자 폭 변환에서
/// **트랩(프로세스 종료)이 아니라 validation 오류**가 되어야 한다.
final class DawnHardeningTests: XCTestCase {

    func test_적대적_인자에도_크래시_없이_validation으로_거부한다() throws {
        let runtime = try DawnWebGPURuntime()
        defer { runtime.reset() }
        try runtime.attachOffscreenCanvas(identifier: "h", size: CGSize(width: 8, height: 8))

        // 음수 크기·오프셋·인덱스, u32/u16 범위 초과, 없는 핸들 — 전 계열을 한 배치로.
        let hostile: [[String: Any]] = [
            ["op": "createBuffer", "id": 1, "size": -16, "usage": 0x20],
            ["op": "createBuffer", "id": 2, "size": 16, "usage": 0x0040 | 0x0008],
            ["op": "writeBuffer", "buffer": 2, "data": [1, 2, 3, 4], "bufferOffset": -8],
            ["op": "createTexture", "id": 3, "size": ["width": -4, "height": 4],
             "format": "rgba8unorm", "usage": 0x10],
            ["op": "createTexture", "id": 4, "size": ["width": 4, "height": 4],
             "format": "rgba8unorm", "usage": 0x10, "mipLevelCount": -1],
            ["op": "createSampler", "id": 5, "maxAnisotropy": 99999],
            ["op": "configureCanvas", "canvas": "h", "format": "rgba8unorm"],
            ["op": "getCurrentTexture", "id": 10, "canvas": "h"],
            ["op": "createTextureView", "id": 11, "texture": 10, "baseMipLevel": -3],
            ["op": "beginRenderPass",
             "colorAttachments": [["view": 99, "loadOp": "clear", "storeOp": "store"]]],
            ["op": "draw", "vertexCount": -3],
            ["op": "setScissorRect", "x": -1, "y": 0, "width": 4, "height": 4],
            ["op": "setBindGroup", "index": -1, "bindGroup": 999],
            ["op": "dispatchWorkgroups", "x": 4_294_967_296_000],
            ["op": "copyBufferToBuffer", "source": 2, "sourceOffset": -4, "destination": 2],
            ["op": "clearBuffer", "buffer": 2, "offset": -4],
            ["op": "resolveQuerySet", "querySet": 999, "firstQuery": -1, "destination": 2],
        ]
        let result = runtime.execute(commands: hostile, present: true)
        XCTAssertEqual(result["ok"] as? Bool, false, "적대 입력이 성공으로 보고됐다")
        let errors = result["errors"] as? [[String: Any]] ?? []
        XCTAssertFalse(errors.isEmpty, "적대 입력이 오류 없이 지나갔다")
        // 전부 명세 오류 4종 중 하나로 분류돼야 한다 (트랩·크래시가 아니라).
        for error in errors {
            XCTAssertNotNil(error["kind"], "kind 없는 오류: \(error)")
        }

        // 프로세스도 디바이스도 살아 있다 — 이어지는 정상 배치가 그대로 동작한다.
        let sane = runtime.execute(commands: [
            ["op": "createBuffer", "id": 20, "size": 16, "usage": 0x0040],
        ], present: false)
        XCTAssertEqual(sane["ok"] as? Bool, true, "적대 배치 뒤 정상 배치가 실패한다")

        // 크기 공격 — NaN·음수 resize는 무시되고, 잘못된 캔버스 읽기는 던진다 (트랩 아님).
        runtime.resizeCanvas(identifier: "h", drawableSize: CGSize(width: CGFloat.nan, height: -5))
        XCTAssertThrowsError(try runtime.readCanvasPixels(identifier: "없는-캔버스"))
        let info = runtime.canvasInfo(identifier: "h")
        XCTAssertEqual(info["ok"] as? Bool, true, "NaN resize가 캔버스 상태를 오염시켰다")
    }
}
