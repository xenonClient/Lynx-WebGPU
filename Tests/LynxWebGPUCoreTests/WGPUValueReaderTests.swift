import XCTest
@testable import LynxWebGPUCore

/// 커맨드 스트림 디코딩 — JS가 보내는 형태를 그대로 재현해 검증한다.
final class WGPUValueReaderTests: XCTestCase {
    func test_스칼라_읽기와_기본값() throws {
        let reader = WGPUValueReader(["size": 128, "label": "vertices", "flag": true])

        XCTAssertEqual(try reader.requiredInt("size"), 128)
        XCTAssertEqual(reader.int("missing", default: 7), 7)
        XCTAssertEqual(try reader.requiredString("label"), "vertices")
        XCTAssertTrue(reader.bool("flag", default: false))
        XCTAssertFalse(reader.bool("missing", default: false))
    }

    func test_필수필드가_없으면_경로가_붙은_오류다() {
        let reader = WGPUValueReader(["a": 1], path: "commands[3]")

        XCTAssertThrowsError(try reader.requiredInt("size")) { error in
            let wgpu = error as? WGPUError
            XCTAssertEqual(wgpu?.kind, .validation)
            XCTAssertEqual(wgpu?.path, "commands[3].size")
        }
    }

    func test_null은_없는_값으로_취급한다() {
        // JS의 `undefined`는 Lynx를 거치며 NSNull로 온다.
        let reader = WGPUValueReader(["label": NSNull()])
        XCTAssertFalse(reader.has("label"))
        XCTAssertNil(reader.optionalString("label"))
    }

    func test_열거형은_명세_철자로_파싱하고_모르면_후보를_알려준다() throws {
        let reader = WGPUValueReader(["format": "bgra8unorm", "topology": "triangle-strip"])

        XCTAssertEqual(try reader.requiredEnum("format", WGPUTextureFormat.self), .bgra8unorm)
        XCTAssertEqual(try reader.requiredEnum("topology", WGPUPrimitiveTopology.self), .triangleStrip)

        let bad = WGPUValueReader(["format": "rgba8"])
        XCTAssertThrowsError(try bad.requiredEnum("format", WGPUTextureFormat.self)) { error in
            let message = (error as? WGPUError)?.message ?? ""
            XCTAssertTrue(message.contains("rgba8unorm"), "가능한 값 목록이 없다: \(message)")
        }
    }

    func test_비트마스크_플래그를_읽는다() throws {
        let reader = WGPUValueReader(["usage": 0x0020 | 0x0008])
        let usage = try reader.requiredFlags("usage", WGPUBufferUsage.self)

        XCTAssertTrue(usage.contains(.vertex))
        XCTAssertTrue(usage.contains(.copyDst))
        XCTAssertFalse(usage.contains(.uniform))
    }

    func test_색은_객체와_배열_두_표기를_모두_받는다() throws {
        let object = WGPUValueReader(["clearValue": ["r": 1.0, "g": 0.5, "b": 0.0, "a": 1.0]])
        XCTAssertEqual(try object.color("clearValue", default: .black), WGPUColor(red: 1, green: 0.5, blue: 0, alpha: 1))

        let array = WGPUValueReader(["clearValue": [1.0, 0.5, 0.0]])
        XCTAssertEqual(try array.color("clearValue", default: .black), WGPUColor(red: 1, green: 0.5, blue: 0, alpha: 1))

        let missing = WGPUValueReader([:])
        XCTAssertEqual(try missing.color("clearValue", default: .black), .black)
    }

    func test_크기는_객체와_배열_두_표기를_모두_받는다() throws {
        let object = WGPUValueReader(["size": ["width": 64, "height": 32]])
        XCTAssertEqual(try object.requiredExtent("size"), WGPUExtent3D(width: 64, height: 32))

        let array = WGPUValueReader(["size": [64, 32, 6]])
        XCTAssertEqual(try array.requiredExtent("size"), WGPUExtent3D(width: 64, height: 32, depthOrArrayLayers: 6))
    }

    func test_바이너리는_base64와_바이트배열을_모두_받는다() throws {
        let bytes: [UInt8] = [0xDE, 0xAD, 0xBE, 0xEF]

        let base64 = WGPUValueReader(["data": Data(bytes).base64EncodedString()])
        XCTAssertEqual(Array(try base64.requiredData("data")), bytes)

        let array = WGPUValueReader(["data": bytes.map { Int($0) }])
        XCTAssertEqual(Array(try array.requiredData("data")), bytes)

        let broken = WGPUValueReader(["data": "!!not base64!!"])
        XCTAssertThrowsError(try broken.requiredData("data"))
    }

    func test_중첩_배열의_경로가_인덱스까지_남는다() {
        let reader = WGPUValueReader(["entries": [["binding": 0], ["nope": 1]]], path: "cmd")
        let entries = try? reader.requiredObjects("entries")

        XCTAssertEqual(entries?.count, 2)
        XCTAssertThrowsError(try entries?[1].requiredInt("binding")) { error in
            XCTAssertEqual((error as? WGPUError)?.path, "cmd.entries[1].binding")
        }
    }
}
