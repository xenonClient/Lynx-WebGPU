import XCTest
@testable import LynxWebGPUCore

/// Command stream decoding — verified by reproducing exactly what JS sends.
final class WGPUValueReaderTests: XCTestCase {
    func test_scalarReadsAndDefaults() throws {
        let reader = WGPUValueReader(["size": 128, "label": "vertices", "flag": true])

        XCTAssertEqual(try reader.requiredInt("size"), 128)
        XCTAssertEqual(reader.int("missing", default: 7), 7)
        XCTAssertEqual(try reader.requiredString("label"), "vertices")
        XCTAssertTrue(reader.bool("flag", default: false))
        XCTAssertFalse(reader.bool("missing", default: false))
    }

    func test_aMissingRequiredFieldIsAnErrorWithAPath() {
        let reader = WGPUValueReader(["a": 1], path: "commands[3]")

        XCTAssertThrowsError(try reader.requiredInt("size")) { error in
            let wgpu = error as? WGPUError
            XCTAssertEqual(wgpu?.kind, .validation)
            XCTAssertEqual(wgpu?.path, "commands[3].size")
        }
    }

    func test_nullIsTreatedAsAbsent() {
        // JS's `undefined` arrives as NSNull after crossing Lynx.
        let reader = WGPUValueReader(["label": NSNull()])
        XCTAssertFalse(reader.has("label"))
        XCTAssertNil(reader.optionalString("label"))
    }

    func test_enumsParseBySpecSpellingAndListCandidatesWhenUnknown() throws {
        let reader = WGPUValueReader(["format": "bgra8unorm", "topology": "triangle-strip"])

        XCTAssertEqual(try reader.requiredEnum("format", WGPUTextureFormat.self), .bgra8unorm)
        XCTAssertEqual(try reader.requiredEnum("topology", WGPUPrimitiveTopology.self), .triangleStrip)

        let bad = WGPUValueReader(["format": "rgba8"])
        XCTAssertThrowsError(try bad.requiredEnum("format", WGPUTextureFormat.self)) { error in
            let message = (error as? WGPUError)?.message ?? ""
            XCTAssertTrue(message.contains("rgba8unorm"), "no list of possible values: \(message)")
        }
    }

    func test_readsBitmaskFlags() throws {
        let reader = WGPUValueReader(["usage": 0x0020 | 0x0008])
        let usage = try reader.requiredFlags("usage", WGPUBufferUsage.self)

        XCTAssertTrue(usage.contains(.vertex))
        XCTAssertTrue(usage.contains(.copyDst))
        XCTAssertFalse(usage.contains(.uniform))
    }

    func test_colorAcceptsBothObjectAndArrayForms() throws {
        let object = WGPUValueReader(["clearValue": ["r": 1.0, "g": 0.5, "b": 0.0, "a": 1.0]])
        XCTAssertEqual(try object.color("clearValue", default: .black), WGPUColor(red: 1, green: 0.5, blue: 0, alpha: 1))

        let array = WGPUValueReader(["clearValue": [1.0, 0.5, 0.0]])
        XCTAssertEqual(try array.color("clearValue", default: .black), WGPUColor(red: 1, green: 0.5, blue: 0, alpha: 1))

        let missing = WGPUValueReader([:])
        XCTAssertEqual(try missing.color("clearValue", default: .black), .black)
    }

    func test_extentAcceptsBothObjectAndArrayForms() throws {
        let object = WGPUValueReader(["size": ["width": 64, "height": 32]])
        XCTAssertEqual(try object.requiredExtent("size"), WGPUExtent3D(width: 64, height: 32))

        let array = WGPUValueReader(["size": [64, 32, 6]])
        XCTAssertEqual(try array.requiredExtent("size"), WGPUExtent3D(width: 64, height: 32, depthOrArrayLayers: 6))
    }

    func test_binaryAcceptsDataBase64AndByteArrays() throws {
        let bytes: [UInt8] = [0xDE, 0xAD, 0xBE, 0xEF]

        // The path the JS shim uses — what Lynx turned from an ArrayBuffer into NSData arrives as-is.
        let native = WGPUValueReader(["data": Data(bytes)])
        XCTAssertEqual(Array(try native.requiredData("data")), bytes)

        // The same when it arrives as NSData (the type the bridge actually passes).
        let bridged = WGPUValueReader(["data": NSData(data: Data(bytes))])
        XCTAssertEqual(Array(try bridged.requiredData("data")), bytes)

        let base64 = WGPUValueReader(["data": Data(bytes).base64EncodedString()])
        XCTAssertEqual(Array(try base64.requiredData("data")), bytes)

        let array = WGPUValueReader(["data": bytes.map { Int($0) }])
        XCTAssertEqual(Array(try array.requiredData("data")), bytes)

        let broken = WGPUValueReader(["data": "!!not base64!!"])
        XCTAssertThrowsError(try broken.requiredData("data"))
    }

    func test_aNestedArrayPathKeepsTheIndex() {
        let reader = WGPUValueReader(["entries": [["binding": 0], ["nope": 1]]], path: "cmd")
        let entries = try? reader.requiredObjects("entries")

        XCTAssertEqual(entries?.count, 2)
        XCTAssertThrowsError(try entries?[1].requiredInt("binding")) { error in
            XCTAssertEqual((error as? WGPUError)?.path, "cmd.entries[1].binding")
        }
    }
}
