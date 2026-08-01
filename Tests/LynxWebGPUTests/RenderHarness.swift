import Foundation
import Metal
import XCTest
import LynxWebGPUCore
@testable import LynxWebGPU

/// 오프스크린 렌더 검증 하네스.
///
/// GPU 결과를 "눈으로 보는" 대신 **픽셀 값으로 단언**한다. 시뮬레이터 스크린샷과 달리
/// 결정적이고 CI에서도 돌아간다 (`docs/TESTING.md` §4).
struct RenderHarness {
    let context: LynxWebGPUContext
    let surface: WGPUOffscreenSurface
    let width: Int
    let height: Int

    /// Metal을 쓸 수 없는 환경이면 nil — 호출 측이 테스트를 건너뛴다.
    static func make(width: Int = 64, height: Int = 64) -> RenderHarness? {
        guard let device = MTLCreateSystemDefaultDevice(),
              let context = try? LynxWebGPUContext(device: device) else { return nil }
        let surface = WGPUOffscreenSurface(
            identifier: "test", size: CGSize(width: width, height: height), device: device
        )
        context.registerSurface(surface)
        return RenderHarness(context: context, surface: surface, width: width, height: height)
    }

    @discardableResult
    func execute(_ commands: [[String: Any]]) -> [String: Any] {
        context.execute(commands: commands)
    }

    /// 오류 없이 실행됐는지 확인하고, 아니면 오류를 그대로 보여 준다.
    @discardableResult
    func executeExpectingSuccess(
        _ commands: [[String: Any]],
        file: StaticString = #filePath,
        line: UInt = #line
    ) -> [String: Any] {
        let result = execute(commands)
        if (result["ok"] as? Bool) != true {
            XCTFail("커맨드 실행 실패: \(describeErrors(result))", file: file, line: line)
        }
        return result
    }

    func describeErrors(_ result: [String: Any]) -> String {
        guard let errors = result["errors"] as? [[String: Any]] else { return "(오류 정보 없음)" }
        return errors.map { error in
            let path = error["path"].map { " @\($0)" } ?? ""
            return "[\(error["kind"] ?? "?")]\(path) \(error["message"] ?? "")"
        }.joined(separator: "\n")
    }

    /// 렌더 결과를 포맷·행 간격과 함께 되읽는다.
    func readback() throws -> WGPUPixelReadback {
        try surface.readPixels(queue: context.queue)
    }

    /// 렌더 결과 픽셀을 채널 값 그대로 읽는다. float 표면이면 **1.0 초과·음수도 그대로** 나온다.
    func pixelFloat(x: Int, y: Int) throws -> SIMD4<Float> {
        try readback().rgba(x: x, y: y)
    }

    /// 렌더 결과 픽셀 (RGBA, 0~255). 8비트 표면용 — float 표면에는 `pixelFloat`를 쓴다.
    func pixel(x: Int, y: Int) throws -> (r: Int, g: Int, b: Int, a: Int) {
        let color = try pixelFloat(x: x, y: y)
        let byte = { (value: Float) in Int((value * 255).rounded()) }
        return (byte(color.x), byte(color.y), byte(color.z), byte(color.w))
    }

    /// 픽셀 색을 허용 오차와 함께 단언한다 (sRGB 변환·래스터화 오차 흡수).
    func assertPixel(
        x: Int,
        y: Int,
        equals expected: (r: Int, g: Int, b: Int, a: Int),
        tolerance: Int = 2,
        _ message: String = "",
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws {
        let actual = try pixel(x: x, y: y)
        let matches = abs(actual.r - expected.r) <= tolerance
            && abs(actual.g - expected.g) <= tolerance
            && abs(actual.b - expected.b) <= tolerance
            && abs(actual.a - expected.a) <= tolerance
        XCTAssertTrue(
            matches,
            "픽셀 (\(x), \(y)) = \(actual), 기대 \(expected)\(message.isEmpty ? "" : " — \(message)")",
            file: file, line: line
        )
    }

    /// float 채널 값을 허용 오차와 함께 단언한다. SDR 범위 밖(1.0 초과·음수)도 그대로 비교한다.
    func assertPixelFloat(
        x: Int,
        y: Int,
        equals expected: SIMD4<Float>,
        tolerance: Float = 0.01,
        _ message: String = "",
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws {
        let actual = try pixelFloat(x: x, y: y)
        let matches = (0..<4).allSatisfy { abs(actual[$0] - expected[$0]) <= tolerance }
        XCTAssertTrue(
            matches,
            "픽셀 (\(x), \(y)) = \(actual), 기대 \(expected)\(message.isEmpty ? "" : " — \(message)")",
            file: file, line: line
        )
    }

    /// 디버깅용 — 렌더 결과를 PNG로 떨군다 (`.tmp/` 아래).
    /// float 표면은 0~1로 잘라서 8비트로 굽는다 (눈으로 볼 용도이므로 HDR 범위는 버린다).
    @discardableResult
    func dumpPNG(named name: String) -> URL? {
        guard let readback = try? readback() else { return nil }
        var data = Data(capacity: width * height * 4)
        for y in 0..<height {
            for x in 0..<width {
                guard let color = try? readback.rgba(x: x, y: y) else { return nil }
                for channel in 0..<4 {
                    data.append(UInt8((min(max(color[channel], 0), 1) * 255).rounded()))
                }
            }
        }
        let directory = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(".tmp")
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appendingPathComponent("\(name).png")
        guard let png = PNGWriter.encode(rgba: data, width: width, height: height) else { return nil }
        try? png.write(to: url)
        return url
    }
}

/// 의존성 없이 RGBA8 버퍼를 PNG로 인코딩한다 (렌더 결과를 눈으로 확인할 때만 쓴다).
enum PNGWriter {
    static func encode(rgba: Data, width: Int, height: Int) -> Data? {
        var raw = Data()
        for row in 0..<height {
            raw.append(0)   // 필터 타입: None
            let start = row * width * 4
            raw.append(rgba.subdata(in: start..<(start + width * 4)))
        }
        var png = Data([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
        var header = Data()
        header.append(contentsOf: bigEndian(UInt32(width)))
        header.append(contentsOf: bigEndian(UInt32(height)))
        header.append(contentsOf: [8, 6, 0, 0, 0])   // 8bit, RGBA, deflate, adaptive, no interlace
        png.append(chunk("IHDR", header))
        png.append(chunk("IDAT", zlibStored(raw)))
        png.append(chunk("IEND", Data()))
        return png
    }

    private static func chunk(_ type: String, _ payload: Data) -> Data {
        var data = Data(bigEndian(UInt32(payload.count)))
        let body = Data(type.utf8) + payload
        data.append(body)
        data.append(contentsOf: bigEndian(crc32(body)))
        return data
    }

    /// 압축하지 않는 deflate(stored) 스트림 — 인코딩이 단순하고 뷰어 호환성은 동일하다.
    private static func zlibStored(_ payload: Data) -> Data {
        var data = Data([0x78, 0x01])
        var offset = 0
        while offset < payload.count {
            let length = min(65535, payload.count - offset)
            let isLast: UInt8 = offset + length >= payload.count ? 1 : 0
            data.append(isLast)
            data.append(UInt8(length & 0xFF))
            data.append(UInt8((length >> 8) & 0xFF))
            data.append(UInt8(~length & 0xFF))
            data.append(UInt8((~length >> 8) & 0xFF))
            data.append(payload.subdata(in: offset..<(offset + length)))
            offset += length
        }
        data.append(contentsOf: bigEndian(adler32(payload)))
        return data
    }

    private static func bigEndian(_ value: UInt32) -> [UInt8] {
        [UInt8((value >> 24) & 0xFF), UInt8((value >> 16) & 0xFF), UInt8((value >> 8) & 0xFF), UInt8(value & 0xFF)]
    }

    private static func adler32(_ data: Data) -> UInt32 {
        var a: UInt32 = 1
        var b: UInt32 = 0
        for byte in data {
            a = (a + UInt32(byte)) % 65521
            b = (b + a) % 65521
        }
        return (b << 16) | a
    }

    private static let crcTable: [UInt32] = (0..<256).map { index -> UInt32 in
        var c = UInt32(index)
        for _ in 0..<8 {
            c = (c & 1) != 0 ? (0xEDB8_8320 ^ (c >> 1)) : (c >> 1)
        }
        return c
    }

    private static func crc32(_ data: Data) -> UInt32 {
        var c: UInt32 = 0xFFFF_FFFF
        for byte in data {
            c = crcTable[Int((c ^ UInt32(byte)) & 0xFF)] ^ (c >> 8)
        }
        return c ^ 0xFFFF_FFFF
    }
}

/// JS가 쓰는 `GPUBufferUsage` / `GPUTextureUsage` 상수 (테스트에서 그대로 재현).
enum TestUsage {
    static let mapRead = 0x0001
    static let copySrc = 0x0004
    static let copyDst = 0x0008
    static let index = 0x0010
    static let vertex = 0x0020
    static let uniform = 0x0040
    static let storage = 0x0080

    static let textureCopySrc = 0x01
    static let textureCopyDst = 0x02
    static let textureBinding = 0x04
    static let renderAttachment = 0x10
}

extension Array where Element == Float {
    /// 커맨드 스트림에 실어 보내는 base64 바이트열.
    var base64: String {
        withUnsafeBufferPointer { Data(buffer: $0).base64EncodedString() }
    }
}
