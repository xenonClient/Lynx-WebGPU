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

    // MARK: - 프레임 동치성

    /// 지금 프레임 전체를 바이트로 뜬다 — 동치성 비교의 기준값.
    func frameBytes() throws -> Data {
        try readback().data
    }

    /// 프레임 전체가 기준값과 **바이트 단위로** 같은지 단언한다.
    ///
    /// 점 단언은 "같은 결과를 내야 하는 두 경로"를 비교하기에 약하다 — 고른 두 점만 우연히
    /// 맞아도 통과하기 때문이다. 직접 드로우 ↔ 간접 드로우, 직접 인코딩 ↔ 렌더 번들처럼
    /// **계약 자체가 "결과가 같다"**인 경우에는 프레임 전체를 비교한다.
    ///
    /// 다르면 처음 어긋난 픽셀의 좌표와 두 값을 함께 보여 준다 — "N바이트 다름"만으로는 못 고친다.
    func assertFrameEquals(
        _ expected: Data,
        _ message: String = "",
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws {
        let suffix = message.isEmpty ? "" : " — \(message)"
        let actual = try readback()
        guard actual.data.count == expected.count else {
            return XCTFail(
                "프레임 길이가 다르다 — 기준 \(expected.count)B, 실제 \(actual.data.count)B\(suffix)",
                file: file, line: line
            )
        }
        let differences = zip(expected, actual.data).enumerated().filter { $0.element.0 != $0.element.1 }
        guard let first = differences.first else { return }

        // 바이트 오프셋을 픽셀 좌표로 되돌린다.
        let bytesPerPixel = max(actual.bytesPerRow / max(actual.width, 1), 1)
        let y = first.offset / max(actual.bytesPerRow, 1)
        let x = (first.offset % max(actual.bytesPerRow, 1)) / bytesPerPixel
        // 이미 확보한 `actual`로 계산한다 — `readback()`을 다시 부르면 스테이징 버퍼 신규 할당 +
        // 커맨드 버퍼 커밋 + `waitUntilCompleted()`(전체 GPU 동기 대기)가 한 번 더 돈다.
        let detail = (try? actual.rgba(x: x, y: y)).map { " (실제 픽셀 \($0))" } ?? ""
        XCTFail(
            "프레임이 기준과 다르다 — \(differences.count)/\(expected.count)B 불일치, "
                + "처음 어긋난 곳 (\(x), \(y)) 바이트 \(first.offset): "
                + "기준 \(first.element.0) ≠ 실제 \(first.element.1)\(detail)\(suffix)",
            file: file, line: line
        )
    }

    // MARK: - 버퍼 되읽기

    /// 버퍼를 **동기로** 읽는다.
    ///
    /// `LynxWebGPUContext.readBuffer`는 직전 커맨드 버퍼의 GPU 완료를 기다려야 하므로 콜백형이다.
    /// 테스트는 그 뒤에 할 일이 없으니 여기서 기다린다 — `XCTestExpectation` 보일러플레이트가
    /// 리드백 테스트마다 반복되던 것을 없앤다.
    func readBufferSync(
        handle: Int,
        offset: Int = 0,
        size: Int? = nil,
        timeout: TimeInterval = 10
    ) throws -> Data {
        let box = ReadbackBox()
        let semaphore = DispatchSemaphore(value: 0)
        // 이미 완료된 커맨드 버퍼면 콜백이 **이 스레드에서 동기로** 온다 — signal이 wait보다
        // 앞서지만 세마포어가 값을 세므로 그대로 통과한다 (교착 없음).
        context.readBuffer(handle: handle, offset: offset, size: size) { result in
            box.result = result
            semaphore.signal()
        }
        guard semaphore.wait(timeout: .now() + timeout) == .success else {
            throw HarnessError("readBuffer가 \(timeout)초 안에 돌아오지 않았다 (handle \(handle))")
        }
        let result = box.result
        guard (result["ok"] as? Bool) == true, let data = result["data"] as? Data else {
            throw HarnessError("readBuffer 실패 (handle \(handle)): \(describeErrors(result))")
        }
        return data
    }

    /// 버퍼를 원소 타입으로 읽는다 (`try harness.readBufferSync(handle: 3, as: Float.self)`).
    func readBufferSync<T>(
        handle: Int,
        as type: T.Type,
        offset: Int = 0,
        size: Int? = nil,
        timeout: TimeInterval = 10
    ) throws -> [T] {
        let data = try readBufferSync(handle: handle, offset: offset, size: size, timeout: timeout)
        return data.withUnsafeBytes { Array($0.bindMemory(to: T.self)) }
    }

    // MARK: - 기기 조건

    /// 기기마다 갈리는 기능. GPU 유무만 보던 스킵 조건을 기능별로 나눈다.
    enum Capability {
        /// 타임스탬프 쿼리 — 패스 경계에서 GPU 카운터를 샘플링할 수 있는가.
        case timestampQuery
    }

    func supports(_ capability: Capability) -> Bool {
        switch capability {
        case .timestampQuery:
            guard context.device.supportsCounterSampling(.atStageBoundary) else { return false }
            return context.device.counterSets?.contains {
                $0.name == MTLCommonCounterSet.timestamp.rawValue
            } ?? false
        }
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

/// 하네스 자체가 내는 오류 (GPU 오류가 아니라 "테스트를 진행할 수 없다").
struct HarnessError: LocalizedError {
    let message: String
    init(_ message: String) { self.message = message }
    var errorDescription: String? { message }
}

/// 콜백이 다른 스레드에서 올 수 있으므로 값을 참조 타입에 담는다 (세마포어가 가시성을 보장한다).
private final class ReadbackBox {
    var result: [String: Any] = [:]
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
    static let indirect = 0x0100
    static let queryResolve = 0x0200

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
