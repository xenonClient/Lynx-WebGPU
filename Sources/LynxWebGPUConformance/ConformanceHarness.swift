import Foundation
import CoreGraphics
import LynxWebGPUCore

/// 적합성 검사가 실패했다 — GPU 오류가 아니라 **계약 위반**이다.
public struct ConformanceFailure: LocalizedError {
    public let message: String
    public init(_ message: String) { self.message = message }
    public var errorDescription: String? { message }
}

/// JS가 쓰는 `GPUBufferUsage` / `GPUTextureUsage` 비트값.
///
/// 커맨드 스트림은 플래그를 정수로 싣는다 (`docs/COMMAND-STREAM.md` §3-3). 검사가 이 숫자를
/// 손으로 적으면 어느 비트인지 읽을 수 없어서 이름을 붙여 둔다.
public enum WebGPUUsage {
    public static let mapRead = 0x0001
    public static let copySrc = 0x0004
    public static let copyDst = 0x0008
    public static let index = 0x0010
    public static let vertex = 0x0020
    public static let uniform = 0x0040
    public static let storage = 0x0080
    public static let indirect = 0x0100
    public static let queryResolve = 0x0200

    public static let textureCopySrc = 0x01
    public static let textureCopyDst = 0x02
    public static let textureBinding = 0x04
    public static let renderAttachment = 0x10
}

/// 검사 하나가 쓰는 도구 상자.
///
/// **`WebGPURuntime`만 본다** — Metal도, `LynxWebGPUContext`도 모른다. 그래서 여기 적힌
/// 검사는 어느 런타임에든 그대로 돈다. 그것이 이 스위트의 존재 이유다.
public final class ConformanceHarness {
    public let runtime: WebGPURuntime
    public let canvas: String
    public let width: Int
    public let height: Int

    /// 새 검사를 위한 깨끗한 상태를 만든다 — 앞 검사의 객체·오류 스코프가 넘어오지 않게.
    public init(runtime: WebGPURuntime, canvas: String = "conformance", width: Int = 64, height: Int = 64) throws {
        self.runtime = runtime
        self.canvas = canvas
        self.width = width
        self.height = height
        runtime.reset()
        try runtime.attachOffscreenCanvas(
            identifier: canvas, size: CGSize(width: width, height: height)
        )
    }

    deinit {
        runtime.detachCanvas(identifier: canvas)
    }

    // MARK: - 실행

    @discardableResult
    public func execute(_ commands: [[String: Any]], present: Bool = true) -> [String: Any] {
        runtime.execute(["commands": commands, "present": present])
    }

    /// 오류 없이 실행됐는지 확인한다. 실패하면 오류를 그대로 메시지에 담는다.
    /// `present: false`는 프레임 **중간** 제출이다 — 수명 검사가 이 경계를 직접 몬다.
    @discardableResult
    public func executeExpectingSuccess(
        _ commands: [[String: Any]], present: Bool = true
    ) throws -> [String: Any] {
        let result = execute(commands, present: present)
        guard (result["ok"] as? Bool) == true else {
            throw ConformanceFailure("커맨드 실행 실패 — \(Self.describeErrors(result))")
        }
        return result
    }

    /// 오류가 **나야** 정상인 경우. 돌아온 오류 목록을 준다.
    @discardableResult
    public func executeExpectingFailure(_ commands: [[String: Any]]) throws -> [[String: Any]] {
        let result = execute(commands)
        guard (result["ok"] as? Bool) == false else {
            throw ConformanceFailure("오류가 나야 하는데 성공했다")
        }
        return result["errors"] as? [[String: Any]] ?? []
    }

    public static func describeErrors(_ result: [String: Any]) -> String {
        guard let errors = result["errors"] as? [[String: Any]], !errors.isEmpty else {
            return "(오류 정보 없음)"
        }
        return errors.map { error in
            let path = error["path"].map { " @\($0)" } ?? ""
            return "[\(error["kind"] ?? "?")]\(path) \(error["message"] ?? "")"
        }.joined(separator: " / ")
    }

    // MARK: - 픽셀

    public func readback() throws -> WGPUPixelReadback {
        try runtime.readCanvasPixels(identifier: canvas)
    }

    /// 렌더 결과 픽셀 (RGBA, 0~255).
    public func pixel(x: Int, y: Int) throws -> (r: Int, g: Int, b: Int, a: Int) {
        let color = try readback().rgba(x: x, y: y)
        let byte = { (value: Float) in Int((value * 255).rounded()) }
        return (byte(color.x), byte(color.y), byte(color.z), byte(color.w))
    }

    /// 허용 오차와 함께 픽셀을 단언한다 (래스터화·sRGB 변환 오차 흡수).
    public func expectPixel(
        x: Int,
        y: Int,
        equals expected: (r: Int, g: Int, b: Int, a: Int),
        tolerance: Int = 2,
        _ note: String = ""
    ) throws {
        let actual = try pixel(x: x, y: y)
        let matches = abs(actual.r - expected.r) <= tolerance
            && abs(actual.g - expected.g) <= tolerance
            && abs(actual.b - expected.b) <= tolerance
            && abs(actual.a - expected.a) <= tolerance
        guard matches else {
            throw ConformanceFailure(
                "픽셀 (\(x), \(y)) = \(actual), 기대 \(expected)\(note.isEmpty ? "" : " — \(note)")"
            )
        }
    }

    /// 지금 프레임 전체를 바이트로 뜬다 — 동치성 비교의 기준값.
    public func frameBytes() throws -> Data {
        try readback().data
    }

    /// 프레임 전체가 기준과 **바이트 단위로** 같은지 단언한다.
    ///
    /// 점 단언은 "같은 결과를 내야 하는 두 경로"를 비교하기에 약하다 — 고른 두 점만 우연히
    /// 맞아도 통과한다. 직접 드로우 ↔ 간접 드로우, 직접 인코딩 ↔ 렌더 번들처럼 **계약 자체가
    /// "결과가 같다"**인 경우에 쓴다.
    public func expectFrameEquals(_ expected: Data, _ note: String = "") throws {
        let suffix = note.isEmpty ? "" : " — \(note)"
        let actual = try readback()
        guard actual.data.count == expected.count else {
            throw ConformanceFailure(
                "프레임 길이가 다르다 — 기준 \(expected.count)B, 실제 \(actual.data.count)B\(suffix)"
            )
        }
        let differences = zip(expected, actual.data).enumerated().filter { $0.element.0 != $0.element.1 }
        guard let first = differences.first else { return }
        let bytesPerPixel = max(actual.bytesPerRow / max(actual.width, 1), 1)
        let y = first.offset / max(actual.bytesPerRow, 1)
        let x = (first.offset % max(actual.bytesPerRow, 1)) / bytesPerPixel
        throw ConformanceFailure(
            "프레임이 기준과 다르다 — \(differences.count)/\(expected.count)B 불일치, "
                + "처음 어긋난 곳 (\(x), \(y)): 기준 \(first.element.0) ≠ 실제 \(first.element.1)\(suffix)"
        )
    }

    // MARK: - 버퍼

    /// 버퍼를 **동기로** 읽는다. 리드백은 GPU 완료를 기다려야 해서 콜백형이라, 여기서 기다린다.
    public func readBufferSync(
        handle: Int, offset: Int = 0, size: Int? = nil, timeout: TimeInterval = 10
    ) throws -> Data {
        let box = ResultBox()
        let semaphore = DispatchSemaphore(value: 0)
        // 이미 완료된 커맨드 버퍼면 콜백이 **이 스레드에서 동기로** 온다 — signal이 wait보다
        // 앞서지만 세마포어가 값을 세므로 교착이 없다.
        runtime.readBuffer(handle: handle, offset: offset, size: size) { result in
            box.value = result
            semaphore.signal()
        }
        // 한 번에 기다리지 않고 짧게 쪼개어 사이사이 펌프를 돌린다 — 디스플레이 링크가 없는
        // 이 환경에서는 하네스가 유일한 펌프 호출자다 (`WebGPURuntime.processEvents`).
        // 펌프가 필요 없는 런타임(Metal)은 첫 대기에서 바로 통과하므로 동작이 같다.
        try waitPumping(semaphore, timeout: timeout) {
            ConformanceFailure("readBuffer가 \(timeout)초 안에 돌아오지 않았다 (handle \(handle))")
        }
        guard (box.value["ok"] as? Bool) == true, let data = box.value["data"] as? Data else {
            throw ConformanceFailure("readBuffer 실패 (handle \(handle)) — \(Self.describeErrors(box.value))")
        }
        return data
    }

    public func readBufferSync<T>(handle: Int, as type: T.Type, size: Int? = nil) throws -> [T] {
        let data = try readBufferSync(handle: handle, size: size)
        return data.withUnsafeBytes { Array($0.bindMemory(to: T.self)) }
    }

    /// 세마포어를 조각내어 기다리며 사이사이 `processEvents()`를 돌린다.
    /// 콜백형 API를 기다리는 검사(`readBufferSync`·이미지 디코딩)가 공유한다.
    public func waitPumping(
        _ semaphore: DispatchSemaphore,
        timeout: TimeInterval,
        onTimeout: () -> ConformanceFailure
    ) throws {
        let deadline = Date(timeIntervalSinceNow: timeout)
        while semaphore.wait(timeout: .now() + 0.01) != .success {
            guard Date() < deadline else { throw onTimeout() }
            runtime.processEvents()
        }
    }

    // MARK: - 기기 기능

    /// 어댑터가 광고하는 선택 기능 — **명세가 정한 통로 그대로** 본다.
    /// 백엔드 내부를 들여다보지 않으므로 어느 런타임에든 같은 방식으로 물을 수 있다.
    public func advertises(feature: String) -> Bool {
        (runtime.adapterInfo()["features"] as? [String])?.contains(feature) ?? false
    }
}

/// 콜백이 다른 스레드에서 올 수 있어 값을 참조 타입에 담는다 (세마포어가 가시성을 보장한다).
private final class ResultBox {
    var value: [String: Any] = [:]
}

public extension Array where Element == Float {
    /// 커맨드 스트림에 실어 보내는 base64 바이트열.
    var conformanceBase64: String {
        withUnsafeBufferPointer { Data(buffer: $0).base64EncodedString() }
    }
}

public extension Array where Element == UInt16 {
    var conformanceBase64: String {
        withUnsafeBufferPointer { Data(buffer: $0).base64EncodedString() }
    }
}

public extension Array where Element == UInt8 {
    var conformanceBase64: String {
        Data(self).base64EncodedString()
    }
}
