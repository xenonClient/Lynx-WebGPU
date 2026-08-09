import Foundation
import CoreGraphics
import LynxWebGPUCore

/// A conformance check failed — not a GPU error but a **contract violation**.
public struct ConformanceFailure: LocalizedError {
    public let message: String
    public init(_ message: String) { self.message = message }
    public var errorDescription: String? { message }
}

/// The `GPUBufferUsage` / `GPUTextureUsage` bit values JS uses.
///
/// The command stream carries flags as integers (`docs/COMMAND-STREAM.md` §3-3). Writing those
/// numbers by hand in a check makes it unreadable which bit is which, so they get names.
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

/// The toolbox one check uses.
///
/// **It sees `WebGPURuntime` only** — it knows nothing of Metal or `LynxWebGPUContext`. That is what
/// lets every check written here run on any runtime, and it is the reason this suite exists.
public final class ConformanceHarness {
    public let runtime: WebGPURuntime
    public let canvas: String
    public let width: Int
    public let height: Int

    /// Builds clean state for a new check — so the previous check's objects and error scopes do not carry over.
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

    // MARK: - Execution

    @discardableResult
    public func execute(_ commands: [[String: Any]], present: Bool = true) -> [String: Any] {
        runtime.execute(["commands": commands, "present": present])
    }

    /// Verifies it ran without errors. On failure the errors go straight into the message.
    /// `present: false` is a **mid-frame** submit — the lifetime checks drive that boundary directly.
    @discardableResult
    public func executeExpectingSuccess(
        _ commands: [[String: Any]], present: Bool = true
    ) throws -> [String: Any] {
        let result = execute(commands, present: present)
        guard (result["ok"] as? Bool) == true else {
            throw ConformanceFailure("command execution failed — \(Self.describeErrors(result))")
        }
        return result
    }

    /// The case where an error **is** the correct outcome. Returns the error list.
    @discardableResult
    public func executeExpectingFailure(_ commands: [[String: Any]]) throws -> [[String: Any]] {
        let result = execute(commands)
        guard (result["ok"] as? Bool) == false else {
            throw ConformanceFailure("expected an error but it succeeded")
        }
        return result["errors"] as? [[String: Any]] ?? []
    }

    public static func describeErrors(_ result: [String: Any]) -> String {
        guard let errors = result["errors"] as? [[String: Any]], !errors.isEmpty else {
            return "(no error information)"
        }
        return errors.map { error in
            let path = error["path"].map { " @\($0)" } ?? ""
            return "[\(error["kind"] ?? "?")]\(path) \(error["message"] ?? "")"
        }.joined(separator: " / ")
    }

    // MARK: - Pixels

    public func readback() throws -> WGPUPixelReadback {
        try runtime.readCanvasPixels(identifier: canvas)
    }

    /// A render result pixel (RGBA, 0...255).
    public func pixel(x: Int, y: Int) throws -> (r: Int, g: Int, b: Int, a: Int) {
        let color = try readback().rgba(x: x, y: y)
        let byte = { (value: Float) in Int((value * 255).rounded()) }
        return (byte(color.x), byte(color.y), byte(color.z), byte(color.w))
    }

    /// Asserts a pixel within a tolerance (absorbing rasterization and sRGB conversion error).
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
                "pixel (\(x), \(y)) = \(actual), expected \(expected)\(note.isEmpty ? "" : " — \(note)")"
            )
        }
    }

    /// Captures the whole current frame as bytes — the baseline for an equivalence comparison.
    public func frameBytes() throws -> Data {
        try readback().data
    }

    /// Asserts the whole frame matches the baseline **byte for byte**.
    ///
    /// Point assertions are weak for comparing "two paths that must produce the same result" — two
    /// chosen points can match by chance. Use this where **the contract itself is "the results are
    /// equal"**: direct draw ↔ indirect draw, direct encoding ↔ render bundle.
    public func expectFrameEquals(_ expected: Data, _ note: String = "") throws {
        let suffix = note.isEmpty ? "" : " — \(note)"
        let actual = try readback()
        guard actual.data.count == expected.count else {
            throw ConformanceFailure(
                "frame lengths differ — baseline \(expected.count)B, actual \(actual.data.count)B\(suffix)"
            )
        }
        let differences = zip(expected, actual.data).enumerated().filter { $0.element.0 != $0.element.1 }
        guard let first = differences.first else { return }
        let bytesPerPixel = max(actual.bytesPerRow / max(actual.width, 1), 1)
        let y = first.offset / max(actual.bytesPerRow, 1)
        let x = (first.offset % max(actual.bytesPerRow, 1)) / bytesPerPixel
        throw ConformanceFailure(
            "the frame differs from the baseline — \(differences.count)/\(expected.count)B mismatched, "
                + "first divergence at (\(x), \(y)): baseline \(first.element.0) != actual \(first.element.1)\(suffix)"
        )
    }

    // MARK: - Buffers

    /// Reads a buffer **synchronously**. Readback must wait on GPU completion and is therefore callback-based, so we wait here.
    public func readBufferSync(
        handle: Int, offset: Int = 0, size: Int? = nil, timeout: TimeInterval = 10
    ) throws -> Data {
        let box = ResultBox()
        let semaphore = DispatchSemaphore(value: 0)
        // For an already-completed command buffer the callback arrives **synchronously on this thread**
        // — signal precedes wait, but the semaphore counts, so there is no deadlock.
        runtime.readBuffer(handle: handle, offset: offset, size: size) { result in
            box.value = result
            semaphore.signal()
        }
        // Rather than waiting once, we split into short waits and pump in between — with no display
        // link in this environment the harness is the only pump caller (`WebGPURuntime.processEvents`).
        // A runtime that needs no pump (Metal) passes on the first wait, so behaviour is identical.
        try waitPumping(semaphore, timeout: timeout) {
            ConformanceFailure("readBuffer did not return within \(timeout)s (handle \(handle))")
        }
        guard (box.value["ok"] as? Bool) == true, let data = box.value["data"] as? Data else {
            throw ConformanceFailure("readBuffer failed (handle \(handle)) — \(Self.describeErrors(box.value))")
        }
        return data
    }

    public func readBufferSync<T>(handle: Int, as type: T.Type, size: Int? = nil) throws -> [T] {
        let data = try readBufferSync(handle: handle, size: size)
        return data.withUnsafeBytes { Array($0.bindMemory(to: T.self)) }
    }

    /// Waits on the semaphore in slices, running `processEvents()` in between.
    /// Shared by checks waiting on callback-based APIs (`readBufferSync`, image decoding).
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

    // MARK: - Device features

    /// Optional features the adapter advertises — read **through the channel the spec defines**.
    /// It never inspects backend internals, so it can be asked the same way on any runtime.
    public func advertises(feature: String) -> Bool {
        (runtime.adapterInfo()["features"] as? [String])?.contains(feature) ?? false
    }
}

/// The callback can come from another thread, so the value lives in a reference type (the semaphore guarantees visibility).
private final class ResultBox {
    var value: [String: Any] = [:]
}

public extension Array where Element == Float {
    /// The base64 byte string carried in the command stream.
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
