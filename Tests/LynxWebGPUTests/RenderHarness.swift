import Foundation
import Metal
import XCTest
import LynxWebGPUCore
@testable import LynxWebGPU

/// The offscreen render verification harness.
///
/// Instead of "looking at" GPU results, it **asserts on pixel values.** Unlike a simulator screenshot
/// this is deterministic and runs in CI (`docs/TESTING.md` §4).
///
/// **It sees only the `WebGPURuntime` protocol** — so that contract tests cannot reach the default
/// implementation's non-protocol API through the harness. A verification the harness cannot express
/// is a signal that the protocol has a hole (`docs/extra/DAWN-BACKEND-REVIEW.md` §3-4).
struct RenderHarness {
    let runtime: WebGPURuntime
    let canvasId = "test"
    let width: Int
    let height: Int

    /// An escape hatch for observing Metal internals only (device feature gates, the staging pool).
    /// Do not use it for contract verification — an assertion resting on it does not carry to another runtime.
    var context: LynxWebGPUContext? { runtime as? LynxWebGPUContext }

    /// Nil where Metal is unavailable — the caller skips the test.
    static func make(width: Int = 64, height: Int = 64) -> RenderHarness? {
        guard let device = MTLCreateSystemDefaultDevice(),
              let context = try? LynxWebGPUContext(device: device) else { return nil }
        return make(runtime: context, width: width, height: height)
    }

    /// The injection point — the same contract tests can run on another `WebGPURuntime` implementation.
    static func make(runtime: WebGPURuntime, width: Int, height: Int) -> RenderHarness? {
        do {
            try runtime.attachOffscreenCanvas(
                identifier: "test", size: CGSize(width: width, height: height)
            )
        } catch {
            return nil
        }
        return RenderHarness(runtime: runtime, width: width, height: height)
    }

    @discardableResult
    func execute(_ commands: [[String: Any]], present: Bool = true) -> [String: Any] {
        runtime.execute(commands: commands, present: present)
    }

    /// The live native object count as seen through the protocol — the batch result's `objects` field
    /// (`docs/COMMAND-STREAM.md` §2).
    ///
    /// The probe is **an empty present:false batch**: it touches neither frame-scoped handles, error
    /// scopes nor drawable state, and creates no command buffer. Its only side effect is draining the
    /// previous batch's GPU failures one batch early — do not use it in tests that check GPU failures.
    var liveObjects: Int {
        execute([], present: false)["objects"] as? Int ?? -1
    }

    /// Checks it ran without errors, and otherwise shows the errors as they are.
    @discardableResult
    func executeExpectingSuccess(
        _ commands: [[String: Any]],
        file: StaticString = #filePath,
        line: UInt = #line
    ) -> [String: Any] {
        let result = execute(commands)
        if (result["ok"] as? Bool) != true {
            XCTFail("command execution failed: \(describeErrors(result))", file: file, line: line)
        }
        return result
    }

    func describeErrors(_ result: [String: Any]) -> String {
        guard let errors = result["errors"] as? [[String: Any]] else { return "(no error information)" }
        return errors.map { error in
            let path = error["path"].map { " @\($0)" } ?? ""
            return "[\(error["kind"] ?? "?")]\(path) \(error["message"] ?? "")"
        }.joined(separator: "\n")
    }

    /// Reads the render result back together with its format and row stride.
    func readback() throws -> WGPUPixelReadback {
        try runtime.readCanvasPixels(identifier: canvasId)
    }

    /// Reads render result pixels as raw channel values. On a float surface **values above 1.0 and negatives survive.**
    func pixelFloat(x: Int, y: Int) throws -> SIMD4<Float> {
        try readback().rgba(x: x, y: y)
    }

    /// A render result pixel (RGBA, 0...255). For 8-bit surfaces — use `pixelFloat` on float surfaces.
    func pixel(x: Int, y: Int) throws -> (r: Int, g: Int, b: Int, a: Int) {
        let color = try pixelFloat(x: x, y: y)
        let byte = { (value: Float) in Int((value * 255).rounded()) }
        return (byte(color.x), byte(color.y), byte(color.z), byte(color.w))
    }

    /// Asserts a pixel color within a tolerance (absorbing sRGB conversion and rasterization error).
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
            "pixel (\(x), \(y)) = \(actual), expected \(expected)\(message.isEmpty ? "" : " — \(message)")",
            file: file, line: line
        )
    }

    /// Asserts float channel values within a tolerance. Values outside SDR (above 1.0, negative) compare as they are.
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
            "pixel (\(x), \(y)) = \(actual), expected \(expected)\(message.isEmpty ? "" : " — \(message)")",
            file: file, line: line
        )
    }

    // MARK: - Frame equivalence

    /// Captures the whole current frame as bytes — the baseline for an equivalence comparison.
    func frameBytes() throws -> Data {
        try readback().data
    }

    /// Asserts the whole frame matches the baseline **byte for byte**.
    ///
    /// Point assertions are weak for comparing "two paths that must produce the same result" — two chosen
    /// points can match by chance. Where **the contract itself is "the results are equal"** — direct draw
    /// vs indirect draw, direct encoding vs render bundle — the whole frame is compared.
    ///
    /// On a mismatch it shows the first differing pixel's coordinate and both values — "N bytes differ" alone cannot be fixed.
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
                "frame lengths differ — baseline \(expected.count)B, actual \(actual.data.count)B\(suffix)",
                file: file, line: line
            )
        }
        let differences = zip(expected, actual.data).enumerated().filter { $0.element.0 != $0.element.1 }
        guard let first = differences.first else { return }

        // Turn the byte offset back into a pixel coordinate.
        let bytesPerPixel = max(actual.bytesPerRow / max(actual.width, 1), 1)
        let y = first.offset / max(actual.bytesPerRow, 1)
        let x = (first.offset % max(actual.bytesPerRow, 1)) / bytesPerPixel
        // Compute from the `actual` we already have — calling `readback()` again would mean another
        // staging buffer allocation, command buffer commit and `waitUntilCompleted()` (a full GPU wait).
        let detail = (try? actual.rgba(x: x, y: y)).map { " (actual pixel \($0))" } ?? ""
        XCTFail(
            "the frame differs from the baseline — \(differences.count)/\(expected.count)B mismatched, "
                + "first divergence at (\(x), \(y)), byte \(first.offset): "
                + "baseline \(first.element.0) != actual \(first.element.1)\(detail)\(suffix)",
            file: file, line: line
        )
    }

    // MARK: - Buffer readback

    /// Reads a buffer **synchronously**.
    ///
    /// `LynxWebGPUContext.readBuffer` must wait on the previous command buffer's GPU completion, so it
    /// is callback-based. A test has nothing to do afterwards, so it waits here — removing the
    /// `XCTestExpectation` boilerplate that used to repeat in every readback test.
    func readBufferSync(
        handle: Int,
        offset: Int = 0,
        size: Int? = nil,
        timeout: TimeInterval = 10
    ) throws -> Data {
        let box = ReadbackBox()
        let semaphore = DispatchSemaphore(value: 0)
        // For an already-completed command buffer the callback arrives **synchronously on this thread**
        // — signal precedes wait, but the semaphore counts so it passes through (no deadlock).
        runtime.readBuffer(handle: handle, offset: offset, size: size) { result in
            box.result = result
            semaphore.signal()
        }
        guard semaphore.wait(timeout: .now() + timeout) == .success else {
            throw HarnessError("readBuffer did not return within \(timeout)s (handle \(handle))")
        }
        let result = box.result
        guard (result["ok"] as? Bool) == true, let data = result["data"] as? Data else {
            throw HarnessError("readBuffer failed (handle \(handle)): \(describeErrors(result))")
        }
        return data
    }

    /// Reads a buffer as an element type (`try harness.readBufferSync(handle: 3, as: Float.self)`).
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

    // MARK: - Device conditions

    /// Features that vary per device. Splits the old "is there a GPU" skip condition per feature.
    enum Capability {
        /// Timestamp queries — can GPU counters be sampled at pass boundaries?
        case timestampQuery
        /// Indirect draw/dispatch arguments — Metal requires Apple family 3 or above.
        /// **The iOS simulator reports family 2 and drops out** (real devices, A12 and up, support it).
        case indirectArguments
    }

    func supports(_ capability: Capability) -> Bool {
        // The feature gate has to consult the Metal device — another runtime announces it through
        // adapterInfo's features (which is the path the conformance suite uses).
        guard let device = context?.device else { return false }
        switch capability {
        case .timestampQuery:
            guard device.supportsCounterSampling(.atStageBoundary) else { return false }
            return device.counterSets?.contains {
                $0.name == MTLCommonCounterSet.timestamp.rawValue
            } ?? false
        case .indirectArguments:
            return WGPUDeviceCapability.supportsIndirectArguments(device)
        }
    }

    /// For debugging — dumps the render result as a PNG (under `.tmp/`).
    /// A float surface is clipped to 0...1 and baked to 8 bits (this is for the eye, so the HDR range is discarded).
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

/// An error from the harness itself (not a GPU error but "the test cannot proceed").
struct HarnessError: LocalizedError {
    let message: String
    init(_ message: String) { self.message = message }
    var errorDescription: String? { message }
}

/// The callback can come from another thread, so the value lives in a reference type (the semaphore guarantees visibility).
private final class ReadbackBox {
    var result: [String: Any] = [:]
}

/// Encodes an RGBA8 buffer as a PNG with no dependencies (used only to eyeball render results).
enum PNGWriter {
    static func encode(rgba: Data, width: Int, height: Int) -> Data? {
        var raw = Data()
        for row in 0..<height {
            raw.append(0)   // filter type: None
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

    /// An uncompressed deflate (stored) stream — simple to encode with identical viewer compatibility.
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

/// The `GPUBufferUsage` / `GPUTextureUsage` constants JS uses (reproduced here for tests).
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
    /// The base64 byte string carried in the command stream.
    var base64: String {
        withUnsafeBufferPointer { Data(buffer: $0).base64EncodedString() }
    }
}
