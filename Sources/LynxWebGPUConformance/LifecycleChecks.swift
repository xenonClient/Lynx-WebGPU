import Foundation
import CoreGraphics
import LynxWebGPUCore

/// Checks for frame lifetime, callback APIs and query contracts.
///
/// Where `Checks.swift` asks "does drawing draw?", this file looks at **the boundaries** — when
/// present closes a frame, what a callback API returns. Getting these wrong shows nothing in a simple
/// scene and only blows up in code that splits a frame across several batches, such as Three.js-style
/// lazy pipeline creation (`docs/COMMAND-STREAM.md` §2). So the suite drives the boundaries directly.
public extension WebGPUConformance {

    // MARK: - Frame lifetime

    /// A `present: false` batch does not close the frame — frame-scoped handles survive.
    ///
    /// The point is that the middle batch **produces a command buffer from a single writeBuffer**: a
    /// commit happens, yet present and handle expiry must still be deferred. Three.js's lazy pipeline
    /// creation walked exactly this path (flush right after pop).
    static var presentFalsePreservesFrame: Check {
        Check("present-false-preserves-frame") { harness in
            // Batch 1 (open the frame): obtain the drawable texture (10) and view (11).
            try harness.executeExpectingSuccess(harness.canvasSetup, present: false)
            // Batch 2 (mid-frame): the dangerous path where a command buffer appears.
            try harness.executeExpectingSuccess([
                ["op": "createBuffer", "id": 20, "size": 16,
                 "usage": WebGPUUsage.copyDst | WebGPUUsage.uniform],
                ["op": "writeBuffer", "buffer": 20,
                 "data": [UInt8](repeating: 7, count: 16).conformanceBase64],
            ], present: false)
            // Batch 3 (the real frame submit): batch 1's view must still be valid.
            try harness.executeExpectingSuccess([
                ["op": "beginRenderPass", "colorAttachments": [[
                    "view": 11, "loadOp": "clear", "storeOp": "store",
                    "clearValue": ["r": 1, "g": 0, "b": 0, "a": 1],
                ]]],
                ["op": "endPass"],
            ])
            try harness.expectPixel(
                x: 32, y: 32, equals: (255, 0, 0, 255), "drawn with a handle that survived the mid-frame submit"
            )
        }
    }

    /// When present closes the frame, frame-scoped handles expire — reuse is a validation error.
    static var presentExpiresFrame: Check {
        Check("present-expires-frame") { harness in
            try harness.executeExpectingSuccess(
                harness.canvasSetup + [
                    ["op": "beginRenderPass", "colorAttachments": [[
                        "view": 11, "loadOp": "clear", "storeOp": "store",
                        "clearValue": ["r": 0, "g": 1, "b": 0, "a": 1],
                    ]]],
                    ["op": "endPass"],
                ]
            )
            let errors = try harness.executeExpectingFailure([
                ["op": "beginRenderPass", "colorAttachments": [[
                    "view": 11, "loadOp": "clear", "storeOp": "store",
                ]]],
            ])
            guard errors.contains(where: { ($0["kind"] as? String) == "validation" }) else {
                throw ConformanceFailure("reusing an expired handle was not a validation error — \(errors)")
            }
        }
    }

    /// A present batch with **no** commands is a tick's closing batch — it must still close the frame
    /// (handles expire, and what was drawn stays on screen).
    static var emptyPresentClosesFrame: Check {
        Check("empty-present-closes-frame") { harness in
            try harness.executeExpectingSuccess(
                harness.canvasSetup + [
                    ["op": "beginRenderPass", "colorAttachments": [[
                        "view": 11, "loadOp": "clear", "storeOp": "store",
                        "clearValue": ["r": 0.25, "g": 0.5, "b": 0.75, "a": 1.0],
                    ]]],
                    ["op": "endPass"],
                ], present: false
            )
            // Tick close — closes the frame even with no commands.
            try harness.executeExpectingSuccess([], present: true)
            _ = try harness.executeExpectingFailure([
                ["op": "beginRenderPass", "colorAttachments": [[
                    "view": 11, "loadOp": "clear", "storeOp": "store",
                ]]],
            ])
            try harness.expectPixel(
                x: 32, y: 32, equals: (64, 128, 191, 255), "the picture still there after the frame closed"
            )
        }
    }

    // MARK: - Callback APIs

    /// `readBuffer` (mapAsync + getMappedRange) — full and partial reads, refusing re-entry while
    /// mapped, reuse after unmap, and a clean failure for a missing handle.
    static var readBufferContract: Check {
        Check("read-buffer-contract") { harness in
            let bytes = (0..<32).map(UInt8.init)
            try harness.executeExpectingSuccess([
                ["op": "createBuffer", "id": 1,
                 "usage": WebGPUUsage.mapRead | WebGPUUsage.copyDst,
                 "data": bytes.conformanceBase64],
            ], present: false)

            let all = try harness.readBufferSync(handle: 1)
            guard [UInt8](all) == bytes else {
                throw ConformanceFailure("the full read differs from what was written — \([UInt8](all))")
            }

            // The buffer is mapped while being read — re-entry must be refused (the spec's unavailable).
            let box = LifecycleResultBox()
            let semaphore = DispatchSemaphore(value: 0)
            harness.runtime.readBuffer(handle: 1, offset: 0, size: nil) { result in
                box.value = result
                semaphore.signal()
            }
            try harness.waitPumping(semaphore, timeout: 10) {
                ConformanceFailure("readBuffer did not answer while mapped")
            }
            guard (box.value["ok"] as? Bool) == false else {
                throw ConformanceFailure("a re-read while mapped was not refused")
            }

            // After unmap it can be read again, and an offset/size partial read gives just that range.
            try harness.executeExpectingSuccess([
                ["op": "unmapBuffer", "buffer": 1],
            ], present: false)
            let part = try harness.readBufferSync(handle: 1, offset: 8, size: 4)
            guard [UInt8](part) == Array(bytes[8..<12]) else {
                throw ConformanceFailure("the partial read (offset 8, size 4) differs — \([UInt8](part))")
            }

            // A missing handle is an error payload, not a crash.
            let missing = LifecycleResultBox()
            let missingSemaphore = DispatchSemaphore(value: 0)
            harness.runtime.readBuffer(handle: 999, offset: 0, size: nil) { result in
                missing.value = result
                missingSemaphore.signal()
            }
            try harness.waitPumping(missingSemaphore, timeout: 10) {
                ConformanceFailure("readBuffer with a missing handle did not answer")
            }
            guard (missing.value["ok"] as? Bool) == false,
                  (missing.value["errors"] as? [[String: Any]])?.isEmpty == false else {
                throw ConformanceFailure("a missing handle did not answer with an error — \(missing.value)")
            }
        }
    }

    /// After `resizeCanvas`, `canvasInfo`, `getCurrentTexture` and pixel readback all see the new size.
    static var resizeCanvasReflects: Check {
        Check("resize-canvas") { harness in
            try harness.executeExpectingSuccess([
                ["op": "configureCanvas", "canvas": harness.canvas, "format": "rgba8unorm"],
            ])
            harness.runtime.resizeCanvas(
                identifier: harness.canvas, drawableSize: CGSize(width: 32, height: 16)
            )
            let info = harness.runtime.canvasInfo(identifier: harness.canvas)
            guard (info["width"] as? Int) == 32, (info["height"] as? Int) == 16 else {
                throw ConformanceFailure(
                    "after resize canvasInfo was \(info["width"] ?? "?")×\(info["height"] ?? "?") — expected 32×16"
                )
            }
            try harness.executeExpectingSuccess([
                ["op": "getCurrentTexture", "id": 10, "canvas": harness.canvas],
                ["op": "createTextureView", "id": 11, "texture": 10],
                ["op": "beginRenderPass", "colorAttachments": [[
                    "view": 11, "loadOp": "clear", "storeOp": "store",
                    "clearValue": ["r": 1, "g": 0, "b": 0, "a": 1],
                ]]],
                ["op": "endPass"],
            ])
            let readback = try harness.readback()
            guard readback.width == 32, readback.height == 16 else {
                throw ConformanceFailure(
                    "after resize the readback was \(readback.width)×\(readback.height) — expected 32×16"
                )
            }
        }
    }

    /// `shaderCompilationInfo` — broken WGSL is diagnosed as a `GPUCompilationMessage` with all six
    /// keys, a healthy module gives an empty list, and a missing handle fails cleanly.
    static var shaderCompilationInfoShape: Check {
        Check("shader-compilation-info") { harness in
            // Whether errors are reported immediately or deferred may vary per runtime — here we only look at the diagnostic channel.
            _ = harness.execute([
                ["op": "createShaderModule", "id": 1, "code": "fn broken( {"],
            ], present: false)
            let info = harness.runtime.shaderCompilationInfo(handle: 1)
            guard (info["ok"] as? Bool) == true,
                  let messages = info["messages"] as? [[String: Any]], !messages.isEmpty else {
                throw ConformanceFailure("no diagnostic for a broken shader — \(info)")
            }
            // The six keys of the spec's `GPUCompilationMessage` — editors underline by these names.
            let required = ["message", "type", "lineNum", "linePos", "offset", "length"]
            for message in messages {
                let missing = required.filter { message[$0] == nil }
                guard missing.isEmpty else {
                    throw ConformanceFailure(
                        "GPUCompilationMessage keys missing — \(missing.joined(separator: ", "))"
                    )
                }
            }

            try harness.executeExpectingSuccess([
                ["op": "createShaderModule", "id": 2, "code": WebGPUConformance.fullscreenShader],
            ], present: false)
            let clean = harness.runtime.shaderCompilationInfo(handle: 2)
            guard (clean["ok"] as? Bool) == true,
                  (clean["messages"] as? [[String: Any]])?.isEmpty == true else {
                throw ConformanceFailure("a healthy module carries diagnostics — \(clean)")
            }

            guard (harness.runtime.shaderCompilationInfo(handle: 999)["ok"] as? Bool) == false else {
                throw ConformanceFailure("asked about a missing module but got ok: true")
            }
        }
    }

    /// `language: "msl"` is an **optional feature** — succeed if supported, otherwise refuse cleanly
    /// with `unsupported` throughout (no crash, no validation — `docs/COMMAND-STREAM.md` §4-1).
    static var mslOptional: Check {
        Check("msl-optional") { harness in
            let result = harness.execute([
                ["op": "createShaderModule", "id": 1, "language": "msl", "code": """
                #include <metal_stdlib>
                using namespace metal;
                vertex float4 vs_main() { return float4(0.0); }
                """],
            ], present: false)
            if (result["ok"] as? Bool) == true { return }
            let errors = result["errors"] as? [[String: Any]] ?? []
            guard !errors.isEmpty,
                  errors.allSatisfy({ ($0["kind"] as? String) == "unsupported" }) else {
                throw ConformanceFailure(
                    "unsupported msl must be refused as unsupported — \(ConformanceHarness.describeErrors(result))"
                )
            }
        }
    }

    /// `decodeImage` (createImageBitmap) → `copyExternalImageToTexture` — verifies the decode and
    /// upload path down to the pixels using a built-in 2×2 PNG, with no assets.
    static var decodeImageUpload: Check {
        Check("decode-image-upload") { harness in
            let pixels: [UInt8] = [
                255, 0, 0, 255,    0, 255, 0, 255,     // row 1: red, green
                0, 0, 255, 255,    255, 255, 0, 255,   // row 2: blue, yellow
            ]
            let box = LifecycleResultBox()
            let semaphore = DispatchSemaphore(value: 0)
            harness.runtime.decodeImage(
                handle: 30,
                data: LifecyclePNG.encode(rgba: pixels, width: 2, height: 2),
                name: nil,
                options: WGPUImageDecodeOptions(),
                provider: nil
            ) { result in
                box.value = result
                semaphore.signal()
            }
            try harness.waitPumping(semaphore, timeout: 10) {
                ConformanceFailure("decodeImage did not return")
            }
            guard (box.value["ok"] as? Bool) == true,
                  (box.value["width"] as? Int) == 2, (box.value["height"] as? Int) == 2 else {
                throw ConformanceFailure("decodeImage failed — \(box.value)")
            }

            try harness.executeExpectingSuccess([
                ["op": "createTexture", "id": 1, "size": ["width": 2, "height": 2],
                 "format": "rgba8unorm",
                 "usage": WebGPUUsage.textureCopyDst | WebGPUUsage.textureCopySrc],
                ["op": "copyExternalImageToTexture",
                 "source": ["source": 30],
                 "destination": ["texture": 1],
                 "copySize": ["width": 2, "height": 2]],
                ["op": "createBuffer", "id": 2, "size": 512,
                 "usage": WebGPUUsage.copyDst | WebGPUUsage.mapRead],
                // bytesPerRow honours the spec's lower bound (a multiple of 256).
                ["op": "copyTextureToBuffer",
                 "source": ["texture": 1],
                 "destination": ["buffer": 2, "bytesPerRow": 256],
                 "copySize": ["width": 2, "height": 2]],
            ], present: false)

            let data = try harness.readBufferSync(handle: 2)
            let row0 = [UInt8](data.prefix(8))
            let row1 = [UInt8](data.dropFirst(256).prefix(8))
            guard row0 == [255, 0, 0, 255, 0, 255, 0, 255],
                  row1 == [0, 0, 255, 255, 255, 255, 0, 255] else {
                throw ConformanceFailure("the uploaded pixels differ — row 1 \(row0), row 2 \(row1)")
            }
        }
    }

    // MARK: - Frame signals

    /// `execute` (JS thread) and `processEvents` (tick thread) **are contractually called
    /// concurrently** (`docs/COMMAND-STREAM.md` §5-1). If the backend API is not thread-safe the
    /// implementation must serialize them, and one that skipped it shows up here as an internal backend
    /// assertion (process death) — the Dawn prototype really did die on a Metal encoder accounting
    /// assertion here. The success criterion is **finishing**.
    static var pumpConcurrency: Check {
        Check("pump-concurrency") { harness in
            final class StopFlag {
                private let lock = NSLock()
                private var stopped = false
                var isStopped: Bool {
                    lock.lock()
                    defer { lock.unlock() }
                    return stopped
                }
                func stop() {
                    lock.lock()
                    stopped = true
                    lock.unlock()
                }
            }
            let flag = StopFlag()
            let finished = DispatchSemaphore(value: 0)
            let runtime = harness.runtime
            DispatchQueue.global(qos: .userInitiated).async {
                while !flag.isStopped { runtime.processEvents() }
                finished.signal()
            }
            defer {
                flag.stop()
                _ = finished.wait(timeout: .now() + 5)
            }

            // Hammer batches while the pump runs — creation, upload and destruction touch the backend every batch.
            for index in 0..<200 {
                try harness.executeExpectingSuccess([
                    ["op": "createBuffer", "id": 40, "size": 256,
                     "usage": WebGPUUsage.copyDst | WebGPUUsage.copySrc],
                    ["op": "writeBuffer", "buffer": 40,
                     "data": [UInt8](repeating: UInt8(index % 256), count: 4).conformanceBase64],
                    ["op": "destroy", "id": 40],
                ], present: false)
            }
            // The mapping path under contention too — this pokes an implementation whose completion callback comes from the pump.
            try harness.executeExpectingSuccess([
                ["op": "createBuffer", "id": 41,
                 "usage": WebGPUUsage.mapRead | WebGPUUsage.copyDst,
                 "data": [UInt8]([1, 2, 3, 4]).conformanceBase64],
            ], present: false)
            let bytes = try harness.readBufferSync(handle: 41)
            guard [UInt8](bytes) == [1, 2, 3, 4] else {
                throw ConformanceFailure("the readback was corrupted under contention — \([UInt8](bytes))")
            }
        }
    }

    /// `isReadyForNextFrame` is true when idle and non-blocking. `processEvents` is safe to call at
    /// any time (a no-op is allowed).
    static var frameReadiness: Check {
        Check("frame-readiness") { harness in
            let start = Date()
            let ready = harness.runtime.isReadyForNextFrame
            guard Date().timeIntervalSince(start) < 0.1 else {
                throw ConformanceFailure("isReadyForNextFrame blocks")
            }
            guard ready else {
                throw ConformanceFailure("isReadyForNextFrame is false while idle")
            }
            harness.runtime.processEvents()
        }
    }
}

/// The callback can come from another thread, so the value lives in a reference type (the semaphore guarantees visibility).
private final class LifecycleResultBox {
    var value: [String: Any] = [:]
}

/// A minimal PNG encoder for the image check to run without assets (RGBA8, uncompressed stored deflate).
///
/// What is under test is not the encoding but **the runtime's decode path** — uncompressed is still a
/// valid PNG that any decoder (ImageIO and the rest) can open, and the bytes are deterministic so the
/// pixel assertions never wobble.
private enum LifecyclePNG {
    static func encode(rgba: [UInt8], width: Int, height: Int) -> Data {
        var raw = Data()
        for row in 0..<height {
            raw.append(0)   // filter: None
            raw.append(contentsOf: rgba[(row * width * 4)..<((row + 1) * width * 4)])
        }
        var png = Data([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
        var header = Data()
        header.append(contentsOf: bigEndian(UInt32(width)))
        header.append(contentsOf: bigEndian(UInt32(height)))
        header.append(contentsOf: [8, 6, 0, 0, 0])   // 8bit RGBA, deflate, adaptive, no interlace
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

    private static func zlibStored(_ payload: Data) -> Data {
        var data = Data([0x78, 0x01])
        var offset = 0
        while offset < payload.count {
            let length = min(65535, payload.count - offset)
            data.append(offset + length >= payload.count ? 1 : 0)
            data.append(UInt8(length & 0xFF))
            data.append(UInt8((length >> 8) & 0xFF))
            data.append(UInt8(~length & 0xFF))
            data.append(UInt8((~length >> 8) & 0xFF))
            data.append(payload.subdata(in: offset..<(offset + length)))
            offset += length
        }
        var a: UInt32 = 1, b: UInt32 = 0
        for byte in payload {
            a = (a + UInt32(byte)) % 65521
            b = (b + a) % 65521
        }
        data.append(contentsOf: bigEndian((b << 16) | a))
        return data
    }

    private static func bigEndian(_ value: UInt32) -> [UInt8] {
        [UInt8((value >> 24) & 0xFF), UInt8((value >> 16) & 0xFF),
         UInt8((value >> 8) & 0xFF), UInt8(value & 0xFF)]
    }

    private static let crcTable: [UInt32] = (0..<256).map { index -> UInt32 in
        var c = UInt32(index)
        for _ in 0..<8 { c = (c & 1) != 0 ? (0xEDB8_8320 ^ (c >> 1)) : (c >> 1) }
        return c
    }

    private static func crc32(_ data: Data) -> UInt32 {
        var c: UInt32 = 0xFFFF_FFFF
        for byte in data { c = crcTable[Int((c ^ UInt32(byte)) & 0xFF)] ^ (c >> 8) }
        return c ^ 0xFFFF_FFFF
    }
}
