import Foundation
import CoreGraphics
import LynxWebGPUCore

/// 프레임 수명·콜백 API·조회 계약의 검사들.
///
/// `Checks.swift`가 "그리면 그려지는가"를 본다면, 여기는 **경계**를 본다 — present가 프레임을
/// 언제 닫는가, 콜백형 API가 무엇을 돌려주는가. 이 계약들은 어긋나도 단순한 씬에서는 티가
/// 안 나고, Three.js류의 지연 파이프라인 생성처럼 프레임을 여러 배치로 쪼개는 코드에서만
/// 터진다 (`docs/COMMAND-STREAM.md` §2). 그래서 스위트가 직접 경계를 몰아 본다.
public extension WebGPUConformance {

    // MARK: - 프레임 수명

    /// `present: false` 배치는 프레임을 닫지 않는다 — 프레임 스코프 핸들이 살아남는다.
    ///
    /// 가운데 배치가 **writeBuffer 하나로 커맨드 버퍼를 만드는** 것이 요점이다: 커밋이
    /// 일어나는데도 present·핸들 만료는 미뤄야 한다. Three.js의 지연 파이프라인 생성이
    /// 정확히 이 경로를 밟았다 (pop 즉시 flush).
    static var presentFalsePreservesFrame: Check {
        Check("present-false-preserves-frame") { harness in
            // 배치 1 (프레임 열기): 드로어블 텍스처(10)와 뷰(11)를 얻는다.
            try harness.executeExpectingSuccess(harness.canvasSetup, present: false)
            // 배치 2 (프레임 중간): 커맨드 버퍼가 생기는 위험 경로.
            try harness.executeExpectingSuccess([
                ["op": "createBuffer", "id": 20, "size": 16,
                 "usage": WebGPUUsage.copyDst | WebGPUUsage.uniform],
                ["op": "writeBuffer", "buffer": 20,
                 "data": [UInt8](repeating: 7, count: 16).conformanceBase64],
            ], present: false)
            // 배치 3 (진짜 프레임 제출): 배치 1의 뷰가 아직 유효해야 한다.
            try harness.executeExpectingSuccess([
                ["op": "beginRenderPass", "colorAttachments": [[
                    "view": 11, "loadOp": "clear", "storeOp": "store",
                    "clearValue": ["r": 1, "g": 0, "b": 0, "a": 1],
                ]]],
                ["op": "endPass"],
            ])
            try harness.expectPixel(
                x: 32, y: 32, equals: (255, 0, 0, 255), "중간 제출을 지나 살아남은 핸들로 그린 결과"
            )
        }
    }

    /// present가 프레임을 닫으면 프레임 스코프 핸들은 만료된다 — 재사용은 validation 오류다.
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
                throw ConformanceFailure("만료된 핸들 재사용이 validation이 아니다 — \(errors)")
            }
        }
    }

    /// 명령이 **빈** present 배치는 틱의 마무리다 — 그래도 프레임을 닫아야 한다
    /// (핸들 만료 + 그린 내용은 화면에 남는다).
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
            // 틱 마무리 — 커맨드가 없어도 프레임을 닫는다.
            try harness.executeExpectingSuccess([], present: true)
            _ = try harness.executeExpectingFailure([
                ["op": "beginRenderPass", "colorAttachments": [[
                    "view": 11, "loadOp": "clear", "storeOp": "store",
                ]]],
            ])
            try harness.expectPixel(
                x: 32, y: 32, equals: (64, 128, 191, 255), "프레임을 닫은 뒤에도 남아 있는 그림"
            )
        }
    }

    // MARK: - 콜백 API

    /// `readBuffer`(mapAsync + getMappedRange) — 전량/부분 읽기, 매핑 중 재진입 거부,
    /// unmap 후 재사용, 없는 핸들의 깨끗한 실패.
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
                throw ConformanceFailure("전량 읽기가 쓴 값과 다르다 — \([UInt8](all))")
            }

            // 읽는 동안 버퍼는 매핑 상태다 — 재진입은 거부되어야 한다 (명세의 unavailable).
            let box = LifecycleResultBox()
            let semaphore = DispatchSemaphore(value: 0)
            harness.runtime.readBuffer(handle: 1, offset: 0, size: nil) { result in
                box.value = result
                semaphore.signal()
            }
            try harness.waitPumping(semaphore, timeout: 10) {
                ConformanceFailure("매핑 중 readBuffer가 답하지 않았다")
            }
            guard (box.value["ok"] as? Bool) == false else {
                throw ConformanceFailure("매핑 중 재읽기가 거부되지 않았다")
            }

            // unmap하면 다시 읽을 수 있고, offset/size 부분 읽기가 그 구간만 준다.
            try harness.executeExpectingSuccess([
                ["op": "unmapBuffer", "buffer": 1],
            ], present: false)
            let part = try harness.readBufferSync(handle: 1, offset: 8, size: 4)
            guard [UInt8](part) == Array(bytes[8..<12]) else {
                throw ConformanceFailure("부분 읽기(offset 8, size 4)가 다르다 — \([UInt8](part))")
            }

            // 없는 핸들은 크래시가 아니라 오류 페이로드다.
            let missing = LifecycleResultBox()
            let missingSemaphore = DispatchSemaphore(value: 0)
            harness.runtime.readBuffer(handle: 999, offset: 0, size: nil) { result in
                missing.value = result
                missingSemaphore.signal()
            }
            try harness.waitPumping(missingSemaphore, timeout: 10) {
                ConformanceFailure("없는 핸들 readBuffer가 답하지 않았다")
            }
            guard (missing.value["ok"] as? Bool) == false,
                  (missing.value["errors"] as? [[String: Any]])?.isEmpty == false else {
                throw ConformanceFailure("없는 핸들이 오류로 답하지 않았다 — \(missing.value)")
            }
        }
    }

    /// `resizeCanvas` 뒤에는 `canvasInfo`·`getCurrentTexture`·픽셀 읽기가 전부 새 크기를 본다.
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
                    "resize 뒤 canvasInfo가 \(info["width"] ?? "?")×\(info["height"] ?? "?") — 기대 32×16"
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
                    "resize 뒤 리드백이 \(readback.width)×\(readback.height) — 기대 32×16"
                )
            }
        }
    }

    /// `shaderCompilationInfo` — 깨진 WGSL은 6키를 갖춘 `GPUCompilationMessage`로 진단되고,
    /// 정상 모듈은 빈 목록, 없는 핸들은 깨끗한 실패다.
    static var shaderCompilationInfoShape: Check {
        Check("shader-compilation-info") { harness in
            // 오류 보고 여부(즉시 vs 지연)는 런타임마다 갈릴 수 있다 — 여기서는 진단 통로만 본다.
            _ = harness.execute([
                ["op": "createShaderModule", "id": 1, "code": "fn broken( {"],
            ], present: false)
            let info = harness.runtime.shaderCompilationInfo(handle: 1)
            guard (info["ok"] as? Bool) == true,
                  let messages = info["messages"] as? [[String: Any]], !messages.isEmpty else {
                throw ConformanceFailure("깨진 셰이더의 진단이 없다 — \(info)")
            }
            // 명세 `GPUCompilationMessage`의 여섯 키 — 편집기가 이 이름으로 밑줄을 긋는다.
            let required = ["message", "type", "lineNum", "linePos", "offset", "length"]
            for message in messages {
                let missing = required.filter { message[$0] == nil }
                guard missing.isEmpty else {
                    throw ConformanceFailure(
                        "GPUCompilationMessage 키 누락 — \(missing.joined(separator: ", "))"
                    )
                }
            }

            try harness.executeExpectingSuccess([
                ["op": "createShaderModule", "id": 2, "code": WebGPUConformance.fullscreenShader],
            ], present: false)
            let clean = harness.runtime.shaderCompilationInfo(handle: 2)
            guard (clean["ok"] as? Bool) == true,
                  (clean["messages"] as? [[String: Any]])?.isEmpty == true else {
                throw ConformanceFailure("정상 모듈에 진단이 있다 — \(clean)")
            }

            guard (harness.runtime.shaderCompilationInfo(handle: 999)["ok"] as? Bool) == false else {
                throw ConformanceFailure("없는 모듈을 물었는데 ok: true다")
            }
        }
    }

    /// `language: "msl"`은 **선택 기능**이다 — 지원하면 성공, 아니면 전원 `unsupported`로
    /// 깨끗이 거부한다 (크래시·validation 금지 — `docs/COMMAND-STREAM.md` §4-1).
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
                    "msl 미지원은 unsupported로 거부해야 한다 — \(ConformanceHarness.describeErrors(result))"
                )
            }
        }
    }

    /// `decodeImage`(createImageBitmap) → `copyExternalImageToTexture` — 애셋 없이
    /// 내장 2×2 PNG로 디코딩·업로드 경로를 픽셀까지 확인한다.
    static var decodeImageUpload: Check {
        Check("decode-image-upload") { harness in
            let pixels: [UInt8] = [
                255, 0, 0, 255,    0, 255, 0, 255,     // 1행: 빨강, 초록
                0, 0, 255, 255,    255, 255, 0, 255,   // 2행: 파랑, 노랑
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
                ConformanceFailure("decodeImage가 돌아오지 않았다")
            }
            guard (box.value["ok"] as? Bool) == true,
                  (box.value["width"] as? Int) == 2, (box.value["height"] as? Int) == 2 else {
                throw ConformanceFailure("decodeImage 실패 — \(box.value)")
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
                // bytesPerRow는 명세 하한(256의 배수)을 지킨다.
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
                throw ConformanceFailure("업로드된 픽셀이 다르다 — 1행 \(row0), 2행 \(row1)")
            }
        }
    }

    // MARK: - 프레임 신호

    /// `execute`(JS 스레드)와 `processEvents`(틱 스레드)는 **동시에 불리는 것이 계약**이다
    /// (`docs/COMMAND-STREAM.md` §5-1). 백엔드 API가 스레드 안전하지 않으면 구현이 직렬화해야
    /// 하고, 빠뜨린 구현은 여기서 백엔드 내부 단언(프로세스 종료)으로 드러난다 — Dawn 시제품이
    /// 실제로 Metal 인코더 회계 단언으로 죽었던 자리다. 성공 기준은 **완주**다.
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

            // 펌프가 도는 동안 배치를 몰아친다 — 생성·업로드·파괴가 매 배치 백엔드를 만진다.
            for index in 0..<200 {
                try harness.executeExpectingSuccess([
                    ["op": "createBuffer", "id": 40, "size": 256,
                     "usage": WebGPUUsage.copyDst | WebGPUUsage.copySrc],
                    ["op": "writeBuffer", "buffer": 40,
                     "data": [UInt8](repeating: UInt8(index % 256), count: 4).conformanceBase64],
                    ["op": "destroy", "id": 40],
                ], present: false)
            }
            // 매핑 경로도 경쟁 아래에서 — 완료 콜백이 펌프에서 오는 구현을 그대로 찌른다.
            try harness.executeExpectingSuccess([
                ["op": "createBuffer", "id": 41,
                 "usage": WebGPUUsage.mapRead | WebGPUUsage.copyDst,
                 "data": [UInt8]([1, 2, 3, 4]).conformanceBase64],
            ], present: false)
            let bytes = try harness.readBufferSync(handle: 41)
            guard [UInt8](bytes) == [1, 2, 3, 4] else {
                throw ConformanceFailure("경쟁 중 리드백이 오염됐다 — \([UInt8](bytes))")
            }
        }
    }

    /// `isReadyForNextFrame`은 유휴 상태에서 true이고 논블로킹이다. `processEvents`는
    /// 언제 불러도 안전하다 (no-op 허용).
    static var frameReadiness: Check {
        Check("frame-readiness") { harness in
            let start = Date()
            let ready = harness.runtime.isReadyForNextFrame
            guard Date().timeIntervalSince(start) < 0.1 else {
                throw ConformanceFailure("isReadyForNextFrame이 블로킹한다")
            }
            guard ready else {
                throw ConformanceFailure("유휴 상태인데 isReadyForNextFrame이 false다")
            }
            harness.runtime.processEvents()
        }
    }
}

/// 콜백이 다른 스레드에서 올 수 있어 값을 참조 타입에 담는다 (세마포어가 가시성을 보장한다).
private final class LifecycleResultBox {
    var value: [String: Any] = [:]
}

/// 애셋 없이 도는 이미지 검사용 최소 PNG 인코더 (RGBA8 · 무압축 stored deflate).
///
/// 검증 대상은 인코딩이 아니라 **런타임의 디코딩 경로**다 — 무압축이라도 유효한 PNG이므로
/// 어느 디코더(ImageIO 등)든 열 수 있고, 바이트가 결정적이라 픽셀 단언이 흔들리지 않는다.
private enum LifecyclePNG {
    static func encode(rgba: [UInt8], width: Int, height: Int) -> Data {
        var raw = Data()
        for row in 0..<height {
            raw.append(0)   // 필터: None
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
