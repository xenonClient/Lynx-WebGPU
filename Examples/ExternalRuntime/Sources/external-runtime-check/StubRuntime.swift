import Foundation
import CoreGraphics
import QuartzCore
import LynxWebGPUCore

/// GPU 없는 스텁 런타임 — "Core만 링크해 `WebGPURuntime`을 구현할 수 있다"의 실물 증거.
///
/// Dawn 런타임이 채울 자리가 정확히 이 클래스의 멤버들이다. GPU가 없으므로 실행 계열 op은
/// 정직하게 `unsupported`로 거부하지만, **와이어 정책은 전부 Core의 공용 타입으로 조립한다** —
/// 디스패치는 `WGPUCommand`, 오류 스코프는 `WGPUErrorScopeStack`, 응답 모양은 `WGPUBatchResult`,
/// 지연 실패는 `WGPUDeferredErrorQueue`. 외부 구현이 이 타입들만으로 계약 검사
/// (오류 누적·스코프·msl-optional 등)를 통과할 수 있음을 보인다.
///
/// `processEvents()`는 일부러 구현하지 않는다 — 프로토콜 기본 구현(no-op)이 외부 모듈에서도
/// 동작하는지가 검증 대상이다.
final class StubRuntime: WebGPURuntime {
    /// 레지스트리에 넣는 자리표시 객체 — 핸들 수명(레지스트리 계약)만 흉내 낸다.
    private final class Placeholder {}

    private let lock = NSLock()
    private let registry = WGPUObjectRegistry()
    private var errorScopes = WGPUErrorScopeStack()
    private let deferredErrors = WGPUDeferredErrorQueue()
    private var canvases: [String: (size: CGSize, format: WGPUTextureFormat)] = [:]

    // MARK: - 커맨드 스트림

    func execute(_ payload: [String: Any]) -> [String: Any] {
        lock.lock()
        defer { lock.unlock() }

        let reader = WGPUValueReader(payload)
        guard let commands = try? reader.requiredObjects("commands") else {
            return WGPUBatchResult.failure([.validation("commands 배열이 없다")])
        }

        var errors: [WGPUError] = []
        var poppedScopes: [WGPUPoppedErrorScope] = []
        func record(_ error: WGPUError) {
            if errorScopes.capture(error) { return }
            errors.append(error)
        }

        for failure in deferredErrors.drain() { record(failure) }

        for (index, commandReader) in commands.enumerated() {
            do {
                // Core의 디스패치 표 — op 이름 switch를 여기서 다시 쓰지 않는다.
                let command = try WGPUCommand(from: commandReader)
                switch command {
                case .pushErrorScope(let filter, let decodeFailure):
                    errorScopes.push(filter)
                    if let decodeFailure { throw decodeFailure }
                case .popErrorScope:
                    poppedScopes.append(errorScopes.pop())
                case .createBuffer(let create):
                    registry.insert(Placeholder(), at: create.id)
                case .destroy(let destroy):
                    registry.remove(destroy.id)
                case .configureCanvas(let configuration):
                    guard canvases[configuration.canvasId] != nil else {
                        throw WGPUError.validation("캔버스 '\(configuration.canvasId)'이(가) 없다")
                    }
                    canvases[configuration.canvasId]?.format = configuration.format
                case .endPass:
                    break
                default:
                    throw WGPUError.unsupported(
                        "스텁 런타임은 '\(command.opName)'을(를) 실행하지 않는다 (GPU 없음)"
                    )
                }
            } catch let error as WGPUError {
                record(WGPUError(
                    kind: error.kind,
                    message: error.message,
                    path: error.path ?? "commands[\(index)]",
                    line: error.line
                ))
            } catch {
                record(.backend("\(error)", path: "commands[\(index)]"))
            }
        }

        return WGPUBatchResult(
            commandCount: commands.count,
            liveObjectCount: registry.count,
            errors: errors,
            poppedScopes: poppedScopes
        ).payload
    }

    // MARK: - 조회

    func adapterInfo() -> [String: Any] {
        // 명세 철자를 지킨 최소 응답 — 모양 계약(adapter-limits-spelling)은 GPU 없이도 지킬 수 있다.
        [
            "ok": true,
            "info": [
                "vendor": "stub", "architecture": "", "device": "",
                "description": "external-runtime-check stub",
                "isFallbackAdapter": true, "subgroupMinSize": 0, "subgroupMaxSize": 0,
            ],
            "preferredCanvasFormat": WGPUTextureFormat.bgra8unorm.rawValue,
            "limits": [
                "maxTextureDimension2D": 8192, "maxBindGroups": 4, "maxBufferSize": 1 << 28,
                "maxUniformBufferBindingSize": 65536, "maxStorageBufferBindingSize": 1 << 28,
                "minUniformBufferOffsetAlignment": 256, "minStorageBufferOffsetAlignment": 256,
                "maxVertexBuffers": 8, "maxVertexAttributes": 16, "maxColorAttachments": 4,
                "maxComputeWorkgroupSizeX": 256, "maxComputeWorkgroupsPerDimension": 65535,
            ],
            "features": [String](),
        ]
    }

    func shaderCompilationInfo(handle: Int) -> [String: Any] {
        ["ok": false, "errors": [WGPUError.validation("GPUShaderModule #\(handle)이(가) 없다").payload]]
    }

    func canvasInfo(identifier: String) -> [String: Any] {
        lock.lock()
        defer { lock.unlock() }
        guard let canvas = canvases[identifier] else {
            return ["ok": false, "errors": [
                WGPUError.validation("캔버스 '\(identifier)'이(가) 없다").payload,
            ]]
        }
        return [
            "ok": true,
            "width": Int(canvas.size.width),
            "height": Int(canvas.size.height),
            "format": canvas.format.rawValue,
        ]
    }

    // MARK: - 비동기

    func readBuffer(handle: Int, offset: Int, size: Int?, completion: @escaping ([String: Any]) -> Void) {
        completion(["ok": false, "errors": [
            WGPUError.unsupported("스텁 런타임은 GPU 메모리가 없다").payload,
        ]])
    }

    func decodeImage(
        handle: Int, data: Data?, name: String?, options: WGPUImageDecodeOptions,
        provider: WGPUAssetProvider?, completion: @escaping ([String: Any]) -> Void
    ) {
        completion(["ok": false, "errors": [
            WGPUError.unsupported("스텁 런타임은 이미지를 디코딩하지 않는다").payload,
        ]])
    }

    // MARK: - 캔버스

    func attachCanvas(identifier: String, layer: CAMetalLayer) {
        lock.lock()
        canvases[identifier] = (layer.drawableSize, .bgra8unorm)
        lock.unlock()
    }

    func attachOffscreenCanvas(identifier: String, size: CGSize) throws {
        lock.lock()
        canvases[identifier] = (size, .rgba8unorm)
        lock.unlock()
    }

    func resizeCanvas(identifier: String, drawableSize: CGSize) {
        lock.lock()
        canvases[identifier]?.size = drawableSize
        lock.unlock()
    }

    func detachCanvas(identifier: String) {
        lock.lock()
        canvases.removeValue(forKey: identifier)
        lock.unlock()
    }

    func readCanvasPixels(identifier: String) throws -> WGPUPixelReadback {
        throw WGPUError.unsupported("스텁 런타임은 그리지 않으므로 읽을 픽셀이 없다")
    }

    // MARK: - 프레임 · 수명

    var isReadyForNextFrame: Bool { true }

    func reset() {
        lock.lock()
        registry.removeAll()
        errorScopes.discardAll()
        _ = deferredErrors.drain()
        lock.unlock()
    }
}
