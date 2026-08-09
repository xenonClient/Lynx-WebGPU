import Foundation
import CoreGraphics
import QuartzCore
import LynxWebGPUCore

/// A GPU-free stub runtime — the tangible proof that "`WebGPURuntime` can be implemented linking Core alone".
///
/// The members of this class are exactly the slots a Dawn runtime would fill. With no GPU it honestly
/// rejects the execution family of ops as `unsupported`, but **the wire policy is assembled entirely from
/// Core's public types** — `WGPUCommand` for dispatch, `WGPUErrorScopeStack` for error scopes,
/// `WGPUBatchResult` for the response shape, `WGPUDeferredErrorQueue` for deferred failures. It shows that
/// an external implementation can pass the contract checks (error accumulation, scopes, msl-optional, …) with those types alone.
///
/// `processEvents()` is deliberately left unimplemented — whether the protocol's default implementation
/// (a no-op) works from an external module too is part of what is being verified.
final class StubRuntime: WebGPURuntime {
    /// A placeholder object put into the registry — it only mimics handle lifetime (the registry contract).
    private final class Placeholder {}

    private let lock = NSLock()
    private let registry = WGPUObjectRegistry()
    private var errorScopes = WGPUErrorScopeStack()
    private let deferredErrors = WGPUDeferredErrorQueue()
    private var canvases: [String: (size: CGSize, format: WGPUTextureFormat)] = [:]

    // MARK: - Command stream

    func execute(_ payload: [String: Any]) -> [String: Any] {
        lock.lock()
        defer { lock.unlock() }

        let reader = WGPUValueReader(payload)
        guard let commands = try? reader.requiredObjects("commands") else {
            return WGPUBatchResult.failure([.validation("there is no commands array")])
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
                // Core's dispatch table — the op name switch is not written again here.
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
                        throw WGPUError.validation("canvas '\(configuration.canvasId)' does not exist")
                    }
                    canvases[configuration.canvasId]?.format = configuration.format
                case .endPass:
                    break
                default:
                    throw WGPUError.unsupported(
                        "the stub runtime does not execute '\(command.opName)' (no GPU)"
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

    // MARK: - Queries

    func adapterInfo() -> [String: Any] {
        // A minimal response that keeps the spec spelling — the shape contract (adapter-limits-spelling) can be kept with no GPU.
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
        ["ok": false, "errors": [WGPUError.validation("GPUShaderModule #\(handle) does not exist").payload]]
    }

    func canvasInfo(identifier: String) -> [String: Any] {
        lock.lock()
        defer { lock.unlock() }
        guard let canvas = canvases[identifier] else {
            return ["ok": false, "errors": [
                WGPUError.validation("canvas '\(identifier)' does not exist").payload,
            ]]
        }
        return [
            "ok": true,
            "width": Int(canvas.size.width),
            "height": Int(canvas.size.height),
            "format": canvas.format.rawValue,
        ]
    }

    // MARK: - Asynchronous

    func readBuffer(handle: Int, offset: Int, size: Int?, completion: @escaping ([String: Any]) -> Void) {
        completion(["ok": false, "errors": [
            WGPUError.unsupported("the stub runtime has no GPU memory").payload,
        ]])
    }

    func decodeImage(
        handle: Int, data: Data?, name: String?, options: WGPUImageDecodeOptions,
        provider: WGPUAssetProvider?, completion: @escaping ([String: Any]) -> Void
    ) {
        completion(["ok": false, "errors": [
            WGPUError.unsupported("the stub runtime does not decode images").payload,
        ]])
    }

    // MARK: - Canvases

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
        throw WGPUError.unsupported("the stub runtime does not draw, so there are no pixels to read")
    }

    // MARK: - Frames and lifetime

    var isReadyForNextFrame: Bool { true }

    func reset() {
        lock.lock()
        registry.removeAll()
        errorScopes.discardAll()
        _ = deferredErrors.drain()
        lock.unlock()
    }
}
