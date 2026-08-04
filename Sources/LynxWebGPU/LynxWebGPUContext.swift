import Foundation
import Metal
import LynxWebGPUCore
import LynxWebGPUShader

/// WebGPU 런타임 하나. 디바이스·큐·객체 레지스트리·표면 등록부를 소유한다.
///
/// 호스트 앱이 이 객체를 만들어 Lynx 브리지에 넘기거나 (`LynxWebGPUBridge`),
/// Lynx 없이 Swift에서 직접 커맨드 스트림을 실행할 수 있다 (테스트 하네스가 그렇게 쓴다).
///
/// **스레딩** — `execute(_:)`는 Lynx의 JS 스레드에서 호출되는 것을 전제로 하며 내부 락으로
/// 직렬화된다. 표면 등록/해제는 메인 스레드에서 일어나므로 등록부도 락으로 보호한다.
public final class LynxWebGPUContext {
    public let device: MTLDevice
    public let queue: MTLCommandQueue

    private let registry = WGPUObjectRegistry()
    private var surfaces: [String: WGPUSurface] = [:]
    private let surfaceLock = NSLock()
    private let executionLock = NSLock()
    private var interpreter: WGPUCommandInterpreter!

    public init(device: MTLDevice? = nil) throws {
        guard let resolved = device ?? MTLCreateSystemDefaultDevice() else {
            throw WGPUError.backend("Metal 디바이스를 만들 수 없다 (시뮬레이터/기기 지원 확인)")
        }
        guard let queue = resolved.makeCommandQueue() else {
            throw WGPUError.backend("MTLCommandQueue 생성 실패")
        }
        self.device = resolved
        self.queue = queue
        self.interpreter = WGPUCommandInterpreter(
            device: resolved,
            queue: queue,
            registry: registry,
            surfaceProvider: { [weak self] identifier in self?.surface(for: identifier) }
        )
        WGPULog.device.info("WebGPU 컨텍스트 시작 — \(resolved.name, privacy: .public)")
    }

    // MARK: - 표면 등록

    public func registerSurface(_ surface: WGPUSurface) {
        surfaceLock.lock()
        surfaces[surface.identifier] = surface
        surfaceLock.unlock()
    }

    public func unregisterSurface(identifier: String) {
        surfaceLock.lock()
        surfaces.removeValue(forKey: identifier)
        surfaceLock.unlock()
    }

    public func surface(for identifier: String) -> WGPUSurface? {
        surfaceLock.lock()
        defer { surfaceLock.unlock() }
        return surfaces[identifier]
    }

    public var registeredSurfaceIdentifiers: [String] {
        surfaceLock.lock()
        defer { surfaceLock.unlock() }
        return Array(surfaces.keys).sorted()
    }

    /// 등록된 모든 표면이 새 프레임을 받을 수 있는가.
    ///
    /// 프레임 티커가 이 값을 보고 포화 시 `webgpu:frame` 틱을 건너뛴다 — GPU가 in-flight
    /// 한도(표면당 3프레임)만큼 밀려 있을 때 JS가 프레임을 만들면 `nextDrawable()`이
    /// JS 스레드 전체(터치 핸들러·타이머 포함)를 최대 1초까지 세우기 때문이다.
    /// 완료 핸들러가 돌아오면 다음 틱부터 자연히 재개된다.
    public var isReadyForNextFrame: Bool {
        surfaceLock.lock()
        defer { surfaceLock.unlock() }
        return surfaces.values.allSatisfy { $0.isReadyForNextFrame }
    }

    // MARK: - 커맨드 실행

    /// 커맨드 스트림 하나를 실행한다.
    ///
    /// - Parameter payload: `{"commands": [ {op: …}, … ]}`
    /// - Returns: `{"ok": Bool, "errors": [...], "canvases": {...}}` — JS로 그대로 돌려준다.
    public func execute(_ payload: [String: Any]) -> [String: Any] {
        executionLock.lock()
        defer { executionLock.unlock() }

        let reader = WGPUValueReader(payload)
        let commands: [WGPUValueReader]
        do {
            commands = try reader.requiredObjects("commands")
        } catch let error as WGPUError {
            return ["ok": false, "errors": [error.payload]]
        } catch {
            return ["ok": false, "errors": [WGPUError.validation("\(error)").payload]]
        }
        return interpreter.execute(commands)
    }

    /// 편의 오버로드 — 배열을 그대로 넘긴다.
    @discardableResult
    public func execute(commands: [[String: Any]]) -> [String: Any] {
        execute(["commands": commands])
    }

    /// 버퍼 내용을 읽는다 (`GPUBuffer.mapAsync` + `getMappedRange`에 해당).
    ///
    /// 직전에 제출한 GPU 작업이 끝난 뒤에 읽어야 하므로 비동기다.
    public func readBuffer(
        handle: Int,
        offset: Int = 0,
        size: Int? = nil,
        completion: @escaping ([String: Any]) -> Void
    ) {
        let target: WGPUBufferObject
        do {
            target = try registry.lookup(WGPUHandle(handle), as: WGPUBufferObject.self, kind: "GPUBuffer")
        } catch let error as WGPUError {
            completion(["ok": false, "errors": [error.payload]])
            return
        } catch {
            completion(["ok": false, "errors": [WGPUError.backend("\(error)").payload]])
            return
        }

        // 읽는 동안 이 버퍼는 "unavailable"이다 — 다음 프레임의 쓰기가 같은 메모리에 겹치면
        // JS가 받는 값이 어느 프레임 것인지 보장되지 않는다 (`WGPUBufferObject.isMapped`).
        executionLock.lock()
        let alreadyMapped = target.isMapped
        if !alreadyMapped { target.isMapped = true }
        let pending = interpreter.lastCommittedBuffer
        executionLock.unlock()

        guard !alreadyMapped else {
            completion([
                "ok": false,
                "errors": [WGPUError.validation(
                    "GPUBuffer \(WGPUHandle(handle))은(는) 이미 매핑 중이다 (unmap()을 먼저 부를 것)"
                ).payload],
            ])
            return
        }

        let length = size ?? (target.size - offset)
        let deliver = { [weak self] (failure: WGPUError?) in
            guard let self else { return }
            self.executionLock.lock()
            defer { self.executionLock.unlock() }
            // 실패했으면 매핑을 세우지 않는다 — 명세도 실패한 mapAsync는 버퍼를 매핑하지 않는다.
            let fail = { (error: WGPUError) in
                target.isMapped = false
                completion(["ok": false, "errors": [error.payload]])
            }
            if let failure { return fail(failure) }
            do {
                let data = try target.read(offset: offset, length: length)
                // `Data`를 그대로 싣는다 — Lynx가 `NSData`를 JS의 `ArrayBuffer`로 바꿔 준다.
                // base64로 만들면 33% 팽창에 JS 쪽 디코딩 루프까지 붙는다.
                completion(["ok": true, "data": data, "byteLength": data.count])
            } catch let error as WGPUError {
                fail(error)
            } catch {
                fail(.backend("\(error)"))
            }
        }

        // 제출한 작업이 아직 돌고 있으면 완료 후에 읽는다.
        // `addCompletedHandler`는 commit 이후에 붙일 수 없으므로(Metal 단언) 전용 큐에서 기다린다.
        //
        // `.error`는 **완료가 아니라 실패**다. 성공 경로로 흘려 보내면 GPU 작업이 실패한 버퍼의
        // 내용을 그대로 읽어 `ok: true`로 돌려주게 된다 — 호출자는 실패를 성공으로 받는다.
        if let pending, pending.status == .error {
            deliver(Self.commandBufferError(pending))
            return
        }
        guard let pending, pending.status != .completed else {
            deliver(nil)
            return
        }
        Self.readbackQueue.async {
            pending.waitUntilCompleted()
            deliver(pending.status == .error ? Self.commandBufferError(pending) : nil)
        }
    }

    /// 실패한 커맨드 버퍼를 보고 가능한 오류로 바꾼다.
    static func commandBufferError(_ buffer: MTLCommandBuffer) -> WGPUError {
        .backend("GPU 작업이 실패했다: \(buffer.error?.localizedDescription ?? "원인 불명")")
    }

    /// GPU 완료를 기다리는 전용 큐 — JS 스레드를 막지 않는다.
    private static let readbackQueue = DispatchQueue(label: "org.lynxwebgpu.readback")

    /// `navigator.gpu.requestAdapter()` 가 돌려줄 어댑터 정보와 한계값.
    public func adapterInfo() -> [String: Any] {
        var limits: [String: Any] = [
            "maxBindGroups": WGSLMetalLimits.bufferSlotCount > 0 ? 4 : 0,
            "maxVertexBuffers": WGSLMetalLimits.maxVertexBufferSlots,
            "maxBindGroupBuffers": WGSLMetalLimits.maxBindGroupBuffers,
            "maxTexturesPerStage": WGSLMetalLimits.textureSlotCount,
            "maxSamplersPerStage": WGSLMetalLimits.samplerSlotCount,
            "maxBufferSize": device.maxBufferLength,
        ]
        limits["maxThreadsPerThreadgroup"] = device.maxThreadsPerThreadgroup.width

        return [
            "ok": true,
            "name": device.name,
            "backend": "metal",
            "hasUnifiedMemory": device.hasUnifiedMemory,
            "supportsFamilyApple7": device.supportsFamily(.apple7),
            "preferredCanvasFormat": WGPUTextureFormat.bgra8unorm.rawValue,
            "limits": limits,
            "features": features(),
        ]
    }

    /// 기기마다 갈리는 기능 (`adapter.features` — 명세 철자 그대로).
    ///
    /// JS가 만들기 전에 물어볼 수 있어야 하는 것만 싣는다. 못 만드는 것을 만들려다 오류를
    /// 받는 것보다, 미리 알고 다른 길로 가는 편이 낫다.
    private func features() -> [String] {
        var result: [String] = []
        if device.supportsCounterSampling(.atStageBoundary),
           device.counterSets?.contains(where: { $0.name == MTLCommonCounterSet.timestamp.rawValue }) == true {
            result.append("timestamp-query")
        }
        // 간접 드로우 인자의 `firstInstance`를 존중하는가. 명세는 이것을 선택 기능으로 두고,
        // 기능이 없으면 non-zero인 드로우를 **통째로 no-op**으로 만든다. Metal은 인자 배치가
        // WebGPU와 같아 `baseInstance`를 그대로 존중하므로, 여기서는 항상 "기능이 활성된
        // 어댑터"와 같은 자리에 선다. 인자 값이 GPU 버퍼 안에 있어 인코딩 시점에 검사할 수
        // 없으므로, 이 기능을 알리는 것이 앱에 상황을 전달하는 유일한 수단이다.
        result.append("indirect-first-instance")
        return result
    }

    /// 모든 GPU 객체를 버린다 (페이지 이탈 등).
    public func reset() {
        executionLock.lock()
        registry.removeAll()
        interpreter.discardErrorScopes()
        interpreter.discardFrameState()
        executionLock.unlock()
    }

    public var liveObjectCount: Int { registry.count }

    /// 테스트 관찰용 — 업로드 스테이징 풀.
    var stagingPool: WGPUStagingPool { interpreter.stagingPool }
}
