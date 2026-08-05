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
        // `present: false`는 프레임 **중간**의 내부 제출이라는 뜻이다 (shim의 popErrorScope·
        // mapAsync). 커밋은 하되 드로어블 present와 프레임 핸들 만료를 미룬다 — 진짜 프레임
        // 제출(queue.submit)이 뒤따라온다.
        return interpreter.execute(commands, present: reader.bool("present", default: true))
    }

    /// 편의 오버로드 — 배열을 그대로 넘긴다.
    @discardableResult
    public func execute(commands: [[String: Any]]) -> [String: Any] {
        execute(["commands": commands])
    }

    /// 버퍼 내용을 읽는다 (`GPUBuffer.mapAsync` + `getMappedRange`에 해당).
    ///
    /// 직전에 제출한 GPU 작업이 끝난 뒤에 읽어야 하므로 비동기다.
    /// 인코딩된 이미지를 풀어 `ImageBitmap` 자리의 객체로 등록한다 (JS `createImageBitmap`).
    ///
    /// **핸들은 JS가 발급한다** — 커맨드 스트림과 같은 규칙이다. 디코딩은 느리므로
    /// 백그라운드 큐에서 하고, 등록만 실행 락 안에서 한다.
    ///
    /// - Parameter data: 이미지 바이트. nil이면 `name`을 애셋 공급자로 해석한다.
    /// - Parameter callback: `{"ok": true, "width": Int, "height": Int}` 또는 오류 페이로드.
    public func decodeImage(
        handle: Int,
        data: Data?,
        name: String?,
        options: WGPUImageDecoder.Options,
        provider: WGPUAssetProvider?,
        completion: @escaping ([String: Any]) -> Void
    ) {
        let fail = { (error: WGPUError) in completion(["ok": false, "errors": [error.payload]]) }
        let finish = { [weak self] (bytes: Data) in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let bitmap = try WGPUImageDecoder.decode(bytes, options: options)
                    guard let self else { return }
                    self.executionLock.lock()
                    self.registry.insert(bitmap, at: WGPUHandle(handle))
                    self.executionLock.unlock()
                    completion(["ok": true, "width": bitmap.width, "height": bitmap.height])
                } catch let error as WGPUError {
                    fail(error)
                } catch {
                    fail(WGPUError.backend("\(error)"))
                }
            }
        }

        if let data {
            finish(data)
        } else if let name {
            guard let provider else {
                return fail(WGPUError.validation("애셋 공급자가 없다 — 이미지 바이트를 직접 넘길 것"))
            }
            provider.loadAsset(named: name) { result in
                switch result {
                case .success(let bytes): finish(bytes)
                case .failure(let error): fail(error)
                }
            }
        } else {
            fail(WGPUError.validation("createImageBitmap에는 이미지 바이트나 애셋 이름이 필요하다"))
        }
    }

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

    /// `GPUShaderModule.getCompilationInfo()` — 그 모듈의 컴파일 진단.
    ///
    /// 명세의 `GPUCompilationMessage` 모양(`message`·`type`·`lineNum`·`linePos`·`offset`·`length`)
    /// 으로 돌려준다. 우리가 실제로 아는 것은 메시지와 줄 번호뿐이라 나머지는 0이다 —
    /// **모르는 값을 지어내면 편집기가 엉뚱한 곳에 밑줄을 긋는다.**
    public func shaderCompilationInfo(handle: Int) -> [String: Any] {
        executionLock.lock()
        defer { executionLock.unlock() }

        guard let module = try? registry.lookup(
            WGPUHandle(handle), as: WGPUShaderModuleObject.self, kind: "GPUShaderModule"
        ) else {
            return ["ok": false, "errors": [
                WGPUError.validation("GPUShaderModule #\(handle)이(가) 없다").payload,
            ]]
        }
        let messages = module.compilationMessages.map { error -> [String: Any] in
            [
                "message": error.message,
                // 이 구현의 진단은 전부 오류다 — Metal 런타임 API가 경고를 따로 주지 않는다.
                "type": "error",
                "lineNum": error.line ?? 0,
                "linePos": 0,
                "offset": 0,
                "length": 0,
            ]
        }
        return ["ok": true, "messages": messages]
    }

    /// `navigator.gpu.requestAdapter()` 가 돌려줄 어댑터 정보와 한계값.
    ///
    /// 키는 **명세의 `GPUSupportedLimits` 철자 그대로**다. 웹 라이브러리가 이 이름으로 읽고
    /// 자기 예산을 정한다 (Three.js는 `maxComputeWorkgroupsPerDimension`·
    /// `maxUniformBufferBindingSize`를 본다). 우리 식으로 이름을 지으면 그 코드가
    /// `undefined`를 보고 잘못된 가정을 세운다 — 값이 있는데도 없는 것처럼 동작한다.
    ///
    /// 값은 가능한 한 **Metal 디바이스에서 실제로 읽고**, 런타임 조회가 없는 것은
    /// Metal 기능 집합 표의 보장값을 쓴다 (아래 주석에 근거를 남긴다).
    public func adapterInfo() -> [String: Any] {
        let threadgroup = device.maxThreadsPerThreadgroup
        // Apple GPU family 3 이상(A9+)과 Mac2는 2D 텍스처가 16384까지다. 그 아래는 8192.
        // 이 프로젝트의 최소 타깃(iOS 17 = A12+)은 항상 위쪽이지만, macOS 구형까지 감안해 나눈다.
        let maxTexture2D = device.supportsFamily(.apple3) || device.supportsFamily(.mac2) ? 16384 : 8192

        let limits: [String: Any] = [
            // 텍스처
            "maxTextureDimension1D": maxTexture2D,
            "maxTextureDimension2D": maxTexture2D,
            "maxTextureDimension3D": 2048,
            "maxTextureArrayLayers": 2048,
            // 바인딩 — 우리 인자 테이블 배정 규칙에서 그대로 나온다 (`WGSLMetalLimits`)
            "maxBindGroups": 4,
            "maxBindGroupsPlusVertexBuffers": 4 + WGSLMetalLimits.maxVertexBufferSlots,
            "maxBindingsPerBindGroup": 1000,
            "maxSampledTexturesPerShaderStage": WGSLMetalLimits.textureSlotCount,
            "maxSamplersPerShaderStage": WGSLMetalLimits.samplerSlotCount,
            "maxStorageBuffersPerShaderStage": WGSLMetalLimits.maxBindGroupBuffers,
            "maxStorageTexturesPerShaderStage": WGSLMetalLimits.textureSlotCount,
            "maxUniformBuffersPerShaderStage": WGSLMetalLimits.maxBindGroupBuffers,
            "maxDynamicUniformBuffersPerPipelineLayout": 8,
            "maxDynamicStorageBuffersPerPipelineLayout": 4,
            // 버퍼 — 오프셋 정렬은 명세 기본값(256)을 그대로 쓴다. Metal이 요구하는 값
            // (Apple GPU 32B)보다 크므로 이걸 지키면 Metal도 만족한다. 반대로 32를 보고하면
            // 브라우저에서만 깨지는 코드가 나온다.
            "maxBufferSize": device.maxBufferLength,
            "maxUniformBufferBindingSize": 65536,
            "maxStorageBufferBindingSize": device.maxBufferLength,
            "minUniformBufferOffsetAlignment": 256,
            "minStorageBufferOffsetAlignment": 256,
            // 정점
            "maxVertexBuffers": WGSLMetalLimits.maxVertexBufferSlots,
            "maxVertexAttributes": 30,
            "maxVertexBufferArrayStride": 2048,
            "maxInterStageShaderVariables": 16,
            // 어태치먼트
            "maxColorAttachments": 8,
            "maxColorAttachmentBytesPerSample": 32,
            // 컴퓨트
            "maxComputeWorkgroupStorageSize": device.maxThreadgroupMemoryLength,
            "maxComputeInvocationsPerWorkgroup": threadgroup.width,
            "maxComputeWorkgroupSizeX": threadgroup.width,
            "maxComputeWorkgroupSizeY": threadgroup.height,
            "maxComputeWorkgroupSizeZ": threadgroup.depth,
            // Metal에는 디스패치 그리드 상한 조회가 없다. Dawn과 같은 보수적 값을 쓴다.
            "maxComputeWorkgroupsPerDimension": 65535,
        ]

        // 명세의 `GPUAdapterInfo` — 웹 코드가 GPU 종류로 분기할 때 읽는 표준 이름이다.
        // 값을 모르는 자리는 **빈 문자열**로 둔다 (명세가 그렇게 정한다) — 지어내면
        // 그 문자열로 분기하는 코드가 잘못된 우회로를 탄다.
        let info: [String: Any] = [
            "vendor": "apple",
            "architecture": architectureName(),
            // 명세의 `device`는 벤더별 식별자다 (PCI device ID 같은 것). Metal에는 없다.
            "device": "",
            "description": device.name,
            "isFallbackAdapter": false,
            // `subgroups` 기능을 광고하지 않으므로 명세대로 0이다.
            "subgroupMinSize": 0,
            "subgroupMaxSize": 0,
        ]

        return [
            "ok": true,
            "info": info,
            "name": device.name,
            "backend": "metal",
            "hasUnifiedMemory": device.hasUnifiedMemory,
            "supportsFamilyApple7": device.supportsFamily(.apple7),
            "preferredCanvasFormat": WGPUTextureFormat.bgra8unorm.rawValue,
            "limits": limits,
            "features": features(),
        ]
    }

    /// `GPUAdapterInfo.architecture` — GPU 계열 이름.
    ///
    /// 명세는 "가족/계열 이름, 모르면 빈 문자열"이라고만 정한다. Metal에는 계열을 묻는 API가
    /// 없고 `supportsFamily`로 **아래에서부터 확인**하는 것만 된다. 가장 높은 것부터 짚어
    /// 알아낸 만큼만 답한다 — 모르면 빈 문자열이다.
    private func architectureName() -> String {
        let families: [(MTLGPUFamily, String)] = [
            (.apple9, "apple-9"), (.apple8, "apple-8"), (.apple7, "apple-7"),
            (.apple6, "apple-6"), (.apple5, "apple-5"), (.apple4, "apple-4"),
            (.apple3, "apple-3"), (.apple2, "apple-2"), (.apple1, "apple-1"),
        ]
        for (family, name) in families where device.supportsFamily(family) { return name }
        return device.supportsFamily(.mac2) ? "mac-2" : ""
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
        // 기기가 간접 인자 자체를 못 하면 이 기능도 광고하지 않는다 — 시뮬레이터가 그렇다.
        // 있다고 알려 놓고 첫 호출에서 거부하면, 확인하고 쓴 앱이 오히려 배신당한다.
        if WGPUDeviceCapability.supportsIndirectArguments(device) {
            result.append("indirect-first-instance")
        }
        // 블록 압축 계열. 없는 계열로 텍스처를 만들면 Metal이 단언으로 죽으므로,
        // 앱이 미리 갈라설 수 있게 여기서 알린다 (생성 시점에도 오류로 한 번 더 막는다).
        for probe: WGPUTextureFormat in [.bc1RGBAUnorm, .etc2RGB8Unorm, .astc4x4Unorm] {
            guard let name = WGPUDeviceCapability.compressionFamily(probe).featureName else { continue }
            if WGPUDeviceCapability.supportsCompression(probe, on: device) { result.append(name) }
        }
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
