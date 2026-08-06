import Foundation
import Metal
import LynxWebGPUCore
import LynxWebGPUShader

/// 커맨드 스트림 해석기.
///
/// JS는 한 프레임 분량의 WebGPU 호출을 **명령 배열 하나**로 모아 보낸다. 여기서 그 배열을 순서대로
/// 해석해 Metal 인코딩으로 옮긴다. 프레임당 JS↔네이티브 왕복이 1회로 줄어드는 것이 이 설계의 요점이다
/// (`docs/ARCHITECTURE.md` §3).
///
/// 오류는 던져서 실행을 중단하지 않고 **모아서 돌려준다** — WebGPU가 그렇듯 잘못된 호출 하나가
/// 프레임 전체를 죽이지 않게 하기 위해서다.
final class WGPUCommandInterpreter {
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let registry: WGPUObjectRegistry
    private let surfaceProvider: (String) -> WGPUSurface?
    /// 업로드 스테이징 버퍼 재사용 풀 (writeBuffer/writeTexture 공용).
    let stagingPool: WGPUStagingPool

    // 실행 중 상태 — execute() 하나의 수명 동안만 유효하다.
    private var commandBuffer: MTLCommandBuffer?
    private var renderEncoder: MTLRenderCommandEncoder?
    private var computeEncoder: MTLComputeCommandEncoder?
    private var blitEncoder: MTLBlitCommandEncoder?
    /// 현재 인코더 / 커맨드 버퍼에서 연 디버그 그룹 수 — 짝이 안 맞으면 Metal이 단언으로 죽는다.
    private var encoderDebugDepth = 0
    private var bufferDebugDepth = 0
    private var currentRenderPipeline: WGPURenderPipelineObject?
    private var currentComputePipeline: WGPUComputePipelineObject?
    private var boundGroups: [Int: (group: WGPUBindGroupObject, offsets: [Int])] = [:]
    private var dirtyGroups: Set<Int> = []
    /// Metal 버퍼 인덱스별 바인딩 크기 — `arrayLength()`가 이 표를 조회한다.
    /// 바인드 그룹을 적용할 때마다 갱신하고, 셰이더가 쓸 때만 인코더에 올린다.
    private var bufferSizes = [UInt32](repeating: 0, count: WGSLMetalLimits.maxBindGroupBuffers)
    private var indexBinding: (buffer: MTLBuffer, offset: Int, type: MTLIndexType, stride: Int)?
    /// 슬롯별 정점 버퍼 바인딩. **인코더에 직행하지 않고 여기 모아 두었다가** 드로우 직전에 올린다.
    /// 그래야 `resetPassBindings()`가 번들 경계에서 바인딩을 실제로 무효화할 수 있다 —
    /// Metal 인코더에는 "바인딩 해제"가 없으므로, 무효화는 그림자 상태로만 표현된다.
    private var vertexBindings: [Int: (buffer: MTLBuffer, offset: Int)] = [:]
    private var dirtyVertexSlots: Set<Int> = []
    /// 지금 렌더 패스의 어태치먼트 모양 — 렌더 번들이 이 패스에서 유효한지 볼 때 쓴다.
    private var passFormats: (color: [WGPUTextureFormat], depthStencil: WGPUTextureFormat?, sampleCount: Int)?
    /// 지금 렌더 패스가 깊이/스텐실을 **쓰지 않겠다**고 선언했는가 (`depthReadOnly`/`stencilReadOnly`).
    /// 선언해 놓고 쓰는 파이프라인·번들을 `setPipeline`/`executeBundles`에서 막는다.
    private var passDepthReadOnly = false
    private var passStencilReadOnly = false
    /// 지금 렌더 패스가 물고 있는 occlusion 쿼리셋 (`beginRenderPass`에서만 붙일 수 있다).
    private var passOcclusionQuerySet: WGPUQuerySetObject?
    /// 열려 있는 occlusion 쿼리 인덱스 — 중첩·미종료를 잡는다.
    private var openOcclusionQuery: Int?
    /// 지금 패스에서 이미 쓴 occlusion 쿼리 인덱스 — 명세는 같은 패스에서 재사용을 금지한다.
    /// (같은 8바이트 슬롯을 두 구간이 나눠 쓰면 남는 값이 Metal 동작에 달린 값이 된다.)
    private var usedOcclusionQueries: Set<Int> = []
    private var acquiredDrawables: [(handle: WGPUHandle, drawable: WGPUDrawable, surface: WGPUSurface)] = []
    /// 이번 프레임 업로드에 쓴 스테이징 버퍼 — 커맨드 버퍼 완료 시 풀로 돌아간다.
    private var frameStagingBuffers: [MTLBuffer] = []
    /// 프레임이 끝나면 무효해지는 핸들 (드로어블 텍스처와 그 뷰).
    private var frameScopedHandles: [WGPUHandle] = []
    private var touchedCanvases: [String: WGPUSurface] = [:]
    private var errors: [WGPUError] = []

    /// 앞선 배치의 GPU 실행이 실패했다는 보고 — **완료 핸들러(Metal 스레드)가 채운다.**
    ///
    /// `record()`가 도는 시점은 이미 커밋 전이라 GPU 측 실패는 구조상 그때 잡을 수 없다.
    /// 그래서 완료 핸들러가 여기 모아 두었다가 **다음 배치 결과**에 실어 보낸다. 이것이 없으면
    /// `.outOfMemory`·`.timeout` 같은 실패가 어디에도 나타나지 않고 무성으로 남는다.
    private let gpuFailureLock = NSLock()
    private var gpuFailures: [WGPUError] = []

    /// 열려 있는 오류 스코프 (안쪽이 뒤). **배치 사이에도 살아 있다** — WebGPU에서 오류 스코프는
    /// 디바이스 상태이고, `push`와 `pop` 사이에 `submit`이 몇 번이든 들어갈 수 있기 때문이다.
    /// `filter`가 nil이면 **아무것도 잡지 않는 자리표시자**다 — 필터 파싱이 실패했을 때
    /// 스택 깊이를 맞추려고 쌓는다 (안 쌓으면 이후 pop이 바깥 스코프를 가져간다).
    private var errorScopes: [(filter: WGPUErrorFilter?, captured: WGPUError?)] = []
    /// 이번 배치에서 pop된 스코프의 결과 (pop 순서 — JS의 Promise 순서와 1:1로 맞춘다).
    private var poppedScopes: [PoppedScope] = []

    /// pop 결과의 세 가지 상태. JS는 이것을 보고 Promise를 resolve할지 reject할지 정한다.
    private enum PoppedScope {
        /// 스코프는 있었고 잡힌 오류는 없었다 → `null`로 resolve.
        case clean
        /// 스코프가 오류를 잡았다 → 그 오류로 resolve.
        case captured(WGPUError)
        /// `push`와 짝이 맞지 않는다 → 명세대로 `OperationError`로 **reject**한다.
        /// 이 실패는 GPUError가 아니므로 전역 오류 핸들러로 내보내지 않는다.
        case unmatched

        var payload: Any {
            switch self {
            case .clean: return NSNull()
            case .captured(let error): return error.payload
            case .unmatched: return ["rejected": true]
            }
        }
    }

    init(
        device: MTLDevice,
        queue: MTLCommandQueue,
        registry: WGPUObjectRegistry,
        surfaceProvider: @escaping (String) -> WGPUSurface?
    ) {
        self.device = device
        self.queue = queue
        self.registry = registry
        self.surfaceProvider = surfaceProvider
        self.stagingPool = WGPUStagingPool(device: device)
    }

    /// 마지막으로 커밋한 커맨드 버퍼 — `readBuffer`가 GPU 완료를 기다릴 때 쓴다.
    private(set) var lastCommittedBuffer: MTLCommandBuffer?

    // MARK: - 실행

    func execute(_ commands: [WGPUValueReader], present: Bool = true) -> [String: Any] {
        reset()

        // 앞선 배치의 GPU 실행 실패를 먼저 흘려보낸다 — 오류 스코프가 열려 있으면 그쪽이 잡는다.
        for failure in drainGPUFailures() { record(failure) }

        for (index, command) in commands.enumerated() {
            do {
                try perform(command, at: index)
            } catch let error as WGPUError {
                // 경로만 채우고 나머지는 **그대로 옮긴다** — 여기서 필드를 빠뜨리면
                // (줄 번호처럼) 아래 계층이 애써 붙인 단서가 조용히 사라진다.
                record(WGPUError(
                    kind: error.kind,
                    message: error.message,
                    path: error.path ?? "commands[\(index)].\(command.optionalString("op") ?? "?")",
                    line: error.line
                ))
            } catch {
                record(.backend(error.localizedDescription, path: "commands[\(index)]"))
            }
        }

        // 명령이 비어 있는데 present라면 **틱의 마무리 배치**다 (프레임 루프 콜백의 끝).
        // 그 배치는 커맨드 버퍼가 없어도 드로어블을 내보내야 한다.
        finish(present: present, closingFrame: present && commands.isEmpty)

        // objects: live 네이티브 객체 수 — JS가 destroy 누락(레지스트리 증식)을 감시할 수 있게.
        var result: [String: Any] = [
            "ok": errors.isEmpty,
            "commandCount": commands.count,
            "objects": registry.count,
        ]
        if !errors.isEmpty {
            result["errors"] = errors.map(\.payload)
        }
        if !touchedCanvases.isEmpty {
            result["canvases"] = touchedCanvases.mapValues { surface in
                ["width": Int(surface.pixelSize.width), "height": Int(surface.pixelSize.height)]
            }
        }
        if !poppedScopes.isEmpty {
            // pop 순서 그대로 — JS는 popErrorScope()가 돌려준 Promise를 같은 순서로 풀어 준다.
            result["errorScopes"] = poppedScopes.map(\.payload)
        }
        return result
    }

    // MARK: - 오류 스코프

    /// 오류 하나를 **가장 안쪽의 맞는 스코프**에 넣거나, 없으면 배치 결과로 내보낸다.
    ///
    /// 스코프에 잡힌 오류는 결과의 `errors`에 실리지 않는다 — 그래야 JS의 전역 핸들러
    /// (`device.onError`)가 "내가 이미 처리하기로 한 오류"를 다시 보고하지 않는다.
    private func record(_ error: WGPUError) {
        for index in errorScopes.indices.reversed()
        where errorScopes[index].filter?.captures(error.kind) == true {
            // 명세상 스코프가 돌려주는 것은 **처음 잡힌 오류 하나**다.
            if errorScopes[index].captured == nil { errorScopes[index].captured = error }
            return
        }
        errors.append(error)
    }

    /// 디바이스를 버릴 때 (`GPUDevice.destroy`) 열려 있던 스코프도 함께 버린다 —
    /// 남겨 두면 다음 페이지의 오류가 죽은 스코프에 잡혀 아무 데도 보고되지 않는다.
    func discardErrorScopes() {
        errorScopes.removeAll()
    }

    private func pushErrorScope(_ command: WGPUValueReader) throws {
        do {
            errorScopes.append((try WGPUPushErrorScopeCommand(from: command).filter, nil))
        } catch {
            // 필터를 못 읽어도 스택 깊이는 맞춰 둔다 — 안 그러면 이후 pop이 **바깥 스코프**를
            // 가져가서, 앱이 안쪽 구간의 결과라고 믿는 값이 실제로는 바깥 구간의 결과가 된다.
            errorScopes.append((nil, nil))
            throw error
        }
    }

    private func popErrorScope() {
        guard let scope = errorScopes.popLast() else {
            // 명세는 이 경우 Promise를 `OperationError`로 **reject**하라고만 하고, 오류를
            // 생성하라고 하지 않는다. 그래서 throw하지 않고 상태만 실어 보낸다 —
            // 명세에 없는 GPUError가 앱의 전역 핸들러·텔레메트리에 섞이지 않게.
            poppedScopes.append(.unmatched)
            return
        }
        poppedScopes.append(scope.captured.map(PoppedScope.captured) ?? .clean)
    }

    private func reset() {
        commandBuffer = nil
        renderEncoder = nil
        computeEncoder = nil
        blitEncoder = nil
        encoderDebugDepth = 0
        bufferDebugDepth = 0
        currentRenderPipeline = nil
        currentComputePipeline = nil
        boundGroups.removeAll()
        dirtyGroups.removeAll()
        indexBinding = nil
        passFormats = nil
        passOcclusionQuerySet = nil
        passDepthReadOnly = false
        passStencilReadOnly = false
        openOcclusionQuery = nil
        usedOcclusionQueries.removeAll()
        touchedCanvases.removeAll()
        errors.removeAll()
        poppedScopes.removeAll()
        // `errorScopes`는 일부러 비우지 않는다 — 디바이스 상태이므로 배치를 넘어 이어진다.
        // `acquiredDrawables`·`frameScopedHandles`도 마찬가지다 — 프레임의 경계는 배치가
        // 아니라 **present**이고, 한 프레임이 배치 여러 개로 쪼개질 수 있다 (아래 finish() 참고).
    }

    /// 배치 하나를 마무리한다.
    ///
    /// `present`가 false면 프레임 **중간**의 내부 제출이다 — shim의 `popErrorScope`·`mapAsync`가
    /// 결과를 받으려고 미리 흘려보낸 배치. GPU 작업은 커밋하되(리드백이 완료를 기다린다),
    /// 드로어블 present와 프레임 스코프 핸들 만료는 **뒤따라올 진짜 프레임 제출로 미룬다.**
    /// 안 미루면: 그 배치가 `writeBuffer` 하나로라도 커맨드 버퍼를 만든 경우, 방금 획득한
    /// 드로어블이 그리기도 전에 present되고 핸들이 만료되어, 이어지는 `beginRenderPass`가
    /// "GPUTextureView가 존재하지 않는다"로 통째로 거부된다 — Three.js의 지연 파이프라인
    /// 생성(pop 즉시 flush)이 정확히 이 경로를 밟았다.
    /// - Parameter closingFrame: 명령 없이 **present만 하는** 배치인가 (틱의 끝).
    ///   프레임 경계가 `submit()`이 아니라 프레임 루프 콜백의 끝이라 이런 배치가 온다.
    private func finish(present: Bool, closingFrame: Bool = false) {
        endActiveEncoders()
        // 커맨드 버퍼에 연 그룹도 커밋 전에 닫는다 (인코더와 같은 이유 — Metal이 단언으로 죽는다).
        if let commandBuffer, bufferDebugDepth > 0 {
            record(.validation(
                "디버그 그룹 \(bufferDebugDepth)개가 열린 채로 제출됐다 (popDebugGroup을 빠뜨렸다)"
            ))
            while bufferDebugDepth > 0 {
                commandBuffer.popDebugGroup()
                bufferDebugDepth -= 1
            }
        }
        // 틱의 마무리 배치는 명령이 없어도 **드로어블을 내보내야 한다** — 커맨드 버퍼가
        // 없다고 지나가면 화면이 멈춘 채 아무 말이 없다.
        //
        // 조건을 "명령이 비어 있을 때"로 좁힌 것이 중요하다. 명령은 있는데 커맨드 버퍼가
        // 안 생긴 배치(드로어블만 얻고 끝난 경우 등)는 예전처럼 present하지 않는다 —
        // 그리지도 않은 드로어블을 내보내면 그 프레임이 빈 화면으로 나간다.
        if closingFrame, commandBuffer == nil, !acquiredDrawables.isEmpty {
            _ = try? activeCommandBuffer()
        }
        if let commandBuffer {
            if present {
                for acquired in acquiredDrawables {
                    acquired.drawable.present(with: commandBuffer)
                }
            }
            // 완료 핸들러는 commit 전에만 붙일 수 있다 (Metal 단언).
            if !frameStagingBuffers.isEmpty {
                let buffers = frameStagingBuffers
                let pool = stagingPool
                commandBuffer.addCompletedHandler { _ in pool.recycle(buffers) }
            }
            // in-flight 회계 — 프레임 티커가 이 수를 보고 포화 시 틱을 건너뛴다.
            // present하지 않는 배치는 프레임이 아니므로 세지 않는다.
            if present {
                let presentedSurfaces = uniquePresentedSurfaces()
                if !presentedSurfaces.isEmpty {
                    for surface in presentedSurfaces { surface.noteFrameCommitted() }
                    commandBuffer.addCompletedHandler { _ in
                        for surface in presentedSurfaces { surface.noteFrameCompleted() }
                    }
                }
            }
            // GPU 측 실패(.outOfMemory / .timeout / .deviceRemoved 등)를 주워 담는다.
            commandBuffer.addCompletedHandler { [weak self] buffer in
                guard buffer.status == .error else { return }
                guard let self else { return }
                self.gpuFailureLock.lock()
                self.gpuFailures.append(LynxWebGPUContext.commandBufferError(buffer))
                self.gpuFailureLock.unlock()
            }
            commandBuffer.commit()
            lastCommittedBuffer = commandBuffer
            // 드로어블 텍스처와 그 뷰는 **present할 때** 무효해진다 (명세의 "Expire the current
            // texture"가 정한 시점). 배치가 끝날 때마다 회수하면 프레임 중간 제출이 그 프레임의
            // 스왑체인 핸들을 지워 버려 뒤이은 `beginRenderPass`가 "없는 핸들"로 깨진다.
            if present, !acquiredDrawables.isEmpty {
                for handle in frameScopedHandles { registry.remove(handle) }
                frameScopedHandles.removeAll()
                acquiredDrawables.removeAll()
            }
        } else if !frameStagingBuffers.isEmpty {
            // 커밋할 커맨드 버퍼가 없으면 GPU가 이 버퍼들을 참조하지 않는다 — 바로 회수한다.
            stagingPool.recycle(frameStagingBuffers)
        }
        frameStagingBuffers.removeAll()
        commandBuffer = nil
    }

    /// 디바이스를 버릴 때 프레임 중간 상태도 함께 버린다 — 남겨 두면 다음 디바이스의 첫
    /// 프레임이 죽은 드로어블을 present하려 든다.
    func discardFrameState() {
        acquiredDrawables.removeAll()
        frameScopedHandles.removeAll()
        lastCommittedBuffer = nil
        _ = drainGPUFailures()
    }

    /// 모아 둔 GPU 실행 실패를 꺼내 비운다 (완료 핸들러가 다른 스레드에서 채운다).
    private func drainGPUFailures() -> [WGPUError] {
        gpuFailureLock.lock()
        defer { gpuFailureLock.unlock() }
        let failures = gpuFailures
        gpuFailures.removeAll()
        return failures
    }

    /// 이번 프레임에 드로어블을 내준 표면들 (중복 제거 — 한 표면에서 여러 번 얻어도 프레임은 하나다).
    private func uniquePresentedSurfaces() -> [WGPUSurface] {
        var seen = Set<ObjectIdentifier>()
        var surfaces: [WGPUSurface] = []
        for acquired in acquiredDrawables where seen.insert(ObjectIdentifier(acquired.surface)).inserted {
            surfaces.append(acquired.surface)
        }
        return surfaces
    }

    // MARK: - 인코더 수명

    private func activeCommandBuffer() throws -> MTLCommandBuffer {
        if let commandBuffer { return commandBuffer }
        guard let created = queue.makeCommandBuffer() else {
            throw WGPUError.backend("MTLCommandBuffer 생성 실패")
        }
        created.label = "webgpu.frame"
        commandBuffer = created
        return created
    }

    private func endActiveEncoders() {
        if renderEncoder != nil {
            // 명세는 패스를 닫을 때 열려 있는 occlusion 쿼리가 없기를 요구한다. Metal은 그냥
            // 값을 써 주므로 여기서 안 잡으면 **값까지 정상으로 보이고**, 브라우저에서만 프레임이
            // 통째로 날아간다. 패스는 이미 닫히는 중이라 throw 대신 기록한다.
            if let index = openOcclusionQuery {
                record(.validation(
                    "occlusion 쿼리 \(index)이(가) 열린 채로 렌더 패스가 끝났다 "
                        + "(endOcclusionQuery를 빠뜨렸다)"
                ))
            }
            // 패스 상태는 패스 밖으로 새면 안 된다 — 지금은 뒤따르는 beginRenderPass가 다시
            // 설정해서 드러나지 않지만, 새 op이 추가될 때 걸리기 쉬운 자리다.
            openOcclusionQuery = nil
            usedOcclusionQueries.removeAll()
            passOcclusionQuerySet = nil
            passFormats = nil
            passDepthReadOnly = false
            passStencilReadOnly = false
        }
        // 디버그 그룹이 열린 채 인코더를 닫으면 Metal이 단언으로 죽는다 — 닫아 주고 오류로 알린다.
        if let encoder = debugScope { closeDanglingDebugGroups(on: encoder) }
        renderEncoder?.endEncoding()
        renderEncoder = nil
        computeEncoder?.endEncoding()
        computeEncoder = nil
        blitEncoder?.endEncoding()
        blitEncoder = nil
    }

    private func activeBlitEncoder() throws -> MTLBlitCommandEncoder {
        if let blitEncoder { return blitEncoder }
        guard renderEncoder == nil, computeEncoder == nil else {
            throw WGPUError.validation("렌더/컴퓨트 패스 안에서는 복사·업로드 명령을 쓸 수 없다")
        }
        guard let encoder = try activeCommandBuffer().makeBlitCommandEncoder() else {
            throw WGPUError.backend("MTLBlitCommandEncoder 생성 실패")
        }
        blitEncoder = encoder
        return encoder
    }

    private func requireRenderEncoder() throws -> MTLRenderCommandEncoder {
        guard let renderEncoder else {
            throw WGPUError.validation("렌더 패스가 시작되지 않았다 (beginRenderPass 먼저)")
        }
        return renderEncoder
    }

    private func requireComputeEncoder() throws -> MTLComputeCommandEncoder {
        guard let computeEncoder else {
            throw WGPUError.validation("컴퓨트 패스가 시작되지 않았다 (beginComputePass 먼저)")
        }
        return computeEncoder
    }

    // MARK: - 명령 분기

    /// op 이름 → 처리. **디코딩은 전부 이 자리에서 끝난다** — 아래 함수들은 리더를 보지 않고
    /// `LynxWebGPUCore`가 옮겨 준 값만 받는다. 백엔드를 갈아끼울 때 다시 써야 하는 것이
    /// "인코딩"으로만 좁혀지도록 한 배치다 (`WGPUCommands.swift` 머리말).
    ///
    /// 예외는 둘뿐이고 이유가 있다:
    /// - `pushErrorScope` — 필터 디코딩이 **실패해도** 스택 깊이를 맞춰야 해서 리더를 받는다.
    /// - `createRenderBundle` — 명령 목록을 값으로 **저장**했다가 되풀이하므로 리더 자체가 자료다.
    private func perform(_ command: WGPUValueReader, at index: Int) throws {
        let op = try command.requiredString("op")
        switch op {
        // 리소스
        case "createBuffer": try createBuffer(WGPUCreateCommand(from: command))
        case "writeBuffer": try writeBuffer(WGPUWriteBufferCommand(from: command))
        case "unmapBuffer": try unmapBuffer(WGPUUnmapBufferCommand(from: command))
        case "createTexture": try createTexture(WGPUCreateCommand(from: command))
        case "writeTexture": try writeTexture(WGPUWriteTextureCommand(from: command))
        case "copyExternalImageToTexture":
            try copyExternalImageToTexture(WGPUCopyExternalImageCommand(from: command))
        case "createTextureView": try createTextureView(WGPUCreateTextureViewCommand(from: command))
        case "createSampler": try createSampler(WGPUCreateCommand(from: command))
        case "createShaderModule": try createShaderModule(WGPUCreateCommand(from: command))
        case "createBindGroupLayout": try createBindGroupLayout(WGPUCreateCommand(from: command))
        case "createPipelineLayout": try createPipelineLayout(WGPUCreateCommand(from: command))
        case "createBindGroup": try createBindGroup(WGPUCreateCommand(from: command))
        case "createQuerySet": try createQuerySet(WGPUCreateCommand(from: command))
        case "createRenderBundle": try createRenderBundle(WGPUCreateRenderBundleCommand(from: command))
        case "createRenderPipeline": try createRenderPipeline(WGPUCreateCommand(from: command))
        case "createComputePipeline": try createComputePipeline(WGPUCreateCommand(from: command))
        case "getBindGroupLayout": try getBindGroupLayout(WGPUGetBindGroupLayoutCommand(from: command))
        case "destroy": registry.remove(try WGPUDestroyCommand(from: command).id)

        // 오류 스코프
        case "pushErrorScope": try pushErrorScope(command)
        case "popErrorScope": popErrorScope()

        // 캔버스
        case "configureCanvas": try configureCanvas(WGPUCanvasConfiguration(from: command))
        case "getCurrentTexture": try getCurrentTexture(WGPUGetCurrentTextureCommand(from: command))

        // 렌더 패스
        case "beginRenderPass": try beginRenderPass(WGPURenderPassDescriptor(from: command))
        case "setPipeline": try setPipeline(WGPUSetPipelineCommand(from: command))
        case "setBindGroup": try setBindGroup(WGPUSetBindGroupCommand(from: command))
        case "setVertexBuffer": try setVertexBuffer(WGPUSetVertexBufferCommand(from: command))
        case "setIndexBuffer": try setIndexBuffer(WGPUSetIndexBufferCommand(from: command))
        case "setViewport": try setViewport(WGPUSetViewportCommand(from: command))
        case "setScissorRect": try setScissorRect(WGPUSetScissorRectCommand(from: command))
        case "setBlendConstant": try setBlendConstant(WGPUSetBlendConstantCommand(from: command))
        case "setStencilReference":
            try requireRenderEncoder()
                .setStencilReferenceValue(WGPUSetStencilReferenceCommand(from: command).reference)
        case "draw": try draw(WGPUDrawCommand(from: command))
        case "drawIndexed": try drawIndexed(WGPUDrawIndexedCommand(from: command))
        case "drawIndirect": try drawIndirect(WGPUIndirectCommand(from: command))
        case "drawIndexedIndirect": try drawIndexedIndirect(WGPUIndirectCommand(from: command))
        case "executeBundles": try executeBundles(WGPUExecuteBundlesCommand(from: command))
        case "beginOcclusionQuery": try beginOcclusionQuery(WGPUBeginOcclusionQueryCommand(from: command))
        case "endOcclusionQuery": try endOcclusionQuery()

        // 컴퓨트 패스
        case "beginComputePass": try beginComputePass(WGPUComputePassDescriptor(from: command))
        case "dispatchWorkgroups": try dispatchWorkgroups(WGPUDispatchWorkgroupsCommand(from: command))
        case "dispatchWorkgroupsIndirect":
            try dispatchWorkgroupsIndirect(WGPUIndirectCommand(from: command))

        case "endPass": endActiveEncoders()

        // 복사
        case "copyBufferToBuffer": try copyBufferToBuffer(WGPUCopyBufferToBufferCommand(from: command))
        case "clearBuffer": try clearBuffer(WGPUClearBufferCommand(from: command))

        // 디버그 마커
        case "pushDebugGroup": try pushDebugGroup(WGPUPushDebugGroupCommand(from: command))
        case "popDebugGroup": popDebugGroup()
        case "insertDebugMarker": try insertDebugMarker(WGPUInsertDebugMarkerCommand(from: command))
        case "copyTextureToBuffer": try copyTextureToBuffer(WGPUCopyTextureToBufferCommand(from: command))
        case "copyBufferToTexture": try copyBufferToTexture(WGPUCopyBufferToTextureCommand(from: command))
        case "copyTextureToTexture":
            try copyTextureToTexture(WGPUCopyTextureToTextureCommand(from: command))
        case "resolveQuerySet": try resolveQuerySet(WGPUResolveQuerySetCommand(from: command))

        default:
            throw WGPUError.unsupported("알 수 없는 명령 '\(op)'", path: "commands[\(index)].op")
        }
    }

    // MARK: - 리소스 생성

    private func createBuffer(_ command: WGPUCreateCommand<WGPUBufferDescriptor>) throws {
        let object = try WGPUBufferObject(device: device, descriptor: command.descriptor)
        registry.insert(object, at: command.id)
    }

    /// 큐 작업에 쓸 버퍼를 꺼낸다 — **매핑 중이면 거부한다.**
    ///
    /// 명세는 `mapAsync`가 버퍼를 "unavailable"로 만들어 `unmap()` 전까지 큐 작업에 못 쓰게 해
    /// 경쟁 자체를 없앤다. 이 구현은 `.storageModeShared` 버퍼를 스테이징 없이 읽으므로,
    /// 이 검사가 없으면 리드백이 GPU 완료를 기다리는 동안 다음 프레임의 쓰기가 같은 메모리에
    /// 겹쳐 **JS가 받는 값이 어느 프레임 것인지 보장되지 않는다.**
    ///
    /// 버퍼를 쓰는 모든 명령이 이 함수를 지나야 한다 — 한 곳이라도 빠지면 그 경로로 경쟁이 샌다.
    ///
    /// - Parameter path: 오류에 붙일 커맨드 스트림 경로 (`commands[3].buffer`).
    ///   커맨드 구조체의 `fieldPath(_:)`가 만들어 준다.
    private func unmappedBuffer(_ handle: WGPUHandle, path: String? = nil) throws -> WGPUBufferObject {
        let object = try registry.lookup(handle, as: WGPUBufferObject.self, kind: "GPUBuffer", path: path)
        guard !object.isMapped else {
            throw WGPUError.validation(
                "매핑 중인 GPUBuffer \(handle)은(는) 큐 작업에 쓸 수 없다 "
                    + "(mapAsync로 읽은 뒤 unmap()을 부를 것)",
                path: path
            )
        }
        return object
    }

    private func unmapBuffer(_ command: WGPUUnmapBufferCommand) throws {
        try registry.lookup(
            command.buffer, as: WGPUBufferObject.self, kind: "GPUBuffer"
        ).isMapped = false
    }

    private func writeBuffer(_ command: WGPUWriteBufferCommand) throws {
        let target = try unmappedBuffer(command.buffer, path: command.fieldPath("buffer"))
        let offset = command.bufferOffset
        let data = command.data
        guard offset >= 0, offset + data.count <= target.size else {
            throw WGPUError.validation(
                "writeBuffer 범위 초과 — offset \(offset) + \(data.count)B > 버퍼 크기 \(target.size)B"
            )
        }
        guard !data.isEmpty else { return }   // 크기 0은 no-op (Metal blit은 0바이트 복사를 거부한다)
        let staging = try makeStagingBuffer(data)
        // 직접 memcpy 하면 이전 프레임 GPU 작업과 경쟁한다. blit으로 큐에 순서를 태운다.
        let encoder = try activeBlitEncoder()
        encoder.copy(
            from: staging, sourceOffset: 0,
            to: target.buffer, destinationOffset: offset, size: data.count
        )
    }

    private func createTexture(_ command: WGPUCreateCommand<WGPUTextureDescriptor>) throws {
        let object = try WGPUTextureObject(device: device, descriptor: command.descriptor)
        registry.insert(object, at: command.id)
    }

    private func writeTexture(_ command: WGPUWriteTextureCommand) throws {
        let target = try registry.lookup(
            command.texture, as: WGPUTextureObject.self, kind: "GPUTexture",
            path: command.fieldPath("texture")
        )
        let data = command.data
        let size = command.size
        let format = target.format
        // 생략된 스트라이드의 기본값은 **포맷을 알아야** 나온다 — 압축 포맷에서 행은 픽셀이
        // 아니라 **블록** 단위이기 때문이다 (명세 GPUTexelCopyBufferLayout). 그래서 디코딩이
        // 아니라 여기서 채운다.
        let bytesPerRow = command.bytesPerRow ?? format.bytesPerRow(width: size.width)
        let blockRows = format.blockRows(height: size.height)
        let rowsPerImage = command.rowsPerImage ?? blockRows
        guard size.width > 0, size.height > 0, size.depthOrArrayLayers > 0 else { return }   // no-op
        try validateBlockAlignment(format: format, origin: command.origin, size: size,
                                   texture: target, mipLevel: command.mipLevel,
                                   label: "writeTexture")
        let bytesPerImage = bytesPerRow * max(rowsPerImage, blockRows)
        let layers = max(size.depthOrArrayLayers, 1)
        let required = bytesPerImage * (layers - 1) + bytesPerRow * blockRows
        guard data.count >= required else {
            throw WGPUError.validation("writeTexture 데이터가 부족하다 (\(data.count)B, 최소 \(required)B 필요)")
        }
        // 스테이징은 이미지 스트라이드 전체만큼 잡는다 — Metal 검증 레이어가 마지막 이미지도
        // bytesPerImage 범위로 계산하기 때문이다. 남는 꼬리는 텍스처로 복사되지 않는다.
        let staging = try makeStagingBuffer(data, minimumLength: bytesPerImage * layers)
        // writeBuffer와 같은 이유로 blit으로 큐에 순서를 태운다 — 앞선 렌더/복사와 직렬화된다.
        target.encodeWrite(
            from: staging,
            origin: command.origin,
            size: size,
            mipLevel: command.mipLevel,
            bytesPerRow: bytesPerRow,
            rowsPerImage: rowsPerImage,
            blit: try activeBlitEncoder()
        )
    }

    /// 디코딩해 둔 이미지(`ImageBitmap`)를 텍스처로 올린다 — 명세
    /// `queue.copyExternalImageToTexture()`.
    ///
    /// 웹에서는 브라우저가 `<img>`·`<canvas>`·`VideoFrame`을 소스로 받는다. Lynx에는 그런
    /// 엘리먼트가 없으므로 **`createImageBitmap()`이 만든 네이티브 이미지**가 그 자리다.
    /// 픽셀은 이미 RGBA8이라 여기서는 잘라내고 스테이징에 실어 blit하는 일만 한다.
    private func copyExternalImageToTexture(_ command: WGPUCopyExternalImageCommand) throws {
        let bitmap = try registry.lookup(
            command.source.image, as: WGPUImageBitmapObject.self, kind: "ImageBitmap",
            path: command.source.fieldPath("source")
        )
        let target = try registry.lookup(
            command.destination.texture, as: WGPUTextureObject.self, kind: "GPUTexture",
            path: command.destination.fieldPath("texture")
        )
        guard !target.format.isCompressed else {
            throw WGPUError.validation(
                "copyExternalImageToTexture는 압축 텍스처에 쓸 수 없다 (\(target.format.rawValue)) "
                + "— GPU에는 블록 인코더가 없다"
            )
        }
        // 명세는 소스와 대상의 바이트 폭이 같기를 요구한다. 디코딩 결과가 RGBA8이므로
        // 4바이트 포맷만 받는다 — 그 밖은 화면이 조용히 어긋나느니 여기서 막는 편이 낫다.
        guard target.format.bytesPerBlock == 4, !target.format.rawValue.hasPrefix("depth"),
              !target.format.rawValue.hasPrefix("stencil") else {
            throw WGPUError.validation(
                "copyExternalImageToTexture의 대상은 4바이트 컬러 포맷이어야 한다 "
                + "(\(target.format.rawValue)) — 그 밖은 writeTexture로 직접 올릴 것"
            )
        }

        let sourceOrigin = command.source.origin
        // 생략된 복사 크기는 **이미지의 남은 부분 전부**다 — 이미지 크기를 알아야 나오므로 여기서 채운다.
        let size = command.copySize
            ?? WGPUExtent3D(width: bitmap.width - sourceOrigin.x, height: bitmap.height - sourceOrigin.y)
        guard size.width > 0, size.height > 0 else { return }   // no-op
        guard sourceOrigin.x + size.width <= bitmap.width,
              sourceOrigin.y + size.height <= bitmap.height else {
            throw WGPUError.validation(
                "복사 영역이 이미지를 넘는다 — (\(sourceOrigin.x), \(sourceOrigin.y)) + "
                + "\(size.width)x\(size.height) > \(bitmap.width)x\(bitmap.height)"
            )
        }

        // 명세 `GPUCopyExternalImageSourceInfo.flipY` — **복사 시점**에 위아래를 뒤집는다.
        // (`createImageBitmap`의 flipY는 디코딩 시점이라 별개다. 웹 라이브러리는 이쪽을 쓴다 —
        // three.js의 `Texture.flipY`가 기본 true라, 무시하면 텍스처가 조용히 뒤집힌다.)
        let flipY = command.source.flipY

        // 필요한 만큼만 잘라 스테이징에 싣는다. 전체 폭을 그대로 쓰면 부분 복사에서
        // bytesPerRow가 맞지 않는다.
        let rowBytes = size.width * 4
        var slice = Data(count: rowBytes * size.height)
        bitmap.pixels.withUnsafeBytes { source in
            slice.withUnsafeMutableBytes { destination in
                guard let sourceBase = source.baseAddress,
                      let destinationBase = destination.baseAddress else { return }
                for row in 0..<size.height {
                    let sourceRow = flipY ? (sourceOrigin.y + size.height - 1 - row)
                                          : (sourceOrigin.y + row)
                    let from = sourceRow * bitmap.bytesPerRow + sourceOrigin.x * 4
                    memcpy(destinationBase + row * rowBytes, sourceBase + from, rowBytes)
                }
            }
        }

        let staging = try makeStagingBuffer(slice)
        target.encodeWrite(
            from: staging,
            origin: command.destination.origin,
            size: WGPUExtent3D(width: size.width, height: size.height),
            mipLevel: command.destination.mipLevel,
            bytesPerRow: rowBytes,
            rowsPerImage: size.height,
            blit: try activeBlitEncoder()
        )
    }

    /// 압축 텍스처 복사의 블록 정렬 (명세 "validating texel copy range").
    ///
    /// origin은 블록 경계에 있어야 하고, 크기는 블록 배수이거나 **밉 레벨의 끝에 닿아야** 한다
    /// (가장자리 블록은 잘려 있으므로 예외다). 어기면 Metal이 단언으로 죽어서 여기서 먼저 막는다.
    /// 비압축 포맷은 블록이 1×1이라 이 검사가 항상 통과한다.
    private func validateBlockAlignment(
        format: WGPUTextureFormat,
        origin: WGPUOrigin3D,
        size: WGPUExtent3D,
        texture: WGPUTextureObject,
        mipLevel: Int,
        label: String
    ) throws {
        guard format.isCompressed else { return }
        let (blockWidth, blockHeight) = format.blockSize
        guard origin.x % blockWidth == 0, origin.y % blockHeight == 0 else {
            throw WGPUError.validation(
                "\(label): 압축 텍스처의 origin은 블록 경계여야 한다 "
                + "(\(origin.x), \(origin.y)) / 블록 \(blockWidth)x\(blockHeight)"
            )
        }
        let levelWidth = max(texture.size.width >> mipLevel, 1)
        let levelHeight = max(texture.size.height >> mipLevel, 1)
        guard size.width % blockWidth == 0 || origin.x + size.width == levelWidth,
              size.height % blockHeight == 0 || origin.y + size.height == levelHeight else {
            throw WGPUError.validation(
                "\(label): 압축 텍스처의 복사 크기는 블록 배수이거나 밉 레벨 끝에 닿아야 한다 "
                + "(\(size.width)x\(size.height) @ 레벨 \(mipLevel) 크기 \(levelWidth)x\(levelHeight))"
            )
        }
    }

    /// 풀에서 스테이징 버퍼를 받아 데이터를 채우고, 프레임 완료 시 회수 목록에 올린다.
    private func makeStagingBuffer(_ data: Data, minimumLength: Int = 0) throws -> MTLBuffer {
        let staging = try stagingPool.acquire(data, minimumLength: minimumLength)
        frameStagingBuffers.append(staging)
        return staging
    }

    private func createTextureView(_ command: WGPUCreateTextureViewCommand) throws {
        let source = try registry.lookup(
            command.texture, as: WGPUTextureObject.self, kind: "GPUTexture",
            path: command.fieldPath("texture")
        )
        let drawable = acquiredDrawables.first { $0.handle == command.texture }?.drawable
        let view = try WGPUTextureViewObject(
            source: source, descriptor: command.descriptor, drawable: drawable
        )
        registry.insert(view, at: command.id)
        if source.isDrawable { frameScopedHandles.append(command.id) }
    }

    private func createSampler(_ command: WGPUCreateCommand<WGPUSamplerDescriptor>) throws {
        let object = try WGPUSamplerObject(device: device, descriptor: command.descriptor)
        registry.insert(object, at: command.id)
    }

    /// 명세에서 **셰이더 모듈은 컴파일에 실패해도 만들어진다** — 오류는 `getCompilationInfo()`와
    /// 파이프라인 생성 실패로 드러난다. 그래서 파싱이 깨져도 등록하고 진단을 담아 둔다.
    ///
    /// 핸들이 아예 없으면 이후 명령이 전부 "존재하지 않는다"로만 깨져 **진짜 원인(파싱 실패)이
    /// 화면에서 사라진다.** 여기서는 원인도 그 자리에서 보고한다.
    private func createShaderModule(_ command: WGPUCreateCommand<WGPUShaderModuleDescriptor>) throws {
        let object = WGPUShaderModuleObject(descriptor: command.descriptor)
        registry.insert(object, at: command.id)
        if let failure = object.compilationMessages.first, !object.isValid {
            throw WGPUError(
                kind: failure.kind, message: failure.message,
                path: failure.path ?? command.fieldPath("code"), line: failure.line
            )
        }
    }

    private func createBindGroupLayout(_ command: WGPUCreateCommand<WGPUBindGroupLayoutDescriptor>) throws {
        registry.insert(WGPUBindGroupLayoutObject(entries: command.descriptor.entries), at: command.id)
    }

    private func createPipelineLayout(_ command: WGPUCreateCommand<WGPUPipelineLayoutDescriptor>) throws {
        let groups = try command.descriptor.bindGroupLayouts.map {
            try registry.lookup(
                $0, as: WGPUBindGroupLayoutObject.self, kind: "GPUBindGroupLayout",
                path: command.fieldPath("bindGroupLayouts")
            )
        }
        registry.insert(try WGPUPipelineLayoutObject(groups: groups), at: command.id)
    }

    private func createBindGroup(_ command: WGPUCreateCommand<WGPUBindGroupDescriptor>) throws {
        let layout = try registry.lookup(
            command.descriptor.layout, as: WGPUBindGroupLayoutObject.self, kind: "GPUBindGroupLayout",
            path: command.fieldPath("layout")
        )
        registry.insert(
            try WGPUBindGroupObject(layout: layout, descriptor: command.descriptor, registry: registry),
            at: command.id
        )
    }

    private func createQuerySet(_ command: WGPUCreateCommand<WGPUQuerySetDescriptor>) throws {
        let object = try WGPUQuerySetObject(device: device, descriptor: command.descriptor)
        registry.insert(object, at: command.id)
    }

    /// `bundleEncoder.finish()` — JS가 모아 둔 명령 목록을 번들 객체로 등록한다.
    ///
    /// 번들 인코더 자체는 네이티브에 없다. JS가 명령을 배열에 모으고 `finish()`에서 한 번에
    /// 내려보내므로, 인코더의 수명을 양쪽에서 맞출 이유가 없다.
    private func createRenderBundle(_ command: WGPUCreateRenderBundleCommand) throws {
        let bundle = try WGPURenderBundleObject(
            commands: command.commands, descriptor: command.descriptor
        )
        registry.insert(bundle, at: command.id)
    }

    private func createRenderPipeline(_ command: WGPUCreateCommand<WGPURenderPipelineDescriptor>) throws {
        var descriptor = command.descriptor
        let vertexModule = try registry.lookup(
            descriptor.vertex.module, as: WGPUShaderModuleObject.self, kind: "GPUShaderModule",
            path: command.fieldPath("vertex.module")
        )
        let fragmentModule = try descriptor.fragment.map {
            try registry.lookup(
                $0.module, as: WGPUShaderModuleObject.self, kind: "GPUShaderModule",
                path: command.fieldPath("fragment.module")
            )
        }

        // 명세의 "get the entry point" — 이름을 생략하면 그 스테이지의 유일한 진입점을 쓴다.
        // 여기서 한 번 확정해 두면 아래 계층은 전부 결정된 이름만 다룬다.
        descriptor.vertex.entryPoint = try vertexModule.resolveEntryPoint(
            descriptor.vertex.entryPoint, stage: .vertex, path: command.fieldPath("vertex.entryPoint")
        )
        if let fragmentModule, let fragment = descriptor.fragment {
            descriptor.fragment?.entryPoint = try fragmentModule.resolveEntryPoint(
                fragment.entryPoint, stage: .fragment, path: command.fieldPath("fragment.entryPoint")
            )
        }
        let vertexEntry = descriptor.vertex.entryPoint!

        var stages: [(module: WGPUShaderModuleObject, entryPoints: [String])] = []
        if let fragmentModule, fragmentModule === vertexModule, let fragment = descriptor.fragment {
            stages = [(vertexModule, [vertexEntry, fragment.entryPoint!])]
        } else {
            stages = [(vertexModule, [vertexEntry])]
            if let fragmentModule, let fragment = descriptor.fragment {
                stages.append((fragmentModule, [fragment.entryPoint!]))
            }
        }
        let layout = try WGPUPipelineLayoutResolver.resolve(descriptor.layout, stages: stages, registry: registry)
        let pipeline = try WGPURenderPipelineObject(
            device: device, descriptor: descriptor, layout: layout,
            vertexModule: vertexModule, fragmentModule: fragmentModule
        )
        registry.insert(pipeline, at: command.id)
    }

    private func createComputePipeline(_ command: WGPUCreateCommand<WGPUComputePipelineDescriptor>) throws {
        var descriptor = command.descriptor
        let module = try registry.lookup(
            descriptor.module, as: WGPUShaderModuleObject.self, kind: "GPUShaderModule",
            path: command.fieldPath("compute.module")
        )
        descriptor.entryPoint = try module.resolveEntryPoint(
            descriptor.entryPoint, stage: .compute, path: command.fieldPath("compute.entryPoint")
        )
        let layout = try WGPUPipelineLayoutResolver.resolve(
            descriptor.layout, stages: [(module, [descriptor.entryPoint!])], registry: registry
        )
        registry.insert(
            try WGPUComputePipelineObject(device: device, descriptor: descriptor, layout: layout, module: module),
            at: command.id
        )
    }

    /// `pipeline.getBindGroupLayout(index)` — `layout: "auto"`로 유도된 레이아웃을 핸들로 꺼낸다.
    private func getBindGroupLayout(_ command: WGPUGetBindGroupLayoutCommand) throws {
        let pipelineHandle = command.pipeline
        let layout: WGPUPipelineLayoutObject
        if let render = try? registry.lookup(pipelineHandle, as: WGPURenderPipelineObject.self, kind: "x") {
            layout = render.layout
        } else {
            layout = try registry.lookup(
                pipelineHandle, as: WGPUComputePipelineObject.self, kind: "GPUPipeline",
                path: command.fieldPath("pipeline")
            ).layout
        }
        guard let group = layout.group(at: command.index) else {
            throw WGPUError.validation("파이프라인에 바인드 그룹 \(command.index)이(가) 없다")
        }
        registry.insert(group, at: command.id)
    }

    // MARK: - 캔버스

    private func configureCanvas(_ configuration: WGPUCanvasConfiguration) throws {
        guard let surface = surfaceProvider(configuration.canvasId) else {
            throw WGPUError.validation(
                "캔버스 '\(configuration.canvasId)'이(가) 등록되지 않았다 "
                    + "(<webgpu-canvas canvas-id=\"…\">가 화면에 붙어 있는지 확인)"
            )
        }
        try surface.configure(configuration, device: device)
        touchedCanvases[configuration.canvasId] = surface
    }

    private func getCurrentTexture(_ command: WGPUGetCurrentTextureCommand) throws {
        let handle = command.id
        let canvasId = command.canvas
        guard let surface = surfaceProvider(canvasId) else {
            throw WGPUError.validation("캔버스 '\(canvasId)'이(가) 등록되지 않았다")
        }
        touchedCanvases[canvasId] = surface
        guard let drawable = surface.nextDrawable() else {
            throw WGPUError.validation(
                "캔버스 '\(canvasId)'의 드로어블을 얻지 못했다 (크기가 0이거나 드로어블이 고갈됨)"
            )
        }
        // 실제 드로어블 텍스처의 포맷을 쓴다 — 캔버스 설정 반영이 한 프레임 늦을 수 있기 때문.
        let format = WGPUMetalMapping.textureFormat(from: drawable.texture.pixelFormat)
            ?? surface.configuredFormat
        let texture = WGPUTextureObject(drawableTexture: drawable.texture, format: format)
        registry.insert(texture, at: handle)
        acquiredDrawables.append((handle, drawable, surface))
        frameScopedHandles.append(handle)
    }

    // MARK: - 렌더 패스

    private func beginRenderPass(_ descriptor: WGPURenderPassDescriptor) throws {
        endActiveEncoders()
        let passDescriptor = MTLRenderPassDescriptor()
        var colorFormats: [WGPUTextureFormat] = []
        var sampleCount = 1

        for (index, attachment) in descriptor.colorAttachments.enumerated() {
            let view = try registry.lookup(
                attachment.view, as: WGPUTextureViewObject.self, kind: "GPUTextureView"
            )
            colorFormats.append(view.format)
            sampleCount = max(sampleCount, view.sampleCount)
            let target = passDescriptor.colorAttachments[index]!
            target.texture = view.texture
            target.loadAction = WGPUMetalMapping.loadAction(attachment.loadOp)
            target.storeAction = WGPUMetalMapping.storeAction(attachment.storeOp)
            target.clearColor = MTLClearColor(
                red: attachment.clearValue.red,
                green: attachment.clearValue.green,
                blue: attachment.clearValue.blue,
                alpha: attachment.clearValue.alpha
            )
            if let resolveHandle = attachment.resolveTarget {
                let resolve = try registry.lookup(
                    resolveHandle, as: WGPUTextureViewObject.self, kind: "GPUTextureView"
                )
                target.resolveTexture = resolve.texture
                target.storeAction = .multisampleResolve
            }
        }

        var depthStencilFormat: WGPUTextureFormat?
        var depthReadOnly = false
        var stencilReadOnly = false
        if let depth = descriptor.depthStencilAttachment {
            let view = try registry.lookup(depth.view, as: WGPUTextureViewObject.self, kind: "GPUTextureView")
            depthStencilFormat = view.format
            // 깊이 뷰도 패스 레이아웃의 sampleCount에 반영한다 — 컬러 어태치먼트가 없는 MSAA 패스
            // (그림자 맵·깊이 프리패스)에서 이걸 빠뜨리면 올바르게 선언한 번들이 거부된다.
            // 명세는 모든 어태치먼트의 sampleCount가 같기를 요구하므로 max로 충분하다.
            sampleCount = max(sampleCount, view.sampleCount)
            if view.format.hasDepth {
                let target = passDescriptor.depthAttachment!
                target.texture = view.texture
                // readOnly면 load/store op을 줄 수 없으므로(디코딩에서 막는다) 내용을 그대로
                // 읽고 그대로 남기는 조합이 된다.
                target.loadAction = WGPUMetalMapping.loadAction(depth.depthLoadOp ?? .load)
                target.storeAction = WGPUMetalMapping.storeAction(depth.depthStoreOp ?? .store)
                target.clearDepth = depth.depthClearValue
            }
            if view.format.hasStencil {
                let target = passDescriptor.stencilAttachment!
                target.texture = view.texture
                target.loadAction = WGPUMetalMapping.loadAction(depth.stencilLoadOp ?? .load)
                target.storeAction = WGPUMetalMapping.storeAction(depth.stencilStoreOp ?? .store)
                target.clearStencil = UInt32(truncatingIfNeeded: depth.stencilClearValue)
            }
            depthReadOnly = depth.depthReadOnly
            stencilReadOnly = depth.stencilReadOnly
        }

        // occlusion 쿼리는 **패스를 열 때만** 붙일 수 있다 (Metal도 WebGPU도 같은 제약).
        var occlusionQuerySet: WGPUQuerySetObject?
        if let handle = descriptor.occlusionQuerySet {
            let querySet = try registry.lookup(handle, as: WGPUQuerySetObject.self, kind: "GPUQuerySet")
            guard querySet.type == .occlusion else {
                throw WGPUError.validation(
                    "occlusionQuerySet은 type: \"occlusion\"이어야 한다 (받은 것: \(querySet.type.rawValue))"
                )
            }
            passDescriptor.visibilityResultBuffer = querySet.visibilityBuffer
            occlusionQuerySet = querySet
        }

        if let writes = descriptor.timestampWrites {
            let attachment = passDescriptor.sampleBufferAttachments[0]!
            let querySet = try timestampSampleBuffer(writes)
            attachment.sampleBuffer = querySet
            // WebGPU의 "패스 시작/끝"을 Metal의 스테이지 경계에 맞춘다 —
            // 시작은 정점 스테이지 진입, 끝은 프래그먼트 스테이지 종료다.
            attachment.startOfVertexSampleIndex = writes.beginningOfPassWriteIndex ?? MTLCounterDontSample
            attachment.endOfVertexSampleIndex = MTLCounterDontSample
            attachment.startOfFragmentSampleIndex = MTLCounterDontSample
            attachment.endOfFragmentSampleIndex = writes.endOfPassWriteIndex ?? MTLCounterDontSample
        }

        guard let encoder = try activeCommandBuffer().makeRenderCommandEncoder(descriptor: passDescriptor) else {
            throw WGPUError.backend("MTLRenderCommandEncoder 생성 실패 — 어태치먼트 설정을 확인할 것")
        }
        if let label = descriptor.label { encoder.label = label }
        renderEncoder = encoder
        passFormats = (colorFormats, depthStencilFormat, sampleCount)
        passOcclusionQuerySet = occlusionQuerySet
        passDepthReadOnly = depthReadOnly
        passStencilReadOnly = stencilReadOnly
        openOcclusionQuery = nil
        usedOcclusionQueries.removeAll()
        resetPassBindings()
    }

    // MARK: - 쿼리

    /// 타임스탬프 쓰기 자리를 확인하고 카운터 샘플 버퍼를 꺼낸다.
    private func timestampSampleBuffer(_ writes: WGPUPassTimestampWrites) throws -> MTLCounterSampleBuffer {
        let querySet = try registry.lookup(writes.querySet, as: WGPUQuerySetObject.self, kind: "GPUQuerySet")
        guard querySet.type == .timestamp, let buffer = querySet.counterBuffer else {
            throw WGPUError.validation(
                "timestampWrites의 쿼리셋은 type: \"timestamp\"여야 한다 (받은 것: \(querySet.type.rawValue))"
            )
        }
        // 둘 다 생략하면 Metal 샘플 인덱스가 전부 `MTLCounterDontSample`이 되어 **조용한 no-op
        // 패스**가 된다. 오류 없이 쿼리셋 초기값(0)이 resolve되므로 앱은 GPU 시간을 0ns로 읽는다.
        guard writes.beginningOfPassWriteIndex != nil || writes.endOfPassWriteIndex != nil else {
            throw WGPUError.validation(
                "timestampWrites는 beginningOfPassWriteIndex와 endOfPassWriteIndex 중 "
                    + "최소 하나를 줘야 한다",
                path: "timestampWrites"
            )
        }
        // 같은 슬롯을 가리키면 나중 샘플이 앞의 것을 덮어 델타가 의미를 잃는다.
        if let begin = writes.beginningOfPassWriteIndex, begin == writes.endOfPassWriteIndex {
            throw WGPUError.validation(
                "timestampWrites의 두 인덱스는 서로 달라야 한다 (둘 다 \(begin))",
                path: "timestampWrites"
            )
        }
        for index in [writes.beginningOfPassWriteIndex, writes.endOfPassWriteIndex].compactMap({ $0 }) {
            try querySet.checkRange(first: index, count: 1, path: "timestampWrites")
        }
        return buffer
    }

    private func beginOcclusionQuery(_ command: WGPUBeginOcclusionQueryCommand) throws {
        let encoder = try requireRenderEncoder()
        guard let querySet = passOcclusionQuerySet else {
            throw WGPUError.validation(
                "beginOcclusionQuery를 쓰려면 beginRenderPass에 occlusionQuerySet을 줘야 한다"
            )
        }
        guard openOcclusionQuery == nil else {
            throw WGPUError.validation("occlusion 쿼리는 중첩할 수 없다 (앞의 것을 endOcclusionQuery로 닫을 것)")
        }
        let index = command.queryIndex
        try querySet.checkRange(first: index, count: 1, path: command.fieldPath("queryIndex"))
        // 한 패스에서 같은 인덱스를 두 번 쓰면 두 구간이 같은 8바이트 슬롯을 나눠 쓴다 —
        // 최종 값이 Metal의 누적/덮어쓰기 동작에 달린 값이 되어 브라우저와 결과가 갈린다.
        guard usedOcclusionQueries.insert(index).inserted else {
            throw WGPUError.validation(
                "occlusion 쿼리 인덱스 \(index)은(는) 이 패스에서 이미 썼다",
                path: command.fieldPath("queryIndex")
            )
        }
        // `.counting`은 통과한 **샘플 수**를 센다 — 명세의 occlusion 결과와 같은 뜻이다.
        encoder.setVisibilityResultMode(.counting, offset: index * WGPUQuerySetObject.resultStride)
        openOcclusionQuery = index
    }

    private func endOcclusionQuery() throws {
        let encoder = try requireRenderEncoder()
        guard let index = openOcclusionQuery else {
            throw WGPUError.validation("endOcclusionQuery: 열려 있는 occlusion 쿼리가 없다")
        }
        encoder.setVisibilityResultMode(.disabled, offset: index * WGPUQuerySetObject.resultStride)
        openOcclusionQuery = nil
    }

    /// 쿼리 결과를 버퍼로 내린다. 종류마다 blit 명령이 다르다.
    private func resolveQuerySet(_ command: WGPUResolveQuerySetCommand) throws {
        let querySet = try registry.lookup(
            command.querySet, as: WGPUQuerySetObject.self, kind: "GPUQuerySet",
            path: command.fieldPath("querySet")
        )
        let destination = try unmappedBuffer(command.destination, path: command.fieldPath("destination"))
        let first = command.firstQuery
        // 생략하면 쿼리셋의 남은 전부 — 쿼리셋 크기를 알아야 하므로 여기서 채운다.
        let count = command.queryCount ?? (querySet.count - first)
        let offset = command.destinationOffset
        try querySet.checkRange(first: first, count: count, path: command.fieldPath("firstQuery"))

        // 명세가 요구하는 정렬. Metal은 이보다 느슨해서 여기서 안 막으면 브라우저에서만 깨진다.
        guard offset >= 0, offset % 256 == 0 else {
            throw WGPUError.validation(
                "destinationOffset은 256의 배수여야 한다 (받은 값 \(offset))",
                path: command.fieldPath("destinationOffset")
            )
        }
        let byteCount = count * WGPUQuerySetObject.resultStride
        guard offset + byteCount <= destination.size else {
            throw WGPUError.validation(
                "쿼리 결과 \(byteCount)B가 버퍼 범위를 넘는다 — "
                    + "offset \(offset) + \(byteCount)B > 버퍼 크기 \(destination.size)B",
                path: command.fieldPath("destinationOffset")
            )
        }
        guard destination.usage.contains(.queryResolve) else {
            throw WGPUError.validation(
                "resolveQuerySet의 목적지 버퍼는 GPUBufferUsage.QUERY_RESOLVE로 만들어야 한다",
                path: command.fieldPath("destination")
            )
        }
        guard count > 0 else { return }   // no-op (Metal은 0바이트 복사를 거부한다)

        let encoder = try activeBlitEncoder()
        switch querySet.type {
        case .occlusion:
            guard let source = querySet.visibilityBuffer else {
                throw WGPUError.backend("occlusion 쿼리 버퍼가 없다")
            }
            encoder.copy(
                from: source, sourceOffset: first * WGPUQuerySetObject.resultStride,
                to: destination.buffer, destinationOffset: offset, size: byteCount
            )
        case .timestamp:
            guard let source = querySet.counterBuffer else {
                throw WGPUError.backend("타임스탬프 샘플 버퍼가 없다")
            }
            // 카운터는 평범한 버퍼가 아니라 전용 저장소라 resolveCounters로만 꺼낼 수 있다.
            encoder.resolveCounters(
                source, range: first..<(first + count),
                destinationBuffer: destination.buffer, destinationOffset: offset
            )
        }
    }

    /// 파이프라인·바인드 그룹·정점/인덱스 버퍼 바인딩을 "지정되지 않음"으로 되돌린다.
    ///
    /// 패스를 새로 열 때와 `executeBundles` 앞뒤에 쓴다. 명세는 번들 실행이 **이전 상태를
    /// 복원하는 것이 아니라 무효화한다**고 정한다 — 번들은 패스 상태를 물려받지 않고,
    /// 실행이 끝나면 패스도 번들이 남긴 상태를 물려받지 않는다. 그래서 양쪽 다 초기화한다.
    /// (뷰포트·시저·블렌드 상수·스텐실 참조는 이 목록에 없다 — 그대로 남는다.)
    private func resetPassBindings() {
        currentRenderPipeline = nil
        boundGroups.removeAll()
        dirtyGroups.removeAll()
        indexBinding = nil
        vertexBindings.removeAll()
        dirtyVertexSlots.removeAll()
    }

    private func executeBundles(_ command: WGPUExecuteBundlesCommand) throws {
        _ = try requireRenderEncoder()
        guard let formats = passFormats else {
            throw WGPUError.validation("executeBundles는 렌더 패스 안에서만 쓸 수 있다")
        }
        let bundles = try command.bundles.map {
            try registry.lookup(
                $0, as: WGPURenderBundleObject.self, kind: "GPURenderBundle",
                path: command.fieldPath("bundles")
            )
        }

        for bundle in bundles {
            try bundle.checkCompatibility(
                color: formats.color,
                depthStencil: formats.depthStencil,
                sampleCount: formats.sampleCount,
                depthReadOnly: passDepthReadOnly,
                stencilReadOnly: passStencilReadOnly
            )
        }
        // 명세의 "Reset the render pass binding state"(step 4)는 호환성 검증만 통과하면 **무조건**
        // 실행된다. 번들 명령 하나가 실패해 throw해도 마찬가지다 — 여기서 빠뜨리면 그 뒤의 패스 명령이
        // 번들이 남긴 파이프라인·바인드 그룹을 물고 그려져 잘못된 픽셀이 나간다.
        defer { resetPassBindings() }
        // 하나라도 맞지 않으면 아무것도 실행하지 않는다 — 절반만 그려진 프레임을 남기지 않는다.
        for bundle in bundles {
            resetPassBindings()
            for bundleCommand in bundle.commands {
                try perform(bundleCommand, at: 0)
            }
        }
    }

    private func setPipeline(_ command: WGPUSetPipelineCommand) throws {
        let handle = command.pipeline
        if let encoder = renderEncoder {
            let pipeline = try registry.lookup(
                handle, as: WGPURenderPipelineObject.self, kind: "GPURenderPipeline",
                path: command.fieldPath("pipeline")
            )
            // read-only로 선언한 어태치먼트를 쓰는 파이프라인은 여기서 막는다 — Metal은 그냥
            // 써 버리므로, 안 막으면 read-only라고 적어 둔 깊이 버퍼가 실제로 변조된다.
            guard !passDepthReadOnly || !pipeline.writesDepth else {
                throw WGPUError.validation(
                    "depthReadOnly 패스에서는 depthWriteEnabled: true 파이프라인을 쓸 수 없다"
                )
            }
            guard !passStencilReadOnly || !pipeline.writesStencil else {
                throw WGPUError.validation(
                    "stencilReadOnly 패스에서는 스텐실을 쓰는 파이프라인을 쓸 수 없다 "
                        + "(failOp·depthFailOp·passOp가 모두 \"keep\"이어야 한다)"
                )
            }
            encoder.setRenderPipelineState(pipeline.state)
            encoder.setCullMode(pipeline.cullMode)
            encoder.setFrontFacing(pipeline.winding)
            if let depthStencilState = pipeline.depthStencilState {
                encoder.setDepthStencilState(depthStencilState)
            }
            if pipeline.depthBias != 0 || pipeline.depthBiasSlopeScale != 0 {
                encoder.setDepthBias(
                    pipeline.depthBias, slopeScale: pipeline.depthBiasSlopeScale, clamp: pipeline.depthBiasClamp
                )
            }
            currentRenderPipeline = pipeline
        } else if let encoder = computeEncoder {
            let pipeline = try registry.lookup(
                handle, as: WGPUComputePipelineObject.self, kind: "GPUComputePipeline",
                path: command.fieldPath("pipeline")
            )
            encoder.setComputePipelineState(pipeline.state)
            currentComputePipeline = pipeline
        } else {
            throw WGPUError.validation("setPipeline은 패스 안에서만 쓸 수 있다")
        }
        // 파이프라인이 바뀌면 레이아웃이 달라질 수 있으므로 바인드 그룹을 다시 적용한다.
        dirtyGroups = Set(boundGroups.keys)
    }

    private func setBindGroup(_ command: WGPUSetBindGroupCommand) throws {
        let group = try registry.lookup(
            command.bindGroup, as: WGPUBindGroupObject.self, kind: "GPUBindGroup",
            path: command.fieldPath("bindGroup")
        )
        boundGroups[command.index] = (group, command.dynamicOffsets)
        dirtyGroups.insert(command.index)
    }

    /// 드로우·디스패치 직전에 파이프라인이 요구하는 상태를 전부 확인하고 인코더에 올린다.
    ///
    /// 바인드 그룹과 정점 버퍼를 한자리에서 다루는 이유는, 둘 다 **번들 경계에서 무효화되는
    /// 상태**라 검사 시점이 같아야 하기 때문이다. 새 드로우 op을 추가할 때 이 함수 하나만
    /// 부르면 격리 계약이 자동으로 따라온다.
    ///
    /// 파이프라인 가드는 각 드로우 op이 자기 이름이 든 메시지로 **이 함수보다 먼저** 세운다 —
    /// 아래의 같은 검사는 그 가드를 빠뜨린 op을 위한 안전망이라 메시지가 일반형이다.
    private func applyDrawState() throws {
        let layout: WGPUPipelineLayoutObject
        let needsSizes: Bool
        if renderEncoder != nil {
            guard let pipeline = currentRenderPipeline else {
                throw WGPUError.validation("draw 전에 setPipeline이 필요하다")
            }
            layout = pipeline.layout
            needsSizes = pipeline.needsBufferSizes
        } else {
            guard let pipeline = currentComputePipeline else {
                throw WGPUError.validation("dispatch 전에 setPipeline이 필요하다")
            }
            layout = pipeline.layout
            needsSizes = pipeline.needsBufferSizes
        }

        // 레이아웃이 요구하는 그룹이 전부 바인드되어 있어야 한다. 이 검사가 없으면 번들이
        // 남긴 바인딩(또는 패스가 미리 올려 둔 바인딩)으로 조용히 그려진다 — Metal 인코더에는
        // "바인딩 해제"가 없으므로 `resetPassBindings()`만으로는 실제로 격리되지 않는다.
        for groupIndex in layout.requiredGroups.sorted() where boundGroups[groupIndex] == nil {
            throw WGPUError.validation(
                "파이프라인 레이아웃이 요구하는 @group(\(groupIndex))이 바인드되지 않았다 "
                    + "(번들 실행 앞뒤로는 바인딩이 무효화된다 — setBindGroup을 다시 할 것)"
            )
        }

        // 바인드 그룹이 물고 있는 버퍼가 매핑 중이면 이 드로우도 큐 작업이므로 거부한다.
        // (그룹은 만들 때 버퍼를 고정하므로, 만든 뒤에 매핑된 경우가 여기서 걸린다.)
        for (_, bound) in boundGroups {
            for buffer in bound.group.bufferObjects where buffer.isMapped {
                throw WGPUError.validation(
                    "매핑 중인 버퍼를 물고 있는 바인드 그룹으로는 그릴 수 없다 (unmap()을 먼저 부를 것)"
                )
            }
        }

        for groupIndex in dirtyGroups.sorted() {
            guard let bound = boundGroups[groupIndex] else { continue }
            try apply(bound.group, at: groupIndex, dynamicOffsets: bound.offsets, layout: layout)
        }
        dirtyGroups.removeAll()

        try applyVertexBuffers()

        // `arrayLength()`용 버퍼 크기 표. 88바이트라 setBytes로 매 드로우 올려도 부담이 없다.
        guard needsSizes else { return }
        let byteLength = bufferSizes.count * MemoryLayout<UInt32>.stride
        let index = WGSLMetalLimits.bufferSizesIndex
        bufferSizes.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return }
            if let encoder = renderEncoder {
                encoder.setVertexBytes(base, length: byteLength, index: index)
                encoder.setFragmentBytes(base, length: byteLength, index: index)
            } else if let encoder = computeEncoder {
                encoder.setBytes(base, length: byteLength, index: index)
            }
        }
    }

    private func apply(
        _ group: WGPUBindGroupObject,
        at groupIndex: Int,
        dynamicOffsets: [Int],
        layout: WGPUPipelineLayoutObject
    ) throws {
        var offsetCursor = 0
        for binding in group.bindings.sorted(by: { $0.binding < $1.binding }) {
            guard let metalIndex = layout.assignment.index(group: groupIndex, binding: binding.binding) else {
                throw WGPUError.validation(
                    "파이프라인 레이아웃에 @group(\(groupIndex)) @binding(\(binding.binding))이 없다"
                )
            }
            switch binding.resource {
            case .buffer(let buffer, let offset, let boundSize):
                // `arrayLength()`가 볼 크기 표를 채운다 (셰이더가 쓸 때만 업로드한다).
                if metalIndex < bufferSizes.count { bufferSizes[metalIndex] = UInt32(boundSize) }
                var finalOffset = offset
                if group.layout.entry(binding: binding.binding).map(Self.hasDynamicOffset) == true {
                    guard offsetCursor < dynamicOffsets.count else {
                        throw WGPUError.validation("dynamicOffsets 개수가 레이아웃 선언보다 적다")
                    }
                    finalOffset += dynamicOffsets[offsetCursor]
                    offsetCursor += 1
                }
                if let encoder = renderEncoder {
                    if binding.visibility.contains(.vertex) {
                        encoder.setVertexBuffer(buffer, offset: finalOffset, index: metalIndex)
                    }
                    if binding.visibility.contains(.fragment) {
                        encoder.setFragmentBuffer(buffer, offset: finalOffset, index: metalIndex)
                    }
                } else if let encoder = computeEncoder {
                    encoder.setBuffer(buffer, offset: finalOffset, index: metalIndex)
                }
            case .texture(let texture):
                if let encoder = renderEncoder {
                    if binding.visibility.contains(.vertex) { encoder.setVertexTexture(texture, index: metalIndex) }
                    if binding.visibility.contains(.fragment) {
                        encoder.setFragmentTexture(texture, index: metalIndex)
                    }
                } else if let encoder = computeEncoder {
                    encoder.setTexture(texture, index: metalIndex)
                }
            case .sampler(let sampler):
                if let encoder = renderEncoder {
                    if binding.visibility.contains(.vertex) {
                        encoder.setVertexSamplerState(sampler, index: metalIndex)
                    }
                    if binding.visibility.contains(.fragment) {
                        encoder.setFragmentSamplerState(sampler, index: metalIndex)
                    }
                } else if let encoder = computeEncoder {
                    encoder.setSamplerState(sampler, index: metalIndex)
                }
            }
        }
    }

    private static func hasDynamicOffset(_ entry: WGPUBindGroupLayoutEntry) -> Bool {
        if case .buffer(let buffer) = entry.layout { return buffer.hasDynamicOffset }
        return false
    }

    private func setVertexBuffer(_ command: WGPUSetVertexBufferCommand) throws {
        _ = try requireRenderEncoder()
        let slot = command.slot
        guard slot >= 0, slot < WGSLMetalLimits.maxVertexBufferSlots else {
            throw WGPUError.validation("정점 버퍼 슬롯은 0~\(WGSLMetalLimits.maxVertexBufferSlots - 1) 범위다")
        }
        let buffer = try unmappedBuffer(command.buffer, path: command.fieldPath("buffer"))
        let offset = command.offset
        guard offset >= 0, offset <= buffer.size else {
            throw WGPUError.validation(
                "정점 버퍼 offset(\(offset))이 버퍼 크기(\(buffer.size)B)를 벗어난다",
                path: command.fieldPath("offset")
            )
        }
        // 인코더에 바로 올리지 않는다 — 드로우 직전에 올려야 번들 경계의 무효화가 성립한다.
        vertexBindings[slot] = (buffer.buffer, offset)
        dirtyVertexSlots.insert(slot)
    }

    /// 드로우 직전에 파이프라인이 요구하는 정점 버퍼가 다 있는지 보고 인코더에 올린다.
    ///
    /// 명세는 "`vertex.buffers[slot]`이 null이 아니면 `[[vertex_buffers]]`가 그 슬롯을 담아야
    /// 한다"고 정한다. 이 검사가 없으면 패스가 미리 올려 둔 정점 버퍼로 번들이 그려지고,
    /// 번들이 올린 것으로 패스가 그려진다 — 브라우저에서는 둘 다 무효인 코드다.
    private func applyVertexBuffers() throws {
        guard let encoder = renderEncoder, let pipeline = currentRenderPipeline else { return }
        for slot in pipeline.requiredVertexSlots.sorted() where vertexBindings[slot] == nil {
            throw WGPUError.validation(
                "파이프라인이 요구하는 정점 버퍼 슬롯 \(slot)이 바인드되지 않았다 "
                    + "(번들 실행 앞뒤로는 바인딩이 무효화된다 — setVertexBuffer를 다시 할 것)"
            )
        }
        for slot in dirtyVertexSlots.sorted() {
            guard let binding = vertexBindings[slot] else { continue }
            encoder.setVertexBuffer(
                binding.buffer,
                offset: binding.offset,
                index: WGSLMetalLimits.vertexBufferIndex(slot: slot)
            )
        }
        dirtyVertexSlots.removeAll()
    }

    private func setIndexBuffer(_ command: WGPUSetIndexBufferCommand) throws {
        _ = try requireRenderEncoder()
        let buffer = try unmappedBuffer(command.buffer, path: command.fieldPath("buffer"))
        indexBinding = (
            buffer.buffer,
            command.offset,
            WGPUMetalMapping.indexType(command.format),
            command.indexStride
        )
    }

    private func setViewport(_ command: WGPUSetViewportCommand) throws {
        let encoder = try requireRenderEncoder()
        encoder.setViewport(MTLViewport(
            originX: command.x,
            originY: command.y,
            width: command.width,
            height: command.height,
            znear: command.minDepth,
            zfar: command.maxDepth
        ))
    }

    private func setScissorRect(_ command: WGPUSetScissorRectCommand) throws {
        let encoder = try requireRenderEncoder()
        encoder.setScissorRect(MTLScissorRect(
            x: command.x, y: command.y, width: command.width, height: command.height
        ))
    }

    private func setBlendConstant(_ command: WGPUSetBlendConstantCommand) throws {
        let encoder = try requireRenderEncoder()
        let color = command.color
        encoder.setBlendColor(
            red: Float(color.red), green: Float(color.green), blue: Float(color.blue), alpha: Float(color.alpha)
        )
    }

    private func draw(_ command: WGPUDrawCommand) throws {
        let encoder = try requireRenderEncoder()
        // 파이프라인 가드는 `applyDrawState()`보다 **먼저** 둔다 — 그 안의 같은 검사가 먼저
        // 던지면 이 op 이름이 든 메시지가 영영 나가지 못하는 죽은 코드가 된다 (아래 draw 계열 공통).
        guard let pipeline = currentRenderPipeline else {
            throw WGPUError.validation("draw 전에 setPipeline이 필요하다")
        }
        try applyDrawState()
        encoder.drawPrimitives(
            type: pipeline.primitiveType,
            vertexStart: command.firstVertex,
            vertexCount: command.vertexCount,
            instanceCount: command.instanceCount,
            baseInstance: command.firstInstance
        )
    }

    private func drawIndexed(_ command: WGPUDrawIndexedCommand) throws {
        let encoder = try requireRenderEncoder()
        guard let pipeline = currentRenderPipeline else {
            throw WGPUError.validation("drawIndexed 전에 setPipeline이 필요하다")
        }
        guard let indexBinding else {
            throw WGPUError.validation("drawIndexed 전에 setIndexBuffer가 필요하다")
        }
        try applyDrawState()
        encoder.drawIndexedPrimitives(
            type: pipeline.primitiveType,
            indexCount: command.indexCount,
            indexType: indexBinding.type,
            indexBuffer: indexBinding.buffer,
            indexBufferOffset: indexBinding.offset + command.firstIndex * indexBinding.stride,
            instanceCount: command.instanceCount,
            baseVertex: command.baseVertex,
            baseInstance: command.firstInstance
        )
    }

    // MARK: - 간접 드로우 / 디스패치

    /// 간접 인자 버퍼를 찾고 오프셋을 검증한다.
    ///
    /// WebGPU와 Metal의 인자 구조체가 **필드 순서까지 1:1로 같아서** 변환 없이 버퍼를 그대로
    /// 넘긴다. 그래서 여기서 막을 것은 "몇 바이트를 읽을 것인가"뿐이다:
    ///
    /// - 4바이트 정렬과 범위는 **Metal이 단언(=프로세스 종료)으로 처리**하므로 여기서 잡는다.
    /// - `INDIRECT` usage는 Metal에 대응하는 개념이 아예 없어 Metal이 봐 주지 않는다.
    ///   확인하지 않으면 여기서는 돌고 브라우저에서만 깨지는 코드가 나온다.
    private func indirectArguments(
        _ command: WGPUIndirectCommand,
        argumentSize: Int
    ) throws -> (buffer: MTLBuffer, offset: Int) {
        // 기기가 간접 인자를 지원하지 않으면 **여기서 막는다.** 그대로 Metal에 넘기면
        // `MTLValidateFeatureSupport ... failed assertion`으로 프로세스가 죽어, 앱은
        // 이유를 남기지도 못한다. 세 간접 op이 모두 이 함수를 지나므로 한 자리로 충분하다.
        guard WGPUDeviceCapability.supportsIndirectArguments(device) else {
            throw WGPUError.unsupported(
                "이 기기는 간접 드로우·디스패치 인자를 지원하지 않는다 (Metal이 Apple GPU family 3 "
                    + "이상을 요구한다). **iOS 시뮬레이터가 여기 해당한다** — 실기기(A12 이상)에서는 "
                    + "동작하므로, 직접 드로우로 대체하거나 실기기에서 확인할 것"
            )
        }
        let object = try unmappedBuffer(command.indirectBuffer, path: command.fieldPath("indirectBuffer"))
        let offset = command.indirectOffset
        guard offset >= 0, offset % 4 == 0 else {
            throw WGPUError.validation(
                "indirectOffset은 4의 배수여야 한다 (받은 값 \(offset))",
                path: command.fieldPath("indirectOffset")
            )
        }
        guard offset + argumentSize <= object.size else {
            throw WGPUError.validation(
                "간접 인자 \(argumentSize)B가 버퍼 범위를 넘는다 — "
                    + "offset \(offset) + \(argumentSize)B > 버퍼 크기 \(object.size)B",
                path: command.fieldPath("indirectOffset")
            )
        }
        guard object.usage.contains(.indirect) else {
            throw WGPUError.validation(
                "간접 드로우/디스패치의 인자 버퍼는 GPUBufferUsage.INDIRECT로 만들어야 한다",
                path: command.fieldPath("indirectBuffer")
            )
        }
        return (object.buffer, offset)
    }

    private func drawIndirect(_ command: WGPUIndirectCommand) throws {
        let encoder = try requireRenderEncoder()
        // 인자 검증을 `applyDrawState()`보다 **먼저** 한다 — 거부할 명령이 인코더 상태를
        // 이미 바꿔 놓는 일이 없어야 한다 (오류는 프레임을 죽이지 않고 누적되므로 더 그렇다).
        // vertexCount, instanceCount, firstVertex, firstInstance — u32 4개.
        let arguments = try indirectArguments(command, argumentSize: 16)
        guard let pipeline = currentRenderPipeline else {
            throw WGPUError.validation("drawIndirect 전에 setPipeline이 필요하다")
        }
        try applyDrawState()
        encoder.drawPrimitives(
            type: pipeline.primitiveType,
            indirectBuffer: arguments.buffer,
            indirectBufferOffset: arguments.offset
        )
    }

    private func drawIndexedIndirect(_ command: WGPUIndirectCommand) throws {
        let encoder = try requireRenderEncoder()
        // indexCount, instanceCount, firstIndex, baseVertex(i32), firstInstance — 5칸.
        let arguments = try indirectArguments(command, argumentSize: 20)
        guard let pipeline = currentRenderPipeline else {
            throw WGPUError.validation("drawIndexedIndirect 전에 setPipeline이 필요하다")
        }
        guard let indexBinding else {
            throw WGPUError.validation("drawIndexedIndirect 전에 setIndexBuffer가 필요하다")
        }
        try applyDrawState()
        encoder.drawIndexedPrimitives(
            type: pipeline.primitiveType,
            indexType: indexBinding.type,
            indexBuffer: indexBinding.buffer,
            // 직접 경로(`drawIndexed`)와 달리 `firstIndex`를 여기 더하지 않는다 —
            // 그 값은 인자 버퍼 안에 있고 GPU가 읽는다. 더하면 두 번 세어 조용히 틀린다.
            indexBufferOffset: indexBinding.offset,
            indirectBuffer: arguments.buffer,
            indirectBufferOffset: arguments.offset
        )
    }

    private func dispatchWorkgroupsIndirect(_ command: WGPUIndirectCommand) throws {
        let encoder = try requireComputeEncoder()
        // x, y, z — u32 3개.
        let arguments = try indirectArguments(command, argumentSize: 12)
        guard let pipeline = currentComputePipeline else {
            throw WGPUError.validation("dispatchWorkgroupsIndirect 전에 setPipeline이 필요하다")
        }
        try applyDrawState()
        encoder.dispatchThreadgroups(
            indirectBuffer: arguments.buffer,
            indirectBufferOffset: arguments.offset,
            threadsPerThreadgroup: pipeline.threadsPerThreadgroup
        )
    }

    // MARK: - 컴퓨트 패스

    private func beginComputePass(_ descriptor: WGPUComputePassDescriptor) throws {
        endActiveEncoders()
        let buffer = try activeCommandBuffer()

        let encoder: MTLComputeCommandEncoder?
        if let writes = descriptor.timestampWrites {
            let passDescriptor = MTLComputePassDescriptor()
            let attachment = passDescriptor.sampleBufferAttachments[0]!
            attachment.sampleBuffer = try timestampSampleBuffer(writes)
            attachment.startOfEncoderSampleIndex = writes.beginningOfPassWriteIndex ?? MTLCounterDontSample
            attachment.endOfEncoderSampleIndex = writes.endOfPassWriteIndex ?? MTLCounterDontSample
            encoder = buffer.makeComputeCommandEncoder(descriptor: passDescriptor)
        } else {
            encoder = buffer.makeComputeCommandEncoder()
        }
        guard let encoder else {
            throw WGPUError.backend("MTLComputeCommandEncoder 생성 실패")
        }
        if let label = descriptor.label { encoder.label = label }
        computeEncoder = encoder
        currentComputePipeline = nil
        boundGroups.removeAll()
        dirtyGroups.removeAll()
    }

    private func dispatchWorkgroups(_ command: WGPUDispatchWorkgroupsCommand) throws {
        let encoder = try requireComputeEncoder()
        guard let pipeline = currentComputePipeline else {
            throw WGPUError.validation("dispatchWorkgroups 전에 setPipeline이 필요하다")
        }
        try applyDrawState()
        encoder.dispatchThreadgroups(
            MTLSize(width: command.x, height: command.y, depth: command.z),
            threadsPerThreadgroup: pipeline.threadsPerThreadgroup
        )
    }

    // MARK: - 복사

    private func copyBufferToBuffer(_ command: WGPUCopyBufferToBufferCommand) throws {
        let source = try unmappedBuffer(command.source, path: command.fieldPath("source"))
        let destination = try unmappedBuffer(command.destination, path: command.fieldPath("destination"))
        let sourceOffset = command.sourceOffset
        let destinationOffset = command.destinationOffset
        // 명세의 짧은 형태 `copyBufferToBuffer(src, dst)`는 "원본의 남은 전부"다.
        // JS shim이 채워 보내지만, 커맨드 스트림을 직접 만드는 쪽(네이티브 단독 사용)에도
        // 같은 기본값을 준다 — `clearBuffer`와 규칙을 맞춘다.
        let size = command.size ?? max(0, source.size - sourceOffset)
        // 범위를 넘는 복사는 **Metal이 단언으로 죽인다.** 여기서 검증 오류로 바꾼다.
        guard sourceOffset >= 0, destinationOffset >= 0, size >= 0 else {
            throw WGPUError.validation(
                "copyBufferToBuffer의 오프셋·크기는 음수일 수 없다 "
                + "(sourceOffset \(sourceOffset), destinationOffset \(destinationOffset), size \(size))"
            )
        }
        guard sourceOffset + size <= source.size else {
            throw WGPUError.validation(
                "copyBufferToBuffer 원본 범위가 버퍼를 넘는다 — "
                + "\(sourceOffset) + \(size)B > 크기 \(source.size)B",
                path: command.fieldPath("size")
            )
        }
        guard destinationOffset + size <= destination.size else {
            throw WGPUError.validation(
                "copyBufferToBuffer 대상 범위가 버퍼를 넘는다 — "
                + "\(destinationOffset) + \(size)B > 크기 \(destination.size)B",
                path: command.fieldPath("size")
            )
        }
        guard size > 0 else { return }   // Metal blit은 0바이트 복사를 거부한다
        let encoder = try activeBlitEncoder()
        encoder.copy(
            from: source.buffer, sourceOffset: sourceOffset,
            to: destination.buffer, destinationOffset: destinationOffset,
            size: size
        )
    }

    // MARK: - 디버그 마커

    /// 디버그 그룹·마커를 받을 대상 — **사용자 패스**가 열려 있으면 그 인코더, 아니면 커맨드 버퍼.
    ///
    /// 여기서 blit 인코더를 빼는 것이 핵심이다. blit은 `writeBuffer`·복사가 **속으로** 여닫는
    /// 내부 인코더라, 사용자가 보기엔 "패스 밖"이다. 그걸 스코프로 삼으면 프레임 단위로 연 그룹이
    /// blit 인코더에서 닫히려 하고, Metal이 요구하는 "인코더마다 짝 맞추기"가 깨진다
    /// (실제로 `pushDebugGroup` → `writeBuffer` → `popDebugGroup` 순서가 그렇게 어긋났다).
    ///
    /// 그래서 스코프는 두 층이다 — 패스 안 구간(렌더/컴퓨트 인코더)과 프레임 구간(커맨드 버퍼).
    private var debugScope: MTLCommandEncoder? {
        renderEncoder ?? computeEncoder
    }

    private func pushDebugGroup(_ command: WGPUPushDebugGroupCommand) throws {
        let label = command.groupLabel
        if let encoder = debugScope {
            encoder.pushDebugGroup(label)
            encoderDebugDepth += 1
        } else {
            // 프레임 구간 — 아직 커맨드 버퍼가 없으면 만든다. 그래야 뒤따르는 pop과 짝이 맞는다.
            try activeCommandBuffer().pushDebugGroup(label)
            bufferDebugDepth += 1
        }
    }

    /// 짝이 맞지 않는 `pop`은 **Metal이 단언으로 프로세스를 죽인다.** 그래서 깊이를 세어
    /// 여기서 막는다 — 명세도 이 경우를 오류로 정하므로 동작이 같고, 앱은 살아남는다.
    private func popDebugGroup() {
        if let encoder = debugScope {
            guard encoderDebugDepth > 0 else {
                record(.validation("popDebugGroup: 짝이 맞는 pushDebugGroup이 없다 (패스 안)"))
                return
            }
            encoderDebugDepth -= 1
            encoder.popDebugGroup()
        } else if let commandBuffer {
            guard bufferDebugDepth > 0 else {
                record(.validation("popDebugGroup: 짝이 맞는 pushDebugGroup이 없다"))
                return
            }
            bufferDebugDepth -= 1
            commandBuffer.popDebugGroup()
        }
    }

    /// 인코더를 닫기 전에 열려 있는 디버그 그룹을 정리한다.
    ///
    /// Metal은 그룹이 열린 채 인코더가 끝나도 단언으로 죽는다. 명세는 "패스가 끝날 때
    /// 디버그 스택이 비어 있어야 한다"고 정하므로, **오류로 알리되 닫아 주고 계속 간다** —
    /// 여기서 프로세스가 죽으면 진단할 기회조차 없다.
    private func closeDanglingDebugGroups(on encoder: MTLCommandEncoder) {
        guard encoderDebugDepth > 0 else { return }
        record(.validation(
            "디버그 그룹 \(encoderDebugDepth)개가 열린 채로 패스가 끝났다 (popDebugGroup을 빠뜨렸다)"
        ))
        while encoderDebugDepth > 0 {
            encoder.popDebugGroup()
            encoderDebugDepth -= 1
        }
    }

    private func insertDebugMarker(_ command: WGPUInsertDebugMarkerCommand) throws {
        let label = command.markerLabel
        // 인코더에는 signpost(점 이벤트)가 있다. 커맨드 버퍼에는 그룹밖에 없어 여닫아 흉내 낸다.
        if let encoder = debugScope {
            encoder.insertDebugSignpost(label)
        } else {
            let buffer = try activeCommandBuffer()
            buffer.pushDebugGroup(label)
            buffer.popDebugGroup()
        }
    }

    /// `clearBuffer` — 버퍼의 한 구간을 0으로 채운다.
    ///
    /// `writeBuffer`로 0을 밀어 넣는 것과 결과는 같지만 **CPU에서 0 배열을 만들어 브리지로
    /// 실어 보내지 않는다.** 큰 스토리지 버퍼를 프레임마다 초기화하는 컴퓨트 경로에서 차이가 크다.
    private func clearBuffer(_ command: WGPUClearBufferCommand) throws {
        let object = try unmappedBuffer(command.buffer, path: command.fieldPath("buffer"))
        let offset = command.offset
        // 명세: size를 생략하면 버퍼 끝까지다.
        let size = command.size ?? max(0, object.size - offset)

        guard object.usage.contains(.copyDst) else {
            throw WGPUError.validation(
                "clearBuffer의 대상은 GPUBufferUsage.COPY_DST로 만들어야 한다",
                path: command.fieldPath("buffer")
            )
        }
        // 4의 배수 요구는 명세 규칙이다. Metal은 바이트 단위로도 채워 주므로 안 막으면
        // 브라우저에서만 거부되는 코드가 나온다.
        guard offset % 4 == 0, size % 4 == 0 else {
            throw WGPUError.validation(
                "clearBuffer의 offset·size는 4의 배수여야 한다 (받은 값 \(offset), \(size))",
                path: command.fieldPath("offset")
            )
        }
        guard offset >= 0, size >= 0, offset + size <= object.size else {
            throw WGPUError.validation(
                "clearBuffer 범위가 버퍼를 넘는다 — offset \(offset) + \(size)B > 크기 \(object.size)B",
                path: command.fieldPath("size")
            )
        }
        guard size > 0 else { return }

        let encoder = try activeBlitEncoder()
        encoder.fill(buffer: object.buffer, range: offset..<(offset + size), value: 0)
    }

    private func copyTextureToBuffer(_ command: WGPUCopyTextureToBufferCommand) throws {
        let source = command.source
        let destination = command.destination
        let texture = try registry.lookup(
            source.texture, as: WGPUTextureObject.self, kind: "GPUTexture",
            path: source.fieldPath("texture")
        )
        let buffer = try unmappedBuffer(destination.buffer, path: destination.fieldPath("buffer"))
        let size = command.copySize
        let format = texture.format
        let bytesPerRow = destination.bytesPerRow ?? format.bytesPerRow(width: size.width)
        try validateBlockAlignment(format: format, origin: source.origin, size: size,
                                   texture: texture, mipLevel: source.mipLevel,
                                   label: "copyTextureToBuffer")

        let encoder = try activeBlitEncoder()
        encoder.copy(
            from: texture.texture,
            sourceSlice: source.origin.z,
            sourceLevel: source.mipLevel,
            sourceOrigin: MTLOrigin(x: source.origin.x, y: source.origin.y, z: 0),
            sourceSize: MTLSize(width: size.width, height: size.height, depth: 1),
            to: buffer.buffer,
            destinationOffset: destination.offset,
            destinationBytesPerRow: bytesPerRow,
            // 한 슬라이스만 복사하므로 `rowsPerImage`는 쓰이지 않는다
            // (`docs/COMMAND-STREAM.md`의 알려진 차이).
            destinationBytesPerImage: bytesPerRow * format.blockRows(height: size.height)
        )
    }

    private func copyBufferToTexture(_ command: WGPUCopyBufferToTextureCommand) throws {
        let source = command.source
        let destination = command.destination
        let buffer = try unmappedBuffer(source.buffer, path: source.fieldPath("buffer"))
        let texture = try registry.lookup(
            destination.texture, as: WGPUTextureObject.self, kind: "GPUTexture",
            path: destination.fieldPath("texture")
        )
        let size = command.copySize
        let format = texture.format
        let bytesPerRow = source.bytesPerRow ?? format.bytesPerRow(width: size.width)
        try validateBlockAlignment(format: format, origin: destination.origin, size: size,
                                   texture: texture, mipLevel: destination.mipLevel,
                                   label: "copyBufferToTexture")

        let encoder = try activeBlitEncoder()
        encoder.copy(
            from: buffer.buffer,
            sourceOffset: source.offset,
            sourceBytesPerRow: bytesPerRow,
            sourceBytesPerImage: bytesPerRow * format.blockRows(height: size.height),
            sourceSize: MTLSize(width: size.width, height: size.height, depth: 1),
            to: texture.texture,
            destinationSlice: destination.origin.z,
            destinationLevel: destination.mipLevel,
            destinationOrigin: MTLOrigin(x: destination.origin.x, y: destination.origin.y, z: 0)
        )
    }

    private func copyTextureToTexture(_ command: WGPUCopyTextureToTextureCommand) throws {
        let source = try registry.lookup(
            command.source.texture, as: WGPUTextureObject.self, kind: "GPUTexture",
            path: command.source.fieldPath("texture")
        )
        let destination = try registry.lookup(
            command.destination.texture, as: WGPUTextureObject.self, kind: "GPUTexture",
            path: command.destination.fieldPath("texture")
        )
        let size = command.copySize
        let sourceOrigin = command.source.origin
        let destinationOrigin = command.destination.origin

        let encoder = try activeBlitEncoder()
        encoder.copy(
            from: source.texture,
            sourceSlice: sourceOrigin.z,
            sourceLevel: command.source.mipLevel,
            sourceOrigin: MTLOrigin(x: sourceOrigin.x, y: sourceOrigin.y, z: 0),
            sourceSize: MTLSize(width: size.width, height: size.height, depth: 1),
            to: destination.texture,
            destinationSlice: destinationOrigin.z,
            destinationLevel: command.destination.mipLevel,
            destinationOrigin: MTLOrigin(x: destinationOrigin.x, y: destinationOrigin.y, z: 0)
        )
    }
}
