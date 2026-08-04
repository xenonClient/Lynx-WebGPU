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

    func execute(_ commands: [WGPUValueReader]) -> [String: Any] {
        reset()

        for (index, command) in commands.enumerated() {
            do {
                try perform(command, at: index)
            } catch let error as WGPUError {
                record(WGPUError(
                    kind: error.kind,
                    message: error.message,
                    path: error.path ?? "commands[\(index)].\(command.optionalString("op") ?? "?")"
                ))
            } catch {
                record(.backend(error.localizedDescription, path: "commands[\(index)]"))
            }
        }

        finish()

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
            errorScopes.append((try command.requiredEnum("filter", WGPUErrorFilter.self), nil))
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

    private func finish() {
        endActiveEncoders()
        if let commandBuffer {
            for acquired in acquiredDrawables {
                acquired.drawable.present(with: commandBuffer)
            }
            // 완료 핸들러는 commit 전에만 붙일 수 있다 (Metal 단언).
            if !frameStagingBuffers.isEmpty {
                let buffers = frameStagingBuffers
                let pool = stagingPool
                commandBuffer.addCompletedHandler { _ in pool.recycle(buffers) }
            }
            // in-flight 회계 — 프레임 티커가 이 수를 보고 포화 시 틱을 건너뛴다.
            let presentedSurfaces = uniquePresentedSurfaces()
            if !presentedSurfaces.isEmpty {
                for surface in presentedSurfaces { surface.noteFrameCommitted() }
                commandBuffer.addCompletedHandler { _ in
                    for surface in presentedSurfaces { surface.noteFrameCompleted() }
                }
            }
            commandBuffer.commit()
            lastCommittedBuffer = commandBuffer
            // 드로어블 텍스처와 그 뷰는 **present할 때** 무효해진다 (명세의 "Expire the current
            // texture"가 정한 시점). 배치가 끝날 때마다 회수하면, `popErrorScope`·`mapAsync`처럼
            // 프레임 중간에 제출하는 API가 그 프레임의 스왑체인 핸들을 지워 버려 뒤이은
            // `beginRenderPass`가 "없는 핸들"로 깨진다.
            if !acquiredDrawables.isEmpty {
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

    private func perform(_ command: WGPUValueReader, at index: Int) throws {
        let op = try command.requiredString("op")
        switch op {
        // 리소스
        case "createBuffer": try createBuffer(command)
        case "writeBuffer": try writeBuffer(command)
        case "createTexture": try createTexture(command)
        case "writeTexture": try writeTexture(command)
        case "createTextureView": try createTextureView(command)
        case "createSampler": try createSampler(command)
        case "createShaderModule": try createShaderModule(command)
        case "createBindGroupLayout": try createBindGroupLayout(command)
        case "createPipelineLayout": try createPipelineLayout(command)
        case "createBindGroup": try createBindGroup(command)
        case "createQuerySet": try createQuerySet(command)
        case "createRenderBundle": try createRenderBundle(command)
        case "createRenderPipeline": try createRenderPipeline(command)
        case "createComputePipeline": try createComputePipeline(command)
        case "getBindGroupLayout": try getBindGroupLayout(command)
        case "destroy": registry.remove(try command.requiredHandle("id"))

        // 오류 스코프
        case "pushErrorScope": try pushErrorScope(command)
        case "popErrorScope": popErrorScope()

        // 캔버스
        case "configureCanvas": try configureCanvas(command)
        case "getCurrentTexture": try getCurrentTexture(command)

        // 렌더 패스
        case "beginRenderPass": try beginRenderPass(command)
        case "setPipeline": try setPipeline(command)
        case "setBindGroup": try setBindGroup(command)
        case "setVertexBuffer": try setVertexBuffer(command)
        case "setIndexBuffer": try setIndexBuffer(command)
        case "setViewport": try setViewport(command)
        case "setScissorRect": try setScissorRect(command)
        case "setBlendConstant": try setBlendConstant(command)
        // `truncatingIfNeeded`는 WebIDL의 `u32` 변환(modulo)과 같은 동작이다. 비-truncating
        // 이니셜라이저를 쓰면 `setStencilReference(-1)` 한 줄로 Swift 런타임이 트랩한다 —
        // "잘못된 인자로 프로세스를 죽이지 않는다"는 이 라이브러리의 계약(WGPUError.swift)에 어긋난다.
        case "setStencilReference": try requireRenderEncoder()
            .setStencilReferenceValue(UInt32(truncatingIfNeeded: command.int("reference", default: 0)))
        case "draw": try draw(command)
        case "drawIndexed": try drawIndexed(command)
        case "drawIndirect": try drawIndirect(command)
        case "drawIndexedIndirect": try drawIndexedIndirect(command)
        case "executeBundles": try executeBundles(command)
        case "beginOcclusionQuery": try beginOcclusionQuery(command)
        case "endOcclusionQuery": try endOcclusionQuery()

        // 컴퓨트 패스
        case "beginComputePass": try beginComputePass(command)
        case "dispatchWorkgroups": try dispatchWorkgroups(command)
        case "dispatchWorkgroupsIndirect": try dispatchWorkgroupsIndirect(command)

        case "endPass": endActiveEncoders()

        // 복사
        case "copyBufferToBuffer": try copyBufferToBuffer(command)
        case "copyTextureToBuffer": try copyTextureToBuffer(command)
        case "copyBufferToTexture": try copyBufferToTexture(command)
        case "copyTextureToTexture": try copyTextureToTexture(command)
        case "resolveQuerySet": try resolveQuerySet(command)

        default:
            throw WGPUError.unsupported("알 수 없는 명령 '\(op)'", path: "commands[\(index)].op")
        }
    }

    // MARK: - 리소스 생성

    private func createBuffer(_ command: WGPUValueReader) throws {
        let handle = try command.requiredHandle("id")
        let object = try WGPUBufferObject(device: device, descriptor: WGPUBufferDescriptor(from: command))
        registry.insert(object, at: handle)
    }

    private func writeBuffer(_ command: WGPUValueReader) throws {
        let target = try registry.lookup(
            try command.requiredHandle("buffer"), as: WGPUBufferObject.self, kind: "GPUBuffer"
        )
        let data = try command.requiredData("data")
        let offset = command.int("bufferOffset", default: 0)
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

    private func createTexture(_ command: WGPUValueReader) throws {
        let handle = try command.requiredHandle("id")
        let object = try WGPUTextureObject(device: device, descriptor: WGPUTextureDescriptor(from: command))
        registry.insert(object, at: handle)
    }

    private func writeTexture(_ command: WGPUValueReader) throws {
        let target = try registry.lookup(
            try command.requiredHandle("texture"), as: WGPUTextureObject.self, kind: "GPUTexture"
        )
        let data = try command.requiredData("data")
        let size = try command.requiredExtent("size")
        let bytesPerRow = command.int("bytesPerRow", default: size.width * target.format.bytesPerPixel)
        let rowsPerImage = command.int("rowsPerImage", default: size.height)
        guard size.width > 0, size.height > 0, size.depthOrArrayLayers > 0 else { return }   // no-op
        let bytesPerImage = bytesPerRow * max(rowsPerImage, size.height)
        let layers = max(size.depthOrArrayLayers, 1)
        let required = bytesPerImage * (layers - 1) + bytesPerRow * size.height
        guard data.count >= required else {
            throw WGPUError.validation("writeTexture 데이터가 부족하다 (\(data.count)B, 최소 \(required)B 필요)")
        }
        // 스테이징은 이미지 스트라이드 전체만큼 잡는다 — Metal 검증 레이어가 마지막 이미지도
        // bytesPerImage 범위로 계산하기 때문이다. 남는 꼬리는 텍스처로 복사되지 않는다.
        let staging = try makeStagingBuffer(data, minimumLength: bytesPerImage * layers)
        // writeBuffer와 같은 이유로 blit으로 큐에 순서를 태운다 — 앞선 렌더/복사와 직렬화된다.
        target.encodeWrite(
            from: staging,
            origin: try command.origin("origin"),
            size: size,
            mipLevel: command.int("mipLevel", default: 0),
            bytesPerRow: bytesPerRow,
            rowsPerImage: rowsPerImage,
            blit: try activeBlitEncoder()
        )
    }

    /// 풀에서 스테이징 버퍼를 받아 데이터를 채우고, 프레임 완료 시 회수 목록에 올린다.
    private func makeStagingBuffer(_ data: Data, minimumLength: Int = 0) throws -> MTLBuffer {
        let staging = try stagingPool.acquire(data, minimumLength: minimumLength)
        frameStagingBuffers.append(staging)
        return staging
    }

    private func createTextureView(_ command: WGPUValueReader) throws {
        let handle = try command.requiredHandle("id")
        let sourceHandle = try command.requiredHandle("texture")
        let source = try registry.lookup(sourceHandle, as: WGPUTextureObject.self, kind: "GPUTexture")
        let drawable = acquiredDrawables.first { $0.handle == sourceHandle }?.drawable
        let view = try WGPUTextureViewObject(
            source: source,
            descriptor: WGPUTextureViewDescriptor(from: command),
            drawable: drawable
        )
        registry.insert(view, at: handle)
        if source.isDrawable { frameScopedHandles.append(handle) }
    }

    private func createSampler(_ command: WGPUValueReader) throws {
        let handle = try command.requiredHandle("id")
        let object = try WGPUSamplerObject(device: device, descriptor: WGPUSamplerDescriptor(from: command))
        registry.insert(object, at: handle)
    }

    private func createShaderModule(_ command: WGPUValueReader) throws {
        let handle = try command.requiredHandle("id")
        let object = try WGPUShaderModuleObject(descriptor: WGPUShaderModuleDescriptor(from: command))
        registry.insert(object, at: handle)
    }

    private func createBindGroupLayout(_ command: WGPUValueReader) throws {
        let handle = try command.requiredHandle("id")
        let descriptor = try WGPUBindGroupLayoutDescriptor(from: command)
        registry.insert(WGPUBindGroupLayoutObject(entries: descriptor.entries), at: handle)
    }

    private func createPipelineLayout(_ command: WGPUValueReader) throws {
        let handle = try command.requiredHandle("id")
        let descriptor = try WGPUPipelineLayoutDescriptor(from: command)
        let groups = try descriptor.bindGroupLayouts.map {
            try registry.lookup($0, as: WGPUBindGroupLayoutObject.self, kind: "GPUBindGroupLayout")
        }
        registry.insert(try WGPUPipelineLayoutObject(groups: groups), at: handle)
    }

    private func createBindGroup(_ command: WGPUValueReader) throws {
        let handle = try command.requiredHandle("id")
        let descriptor = try WGPUBindGroupDescriptor(from: command)
        let layout = try registry.lookup(
            descriptor.layout, as: WGPUBindGroupLayoutObject.self, kind: "GPUBindGroupLayout"
        )
        registry.insert(
            try WGPUBindGroupObject(layout: layout, descriptor: descriptor, registry: registry), at: handle
        )
    }

    private func createQuerySet(_ command: WGPUValueReader) throws {
        let handle = try command.requiredHandle("id")
        let object = try WGPUQuerySetObject(device: device, descriptor: WGPUQuerySetDescriptor(from: command))
        registry.insert(object, at: handle)
    }

    /// `bundleEncoder.finish()` — JS가 모아 둔 명령 목록을 번들 객체로 등록한다.
    ///
    /// 번들 인코더 자체는 네이티브에 없다. JS가 명령을 배열에 모으고 `finish()`에서 한 번에
    /// 내려보내므로, 인코더의 수명을 양쪽에서 맞출 이유가 없다.
    private func createRenderBundle(_ command: WGPUValueReader) throws {
        let handle = try command.requiredHandle("id")
        let bundle = try WGPURenderBundleObject(
            commands: try command.requiredObjects("commands"),
            descriptor: try WGPURenderBundleDescriptor(from: command)
        )
        registry.insert(bundle, at: handle)
    }

    private func createRenderPipeline(_ command: WGPUValueReader) throws {
        let handle = try command.requiredHandle("id")
        let descriptor = try WGPURenderPipelineDescriptor(from: command)
        let vertexModule = try registry.lookup(
            descriptor.vertex.module, as: WGPUShaderModuleObject.self, kind: "GPUShaderModule"
        )
        let fragmentModule = try descriptor.fragment.map {
            try registry.lookup($0.module, as: WGPUShaderModuleObject.self, kind: "GPUShaderModule")
        }

        var stages: [(module: WGPUShaderModuleObject, entryPoints: [String])] = []
        if let fragmentModule, fragmentModule === vertexModule, let fragment = descriptor.fragment {
            stages = [(vertexModule, [descriptor.vertex.entryPoint, fragment.entryPoint])]
        } else {
            stages = [(vertexModule, [descriptor.vertex.entryPoint])]
            if let fragmentModule, let fragment = descriptor.fragment {
                stages.append((fragmentModule, [fragment.entryPoint]))
            }
        }
        let layout = try WGPUPipelineLayoutResolver.resolve(descriptor.layout, stages: stages, registry: registry)
        let pipeline = try WGPURenderPipelineObject(
            device: device, descriptor: descriptor, layout: layout,
            vertexModule: vertexModule, fragmentModule: fragmentModule
        )
        registry.insert(pipeline, at: handle)
    }

    private func createComputePipeline(_ command: WGPUValueReader) throws {
        let handle = try command.requiredHandle("id")
        let descriptor = try WGPUComputePipelineDescriptor(from: command)
        let module = try registry.lookup(
            descriptor.module, as: WGPUShaderModuleObject.self, kind: "GPUShaderModule"
        )
        let layout = try WGPUPipelineLayoutResolver.resolve(
            descriptor.layout, stages: [(module, [descriptor.entryPoint])], registry: registry
        )
        registry.insert(
            try WGPUComputePipelineObject(device: device, descriptor: descriptor, layout: layout, module: module),
            at: handle
        )
    }

    /// `pipeline.getBindGroupLayout(index)` — `layout: "auto"`로 유도된 레이아웃을 핸들로 꺼낸다.
    private func getBindGroupLayout(_ command: WGPUValueReader) throws {
        let handle = try command.requiredHandle("id")
        let pipelineHandle = try command.requiredHandle("pipeline")
        let index = try command.requiredInt("index")

        let layout: WGPUPipelineLayoutObject
        if let render = try? registry.lookup(pipelineHandle, as: WGPURenderPipelineObject.self, kind: "x") {
            layout = render.layout
        } else {
            layout = try registry.lookup(
                pipelineHandle, as: WGPUComputePipelineObject.self, kind: "GPUPipeline"
            ).layout
        }
        guard let group = layout.group(at: index) else {
            throw WGPUError.validation("파이프라인에 바인드 그룹 \(index)이(가) 없다")
        }
        registry.insert(group, at: handle)
    }

    // MARK: - 캔버스

    private func configureCanvas(_ command: WGPUValueReader) throws {
        let configuration = try WGPUCanvasConfiguration(from: command)
        guard let surface = surfaceProvider(configuration.canvasId) else {
            throw WGPUError.validation(
                "캔버스 '\(configuration.canvasId)'이(가) 등록되지 않았다 "
                    + "(<webgpu-canvas canvas-id=\"…\">가 화면에 붙어 있는지 확인)"
            )
        }
        try surface.configure(configuration, device: device)
        touchedCanvases[configuration.canvasId] = surface
    }

    private func getCurrentTexture(_ command: WGPUValueReader) throws {
        let handle = try command.requiredHandle("id")
        let canvasId = try command.requiredString("canvas")
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

    private func beginRenderPass(_ command: WGPUValueReader) throws {
        endActiveEncoders()
        let descriptor = try WGPURenderPassDescriptor(from: command)
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

    private func beginOcclusionQuery(_ command: WGPUValueReader) throws {
        let encoder = try requireRenderEncoder()
        guard let querySet = passOcclusionQuerySet else {
            throw WGPUError.validation(
                "beginOcclusionQuery를 쓰려면 beginRenderPass에 occlusionQuerySet을 줘야 한다"
            )
        }
        guard openOcclusionQuery == nil else {
            throw WGPUError.validation("occlusion 쿼리는 중첩할 수 없다 (앞의 것을 endOcclusionQuery로 닫을 것)")
        }
        let index = try command.requiredInt("queryIndex")
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
    private func resolveQuerySet(_ command: WGPUValueReader) throws {
        let querySet = try registry.lookup(
            try command.requiredHandle("querySet"), as: WGPUQuerySetObject.self, kind: "GPUQuerySet"
        )
        let destination = try registry.lookup(
            try command.requiredHandle("destination"), as: WGPUBufferObject.self, kind: "GPUBuffer"
        )
        let first = command.int("firstQuery", default: 0)
        let count = command.int("queryCount", default: querySet.count - first)
        let offset = command.int("destinationOffset", default: 0)
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

    private func executeBundles(_ command: WGPUValueReader) throws {
        _ = try requireRenderEncoder()
        guard let formats = passFormats else {
            throw WGPUError.validation("executeBundles는 렌더 패스 안에서만 쓸 수 있다")
        }
        let bundles = try command.handles("bundles").map {
            try registry.lookup($0, as: WGPURenderBundleObject.self, kind: "GPURenderBundle")
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

    private func setPipeline(_ command: WGPUValueReader) throws {
        let handle = try command.requiredHandle("pipeline")
        if let encoder = renderEncoder {
            let pipeline = try registry.lookup(
                handle, as: WGPURenderPipelineObject.self, kind: "GPURenderPipeline"
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
                handle, as: WGPUComputePipelineObject.self, kind: "GPUComputePipeline"
            )
            encoder.setComputePipelineState(pipeline.state)
            currentComputePipeline = pipeline
        } else {
            throw WGPUError.validation("setPipeline은 패스 안에서만 쓸 수 있다")
        }
        // 파이프라인이 바뀌면 레이아웃이 달라질 수 있으므로 바인드 그룹을 다시 적용한다.
        dirtyGroups = Set(boundGroups.keys)
    }

    private func setBindGroup(_ command: WGPUValueReader) throws {
        let index = try command.requiredInt("index")
        let group = try registry.lookup(
            try command.requiredHandle("bindGroup"), as: WGPUBindGroupObject.self, kind: "GPUBindGroup"
        )
        let offsets = (try? command.integers("dynamicOffsets")) ?? []
        boundGroups[index] = (group, offsets)
        dirtyGroups.insert(index)
    }

    /// 드로우·디스패치 직전에 파이프라인이 요구하는 상태를 전부 확인하고 인코더에 올린다.
    ///
    /// 바인드 그룹과 정점 버퍼를 한자리에서 다루는 이유는, 둘 다 **번들 경계에서 무효화되는
    /// 상태**라 검사 시점이 같아야 하기 때문이다. 새 드로우 op을 추가할 때 이 함수 하나만
    /// 부르면 격리 계약이 자동으로 따라온다.
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

    private func setVertexBuffer(_ command: WGPUValueReader) throws {
        _ = try requireRenderEncoder()
        let slot = try command.requiredInt("slot")
        guard slot >= 0, slot < WGSLMetalLimits.maxVertexBufferSlots else {
            throw WGPUError.validation("정점 버퍼 슬롯은 0~\(WGSLMetalLimits.maxVertexBufferSlots - 1) 범위다")
        }
        let buffer = try registry.lookup(
            try command.requiredHandle("buffer"), as: WGPUBufferObject.self, kind: "GPUBuffer"
        )
        let offset = command.int("offset", default: 0)
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

    private func setIndexBuffer(_ command: WGPUValueReader) throws {
        _ = try requireRenderEncoder()
        let buffer = try registry.lookup(
            try command.requiredHandle("buffer"), as: WGPUBufferObject.self, kind: "GPUBuffer"
        )
        let format = try command.requiredEnum("format", WGPUIndexFormat.self)
        indexBinding = (
            buffer.buffer,
            command.int("offset", default: 0),
            WGPUMetalMapping.indexType(format),
            format == .uint16 ? 2 : 4
        )
    }

    private func setViewport(_ command: WGPUValueReader) throws {
        let encoder = try requireRenderEncoder()
        encoder.setViewport(MTLViewport(
            originX: command.double("x", default: 0),
            originY: command.double("y", default: 0),
            width: try command.requiredDouble("width"),
            height: try command.requiredDouble("height"),
            znear: command.double("minDepth", default: 0),
            zfar: command.double("maxDepth", default: 1)
        ))
    }

    private func setScissorRect(_ command: WGPUValueReader) throws {
        let encoder = try requireRenderEncoder()
        encoder.setScissorRect(MTLScissorRect(
            x: command.int("x", default: 0),
            y: command.int("y", default: 0),
            width: try command.requiredInt("width"),
            height: try command.requiredInt("height")
        ))
    }

    private func setBlendConstant(_ command: WGPUValueReader) throws {
        let encoder = try requireRenderEncoder()
        let color = try command.color("color", default: .transparent)
        encoder.setBlendColor(
            red: Float(color.red), green: Float(color.green), blue: Float(color.blue), alpha: Float(color.alpha)
        )
    }

    private func draw(_ command: WGPUValueReader) throws {
        let encoder = try requireRenderEncoder()
        try applyDrawState()
        guard let pipeline = currentRenderPipeline else {
            throw WGPUError.validation("draw 전에 setPipeline이 필요하다")
        }
        encoder.drawPrimitives(
            type: pipeline.primitiveType,
            vertexStart: command.int("firstVertex", default: 0),
            vertexCount: try command.requiredInt("vertexCount"),
            instanceCount: command.int("instanceCount", default: 1),
            baseInstance: command.int("firstInstance", default: 0)
        )
    }

    private func drawIndexed(_ command: WGPUValueReader) throws {
        let encoder = try requireRenderEncoder()
        try applyDrawState()
        guard let pipeline = currentRenderPipeline else {
            throw WGPUError.validation("drawIndexed 전에 setPipeline이 필요하다")
        }
        guard let indexBinding else {
            throw WGPUError.validation("drawIndexed 전에 setIndexBuffer가 필요하다")
        }
        let firstIndex = command.int("firstIndex", default: 0)
        encoder.drawIndexedPrimitives(
            type: pipeline.primitiveType,
            indexCount: try command.requiredInt("indexCount"),
            indexType: indexBinding.type,
            indexBuffer: indexBinding.buffer,
            indexBufferOffset: indexBinding.offset + firstIndex * indexBinding.stride,
            instanceCount: command.int("instanceCount", default: 1),
            baseVertex: command.int("baseVertex", default: 0),
            baseInstance: command.int("firstInstance", default: 0)
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
        _ command: WGPUValueReader,
        argumentSize: Int
    ) throws -> (buffer: MTLBuffer, offset: Int) {
        let object = try registry.lookup(
            try command.requiredHandle("indirectBuffer"), as: WGPUBufferObject.self, kind: "GPUBuffer"
        )
        let offset = command.int("indirectOffset", default: 0)
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

    private func drawIndirect(_ command: WGPUValueReader) throws {
        let encoder = try requireRenderEncoder()
        // 인자 검증을 `applyDrawState()`보다 **먼저** 한다 — 거부할 명령이 인코더 상태를
        // 이미 바꿔 놓는 일이 없어야 한다 (오류는 프레임을 죽이지 않고 누적되므로 더 그렇다).
        // vertexCount, instanceCount, firstVertex, firstInstance — u32 4개.
        let arguments = try indirectArguments(command, argumentSize: 16)
        try applyDrawState()
        guard let pipeline = currentRenderPipeline else {
            throw WGPUError.validation("drawIndirect 전에 setPipeline이 필요하다")
        }
        encoder.drawPrimitives(
            type: pipeline.primitiveType,
            indirectBuffer: arguments.buffer,
            indirectBufferOffset: arguments.offset
        )
    }

    private func drawIndexedIndirect(_ command: WGPUValueReader) throws {
        let encoder = try requireRenderEncoder()
        // indexCount, instanceCount, firstIndex, baseVertex(i32), firstInstance — 5칸.
        let arguments = try indirectArguments(command, argumentSize: 20)
        try applyDrawState()
        guard let pipeline = currentRenderPipeline else {
            throw WGPUError.validation("drawIndexedIndirect 전에 setPipeline이 필요하다")
        }
        guard let indexBinding else {
            throw WGPUError.validation("drawIndexedIndirect 전에 setIndexBuffer가 필요하다")
        }
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

    private func dispatchWorkgroupsIndirect(_ command: WGPUValueReader) throws {
        let encoder = try requireComputeEncoder()
        // x, y, z — u32 3개.
        let arguments = try indirectArguments(command, argumentSize: 12)
        try applyDrawState()
        guard let pipeline = currentComputePipeline else {
            throw WGPUError.validation("dispatchWorkgroupsIndirect 전에 setPipeline이 필요하다")
        }
        encoder.dispatchThreadgroups(
            indirectBuffer: arguments.buffer,
            indirectBufferOffset: arguments.offset,
            threadsPerThreadgroup: pipeline.threadsPerThreadgroup
        )
    }

    // MARK: - 컴퓨트 패스

    private func beginComputePass(_ command: WGPUValueReader) throws {
        endActiveEncoders()
        let descriptor = try WGPUComputePassDescriptor(from: command)
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

    private func dispatchWorkgroups(_ command: WGPUValueReader) throws {
        let encoder = try requireComputeEncoder()
        try applyDrawState()
        guard let pipeline = currentComputePipeline else {
            throw WGPUError.validation("dispatchWorkgroups 전에 setPipeline이 필요하다")
        }
        encoder.dispatchThreadgroups(
            MTLSize(
                width: max(command.int("x", default: 1), 1),
                height: max(command.int("y", default: 1), 1),
                depth: max(command.int("z", default: 1), 1)
            ),
            threadsPerThreadgroup: pipeline.threadsPerThreadgroup
        )
    }

    // MARK: - 복사

    private func copyBufferToBuffer(_ command: WGPUValueReader) throws {
        let source = try registry.lookup(
            try command.requiredHandle("source"), as: WGPUBufferObject.self, kind: "GPUBuffer"
        )
        let destination = try registry.lookup(
            try command.requiredHandle("destination"), as: WGPUBufferObject.self, kind: "GPUBuffer"
        )
        let size = try command.requiredInt("size")
        let encoder = try activeBlitEncoder()
        encoder.copy(
            from: source.buffer, sourceOffset: command.int("sourceOffset", default: 0),
            to: destination.buffer, destinationOffset: command.int("destinationOffset", default: 0),
            size: size
        )
    }

    private func copyTextureToBuffer(_ command: WGPUValueReader) throws {
        let sourceReader = try command.requiredObject("source")
        let destinationReader = try command.requiredObject("destination")
        let texture = try registry.lookup(
            try sourceReader.requiredHandle("texture"), as: WGPUTextureObject.self, kind: "GPUTexture"
        )
        let buffer = try registry.lookup(
            try destinationReader.requiredHandle("buffer"), as: WGPUBufferObject.self, kind: "GPUBuffer"
        )
        let size = try command.requiredExtent("copySize")
        let bytesPerRow = destinationReader.int("bytesPerRow", default: size.width * texture.format.bytesPerPixel)
        let origin = try sourceReader.origin("origin")

        let encoder = try activeBlitEncoder()
        encoder.copy(
            from: texture.texture,
            sourceSlice: origin.z,
            sourceLevel: sourceReader.int("mipLevel", default: 0),
            sourceOrigin: MTLOrigin(x: origin.x, y: origin.y, z: 0),
            sourceSize: MTLSize(width: size.width, height: size.height, depth: 1),
            to: buffer.buffer,
            destinationOffset: destinationReader.int("offset", default: 0),
            destinationBytesPerRow: bytesPerRow,
            destinationBytesPerImage: bytesPerRow * size.height
        )
    }

    private func copyBufferToTexture(_ command: WGPUValueReader) throws {
        let sourceReader = try command.requiredObject("source")
        let destinationReader = try command.requiredObject("destination")
        let buffer = try registry.lookup(
            try sourceReader.requiredHandle("buffer"), as: WGPUBufferObject.self, kind: "GPUBuffer"
        )
        let texture = try registry.lookup(
            try destinationReader.requiredHandle("texture"), as: WGPUTextureObject.self, kind: "GPUTexture"
        )
        let size = try command.requiredExtent("copySize")
        let bytesPerRow = sourceReader.int("bytesPerRow", default: size.width * texture.format.bytesPerPixel)
        let origin = try destinationReader.origin("origin")

        let encoder = try activeBlitEncoder()
        encoder.copy(
            from: buffer.buffer,
            sourceOffset: sourceReader.int("offset", default: 0),
            sourceBytesPerRow: bytesPerRow,
            sourceBytesPerImage: bytesPerRow * size.height,
            sourceSize: MTLSize(width: size.width, height: size.height, depth: 1),
            to: texture.texture,
            destinationSlice: origin.z,
            destinationLevel: destinationReader.int("mipLevel", default: 0),
            destinationOrigin: MTLOrigin(x: origin.x, y: origin.y, z: 0)
        )
    }

    private func copyTextureToTexture(_ command: WGPUValueReader) throws {
        let sourceReader = try command.requiredObject("source")
        let destinationReader = try command.requiredObject("destination")
        let source = try registry.lookup(
            try sourceReader.requiredHandle("texture"), as: WGPUTextureObject.self, kind: "GPUTexture"
        )
        let destination = try registry.lookup(
            try destinationReader.requiredHandle("texture"), as: WGPUTextureObject.self, kind: "GPUTexture"
        )
        let size = try command.requiredExtent("copySize")
        let sourceOrigin = try sourceReader.origin("origin")
        let destinationOrigin = try destinationReader.origin("origin")

        let encoder = try activeBlitEncoder()
        encoder.copy(
            from: source.texture,
            sourceSlice: sourceOrigin.z,
            sourceLevel: sourceReader.int("mipLevel", default: 0),
            sourceOrigin: MTLOrigin(x: sourceOrigin.x, y: sourceOrigin.y, z: 0),
            sourceSize: MTLSize(width: size.width, height: size.height, depth: 1),
            to: destination.texture,
            destinationSlice: destinationOrigin.z,
            destinationLevel: destinationReader.int("mipLevel", default: 0),
            destinationOrigin: MTLOrigin(x: destinationOrigin.x, y: destinationOrigin.y, z: 0)
        )
    }
}
