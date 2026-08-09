import Foundation
import CoreGraphics
import QuartzCore

/// 커맨드 스트림 오케스트레이션 엔진 — `WebGPURuntime`의 **백엔드 무관 구현**.
///
/// JS가 보낸 한 프레임 분량의 명령 배열을 해석해 `WGPUBackend` 동사 호출로 옮긴다.
/// 백엔드(Metal 직접·Dawn 등)가 무엇이든 **와이어 계약과 명세 검증은 여기서 한 번만** 이행된다:
///
/// - 배치 루프·디코딩·디스패치 (exhaustive switch — op 추가 누락은 컴파일러가 잡는다)
/// - 오류 수집(프레임을 죽이지 않는다)·오류 스코프·지연 GPU 오류·응답 조립(`WGPUBatchResult`)
/// - 프레임 수명: 드로어블 핸들의 present 시점 만료, in-flight 회계(`WGPUFrameCoordinator`)
/// - 매핑 게이트(mapAsync 중 버퍼의 큐 사용 금지), 범위·정렬·usage·완전성 검증
/// - 드로우 상태 그림자(바인드 그룹·정점 버퍼) — 번들 경계의 바인딩 무효화가 여기 산다
/// - 렌더 번들 record/replay (네이티브 번들이 없는 백엔드용)
/// - 직렬화: `execute`(JS 스레드)·`processEvents`(메인 틱)·`readBuffer` 등록이 전부
///   하나의 실행 락 아래 돈다 — 펌프 동시성 계약(`docs/COMMAND-STREAM.md` §5-1)의 이행 지점.
///
/// 백엔드에는 이미 해석·검증이 끝난 값으로 자기 GPU API를 부르는 일만 남는다
/// (`WGPUBackend` 문서). Dawn 시제품에서 오케스트레이션 중복이 실제 결함(펌프 경쟁,
/// present 순서, 스코프 드레인)으로 이어진 뒤, 그 중복을 없애려고 이 층을 올렸다.
public final class WGPUBackendEngine<B: WGPUBackend>: WebGPURuntime {
    public let backend: B
    /// in-flight 프레임 회계 — present 시점·포화 판단. 백엔드와 무관한 정책이라 엔진이 몬다.
    public let frameCoordinator: WGPUFrameCoordinator

    private let registry = WGPUObjectRegistry()
    /// 실행 직렬화 락. **재귀 락**이다 — `readBuffer`의 완료가 등록 도중 동기로 도착하는
    /// 백엔드(이미 끝난 작업)에서 완료 래퍼가 같은 락을 다시 잡기 때문이다.
    private let executionLock = NSRecursiveLock()
    private let canvasLock = NSLock()

    // MARK: 배치 상태 — 한 배치의 수명 동안만 유효하다

    private enum PassState { case render, compute }
    private var passState: PassState?
    /// 현재 인코더 / 프레임 구간에서 연 디버그 그룹 수 — 짝이 안 맞으면 백엔드가 단언으로 죽는다.
    private var encoderDebugDepth = 0
    private var bufferDebugDepth = 0
    private var currentRenderPipeline: WGPUEngineRenderPipeline<B>?
    private var currentComputePipeline: WGPUEngineComputePipeline<B>?
    private var boundGroups: [Int: (group: WGPUEngineBindGroup<B>, offsets: [Int])] = [:]
    private var dirtyGroups: Set<Int> = []
    private var indexBinding: (buffer: WGPUEngineBuffer<B>, offset: Int, format: WGPUIndexFormat, stride: Int)?
    /// 슬롯별 정점 버퍼 바인딩. **백엔드에 직행하지 않고 여기 모아 두었다가** 드로우 직전에
    /// 내린다. 그래야 `resetPassBindings()`가 번들 경계에서 바인딩을 실제로 무효화할 수 있다 —
    /// 백엔드 인코더에는 "바인딩 해제"가 없으므로, 무효화는 그림자 상태로만 표현된다.
    private var vertexBindings: [Int: (buffer: WGPUEngineBuffer<B>, offset: Int)] = [:]
    private var dirtyVertexSlots: Set<Int> = []
    /// 지금 렌더 패스의 어태치먼트 모양 — 렌더 번들이 이 패스에서 유효한지 볼 때 쓴다.
    private var passFormats: (color: [WGPUTextureFormat], depthStencil: WGPUTextureFormat?, sampleCount: Int)?
    /// 지금 렌더 패스가 깊이/스텐실을 **쓰지 않겠다**고 선언했는가 (`depthReadOnly`/`stencilReadOnly`).
    private var passDepthReadOnly = false
    private var passStencilReadOnly = false
    private var passOcclusionQuerySet: WGPUEngineQuerySet<B>?
    /// 열려 있는 occlusion 쿼리 인덱스 — 중첩·미종료를 잡는다.
    private var openOcclusionQuery: Int?
    /// 지금 패스에서 이미 쓴 occlusion 쿼리 인덱스 — 명세는 같은 패스에서 재사용을 금지한다.
    private var usedOcclusionQueries: Set<Int> = []
    /// 이번 프레임에 드로어블을 내준 (핸들, 캔버스 id) — present 시 만료·회계 대상이다.
    private var acquiredFrames: [(handle: WGPUHandle, canvas: String)] = []
    /// **이번 배치에서** 드로어블을 내준 캔버스. 한 배치 안의 반복 획득(같은 프레임)과
    /// 배치를 건너뛴 반복 획득(새 프레임)을 가르는 데 쓴다 — `getCurrentTexture` 참고.
    private var acquiredThisBatch: Set<String> = []
    /// 프레임이 끝나면 무효해지는 핸들 (드로어블 텍스처와 그 뷰).
    private var frameScopedHandles: [WGPUHandle] = []
    private var touchedCanvases: [String: B.Surface] = [:]
    private var errors: [WGPUError] = []

    /// 앞선 배치의 GPU 실행이 실패했다는 보고 — 완료 콜백(임의 스레드)이 채우고
    /// 다음 배치가 비운다 (`WGPUDeferredErrorQueue` 문서).
    private let gpuFailures = WGPUDeferredErrorQueue()
    /// 열려 있는 오류 스코프 — 규칙은 전부 `WGPUErrorScopeStack` 문서에 있다.
    private var errorScopes = WGPUErrorScopeStack()
    /// 이번 배치에서 pop된 스코프의 결과 (pop 순서 — JS의 Promise 순서와 1:1로 맞춘다).
    private var poppedScopes: [WGPUPoppedErrorScope] = []

    /// 엔진이 명세 검사를 수행하는가 — 검증기를 통째로 가진 백엔드(Dawn)에서는 거짓이 되어,
    /// 위층에는 브리징과 **최소한의 예외처리**(핸들 조회·와이어 매핑 상태·패스 상태 가드·
    /// CPU 경로 보호)만 남는다 (`WGPUBackendCapabilities.validatesNatively`).
    private let specValidation: Bool

    public init(backend: B, frameCoordinator: WGPUFrameCoordinator = WGPUFrameCoordinator()) {
        self.backend = backend
        self.frameCoordinator = frameCoordinator
        self.specValidation = !backend.capabilities.validatesNatively
    }

    // MARK: - WebGPURuntime: 실행

    public func execute(_ payload: [String: Any]) -> [String: Any] {
        executionLock.lock()
        defer { executionLock.unlock() }

        let reader = WGPUValueReader(payload)
        let commands: [WGPUValueReader]
        do {
            commands = try reader.requiredObjects("commands")
        } catch let error as WGPUError {
            return WGPUBatchResult.failure([error])
        } catch {
            return WGPUBatchResult.failure([WGPUError.validation("\(error)")])
        }
        return run(commands, present: reader.bool("present", default: true))
    }

    private func run(_ commands: [WGPUValueReader], present: Bool) -> [String: Any] {
        resetBatchState()

        // 앞선 배치의 GPU 실행 실패를 먼저 흘려보낸다 — 오류 스코프가 열려 있으면 그쪽이 잡는다.
        for failure in gpuFailures.drain() { record(failure) }
        backend.beginBatch()

        for (index, command) in commands.enumerated() {
            do {
                try perform(command)
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
        finish(WGPUFrameBoundary(requestedPresent: present, commandCount: commands.count))

        // 배치 단위 백엔드 진단 (Dawn의 디바이스 스코프 등) — 제출 후에 회수해야
        // 이 배치의 GPU 검증 오류까지 담긴다.
        for diagnostic in backend.collectBatchDiagnostics() { record(diagnostic) }

        return WGPUBatchResult(
            commandCount: commands.count,
            liveObjectCount: registry.count,
            errors: errors,
            canvases: touchedCanvases.mapValues { surface in
                let report = backend.surfaceReport(surface)
                return WGPUCanvasReport(width: report.width, height: report.height)
            },
            poppedScopes: poppedScopes
        ).payload
    }

    /// 오류 하나를 가장 안쪽의 맞는 스코프에 넣거나, 없으면 배치 결과로 내보낸다.
    private func record(_ error: WGPUError) {
        if errorScopes.capture(error) { return }
        errors.append(error)
    }

    private func resetBatchState() {
        passState = nil
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
        acquiredThisBatch.removeAll()
        errors.removeAll()
        poppedScopes.removeAll()
        // `errorScopes`는 일부러 비우지 않는다 — 디바이스 상태이므로 배치를 넘어 이어진다.
        // `acquiredFrames`·`frameScopedHandles`도 마찬가지다 — 프레임의 경계는 배치가
        // 아니라 **present**이고, 한 프레임이 배치 여러 개로 쪼개질 수 있다 (아래 finish() 참고).
    }

    /// 배치 하나를 마무리한다.
    ///
    /// `present`가 false면 프레임 **중간**의 내부 제출이다 — shim의 `popErrorScope`·`mapAsync`가
    /// 결과를 받으려고 미리 흘려보낸 배치. GPU 작업은 제출하되(리드백이 완료를 기다린다),
    /// 드로어블 present와 프레임 스코프 핸들 만료는 **뒤따라올 진짜 프레임 제출로 미룬다.**
    /// 안 미루면: 그 배치가 `writeBuffer` 하나로라도 GPU 작업을 만든 경우, 방금 획득한
    /// 드로어블이 그리기도 전에 present되고 핸들이 만료되어, 이어지는 `beginRenderPass`가
    /// "GPUTextureView가 존재하지 않는다"로 통째로 거부된다 — Three.js의 지연 파이프라인
    /// 생성(pop 즉시 flush)이 정확히 이 경로를 밟았다.
    private func finish(_ boundary: WGPUFrameBoundary) {
        let present = boundary.presents
        closePass()
        // 프레임 구간에 연 그룹도 제출 전에 닫는다 (인코더와 같은 이유 — Metal이 단언으로 죽는다).
        // 네이티브 검증 백엔드는 Finish에서 스스로 오류를 낸다 — 닫아 주지 않는다.
        if specValidation, backend.hasPendingWork, bufferDebugDepth > 0 {
            record(.validation(
                "디버그 그룹 \(bufferDebugDepth)개가 열린 채로 제출됐다 (popDebugGroup을 빠뜨렸다)"
            ))
            backend.popFrameDebugGroups(count: bufferDebugDepth)
        }
        bufferDebugDepth = 0
        // 틱의 마무리 배치는 명령이 없어도 **드로어블을 내보내야 한다** — 제출 거리가
        // 없다고 지나가면 화면이 멈춘 채 아무 말이 없다.
        //
        // 조건을 "명령이 비어 있을 때"로 좁힌 것이 중요하다. 명령은 있는데 GPU 작업이
        // 안 생긴 배치(드로어블만 얻고 끝난 경우 등)는 present하지 않는다 —
        // 그리지도 않은 드로어블을 내보내면 그 프레임이 빈 화면으로 나간다.
        if boundary.closesFrame, !backend.hasPendingWork, !acquiredFrames.isEmpty {
            backend.ensureSubmittableWork()
        }
        // 제출거리가 없으면 present할 것도 없다. 획득해 둔 드로어블은 **그대로 남긴다** —
        // 이 상태는 "프레임이 실패했다"가 아니라 "아직 진행 중이다"일 수 있고(Three.js의
        // 지연 파이프라인 생성이 그렇다), 여기서 놓으면 뒤따라올 `beginRenderPass`가
        // "없는 핸들"로 깨진다. 붙들린 드로어블은 다음 프레임의 획득이 거둔다
        // (`getCurrentTexture` 참고).
        guard backend.hasPendingWork else { return }

        // in-flight 회계 — 프레임 티커가 이 수를 보고 포화 시 틱을 건너뛴다.
        // present하지 않는 배치는 프레임이 아니므로 세지 않는다.
        let presentedCanvases = present ? uniquePresentedCanvases() : []
        let coordinator = frameCoordinator
        for canvas in presentedCanvases { coordinator.noteCommitted(canvas: canvas) }
        // 완료 콜백은 엔진이 아니라 값(큐·코디네이터)을 잡는다 — 엔진이 먼저 해제되어도 안전하다.
        let failures = gpuFailures
        backend.submit(present: present) { failure in
            if let failure { failures.report(failure) }
            for canvas in presentedCanvases { coordinator.noteCompleted(canvas: canvas) }
        }
        // 드로어블 텍스처와 그 뷰는 **present할 때** 무효해진다 (명세의 "Expire the current
        // texture"가 정한 시점). 배치가 끝날 때마다 회수하면 프레임 중간 제출이 그 프레임의
        // 스왑체인 핸들을 지워 버려 뒤이은 `beginRenderPass`가 "없는 핸들"로 깨진다.
        // (드로어블 자체는 `submit(present: true)`이 내보내며 이미 놓았다.)
        if present, !acquiredFrames.isEmpty { expireFrame() }
    }

    /// 프레임이 끝났다 — 드로어블 텍스처와 그 뷰의 핸들을 만료시킨다
    /// (명세 `GPUCanvasContext`의 "Expire the current texture").
    ///
    /// present한 프레임과 **제출거리 없이 끝난 프레임**이 같은 것을 해야 해서 뽑아 두었다.
    /// 한쪽만 고치면 다른 쪽에서 핸들이 영영 살아남는다.
    private func expireFrame() {
        for handle in frameScopedHandles { registry.remove(handle) }
        frameScopedHandles.removeAll()
        acquiredFrames.removeAll()
    }

    /// 이번 프레임에 드로어블을 내준 캔버스들 (중복 제거 — 한 표면에서 여러 번 얻어도 프레임은 하나다).
    private func uniquePresentedCanvases() -> [String] {
        var seen = Set<String>()
        var canvases: [String] = []
        for acquired in acquiredFrames where seen.insert(acquired.canvas).inserted {
            canvases.append(acquired.canvas)
        }
        return canvases
    }

    /// 열려 있는 패스를 닫는다 — occlusion 미종료·디버그 그룹 잔여를 오류로 알리되
    /// **닫아 주고 계속 간다.** 여기서 백엔드 단언으로 죽으면 진단할 기회조차 없다.
    private func closePass() {
        if passState == .render {
            // 명세는 패스를 닫을 때 열려 있는 occlusion 쿼리가 없기를 요구한다. Metal은 그냥
            // 값을 써 주므로 여기서 안 잡으면 **값까지 정상으로 보이고**, 브라우저에서만 프레임이
            // 통째로 날아간다. 패스는 이미 닫히는 중이라 throw 대신 기록한다.
            // (네이티브 검증 백엔드는 End 시점에 스스로 오류를 낸다.)
            if specValidation, let index = openOcclusionQuery {
                record(.validation(
                    "occlusion 쿼리 \(index)이(가) 열린 채로 렌더 패스가 끝났다 "
                        + "(endOcclusionQuery를 빠뜨렸다)"
                ))
            }
            openOcclusionQuery = nil
            usedOcclusionQueries.removeAll()
            passOcclusionQuerySet = nil
            passFormats = nil
            passDepthReadOnly = false
            passStencilReadOnly = false
        }
        // 디버그 그룹이 열린 채 인코더를 닫으면 Metal이 단언으로 죽는다 — 닫아 주고 오류로 알린다.
        // (네이티브 검증 백엔드는 닫지 않아도 End/Finish에서 오류로 처리한다 — 죽지 않는다.)
        if specValidation, passState != nil, encoderDebugDepth > 0 {
            record(.validation(
                "디버그 그룹 \(encoderDebugDepth)개가 열린 채로 패스가 끝났다 (popDebugGroup을 빠뜨렸다)"
            ))
            while encoderDebugDepth > 0 {
                backend.popDebugGroup(scope: .pass)
                encoderDebugDepth -= 1
            }
        }
        encoderDebugDepth = 0
        passState = nil
        backend.endPass()
    }

    // MARK: - 명령 분기

    /// 디코딩과 분기표는 `WGPUCommand`가 끝낸다 — 여기는 디코딩된 값을 검증·해석해 백엔드
    /// 동사로 옮기는 **exhaustive switch**만 남는다 (`default` 없음). op을 더할 때 케이스를
    /// 빠뜨리면 컴파일이 깨진다 — 백엔드 누락은 `WGPUBackend`의 프로토콜 요구가 잡는다.
    private func perform(_ command: WGPUValueReader) throws {
        try dispatch(WGPUCommand(from: command))
    }

    private func dispatch(_ command: WGPUCommand) throws {
        switch command {
        // 리소스
        case .createBuffer(let c): try createBuffer(c)
        case .writeBuffer(let c): try writeBuffer(c)
        case .unmapBuffer(let c): try unmapBuffer(c)
        case .createTexture(let c): try createTexture(c)
        case .writeTexture(let c): try writeTexture(c)
        case .copyExternalImageToTexture(let c): try copyExternalImageToTexture(c)
        case .createTextureView(let c): try createTextureView(c)
        case .createSampler(let c): try createSampler(c)
        case .createShaderModule(let c): try createShaderModule(c)
        case .createBindGroupLayout(let c): try createBindGroupLayout(c)
        case .createPipelineLayout(let c): try createPipelineLayout(c)
        case .createBindGroup(let c): try createBindGroup(c)
        case .createQuerySet(let c): try createQuerySet(c)
        case .createRenderBundle(let c): try createRenderBundle(c)
        case .createRenderPipeline(let c): try createRenderPipeline(c)
        case .createComputePipeline(let c): try createComputePipeline(c)
        case .getBindGroupLayout(let c): try getBindGroupLayout(c)
        case .destroy(let c): registry.remove(c.id)

        // 오류 스코프
        case .pushErrorScope(let filter, let decodeFailure):
            // 실패해도 push부터 한다 — 깊이 유지 계약 (`WGPUCommand` 문서 참고).
            errorScopes.push(filter)
            if let decodeFailure { throw decodeFailure }
        case .popErrorScope: poppedScopes.append(errorScopes.pop())

        // 캔버스
        case .configureCanvas(let c): try configureCanvas(c)
        case .getCurrentTexture(let c): try getCurrentTexture(c)

        // 렌더 패스
        case .beginRenderPass(let c): try beginRenderPass(c)
        case .setPipeline(let c): try setPipeline(c)
        case .setBindGroup(let c): try setBindGroup(c)
        case .setVertexBuffer(let c): try setVertexBuffer(c)
        case .setIndexBuffer(let c): try setIndexBuffer(c)
        case .setViewport(let c):
            try requireRenderPass()
            try backend.setViewport(c)
        case .setScissorRect(let c):
            try requireRenderPass()
            try backend.setScissorRect(c)
        case .setBlendConstant(let c):
            try requireRenderPass()
            try backend.setBlendConstant(c.color)
        case .setStencilReference(let c):
            try requireRenderPass()
            try backend.setStencilReference(c.reference)
        case .draw(let c): try draw(c)
        case .drawIndexed(let c): try drawIndexed(c)
        case .drawIndirect(let c): try drawIndirect(c)
        case .drawIndexedIndirect(let c): try drawIndexedIndirect(c)
        case .executeBundles(let c): try executeBundles(c)
        case .beginOcclusionQuery(let c): try beginOcclusionQuery(c)
        case .endOcclusionQuery: try endOcclusionQuery()

        // 컴퓨트 패스
        case .beginComputePass(let c): try beginComputePass(c)
        case .dispatchWorkgroups(let c): try dispatchWorkgroups(c)
        case .dispatchWorkgroupsIndirect(let c): try dispatchWorkgroupsIndirect(c)

        case .endPass: closePass()

        // 복사
        case .copyBufferToBuffer(let c): try copyBufferToBuffer(c)
        case .clearBuffer(let c): try clearBuffer(c)
        case .copyTextureToBuffer(let c): try copyTextureToBuffer(c)
        case .copyBufferToTexture(let c): try copyBufferToTexture(c)
        case .copyTextureToTexture(let c): try copyTextureToTexture(c)

        // 쿼리
        case .resolveQuerySet(let c): try resolveQuerySet(c)

        // 디버그 마커
        case .pushDebugGroup(let c): try pushDebugGroup(c)
        case .popDebugGroup: popDebugGroup()
        case .insertDebugMarker(let c):
            try backend.insertDebugMarker(c.markerLabel, scope: passState != nil ? .pass : .frame)
        }
    }

    // MARK: - 조회 헬퍼

    private func buffer(_ handle: WGPUHandle, path: String? = nil) throws -> WGPUEngineBuffer<B> {
        try registry.lookup(handle, as: WGPUEngineBuffer<B>.self, kind: "GPUBuffer", path: path)
    }

    /// 큐 작업에 쓸 버퍼를 꺼낸다 — **매핑 중이면 거부한다.**
    ///
    /// 명세는 `mapAsync`가 버퍼를 "unavailable"로 만들어 `unmap()` 전까지 큐 작업에 못 쓰게 해
    /// 경쟁 자체를 없앤다. 읽기가 GPU 완료를 기다리는 동안 다음 프레임의 쓰기가 같은 메모리에
    /// 겹치면 **JS가 받는 값이 어느 프레임 것인지 보장되지 않는다.**
    ///
    /// 버퍼를 쓰는 모든 명령이 이 함수를 지나야 한다 — 한 곳이라도 빠지면 그 경로로 경쟁이 샌다.
    private func unmappedBuffer(_ handle: WGPUHandle, path: String? = nil) throws -> WGPUEngineBuffer<B> {
        let object = try buffer(handle, path: path)
        guard !object.isMapped else {
            throw WGPUError.validation(
                "매핑 중인 GPUBuffer \(handle)은(는) 큐 작업에 쓸 수 없다 "
                    + "(mapAsync로 읽은 뒤 unmap()을 부를 것)",
                path: path
            )
        }
        return object
    }

    private func texture(_ handle: WGPUHandle, path: String? = nil) throws -> WGPUEngineTexture<B> {
        try registry.lookup(handle, as: WGPUEngineTexture<B>.self, kind: "GPUTexture", path: path)
    }

    private func textureView(_ handle: WGPUHandle, path: String? = nil) throws -> WGPUEngineTextureView<B> {
        try registry.lookup(handle, as: WGPUEngineTextureView<B>.self, kind: "GPUTextureView", path: path)
    }

    private func requireRenderPass() throws {
        guard passState == .render else {
            throw WGPUError.validation("렌더 패스가 시작되지 않았다 (beginRenderPass 먼저)")
        }
    }

    private func requireComputePass() throws {
        guard passState == .compute else {
            throw WGPUError.validation("컴퓨트 패스가 시작되지 않았다 (beginComputePass 먼저)")
        }
    }

    /// 복사·업로드는 패스 밖에서만 — 커맨드 인코더 수준 명령이다 (JS shim도 그렇게만 보낸다).
    /// 네이티브 검증 백엔드에서는 백엔드 검증기가 잡는다 (인코더 복사 — Dawn Finish 시점).
    private func requireNoOpenPass() throws {
        guard specValidation else { return }
        guard passState == nil else {
            throw WGPUError.validation("렌더/컴퓨트 패스 안에서는 복사·업로드 명령을 쓸 수 없다")
        }
    }

    // MARK: - 리소스 생성

    private func createBuffer(_ command: WGPUCreateCommand<WGPUBufferDescriptor>) throws {
        let raw = try backend.makeBuffer(command.descriptor)
        let object = WGPUEngineBuffer<B>(
            raw: raw, size: command.descriptor.size, usage: command.descriptor.usage
        )
        // 명세: mappedAtCreation 버퍼는 unmap 전까지 "unavailable"이다.
        // (JS shim은 이 플래그를 클라이언트에서 접어 initialData로 보낸다 — 이 경로는
        //  커맨드 스트림을 직접 만드는 네이티브 사용자의 것이다.)
        object.isMapped = command.descriptor.mappedAtCreation
        registry.insert(object, at: command.id)
    }

    private func unmapBuffer(_ command: WGPUUnmapBufferCommand) throws {
        let object = try registry.lookup(
            command.buffer, as: WGPUEngineBuffer<B>.self, kind: "GPUBuffer"
        )
        object.isMapped = false
        backend.unmapBuffer(object.raw)
    }

    private func writeBuffer(_ command: WGPUWriteBufferCommand) throws {
        let target = try unmappedBuffer(command.buffer, path: command.fieldPath("buffer"))
        let offset = command.bufferOffset
        let data = command.data
        if specValidation {
            guard offset >= 0, offset + data.count <= target.size else {
                throw WGPUError.validation(
                    "writeBuffer 범위 초과 — offset \(offset) + \(data.count)B > 버퍼 크기 \(target.size)B"
                )
            }
            guard target.usage.contains(.copyDst) else {
                throw WGPUError.validation(
                    "writeBuffer의 대상은 GPUBufferUsage.COPY_DST로 만들어야 한다",
                    path: command.fieldPath("buffer")
                )
            }
            // 4의 배수 요구는 명세 규칙이다. Metal은 바이트 단위 blit도 받아 주므로 안 막으면
            // 브라우저에서만 거부되는 코드가 나온다 (`clearBuffer`와 같은 이유).
            guard offset % 4 == 0, data.count % 4 == 0 else {
                throw WGPUError.validation(
                    "writeBuffer의 bufferOffset·크기는 4의 배수여야 한다 "
                        + "(받은 값 \(offset), \(data.count)B)",
                    path: command.fieldPath("bufferOffset")
                )
            }
        }
        guard !data.isEmpty else { return }   // 크기 0은 no-op
        try requireNoOpenPass()
        try backend.writeBuffer(target.raw, offset: offset, data: data)
    }

    private func createTexture(_ command: WGPUCreateCommand<WGPUTextureDescriptor>) throws {
        if specValidation { try validateCompressedTexture(command.descriptor) }
        let raw = try backend.makeTexture(command.descriptor)
        registry.insert(
            WGPUEngineTexture<B>(
                raw: raw,
                format: command.descriptor.format,
                size: command.descriptor.size,
                sampleCount: command.descriptor.sampleCount,
                isDrawable: false
            ),
            at: command.id
        )
    }

    /// 블록 압축 텍스처의 제약. **백엔드가 단언으로 죽는 조합**이라 여기서 미리 잡아
    /// 명세대로 검증 오류로 돌려준다.
    private func validateCompressedTexture(_ descriptor: WGPUTextureDescriptor) throws {
        let format = descriptor.format
        guard format.isCompressed else { return }
        guard backend.supportsTextureCompression(format) else {
            let feature = format.compressionFamily.featureName ?? "?"
            throw WGPUError.validation(
                "이 기기는 \(format.rawValue)를 지원하지 않는다 — adapter.features의 '\(feature)'를 먼저 확인할 것"
            )
        }
        // 압축 포맷은 샘플링·복사만 된다 (명세: RENDER_ATTACHMENT·STORAGE_BINDING 금지).
        let forbidden: WGPUTextureUsage = [.renderAttachment, .storageBinding]
        guard descriptor.usage.isDisjoint(with: forbidden) else {
            throw WGPUError.validation(
                "압축 텍스처(\(format.rawValue))는 렌더 타깃이나 스토리지로 쓸 수 없다 (usage \(descriptor.usage))"
            )
        }
        guard descriptor.dimension == .twoD else {
            throw WGPUError.validation("압축 텍스처는 2d만 된다 (\(descriptor.dimension.rawValue) 요청)")
        }
        guard descriptor.sampleCount == 1 else {
            throw WGPUError.validation("압축 텍스처는 멀티샘플이 될 수 없다 (sampleCount \(descriptor.sampleCount))")
        }
    }

    private func writeTexture(_ command: WGPUWriteTextureCommand) throws {
        let target = try texture(command.texture, path: command.fieldPath("texture"))
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
        if specValidation {
            try validateBlockAlignment(format: format, origin: command.origin, size: size,
                                       textureSize: target.size, mipLevel: command.mipLevel,
                                       label: "writeTexture")
            let bytesPerImage = bytesPerRow * max(rowsPerImage, blockRows)
            let layers = max(size.depthOrArrayLayers, 1)
            let required = bytesPerImage * (layers - 1) + bytesPerRow * blockRows
            guard data.count >= required else {
                throw WGPUError.validation("writeTexture 데이터가 부족하다 (\(data.count)B, 최소 \(required)B 필요)")
            }
        }
        try requireNoOpenPass()
        try backend.writeTexture(
            target.raw, data: data, origin: command.origin, size: size,
            mipLevel: command.mipLevel, bytesPerRow: bytesPerRow, rowsPerImage: rowsPerImage
        )
    }

    /// 디코딩해 둔 이미지(`ImageBitmap`)를 텍스처로 올린다 — 명세
    /// `queue.copyExternalImageToTexture()`.
    ///
    /// 픽셀은 이미 RGBA8이라 여기서는 잘라내서 `writeTexture` 동사로 수렴시키는 일만 한다.
    private func copyExternalImageToTexture(_ command: WGPUCopyExternalImageCommand) throws {
        let bitmap = try registry.lookup(
            command.source.image, as: WGPUImageBitmapObject.self, kind: "ImageBitmap",
            path: command.source.fieldPath("source")
        )
        let target = try texture(command.destination.texture, path: command.destination.fieldPath("texture"))
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

        // 필요한 만큼만 잘라 싣는다. 전체 폭을 그대로 쓰면 부분 복사에서 bytesPerRow가 맞지 않는다.
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

        try requireNoOpenPass()
        try backend.writeTexture(
            target.raw, data: slice,
            origin: command.destination.origin,
            size: WGPUExtent3D(width: size.width, height: size.height),
            mipLevel: command.destination.mipLevel,
            bytesPerRow: rowBytes, rowsPerImage: size.height
        )
    }

    /// 압축 텍스처 복사의 블록 정렬 (명세 "validating texel copy range").
    ///
    /// origin은 블록 경계에 있어야 하고, 크기는 블록 배수이거나 **밉 레벨의 끝에 닿아야** 한다
    /// (가장자리 블록은 잘려 있으므로 예외다). 어기면 백엔드가 단언으로 죽어서 여기서 먼저 막는다.
    /// 비압축 포맷은 블록이 1×1이라 이 검사가 항상 통과한다.
    private func validateBlockAlignment(
        format: WGPUTextureFormat,
        origin: WGPUOrigin3D,
        size: WGPUExtent3D,
        textureSize: WGPUExtent3D,
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
        let levelWidth = max(textureSize.width >> mipLevel, 1)
        let levelHeight = max(textureSize.height >> mipLevel, 1)
        guard size.width % blockWidth == 0 || origin.x + size.width == levelWidth,
              size.height % blockHeight == 0 || origin.y + size.height == levelHeight else {
            throw WGPUError.validation(
                "\(label): 압축 텍스처의 복사 크기는 블록 배수이거나 밉 레벨 끝에 닿아야 한다 "
                + "(\(size.width)x\(size.height) @ 레벨 \(mipLevel) 크기 \(levelWidth)x\(levelHeight))"
            )
        }
    }

    /// 버퍼↔텍스처 복사의 행 배치 규칙 (명세 "validating GPUTexelCopyBufferInfo" ·
    /// "validating linear texture data").
    ///
    /// **`bytesPerRow`는 256의 배수여야 한다.** Metal은 훨씬 느슨해서(행 크기만 맞으면 받는다)
    /// 여기서 안 막으면 브라우저와 Dawn에서만 거부되는 코드가 통과한다 — 실제로 데모 씬 둘이
    /// 이 자리에서 32를 쓰다가 Dawn 연동 때 드러났다 (`docs/TESTING.md`의 데모 씬 표).
    ///
    /// `queue.writeTexture`에는 이 제약이 **없다** (명세가 큐 업로드와 인코더 복사를 다르게
    /// 정한다). 그래서 이 검사는 두 복사 op에만 붙는다.
    private func validateTexelCopyBufferLayout(
        bytesPerRow: Int?,
        format: WGPUTextureFormat,
        size: WGPUExtent3D,
        label: String,
        path: String?
    ) throws {
        let blockRows = format.blockRows(height: size.height)
        let layers = max(size.depthOrArrayLayers, 1)
        // 행이 여럿이면 스트라이드를 유도할 수 없다 — 명세가 명시를 요구한다.
        guard let bytesPerRow else {
            guard blockRows <= 1, layers <= 1 else {
                throw WGPUError.validation(
                    "\(label): 복사가 여러 행·레이어에 걸치면 bytesPerRow를 줘야 한다 "
                        + "(블록 행 \(blockRows), 레이어 \(layers))",
                    path: path
                )
            }
            return
        }
        guard bytesPerRow % 256 == 0 else {
            throw WGPUError.validation(
                "\(label): bytesPerRow는 256의 배수여야 한다 (받은 값 \(bytesPerRow))",
                path: path
            )
        }
    }

    private func createTextureView(_ command: WGPUCreateTextureViewCommand) throws {
        let source = try texture(command.texture, path: command.fieldPath("texture"))
        let format = command.descriptor.format ?? source.format
        let raw = try backend.makeTextureView(source.raw, descriptor: command.descriptor, format: format)
        registry.insert(
            WGPUEngineTextureView<B>(raw: raw, format: format, sampleCount: source.sampleCount),
            at: command.id
        )
        // 드로어블 텍스처의 뷰는 프레임과 함께 만료된다 (명세 "Expire the current texture").
        if source.isDrawable { frameScopedHandles.append(command.id) }
    }

    private func createSampler(_ command: WGPUCreateCommand<WGPUSamplerDescriptor>) throws {
        let raw = try backend.makeSampler(command.descriptor)
        registry.insert(WGPUEngineSampler<B>(raw: raw), at: command.id)
    }

    /// 명세에서 **셰이더 모듈은 컴파일에 실패해도 만들어진다** — 오류는 `getCompilationInfo()`와
    /// 파이프라인 생성 실패로 드러난다. 그래서 파싱이 깨져도 등록하고 진단을 담아 둔다.
    ///
    /// 핸들이 아예 없으면 이후 명령이 전부 "존재하지 않는다"로만 깨져 **진짜 원인(파싱 실패)이
    /// 화면에서 사라진다.** 여기서는 원인도 그 자리에서 보고한다.
    private func createShaderModule(_ command: WGPUCreateCommand<WGPUShaderModuleDescriptor>) throws {
        let creation = try backend.makeShaderModule(command.descriptor, fieldPath: { command.fieldPath($0) })
        registry.insert(WGPUEngineShaderModule<B>(raw: creation.module), at: command.id)
        if let failure = creation.failure {
            throw WGPUError(
                kind: failure.kind, message: failure.message,
                path: failure.path ?? command.fieldPath("code"), line: failure.line
            )
        }
    }

    private func createBindGroupLayout(_ command: WGPUCreateCommand<WGPUBindGroupLayoutDescriptor>) throws {
        let entries = command.descriptor.entries
        let raw = try backend.makeBindGroupLayout(entries)
        registry.insert(WGPUEngineBindGroupLayout<B>(raw: raw, entries: entries), at: command.id)
    }

    private func createPipelineLayout(_ command: WGPUCreateCommand<WGPUPipelineLayoutDescriptor>) throws {
        let groups = try command.descriptor.bindGroupLayouts.map {
            try registry.lookup(
                $0, as: WGPUEngineBindGroupLayout<B>.self, kind: "GPUBindGroupLayout",
                path: command.fieldPath("bindGroupLayouts")
            )
        }
        let raw = try backend.makePipelineLayout(groups.map(\.raw))
        registry.insert(WGPUEnginePipelineLayout<B>(raw: raw), at: command.id)
    }

    private func createBindGroup(_ command: WGPUCreateCommand<WGPUBindGroupDescriptor>) throws {
        let layout = try registry.lookup(
            command.descriptor.layout, as: WGPUEngineBindGroupLayout<B>.self, kind: "GPUBindGroupLayout",
            path: command.fieldPath("layout")
        )
        var buffers: [WGPUEngineBuffer<B>] = []
        let entries: [WGPUResolvedBindGroupEntry<B>] = try command.descriptor.entries.map { entry in
            // 항목을 아는 레이아웃이면 매칭을 검사한다. 네이티브 검증 백엔드와
            // 파생 레이아웃(항목 미상)은 백엔드 검증에 맡긴다 — visibility도 백엔드가 스스로 안다.
            let layoutEntry: WGPUBindGroupLayoutEntry?
            if specValidation, layout.entries != nil {
                guard let matched = layout.entry(binding: entry.binding) else {
                    throw WGPUError.validation("바인드 그룹 레이아웃에 binding \(entry.binding)이 없다")
                }
                layoutEntry = matched
            } else {
                layoutEntry = nil
            }
            let resource: WGPUResolvedBindGroupEntry<B>.Resource
            switch entry.resource {
            case .buffer(let handle, let offset, let size):
                let object = try registry.lookup(handle, as: WGPUEngineBuffer<B>.self, kind: "GPUBuffer")
                buffers.append(object)
                resource = .buffer(
                    object.raw, offset: offset, boundSize: size ?? max(object.size - offset, 0)
                )
            case .sampler(let handle):
                let object = try registry.lookup(handle, as: WGPUEngineSampler<B>.self, kind: "GPUSampler")
                resource = .sampler(object.raw)
            case .textureView(let handle):
                let object = try registry.lookup(
                    handle, as: WGPUEngineTextureView<B>.self, kind: "GPUTextureView"
                )
                resource = .textureView(object.raw)
            }
            return WGPUResolvedBindGroupEntry<B>(
                binding: entry.binding, layoutEntry: layoutEntry, resource: resource
            )
        }
        let raw = try backend.makeBindGroup(layout: layout.raw, entries: entries)
        registry.insert(WGPUEngineBindGroup<B>(raw: raw, buffers: buffers), at: command.id)
    }

    private func createQuerySet(_ command: WGPUCreateCommand<WGPUQuerySetDescriptor>) throws {
        let raw = try backend.makeQuerySet(command.descriptor)
        registry.insert(
            WGPUEngineQuerySet<B>(raw: raw, type: command.descriptor.type, count: command.descriptor.count),
            at: command.id
        )
    }

    /// `bundleEncoder.finish()` — JS가 모아 둔 명령 목록을 번들 객체로 등록한다.
    ///
    /// 번들 인코더 자체는 와이어에 없다. JS가 명령을 배열에 모으고 `finish()`에서 한 번에
    /// 내려보내므로, 인코더의 수명을 양쪽에서 맞출 이유가 없다. 네이티브 번들을 가진 백엔드는
    /// 여기서 바로 기록까지 끝내고, 아니면 리더를 저장해 두었다가 실행 때 되풀이한다.
    private func createRenderBundle(_ command: WGPUCreateRenderBundleCommand) throws {
        if specValidation { try WGPUEngineRenderBundle<B>.validateOps(command.commands) }
        let native: B.RenderBundle?
        if backend.capabilities.supportsNativeRenderBundles {
            let decoded = try command.commands.map { try WGPUCommand(from: $0) }
            native = try backend.makeRenderBundle(
                command.descriptor, commands: decoded, resolver: bundleResolver()
            )
        } else {
            native = nil
        }
        registry.insert(
            WGPUEngineRenderBundle<B>(
                commands: command.commands, native: native, descriptor: command.descriptor
            ),
            at: command.id
        )
    }

    private func bundleResolver() -> WGPUBundleResolver<B> {
        let registry = self.registry
        return WGPUBundleResolver<B>(
            renderPipeline: { handle, path in
                try registry.lookup(
                    handle, as: WGPUEngineRenderPipeline<B>.self, kind: "GPURenderPipeline", path: path
                ).raw
            },
            bindGroup: { handle, path in
                try registry.lookup(
                    handle, as: WGPUEngineBindGroup<B>.self, kind: "GPUBindGroup", path: path
                ).raw
            },
            buffer: { handle, path in
                let object = try registry.lookup(
                    handle, as: WGPUEngineBuffer<B>.self, kind: "GPUBuffer", path: path
                )
                guard !object.isMapped else {
                    throw WGPUError.validation(
                        "매핑 중인 GPUBuffer \(handle)은(는) 큐 작업에 쓸 수 없다 "
                            + "(mapAsync로 읽은 뒤 unmap()을 부를 것)",
                        path: path
                    )
                }
                return object.raw
            }
        )
    }

    private func createRenderPipeline(_ command: WGPUCreateCommand<WGPURenderPipelineDescriptor>) throws {
        let descriptor = command.descriptor
        let vertexModule = try registry.lookup(
            descriptor.vertex.module, as: WGPUEngineShaderModule<B>.self, kind: "GPUShaderModule",
            path: command.fieldPath("vertex.module")
        )
        let fragmentModule = try descriptor.fragment.map {
            try registry.lookup(
                $0.module, as: WGPUEngineShaderModule<B>.self, kind: "GPUShaderModule",
                path: command.fieldPath("fragment.module")
            )
        }
        let layout = try resolvePipelineLayout(descriptor.layout)
        let creation = try backend.makeRenderPipeline(
            descriptor,
            vertexModule: vertexModule.raw,
            fragmentModule: fragmentModule?.raw,
            layout: layout,
            fieldPath: { command.fieldPath($0) }
        )
        registry.insert(
            WGPUEngineRenderPipeline<B>(raw: creation.pipeline, info: creation.info), at: command.id
        )
    }

    private func createComputePipeline(_ command: WGPUCreateCommand<WGPUComputePipelineDescriptor>) throws {
        let descriptor = command.descriptor
        let module = try registry.lookup(
            descriptor.module, as: WGPUEngineShaderModule<B>.self, kind: "GPUShaderModule",
            path: command.fieldPath("compute.module")
        )
        let layout = try resolvePipelineLayout(descriptor.layout)
        let creation = try backend.makeComputePipeline(
            descriptor, module: module.raw, layout: layout, fieldPath: { command.fieldPath($0) }
        )
        registry.insert(
            WGPUEngineComputePipeline<B>(raw: creation.pipeline, info: creation.info), at: command.id
        )
    }

    private func resolvePipelineLayout(_ reference: WGPUPipelineLayoutRef) throws -> WGPUResolvedPipelineLayout<B> {
        switch reference {
        case .auto:
            return .auto
        case .explicit(let handle):
            let layout = try registry.lookup(
                handle, as: WGPUEnginePipelineLayout<B>.self, kind: "GPUPipelineLayout"
            )
            return .explicit(layout.raw)
        }
    }

    /// `pipeline.getBindGroupLayout(index)` — `layout: "auto"`로 유도된 레이아웃을 핸들로 꺼낸다.
    private func getBindGroupLayout(_ command: WGPUGetBindGroupLayoutCommand) throws {
        let pipelineHandle = command.pipeline
        let pipeline: WGPUResolvedPipeline<B>
        if let render = try? registry.lookup(
            pipelineHandle, as: WGPUEngineRenderPipeline<B>.self, kind: "x"
        ) {
            pipeline = .render(render.raw)
        } else {
            pipeline = .compute(try registry.lookup(
                pipelineHandle, as: WGPUEngineComputePipeline<B>.self, kind: "GPUPipeline",
                path: command.fieldPath("pipeline")
            ).raw)
        }
        guard let creation = try backend.bindGroupLayout(of: pipeline, index: command.index) else {
            throw WGPUError.validation("파이프라인에 바인드 그룹 \(command.index)이(가) 없다")
        }
        registry.insert(
            WGPUEngineBindGroupLayout<B>(raw: creation.layout, entries: creation.entries), at: command.id
        )
    }

    // MARK: - 캔버스 (커맨드 스트림)

    private func configureCanvas(_ configuration: WGPUCanvasConfiguration) throws {
        guard let entry = surfaceEntry(for: configuration.canvasId) else {
            throw WGPUError.validation(
                "캔버스 '\(configuration.canvasId)'이(가) 등록되지 않았다 "
                    + "(<webgpu-canvas canvas-id=\"…\">가 화면에 붙어 있는지 확인)"
            )
        }
        try backend.configureSurface(entry.raw, configuration: configuration)
        touchedCanvases[configuration.canvasId] = entry.raw
    }

    private func getCurrentTexture(_ command: WGPUGetCurrentTextureCommand) throws {
        let handle = command.id
        let canvasId = command.canvas
        guard let entry = surfaceEntry(for: canvasId) else {
            throw WGPUError.validation("캔버스 '\(canvasId)'이(가) 등록되지 않았다")
        }
        // **앞 배치**에서 얻은 드로어블이 present되지 못한 채 이 캔버스가 또 요청됐다면,
        // 그 프레임은 끝난 것이다 — 여기서 거둔다.
        //
        // 제출 없이 끝난 배치가 드로어블을 남기는 것 자체는 의도된 것이다: 프레임이 아직
        // 진행 중일 수 있다 (`finish()` 참고 — Three.js의 지연 파이프라인 생성). 문제는 그
        // 프레임이 **영영 안 그려질 때**로, 첫 인코더 전에 검증 오류가 나면 그렇게 된다.
        // 거두지 않으면 프레임마다 하나씩 쌓여 화면 표면에서는 세 번 만에 드로어블 풀이
        // 마르고, 그 뒤로는 획득이 JS 스레드를 최대 1초씩 세운 뒤 영영 실패한다.
        //
        // **한 배치 안의 반복 획득은 제외한다** — 그건 같은 프레임이고(명세도 같은 텍스처를
        // 돌려주라고 한다), 앞서 내준 뷰가 아직 그 프레임에서 쓰인다.
        if !acquiredThisBatch.contains(canvasId),
           acquiredFrames.contains(where: { $0.canvas == canvasId }) {
            backend.discardAcquiredFrames()
            expireFrame()
        }
        acquiredThisBatch.insert(canvasId)
        touchedCanvases[canvasId] = entry.raw
        guard let acquired = try backend.acquireFrameTexture(entry.raw) else {
            throw WGPUError.validation(
                "캔버스 '\(canvasId)'의 드로어블을 얻지 못했다 (크기가 0이거나 드로어블이 고갈됨)"
            )
        }
        registry.insert(
            WGPUEngineTexture<B>(
                raw: acquired.texture,
                format: acquired.format,
                size: WGPUExtent3D(width: acquired.width, height: acquired.height),
                sampleCount: acquired.sampleCount,
                isDrawable: true
            ),
            at: handle
        )
        acquiredFrames.append((handle, canvasId))
        frameScopedHandles.append(handle)
    }

    // MARK: - 렌더 패스

    private func beginRenderPass(_ descriptor: WGPURenderPassDescriptor) throws {
        closePass()
        var colorFormats: [WGPUTextureFormat] = []
        var sampleCount = 1
        var colors: [WGPUResolvedRenderPass<B>.ColorAttachment] = []

        for attachment in descriptor.colorAttachments {
            let view = try textureView(attachment.view)
            colorFormats.append(view.format)
            sampleCount = max(sampleCount, view.sampleCount)
            var resolveTarget: B.TextureView?
            if let resolveHandle = attachment.resolveTarget {
                resolveTarget = try textureView(resolveHandle).raw
            }
            colors.append(WGPUResolvedRenderPass<B>.ColorAttachment(
                view: view.raw,
                resolveTarget: resolveTarget,
                loadOp: attachment.loadOp,
                storeOp: attachment.storeOp,
                clearValue: attachment.clearValue
            ))
        }

        var depthStencilFormat: WGPUTextureFormat?
        var depthReadOnly = false
        var stencilReadOnly = false
        var depthStencil: WGPUResolvedRenderPass<B>.DepthStencilAttachment?
        if let depth = descriptor.depthStencilAttachment {
            let view = try textureView(depth.view)
            depthStencilFormat = view.format
            // 깊이 뷰도 패스 레이아웃의 sampleCount에 반영한다 — 컬러 어태치먼트가 없는 MSAA 패스
            // (그림자 맵·깊이 프리패스)에서 이걸 빠뜨리면 올바르게 선언한 번들이 거부된다.
            sampleCount = max(sampleCount, view.sampleCount)
            depthStencil = WGPUResolvedRenderPass<B>.DepthStencilAttachment(
                view: view.raw,
                format: view.format,
                depthLoadOp: depth.depthLoadOp,
                depthStoreOp: depth.depthStoreOp,
                depthClearValue: depth.depthClearValue,
                stencilLoadOp: depth.stencilLoadOp,
                stencilStoreOp: depth.stencilStoreOp,
                stencilClearValue: depth.stencilClearValue,
                depthReadOnly: depth.depthReadOnly,
                stencilReadOnly: depth.stencilReadOnly
            )
            depthReadOnly = depth.depthReadOnly
            stencilReadOnly = depth.stencilReadOnly
        }

        // occlusion 쿼리는 **패스를 열 때만** 붙일 수 있다 (WebGPU와 백엔드들이 같은 제약).
        var occlusionQuerySet: WGPUEngineQuerySet<B>?
        if let handle = descriptor.occlusionQuerySet {
            let querySet = try registry.lookup(
                handle, as: WGPUEngineQuerySet<B>.self, kind: "GPUQuerySet"
            )
            if specValidation {
                guard querySet.type == .occlusion else {
                    throw WGPUError.validation(
                        "occlusionQuerySet은 type: \"occlusion\"이어야 한다 (받은 것: \(querySet.type.rawValue))"
                    )
                }
            }
            occlusionQuerySet = querySet
        }

        var timestampWrites: WGPUResolvedTimestampWrites<B>?
        if let writes = descriptor.timestampWrites {
            timestampWrites = try resolveTimestampWrites(writes)
        }

        try backend.beginRenderPass(WGPUResolvedRenderPass<B>(
            label: descriptor.label,
            colorAttachments: colors,
            depthStencil: depthStencil,
            occlusionQuerySet: occlusionQuerySet?.raw,
            timestampWrites: timestampWrites
        ))
        passState = .render
        passFormats = (colorFormats, depthStencilFormat, sampleCount)
        passOcclusionQuerySet = occlusionQuerySet
        passDepthReadOnly = depthReadOnly
        passStencilReadOnly = stencilReadOnly
        openOcclusionQuery = nil
        usedOcclusionQueries.removeAll()
        resetPassBindings()
    }

    /// 타임스탬프 쓰기 자리를 확인한다.
    private func resolveTimestampWrites(_ writes: WGPUPassTimestampWrites) throws -> WGPUResolvedTimestampWrites<B> {
        let querySet = try registry.lookup(writes.querySet, as: WGPUEngineQuerySet<B>.self, kind: "GPUQuerySet")
        if specValidation {
            guard querySet.type == .timestamp else {
                throw WGPUError.validation(
                    "timestampWrites의 쿼리셋은 type: \"timestamp\"여야 한다 (받은 것: \(querySet.type.rawValue))"
                )
            }
            // 둘 다 생략하면 **조용한 no-op 패스**가 된다. 오류 없이 쿼리셋 초기값(0)이 resolve되므로
            // 앱은 GPU 시간을 0ns로 읽는다.
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
        }
        return WGPUResolvedTimestampWrites<B>(
            querySet: querySet.raw,
            beginningOfPassWriteIndex: writes.beginningOfPassWriteIndex,
            endOfPassWriteIndex: writes.endOfPassWriteIndex
        )
    }

    private func beginComputePass(_ descriptor: WGPUComputePassDescriptor) throws {
        closePass()
        var timestampWrites: WGPUResolvedTimestampWrites<B>?
        if let writes = descriptor.timestampWrites {
            timestampWrites = try resolveTimestampWrites(writes)
        }
        try backend.beginComputePass(WGPUResolvedComputePass<B>(
            label: descriptor.label, timestampWrites: timestampWrites
        ))
        passState = .compute
        currentComputePipeline = nil
        boundGroups.removeAll()
        dirtyGroups.removeAll()
    }

    // MARK: - 파이프라인·바인딩 상태

    private func setPipeline(_ command: WGPUSetPipelineCommand) throws {
        let handle = command.pipeline
        switch passState {
        case .render:
            let pipeline = try registry.lookup(
                handle, as: WGPUEngineRenderPipeline<B>.self, kind: "GPURenderPipeline",
                path: command.fieldPath("pipeline")
            )
            // read-only로 선언한 어태치먼트를 쓰는 파이프라인은 여기서 막는다 — 백엔드가 그냥
            // 써 버리면, read-only라고 적어 둔 깊이 버퍼가 실제로 변조된다. 메타데이터를 주지
            // 않는 백엔드(nil)는 스스로 검증한다.
            guard !passDepthReadOnly || !(pipeline.info.writesDepth ?? false) else {
                throw WGPUError.validation(
                    "depthReadOnly 패스에서는 depthWriteEnabled: true 파이프라인을 쓸 수 없다"
                )
            }
            guard !passStencilReadOnly || !(pipeline.info.writesStencil ?? false) else {
                throw WGPUError.validation(
                    "stencilReadOnly 패스에서는 스텐실을 쓰는 파이프라인을 쓸 수 없다 "
                        + "(failOp·depthFailOp·passOp가 모두 \"keep\"이어야 한다)"
                )
            }
            backend.setRenderPipeline(pipeline.raw)
            currentRenderPipeline = pipeline
        case .compute:
            let pipeline = try registry.lookup(
                handle, as: WGPUEngineComputePipeline<B>.self, kind: "GPUComputePipeline",
                path: command.fieldPath("pipeline")
            )
            backend.setComputePipeline(pipeline.raw)
            currentComputePipeline = pipeline
        case nil:
            throw WGPUError.validation("setPipeline은 패스 안에서만 쓸 수 있다")
        }
        // 파이프라인이 바뀌면 레이아웃이 달라질 수 있으므로 바인드 그룹을 다시 적용한다.
        dirtyGroups = Set(boundGroups.keys)
    }

    private func setBindGroup(_ command: WGPUSetBindGroupCommand) throws {
        let group = try registry.lookup(
            command.bindGroup, as: WGPUEngineBindGroup<B>.self, kind: "GPUBindGroup",
            path: command.fieldPath("bindGroup")
        )
        boundGroups[command.index] = (group, command.dynamicOffsets)
        dirtyGroups.insert(command.index)
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

    /// 드로우·디스패치 직전에 파이프라인이 요구하는 상태를 전부 확인하고 백엔드에 내린다.
    ///
    /// 바인드 그룹과 정점 버퍼를 한자리에서 다루는 이유는, 둘 다 **번들 경계에서 무효화되는
    /// 상태**라 검사 시점이 같아야 하기 때문이다. 새 드로우 op을 추가할 때 이 함수 하나만
    /// 부르면 격리 계약이 자동으로 따라온다.
    ///
    /// 파이프라인 가드는 각 드로우 op이 자기 이름이 든 메시지로 **이 함수보다 먼저** 세운다 —
    /// 아래의 같은 검사는 그 가드를 빠뜨린 op을 위한 안전망이라 메시지가 일반형이다.
    private func applyDrawState() throws {
        let requiredGroups: Set<Int>?
        if passState == .render {
            guard let pipeline = currentRenderPipeline else {
                throw WGPUError.validation("draw 전에 setPipeline이 필요하다")
            }
            requiredGroups = pipeline.info.requiredGroups
        } else {
            guard let pipeline = currentComputePipeline else {
                throw WGPUError.validation("dispatch 전에 setPipeline이 필요하다")
            }
            requiredGroups = pipeline.info.requiredGroups
        }

        // 레이아웃이 요구하는 그룹이 전부 바인드되어 있어야 한다. 이 검사가 없으면 번들이
        // 남긴 바인딩(또는 패스가 미리 올려 둔 바인딩)으로 조용히 그려진다 — 인코더에는
        // "바인딩 해제"가 없으므로 `resetPassBindings()`만으로는 실제로 격리되지 않는다.
        // (메타데이터를 주지 않는 백엔드는 스스로 검증한다.)
        if let requiredGroups {
            for groupIndex in requiredGroups.sorted() where boundGroups[groupIndex] == nil {
                throw WGPUError.validation(
                    "파이프라인 레이아웃이 요구하는 @group(\(groupIndex))이 바인드되지 않았다 "
                        + "(번들 실행 앞뒤로는 바인딩이 무효화된다 — setBindGroup을 다시 할 것)"
                )
            }
        }

        // 바인드 그룹이 물고 있는 버퍼가 매핑 중이면 이 드로우도 큐 작업이므로 거부한다.
        // (그룹은 만들 때 버퍼를 고정하므로, 만든 뒤에 매핑된 경우가 여기서 걸린다.)
        for (_, bound) in boundGroups {
            for buffer in bound.group.buffers where buffer.isMapped {
                throw WGPUError.validation(
                    "매핑 중인 버퍼를 물고 있는 바인드 그룹으로는 그릴 수 없다 (unmap()을 먼저 부를 것)"
                )
            }
        }

        for groupIndex in dirtyGroups.sorted() {
            guard let bound = boundGroups[groupIndex] else { continue }
            try backend.applyBindGroup(bound.group.raw, at: groupIndex, dynamicOffsets: bound.offsets)
        }
        dirtyGroups.removeAll()

        try applyVertexBuffers()
    }

    /// 드로우 직전에 파이프라인이 요구하는 정점 버퍼가 다 있는지 보고 백엔드에 내린다.
    ///
    /// 명세는 "`vertex.buffers[slot]`이 null이 아니면 `[[vertex_buffers]]`가 그 슬롯을 담아야
    /// 한다"고 정한다. 이 검사가 없으면 패스가 미리 올려 둔 정점 버퍼로 번들이 그려지고,
    /// 번들이 올린 것으로 패스가 그려진다 — 브라우저에서는 둘 다 무효인 코드다.
    private func applyVertexBuffers() throws {
        guard passState == .render, let pipeline = currentRenderPipeline else { return }
        if let required = pipeline.info.requiredVertexSlots {
            for slot in required.sorted() where vertexBindings[slot] == nil {
                throw WGPUError.validation(
                    "파이프라인이 요구하는 정점 버퍼 슬롯 \(slot)이 바인드되지 않았다 "
                        + "(번들 실행 앞뒤로는 바인딩이 무효화된다 — setVertexBuffer를 다시 할 것)"
                )
            }
        }
        for slot in dirtyVertexSlots.sorted() {
            guard let binding = vertexBindings[slot] else { continue }
            try backend.applyVertexBuffer(binding.buffer.raw, offset: binding.offset, slot: slot)
        }
        dirtyVertexSlots.removeAll()
    }

    private func setVertexBuffer(_ command: WGPUSetVertexBufferCommand) throws {
        try requireRenderPass()
        let slot = command.slot
        let buffer = try unmappedBuffer(command.buffer, path: command.fieldPath("buffer"))
        let offset = command.offset
        if specValidation {
            let maxSlots = backend.capabilities.maxVertexBufferSlots
            guard slot >= 0, slot < maxSlots else {
                throw WGPUError.validation("정점 버퍼 슬롯은 0~\(maxSlots - 1) 범위다")
            }
            guard offset >= 0, offset <= buffer.size else {
                throw WGPUError.validation(
                    "정점 버퍼 offset(\(offset))이 버퍼 크기(\(buffer.size)B)를 벗어난다",
                    path: command.fieldPath("offset")
                )
            }
        }
        // 백엔드에 바로 내리지 않는다 — 드로우 직전에 내려야 번들 경계의 무효화가 성립한다.
        vertexBindings[slot] = (buffer, offset)
        dirtyVertexSlots.insert(slot)
    }

    private func setIndexBuffer(_ command: WGPUSetIndexBufferCommand) throws {
        try requireRenderPass()
        let buffer = try unmappedBuffer(command.buffer, path: command.fieldPath("buffer"))
        indexBinding = (buffer, command.offset, command.format, command.indexStride)
    }

    private func resolvedIndexBinding(
        _ binding: (buffer: WGPUEngineBuffer<B>, offset: Int, format: WGPUIndexFormat, stride: Int)
    ) -> WGPUResolvedIndexBinding<B> {
        WGPUResolvedIndexBinding<B>(
            buffer: binding.buffer.raw, offset: binding.offset,
            format: binding.format, stride: binding.stride
        )
    }

    // MARK: - 드로우 / 디스패치

    private func draw(_ command: WGPUDrawCommand) throws {
        try requireRenderPass()
        // 파이프라인 가드는 `applyDrawState()`보다 **먼저** 둔다 — 그 안의 같은 검사가 먼저
        // 던지면 이 op 이름이 든 메시지가 영영 나가지 못하는 죽은 코드가 된다 (아래 draw 계열 공통).
        guard currentRenderPipeline != nil else {
            throw WGPUError.validation("draw 전에 setPipeline이 필요하다")
        }
        try applyDrawState()
        try backend.draw(command)
    }

    private func drawIndexed(_ command: WGPUDrawIndexedCommand) throws {
        try requireRenderPass()
        guard currentRenderPipeline != nil else {
            throw WGPUError.validation("drawIndexed 전에 setPipeline이 필요하다")
        }
        guard let indexBinding else {
            throw WGPUError.validation("drawIndexed 전에 setIndexBuffer가 필요하다")
        }
        try applyDrawState()
        try backend.drawIndexed(command, index: resolvedIndexBinding(indexBinding))
    }

    /// 간접 인자 버퍼를 찾고 오프셋을 검증한다.
    ///
    /// - 4바이트 정렬과 범위는 **백엔드가 단언(=프로세스 종료)으로 처리**할 수 있으므로 여기서 잡는다.
    /// - `INDIRECT` usage는 백엔드에 대응하는 개념이 없을 수 있어 백엔드가 봐 주지 않는다.
    ///   확인하지 않으면 여기서는 돌고 브라우저에서만 깨지는 코드가 나온다.
    /// - 기기 능력은 백엔드가 먼저 자기 문맥이 담긴 오류로 거른다 (`ensureIndirectSupported`).
    private func indirectArguments(
        _ command: WGPUIndirectCommand,
        argumentSize: Int
    ) throws -> (buffer: B.Buffer, offset: Int) {
        try backend.ensureIndirectSupported()
        let object = try unmappedBuffer(command.indirectBuffer, path: command.fieldPath("indirectBuffer"))
        let offset = command.indirectOffset
        if specValidation {
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
        }
        return (object.raw, offset)
    }

    private func drawIndirect(_ command: WGPUIndirectCommand) throws {
        try requireRenderPass()
        // 인자 검증을 `applyDrawState()`보다 **먼저** 한다 — 거부할 명령이 인코더 상태를
        // 이미 바꿔 놓는 일이 없어야 한다 (오류는 프레임을 죽이지 않고 누적되므로 더 그렇다).
        // vertexCount, instanceCount, firstVertex, firstInstance — u32 4개.
        let arguments = try indirectArguments(command, argumentSize: 16)
        guard currentRenderPipeline != nil else {
            throw WGPUError.validation("drawIndirect 전에 setPipeline이 필요하다")
        }
        try applyDrawState()
        try backend.drawIndirect(buffer: arguments.buffer, offset: arguments.offset)
    }

    private func drawIndexedIndirect(_ command: WGPUIndirectCommand) throws {
        try requireRenderPass()
        // indexCount, instanceCount, firstIndex, baseVertex(i32), firstInstance — 5칸.
        let arguments = try indirectArguments(command, argumentSize: 20)
        guard currentRenderPipeline != nil else {
            throw WGPUError.validation("drawIndexedIndirect 전에 setPipeline이 필요하다")
        }
        guard let indexBinding else {
            throw WGPUError.validation("drawIndexedIndirect 전에 setIndexBuffer가 필요하다")
        }
        try applyDrawState()
        try backend.drawIndexedIndirect(
            buffer: arguments.buffer, offset: arguments.offset,
            index: resolvedIndexBinding(indexBinding)
        )
    }

    private func dispatchWorkgroups(_ command: WGPUDispatchWorkgroupsCommand) throws {
        try requireComputePass()
        guard currentComputePipeline != nil else {
            throw WGPUError.validation("dispatchWorkgroups 전에 setPipeline이 필요하다")
        }
        try applyDrawState()
        try backend.dispatchWorkgroups(command)
    }

    private func dispatchWorkgroupsIndirect(_ command: WGPUIndirectCommand) throws {
        try requireComputePass()
        // x, y, z — u32 3개.
        let arguments = try indirectArguments(command, argumentSize: 12)
        guard currentComputePipeline != nil else {
            throw WGPUError.validation("dispatchWorkgroupsIndirect 전에 setPipeline이 필요하다")
        }
        try applyDrawState()
        try backend.dispatchWorkgroupsIndirect(buffer: arguments.buffer, offset: arguments.offset)
    }

    // MARK: - occlusion 쿼리

    private func beginOcclusionQuery(_ command: WGPUBeginOcclusionQueryCommand) throws {
        try requireRenderPass()
        let index = command.queryIndex
        if specValidation {
            guard let querySet = passOcclusionQuerySet else {
                throw WGPUError.validation(
                    "beginOcclusionQuery를 쓰려면 beginRenderPass에 occlusionQuerySet을 줘야 한다"
                )
            }
            guard openOcclusionQuery == nil else {
                throw WGPUError.validation("occlusion 쿼리는 중첩할 수 없다 (앞의 것을 endOcclusionQuery로 닫을 것)")
            }
            try querySet.checkRange(first: index, count: 1, path: command.fieldPath("queryIndex"))
            // 한 패스에서 같은 인덱스를 두 번 쓰면 두 구간이 같은 8바이트 슬롯을 나눠 쓴다 —
            // 최종 값이 백엔드의 누적/덮어쓰기 동작에 달린 값이 되어 브라우저와 결과가 갈린다.
            guard usedOcclusionQueries.insert(index).inserted else {
                throw WGPUError.validation(
                    "occlusion 쿼리 인덱스 \(index)은(는) 이 패스에서 이미 썼다",
                    path: command.fieldPath("queryIndex")
                )
            }
        } else {
            usedOcclusionQueries.insert(index)
        }
        try backend.beginOcclusionQuery(index: index)
        openOcclusionQuery = index
    }

    private func endOcclusionQuery() throws {
        try requireRenderPass()
        guard let index = openOcclusionQuery else {
            throw WGPUError.validation("endOcclusionQuery: 열려 있는 occlusion 쿼리가 없다")
        }
        try backend.endOcclusionQuery(index: index)
        openOcclusionQuery = nil
    }

    // MARK: - 렌더 번들 실행

    private func executeBundles(_ command: WGPUExecuteBundlesCommand) throws {
        try requireRenderPass()
        guard let formats = passFormats else {
            throw WGPUError.validation("executeBundles는 렌더 패스 안에서만 쓸 수 있다")
        }
        let bundles = try command.bundles.map {
            try registry.lookup(
                $0, as: WGPUEngineRenderBundle<B>.self, kind: "GPURenderBundle",
                path: command.fieldPath("bundles")
            )
        }

        if specValidation {
            for bundle in bundles {
                try bundle.checkCompatibility(
                    color: formats.color,
                    depthStencil: formats.depthStencil,
                    sampleCount: formats.sampleCount,
                    depthReadOnly: passDepthReadOnly,
                    stencilReadOnly: passStencilReadOnly
                )
            }
        }
        // 명세의 "Reset the render pass binding state"(step 4)는 호환성 검증만 통과하면 **무조건**
        // 실행된다. 번들 명령 하나가 실패해 throw해도 마찬가지다 — 여기서 빠뜨리면 그 뒤의 패스 명령이
        // 번들이 남긴 파이프라인·바인드 그룹을 물고 그려져 잘못된 픽셀이 나간다.
        defer { resetPassBindings() }

        if backend.capabilities.supportsNativeRenderBundles {
            let natives = try bundles.map { bundle -> B.RenderBundle in
                guard let native = bundle.native else {
                    throw WGPUError.backend("네이티브 렌더 번들이 없다 (백엔드가 바뀐 채 재사용된 핸들)")
                }
                return native
            }
            try backend.executeBundles(natives)
        } else {
            // 하나라도 맞지 않으면 아무것도 실행하지 않는다 — 절반만 그려진 프레임을 남기지 않는다.
            for bundle in bundles {
                resetPassBindings()
                for bundleCommand in bundle.commands {
                    try perform(bundleCommand)
                }
            }
        }
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
        if specValidation {
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
            guard source.usage.contains(.copySrc) else {
                throw WGPUError.validation(
                    "copyBufferToBuffer의 원본은 GPUBufferUsage.COPY_SRC로 만들어야 한다",
                    path: command.fieldPath("source")
                )
            }
            guard destination.usage.contains(.copyDst) else {
                throw WGPUError.validation(
                    "copyBufferToBuffer의 대상은 GPUBufferUsage.COPY_DST로 만들어야 한다",
                    path: command.fieldPath("destination")
                )
            }
            // 명세 규칙 — Metal의 blit은 바이트 단위로도 복사해 주므로 여기서 안 막으면
            // 브라우저에서만 거부되는 코드가 나온다.
            guard sourceOffset % 4 == 0, destinationOffset % 4 == 0, size % 4 == 0 else {
                throw WGPUError.validation(
                    "copyBufferToBuffer의 오프셋·크기는 4의 배수여야 한다 "
                    + "(받은 값 \(sourceOffset), \(destinationOffset), \(size))",
                    path: command.fieldPath("size")
                )
            }
        }
        if size == 0 { return }   // 0바이트 복사는 no-op
        // 음수는 조용히 삼키지 않는다 — Metal 경로는 위 검사가 먼저 던져 도달하지 않고,
        // 네이티브 검증 경로는 여기서 같은 문구로 거부한다 (음수를 GPU 인자 폭으로 옮기는
        // 순간이 트랩이라, 이 거부는 검증 주체와 무관한 최소한의 예외처리다).
        guard sourceOffset >= 0, destinationOffset >= 0, size > 0 else {
            throw WGPUError.validation(
                "copyBufferToBuffer의 오프셋·크기는 음수일 수 없다 "
                + "(sourceOffset \(sourceOffset), destinationOffset \(destinationOffset), size \(size))"
            )
        }
        try requireNoOpenPass()
        try backend.copyBufferToBuffer(
            source: source.raw, sourceOffset: sourceOffset,
            destination: destination.raw, destinationOffset: destinationOffset, size: size
        )
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

        if specValidation {
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
        }
        if size == 0 { return }   // no-op
        // Range 구성 자체가 음수에서 트랩이다 — 트랩 방지는 검증 주체와 무관하게 여기 몫이다
        // (Metal 경로는 위 검사가 먼저 던져 도달하지 않는다).
        guard offset >= 0, size > 0 else {
            throw WGPUError.validation(
                "clearBuffer의 offset·size는 음수일 수 없다 (받은 값 \(offset), \(size))"
            )
        }
        try requireNoOpenPass()
        try backend.clearBuffer(object.raw, range: offset..<(offset + size))
    }

    private func copyTextureToBuffer(_ command: WGPUCopyTextureToBufferCommand) throws {
        let source = command.source
        let destination = command.destination
        let texture = try self.texture(source.texture, path: source.fieldPath("texture"))
        let buffer = try unmappedBuffer(destination.buffer, path: destination.fieldPath("buffer"))
        let size = command.copySize
        let format = texture.format
        let bytesPerRow = destination.bytesPerRow ?? format.bytesPerRow(width: size.width)
        if specValidation {
            try validateBlockAlignment(format: format, origin: source.origin, size: size,
                                       textureSize: texture.size, mipLevel: source.mipLevel,
                                       label: "copyTextureToBuffer")
            try validateTexelCopyBufferLayout(
                bytesPerRow: destination.bytesPerRow, format: format, size: size,
                label: "copyTextureToBuffer", path: destination.fieldPath("bytesPerRow")
            )
        }
        try requireNoOpenPass()
        try backend.copyTextureToBuffer(
            texture: texture.raw, slice: source.origin.z, mipLevel: source.mipLevel,
            origin: source.origin, size: size,
            buffer: buffer.raw, offset: destination.offset,
            bytesPerRow: bytesPerRow,
            // 한 슬라이스만 복사하므로 `rowsPerImage`는 쓰이지 않는다
            // (`docs/COMMAND-STREAM.md`의 알려진 차이).
            bytesPerImage: bytesPerRow * format.blockRows(height: size.height)
        )
    }

    private func copyBufferToTexture(_ command: WGPUCopyBufferToTextureCommand) throws {
        let source = command.source
        let destination = command.destination
        let buffer = try unmappedBuffer(source.buffer, path: source.fieldPath("buffer"))
        let texture = try self.texture(destination.texture, path: destination.fieldPath("texture"))
        let size = command.copySize
        let format = texture.format
        let bytesPerRow = source.bytesPerRow ?? format.bytesPerRow(width: size.width)
        if specValidation {
            try validateBlockAlignment(format: format, origin: destination.origin, size: size,
                                       textureSize: texture.size, mipLevel: destination.mipLevel,
                                       label: "copyBufferToTexture")
            try validateTexelCopyBufferLayout(
                bytesPerRow: source.bytesPerRow, format: format, size: size,
                label: "copyBufferToTexture", path: source.fieldPath("bytesPerRow")
            )
        }
        try requireNoOpenPass()
        try backend.copyBufferToTexture(
            buffer: buffer.raw, offset: source.offset,
            bytesPerRow: bytesPerRow,
            bytesPerImage: bytesPerRow * format.blockRows(height: size.height),
            texture: texture.raw, slice: destination.origin.z, mipLevel: destination.mipLevel,
            origin: destination.origin, size: size
        )
    }

    private func copyTextureToTexture(_ command: WGPUCopyTextureToTextureCommand) throws {
        let source = try texture(command.source.texture, path: command.source.fieldPath("texture"))
        let destination = try texture(
            command.destination.texture, path: command.destination.fieldPath("texture")
        )
        try requireNoOpenPass()
        try backend.copyTextureToTexture(
            source: source.raw,
            sourceSlice: command.source.origin.z,
            sourceMipLevel: command.source.mipLevel,
            sourceOrigin: command.source.origin,
            destination: destination.raw,
            destinationSlice: command.destination.origin.z,
            destinationMipLevel: command.destination.mipLevel,
            destinationOrigin: command.destination.origin,
            size: command.copySize
        )
    }

    /// 쿼리 결과를 버퍼로 내린다.
    private func resolveQuerySet(_ command: WGPUResolveQuerySetCommand) throws {
        let querySet = try registry.lookup(
            command.querySet, as: WGPUEngineQuerySet<B>.self, kind: "GPUQuerySet",
            path: command.fieldPath("querySet")
        )
        let destination = try unmappedBuffer(command.destination, path: command.fieldPath("destination"))
        let first = command.firstQuery
        // 생략하면 쿼리셋의 남은 전부 — 쿼리셋 크기를 알아야 하므로 여기서 채운다.
        let count = command.queryCount ?? (querySet.count - first)
        let offset = command.destinationOffset
        if specValidation {
            try querySet.checkRange(first: first, count: count, path: command.fieldPath("firstQuery"))

            // 명세가 요구하는 정렬. Metal은 이보다 느슨해서 여기서 안 막으면 브라우저에서만 깨진다.
            guard offset >= 0, offset % 256 == 0 else {
                throw WGPUError.validation(
                    "destinationOffset은 256의 배수여야 한다 (받은 값 \(offset))",
                    path: command.fieldPath("destinationOffset")
                )
            }
            let byteCount = count * WGPUEngineQuerySet<B>.resultStride
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
        }
        if count == 0 { return }   // no-op (0바이트 복사는 백엔드가 거부한다)
        // 음수 count는 넘긴다 — Metal 경로는 위 checkRange가 먼저 던졌고,
        // 네이티브 검증 경로는 백엔드의 안전 변환(dawnU32)이 validation으로 거른다.
        try requireNoOpenPass()
        try backend.resolveQuerySet(
            querySet.raw, first: first, count: count,
            destination: destination.raw, destinationOffset: offset
        )
    }

    // MARK: - 디버그 마커

    private func pushDebugGroup(_ command: WGPUPushDebugGroupCommand) throws {
        if passState != nil {
            try backend.pushDebugGroup(command.groupLabel, scope: .pass)
            encoderDebugDepth += 1
        } else {
            try backend.pushDebugGroup(command.groupLabel, scope: .frame)
            bufferDebugDepth += 1
        }
    }

    /// 짝이 맞지 않는 `pop`은 **Metal이 단언으로 프로세스를 죽인다.** 그래서 깊이를 세어
    /// 여기서 막는다 — 명세도 이 경우를 오류로 정하므로 동작이 같고, 앱은 살아남는다.
    /// 네이티브 검증 백엔드는 죽지 않고 오류를 내므로 그대로 흘려보낸다.
    private func popDebugGroup() {
        if passState != nil {
            guard specValidation else {
                encoderDebugDepth = max(0, encoderDebugDepth - 1)
                backend.popDebugGroup(scope: .pass)
                return
            }
            guard encoderDebugDepth > 0 else {
                record(.validation("popDebugGroup: 짝이 맞는 pushDebugGroup이 없다 (패스 안)"))
                return
            }
            encoderDebugDepth -= 1
            backend.popDebugGroup(scope: .pass)
        } else if backend.hasPendingWork {
            guard specValidation else {
                bufferDebugDepth = max(0, bufferDebugDepth - 1)
                backend.popDebugGroup(scope: .frame)
                return
            }
            guard bufferDebugDepth > 0 else {
                record(.validation("popDebugGroup: 짝이 맞는 pushDebugGroup이 없다"))
                return
            }
            bufferDebugDepth -= 1
            backend.popDebugGroup(scope: .frame)
        }
    }

    // MARK: - WebGPURuntime: 조회·비동기

    public func adapterInfo() -> [String: Any] {
        executionLock.lock()
        defer { executionLock.unlock() }
        return backend.adapterInfo()
    }

    /// `GPUShaderModule.getCompilationInfo()` — 그 모듈의 컴파일 진단.
    public func shaderCompilationInfo(handle: Int) -> [String: Any] {
        executionLock.lock()
        defer { executionLock.unlock() }

        guard let module = try? registry.lookup(
            WGPUHandle(handle), as: WGPUEngineShaderModule<B>.self, kind: "GPUShaderModule"
        ) else {
            return ["ok": false, "errors": [
                WGPUError.validation("GPUShaderModule #\(handle)이(가) 없다").payload,
            ]]
        }
        let messages = backend.compilationMessages(of: module.raw).map { message -> [String: Any] in
            [
                "message": message.message,
                "type": message.type,
                "lineNum": message.lineNum,
                "linePos": message.linePos,
                "offset": message.offset,
                "length": message.length,
            ]
        }
        return ["ok": true, "messages": messages]
    }

    /// 버퍼 내용을 읽는다 (`GPUBuffer.mapAsync` + `getMappedRange`에 해당).
    ///
    /// 읽는 동안 이 버퍼는 "unavailable"이다 — 다음 프레임의 쓰기가 같은 메모리에 겹치면
    /// JS가 받는 값이 어느 프레임 것인지 보장되지 않는다 (`WGPUEngineBuffer.isMapped`).
    public func readBuffer(
        handle: Int,
        offset: Int = 0,
        size: Int? = nil,
        completion: @escaping ([String: Any]) -> Void
    ) {
        let target: WGPUEngineBuffer<B>
        do {
            target = try registry.lookup(WGPUHandle(handle), as: WGPUEngineBuffer<B>.self, kind: "GPUBuffer")
        } catch let error as WGPUError {
            completion(["ok": false, "errors": [error.payload]])
            return
        } catch {
            completion(["ok": false, "errors": [WGPUError.backend("\(error)").payload]])
            return
        }

        executionLock.lock()
        guard !target.isMapped else {
            executionLock.unlock()
            completion([
                "ok": false,
                "errors": [WGPUError.validation(
                    "GPUBuffer \(WGPUHandle(handle))은(는) 이미 매핑 중이다 (unmap()을 먼저 부를 것)"
                ).payload],
            ])
            return
        }
        target.isMapped = true

        let length = size ?? (target.size - offset)
        guard offset >= 0, length >= 0, offset + length <= target.size else {
            // 실패했으면 매핑을 세우지 않는다 — 명세도 실패한 mapAsync는 버퍼를 매핑하지 않는다.
            target.isMapped = false
            executionLock.unlock()
            completion([
                "ok": false,
                "errors": [WGPUError.validation(
                    "readBuffer 범위 초과 — offset \(offset) + \(length)B > 버퍼 크기 \(target.size)B"
                ).payload],
            ])
            return
        }

        // 앞서 제출한 GPU 작업 완료 대기는 백엔드 몫이다. 완료는 임의 스레드에서 오므로
        // 래퍼가 락을 다시 잡는다 — 등록 도중 동기로 와도 재귀 락이라 안전하다.
        noteReadbackStarted()
        backend.readBuffer(target.raw, offset: offset, length: length) { [weak self] result in
            guard let self else { return }
            self.executionLock.lock()
            defer { self.executionLock.unlock() }
            self.noteReadbackFinished()
            switch result {
            case .failure(let error):
                target.isMapped = false
                completion(["ok": false, "errors": [error.payload]])
            case .success(let data):
                // `Data`를 그대로 싣는다 — Lynx가 `NSData`를 JS의 `ArrayBuffer`로 바꿔 준다.
                // base64로 만들면 33% 팽창에 JS 쪽 디코딩 루프까지 붙는다.
                completion(["ok": true, "data": data, "byteLength": data.count])
            }
        }
        executionLock.unlock()
    }

    // MARK: - 리드백 자가 펌프

    /// 미결 리드백 수 — `executionLock` 아래에서만 읽고 쓴다.
    private var pendingReadbacks = 0
    private var readbackPumpRunning = false

    /// 완료가 `pumpEvents()`에서만 나오는 백엔드(Dawn)를 위해, 미결 리드백이 있는 동안
    /// 자가 펌프를 돌린다.
    ///
    /// 호스트의 프레임 티커는 **JS가 프레임 루프를 켠 씬에서만** 돈다. 애니메이션 없는
    /// 씬(정적 검사 화면)이 `mapAsync`를 걸면 아무도 펌프를 밟지 않아 완료가 영영 도착하지
    /// 않았다 — 오류가 아니라 **영원한 대기**라 화면에는 아무 일도 없다. `WebGPURuntime.
    /// processEvents` 문서가 "티커가 없는 구성에서도 완료는 도착해야 한다 — 자체 대기 수단을
    /// 갖출 것"이라 정한 자리의 이행이다. (Metal은 완료 핸들러가 스스로 도착하므로
    /// `needsEventPump`가 거짓이고, 이 경로 전체가 비용 없이 빠진다.)
    ///
    /// `executionLock` 아래에서 부른다.
    private func noteReadbackStarted() {
        guard backend.capabilities.needsEventPump else { return }
        pendingReadbacks += 1
        guard !readbackPumpRunning else { return }
        readbackPumpRunning = true
        Thread.detachNewThread { [weak self] in
            while true {
                guard let self else { return }
                self.executionLock.lock()
                let outstanding = self.pendingReadbacks
                if outstanding > 0 {
                    self.backend.pumpEvents()
                } else {
                    self.readbackPumpRunning = false
                }
                self.executionLock.unlock()
                if outstanding == 0 { return }
                // 락을 놓은 채 쉰다 — JS 스레드의 execute와 1ms 간격으로만 경쟁한다.
                usleep(1_000)
            }
        }
    }

    /// `executionLock` 아래에서 부른다 (완료 래퍼 안).
    private func noteReadbackFinished() {
        guard backend.capabilities.needsEventPump else { return }
        pendingReadbacks -= 1
    }

    /// 인코딩된 이미지를 풀어 `ImageBitmap` 자리의 객체로 등록한다 (JS `createImageBitmap`).
    ///
    /// **핸들은 JS가 발급한다** — 커맨드 스트림과 같은 규칙이다. 디코딩은 느리므로
    /// 백그라운드 큐에서 하고, 등록만 실행 락 안에서 한다.
    public func decodeImage(
        handle: Int,
        data: Data?,
        name: String?,
        options: WGPUImageDecodeOptions,
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

    // MARK: - WebGPURuntime: 캔버스

    private struct CanvasEntry {
        let raw: B.Surface
        let pacesFrames: Bool
    }

    private var canvases: [String: CanvasEntry] = [:]

    public func attachCanvas(identifier: String, layer: CAMetalLayer) {
        let creation = backend.makeLayerSurface(identifier: identifier, layer: layer)
        registerSurface(creation.surface, identifier: identifier, pacesFrames: creation.pacesFrames)
    }

    public func attachOffscreenCanvas(identifier: String, size: CGSize) throws {
        let creation = try backend.makeOffscreenSurface(identifier: identifier, size: size)
        registerSurface(creation.surface, identifier: identifier, pacesFrames: creation.pacesFrames)
    }

    /// 백엔드가 만든 표면을 직접 등록한다 — 커스텀 표면(테스트 더블 등)의 통로다.
    public func registerSurface(_ surface: B.Surface, identifier: String, pacesFrames: Bool) {
        canvasLock.lock()
        canvases[identifier] = CanvasEntry(raw: surface, pacesFrames: pacesFrames)
        canvasLock.unlock()
        // 드로어블 풀이 있는 표면만 페이싱 대상이다 — 오프스크린은 밀릴 일이 없다.
        if pacesFrames { frameCoordinator.track(canvas: identifier) }
    }

    public func detachCanvas(identifier: String) {
        canvasLock.lock()
        canvases.removeValue(forKey: identifier)
        canvasLock.unlock()
        // 죽은 캔버스의 카운터를 남겨 두면 그것이 영원히 프레임 틱을 막는다.
        frameCoordinator.forget(canvas: identifier)
    }

    public func resizeCanvas(identifier: String, drawableSize: CGSize) {
        guard let entry = surfaceEntry(for: identifier) else { return }
        backend.resizeSurface(entry.raw, size: drawableSize)
    }

    public func surface(for identifier: String) -> B.Surface? {
        surfaceEntry(for: identifier)?.raw
    }

    private func surfaceEntry(for identifier: String) -> CanvasEntry? {
        canvasLock.lock()
        defer { canvasLock.unlock() }
        return canvases[identifier]
    }

    public var registeredSurfaceIdentifiers: [String] {
        canvasLock.lock()
        defer { canvasLock.unlock() }
        return Array(canvases.keys).sorted()
    }

    public func canvasInfo(identifier: String) -> [String: Any] {
        guard let entry = surfaceEntry(for: identifier) else {
            return [
                "ok": false,
                "errors": [WGPUError.validation(
                    "캔버스 '\(identifier)'이(가) 없다 (등록된 것: "
                        + "\(registeredSurfaceIdentifiers.joined(separator: ", ")))"
                ).payload],
            ]
        }
        let report = backend.surfaceReport(entry.raw)
        return [
            "ok": true,
            "width": report.width,
            "height": report.height,
            "format": report.format.rawValue,
        ]
    }

    public func readCanvasPixels(identifier: String) throws -> WGPUPixelReadback {
        guard let entry = surfaceEntry(for: identifier) else {
            throw WGPUError.validation(
                "캔버스 '\(identifier)'은(는) 오프스크린 표면이 아니다 — 픽셀을 읽을 수 없다"
            )
        }
        return try backend.readPixels(entry.raw, identifier: identifier)
    }

    // MARK: - WebGPURuntime: 프레임·수명

    /// 등록된 모든 표면이 새 프레임을 받을 수 있는가 — 회계는 `frameCoordinator`가 한다.
    public var isReadyForNextFrame: Bool { frameCoordinator.isReadyForNextFrame }

    public func processEvents() {
        executionLock.lock()
        backend.pumpEvents()
        executionLock.unlock()
    }

    /// 모든 GPU 객체를 버린다 (페이지 이탈 등).
    public func reset() {
        executionLock.lock()
        registry.removeAll()
        errorScopes.discardAll()
        // 프레임 중간 상태도 함께 버린다 — 남겨 두면 다음 디바이스의 첫 프레임이
        // 죽은 드로어블을 present하려 든다.
        acquiredFrames.removeAll()
        frameScopedHandles.removeAll()
        _ = gpuFailures.drain()
        backend.reset()
        executionLock.unlock()
    }

    public var liveObjectCount: Int { registry.count }
}
