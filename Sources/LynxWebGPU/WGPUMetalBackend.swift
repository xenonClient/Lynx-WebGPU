import Foundation
import CoreGraphics
import Metal
import QuartzCore
import LynxWebGPUCore
import LynxWebGPUShader

/// `WGPUBackend`의 **Metal 직접 구현** — 이 패키지가 기본으로 주는 백엔드.
///
/// 오케스트레이션(디코딩·검증·오류 스코프·프레임 수명·직렬화)은 전부
/// `WGPUBackendEngine`(Core)에 있다. 여기 남은 것은 검증이 끝난 값을 Metal 인코딩으로
/// 옮기는 일이다: 커맨드 버퍼/인코더 수명, 스테이징 풀 업로드, MSL 컴파일(파이프라인
/// 생성 시점 — `docs/ARCHITECTURE.md`), 인자 테이블 배정(`WGSLMetalLimits`),
/// `arrayLength()`용 버퍼 크기 표.
///
/// 모든 동사는 엔진의 실행 락 아래에서 불린다 (`WGPUBackend` 문서) — 이 타입 자체는
/// 락을 잡지 않는다.
public final class WGPUMetalBackend: WGPUBackend {
    public typealias Buffer = WGPUBufferObject
    public typealias Texture = WGPUTextureObject
    public typealias TextureView = WGPUTextureViewObject
    public typealias Sampler = WGPUSamplerObject
    public typealias ShaderModule = WGPUShaderModuleObject
    public typealias BindGroupLayout = WGPUBindGroupLayoutObject
    public typealias PipelineLayout = WGPUPipelineLayoutObject
    public typealias BindGroup = WGPUMetalBindGroup
    public typealias RenderPipeline = WGPURenderPipelineObject
    public typealias ComputePipeline = WGPUComputePipelineObject
    public typealias QuerySet = WGPUQuerySetObject
    /// Metal에는 렌더 번들에 대응하는 객체가 없다 (`MTLIndirectCommandBuffer`는 제약이 훨씬
    /// 크고 용도가 다르다) — 엔진이 record/replay로 대신한다.
    public typealias RenderBundle = Never
    public typealias Surface = WGPUSurface

    let device: MTLDevice
    let queue: MTLCommandQueue
    /// 업로드 스테이징 버퍼 재사용 풀 (writeBuffer/writeTexture 공용).
    let stagingPool: WGPUStagingPool

    // 배치 수명 상태 — beginBatch ~ submit 사이에서만 유효하다.
    private var commandBuffer: MTLCommandBuffer?
    private var renderEncoder: MTLRenderCommandEncoder?
    private var computeEncoder: MTLComputeCommandEncoder?
    private var blitEncoder: MTLBlitCommandEncoder?
    private var currentRenderPipeline: WGPURenderPipelineObject?
    private var currentComputePipeline: WGPUComputePipelineObject?
    /// Metal 버퍼 인덱스별 바인딩 크기 — `arrayLength()`가 이 표를 조회한다.
    /// 바인드 그룹을 적용할 때마다 갱신하고, 셰이더가 쓸 때만 인코더에 올린다.
    private var bufferSizes = [UInt32](repeating: 0, count: WGSLMetalLimits.maxBindGroupBuffers)
    /// 이번 프레임 업로드에 쓴 스테이징 버퍼 — 커맨드 버퍼 완료 시 풀로 돌아간다.
    private var frameStagingBuffers: [MTLBuffer] = []
    /// 이번 프레임에 획득한 드로어블 — `submit(present: true)`가 화면으로 보낸다.
    private var acquiredDrawables: [WGPUDrawable] = []
    /// 마지막으로 커밋한 커맨드 버퍼 — `readBuffer`가 GPU 완료를 기다릴 때 쓴다.
    private(set) var lastCommittedBuffer: MTLCommandBuffer?

    public init(device: MTLDevice, queue: MTLCommandQueue) {
        self.device = device
        self.queue = queue
        self.stagingPool = WGPUStagingPool(device: device)
    }

    // MARK: - 능력

    public var capabilities: WGPUBackendCapabilities {
        WGPUBackendCapabilities(
            supportsNativeRenderBundles: false,
            maxVertexBufferSlots: WGSLMetalLimits.maxVertexBufferSlots
        )
    }

    public func supportsTextureCompression(_ format: WGPUTextureFormat) -> Bool {
        WGPUDeviceCapability.supportsCompression(format, on: device)
    }

    /// 기기가 간접 인자를 지원하지 않으면 **여기서 막는다.** 그대로 Metal에 넘기면
    /// `MTLValidateFeatureSupport ... failed assertion`으로 프로세스가 죽어, 앱은
    /// 이유를 남기지도 못한다.
    public func ensureIndirectSupported() throws {
        guard WGPUDeviceCapability.supportsIndirectArguments(device) else {
            throw WGPUError.unsupported(
                "이 기기는 간접 드로우·디스패치 인자를 지원하지 않는다 (Metal이 Apple GPU family 3 "
                    + "이상을 요구한다). **iOS 시뮬레이터가 여기 해당한다** — 실기기(A12 이상)에서는 "
                    + "동작하므로, 직접 드로우로 대체하거나 실기기에서 확인할 것"
            )
        }
    }

    public func pumpEvents() {
        // Metal은 완료가 스스로 도착한다 (완료 핸들러) — 펌프할 것이 없다.
    }

    public func reset() {
        acquiredDrawables.removeAll()
        lastCommittedBuffer = nil
    }

    // MARK: - 배치 수명

    public func beginBatch() {
        // 앞 배치가 제출 없이 끝났다면(오류만 있던 배치 등) 남은 상태를 정리한다.
        if commandBuffer == nil, !frameStagingBuffers.isEmpty {
            stagingPool.recycle(frameStagingBuffers)
            frameStagingBuffers.removeAll()
        }
        commandBuffer = nil
        renderEncoder = nil
        computeEncoder = nil
        blitEncoder = nil
        currentRenderPipeline = nil
        currentComputePipeline = nil
    }

    public func collectBatchDiagnostics() -> [WGPUError] { [] }

    public var hasPendingWork: Bool { commandBuffer != nil }

    public func ensureSubmittableWork() {
        _ = try? activeCommandBuffer()
    }

    public func submit(present: Bool, onCompleted: @escaping (WGPUError?) -> Void) {
        guard let commandBuffer else { return }
        if present {
            for drawable in acquiredDrawables { drawable.present(with: commandBuffer) }
        }
        // 완료 핸들러는 commit 전에만 붙일 수 있다 (Metal 단언).
        if !frameStagingBuffers.isEmpty {
            let buffers = frameStagingBuffers
            let pool = stagingPool
            commandBuffer.addCompletedHandler { _ in pool.recycle(buffers) }
        }
        // GPU 측 실패(.outOfMemory / .timeout / .deviceRemoved 등)를 주워 담는다.
        commandBuffer.addCompletedHandler { buffer in
            onCompleted(buffer.status == .error ? Self.commandBufferError(buffer) : nil)
        }
        commandBuffer.commit()
        lastCommittedBuffer = commandBuffer
        if present { acquiredDrawables.removeAll() }
        frameStagingBuffers.removeAll()
        self.commandBuffer = nil
    }

    /// 실패한 커맨드 버퍼를 보고 가능한 오류로 바꾼다.
    static func commandBufferError(_ buffer: MTLCommandBuffer) -> WGPUError {
        .backend("GPU 작업이 실패했다: \(buffer.error?.localizedDescription ?? "원인 불명")")
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

    public func endPass() {
        renderEncoder?.endEncoding()
        renderEncoder = nil
        computeEncoder?.endEncoding()
        computeEncoder = nil
        blitEncoder?.endEncoding()
        blitEncoder = nil
    }

    /// 디버그 그룹·마커의 패스 스코프 대상 — blit은 내부 인코더라 제외한다
    /// (`WGPUDebugScope` — 스코프 판정 자체는 엔진이 한다).
    private var passEncoder: MTLCommandEncoder? {
        renderEncoder ?? computeEncoder
    }

    // MARK: - 리소스

    public func makeBuffer(_ descriptor: WGPUBufferDescriptor) throws -> WGPUBufferObject {
        try WGPUBufferObject(device: device, descriptor: descriptor)
    }

    public func writeBuffer(_ buffer: WGPUBufferObject, offset: Int, data: Data) throws {
        let staging = try makeStagingBuffer(data)
        // 직접 memcpy 하면 이전 프레임 GPU 작업과 경쟁한다. blit으로 큐에 순서를 태운다.
        let encoder = try activeBlitEncoder()
        encoder.copy(
            from: staging, sourceOffset: 0,
            to: buffer.buffer, destinationOffset: offset, size: data.count
        )
    }

    public func readBuffer(
        _ buffer: WGPUBufferObject, offset: Int, length: Int,
        deliver: @escaping (Result<Data, WGPUError>) -> Void
    ) {
        let finish = { (failed: MTLCommandBuffer?) in
            if let failed {
                // `.error`는 **완료가 아니라 실패**다. 성공 경로로 흘려 보내면 GPU 작업이 실패한
                // 버퍼의 내용을 그대로 읽어 성공으로 돌려주게 된다.
                deliver(.failure(Self.commandBufferError(failed)))
                return
            }
            deliver(.success(Data(
                bytes: buffer.buffer.contents().advanced(by: offset), count: length
            )))
        }

        // 제출한 작업이 아직 돌고 있으면 완료 후에 읽는다.
        // `addCompletedHandler`는 commit 이후에 붙일 수 없으므로(Metal 단언) 전용 큐에서 기다린다.
        let pending = lastCommittedBuffer
        if let pending, pending.status == .error {
            finish(pending)
            return
        }
        guard let pending, pending.status != .completed else {
            finish(nil)
            return
        }
        Self.readbackQueue.async {
            pending.waitUntilCompleted()
            finish(pending.status == .error ? pending : nil)
        }
    }

    /// GPU 완료를 기다리는 전용 큐 — JS 스레드를 막지 않는다.
    private static let readbackQueue = DispatchQueue(label: "org.lynxwebgpu.readback")

    public func makeTexture(_ descriptor: WGPUTextureDescriptor) throws -> WGPUTextureObject {
        try WGPUTextureObject(device: device, descriptor: descriptor)
    }

    public func writeTexture(
        _ texture: WGPUTextureObject, data: Data, origin: WGPUOrigin3D, size: WGPUExtent3D,
        mipLevel: Int, bytesPerRow: Int, rowsPerImage: Int
    ) throws {
        let blockRows = texture.format.blockRows(height: size.height)
        let bytesPerImage = bytesPerRow * max(rowsPerImage, blockRows)
        let layers = max(size.depthOrArrayLayers, 1)
        // 스테이징은 이미지 스트라이드 전체만큼 잡는다 — Metal 검증 레이어가 마지막 이미지도
        // bytesPerImage 범위로 계산하기 때문이다. 남는 꼬리는 텍스처로 복사되지 않는다.
        let staging = try makeStagingBuffer(data, minimumLength: bytesPerImage * layers)
        // writeBuffer와 같은 이유로 blit으로 큐에 순서를 태운다 — 앞선 렌더/복사와 직렬화된다.
        texture.encodeWrite(
            from: staging,
            origin: origin,
            size: size,
            mipLevel: mipLevel,
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

    public func makeTextureView(
        _ texture: WGPUTextureObject, descriptor: WGPUTextureViewDescriptor,
        format: WGPUTextureFormat
    ) throws -> WGPUTextureViewObject {
        try WGPUTextureViewObject(source: texture, descriptor: descriptor, drawable: nil)
    }

    public func makeSampler(_ descriptor: WGPUSamplerDescriptor) throws -> WGPUSamplerObject {
        try WGPUSamplerObject(device: device, descriptor: descriptor)
    }

    public func makeShaderModule(
        _ descriptor: WGPUShaderModuleDescriptor, fieldPath: (String) -> String?
    ) -> WGPUShaderModuleCreation<WGPUMetalBackend> {
        let object = WGPUShaderModuleObject(descriptor: descriptor)
        let failure = object.isValid ? nil : object.compilationMessages.first
        return WGPUShaderModuleCreation(module: object, failure: failure)
    }

    public func unmapBuffer(_ buffer: WGPUBufferObject) {
        // Metal 경로의 매핑은 와이어 상태뿐이다 (shared 메모리 직접 읽기) — 풀 것이 없다.
    }

    public func compilationMessages(of module: WGPUShaderModuleObject) -> [WGPUCompilationMessage] {
        module.compilationMessages.map { error in
            WGPUCompilationMessage(
                message: error.message,
                // 이 구현의 진단은 전부 오류다 — Metal 런타임 API가 경고를 따로 주지 않는다.
                type: "error",
                lineNum: error.line ?? 0
            )
        }
    }

    public func makeBindGroupLayout(_ entries: [WGPUBindGroupLayoutEntry]) throws -> WGPUBindGroupLayoutObject {
        WGPUBindGroupLayoutObject(entries: entries)
    }

    public func makePipelineLayout(_ groups: [WGPUBindGroupLayoutObject]) throws -> WGPUPipelineLayoutObject {
        try WGPUPipelineLayoutObject(groups: groups)
    }

    public func makeBindGroup(
        layout: WGPUBindGroupLayoutObject,
        entries: [WGPUResolvedBindGroupEntry<WGPUMetalBackend>]
    ) throws -> WGPUMetalBindGroup {
        let bindings = try entries.map { entry -> WGPUMetalBindGroup.Binding in
            // Metal 백엔드의 레이아웃은 항상 항목을 안다 (네이티브 파생 레이아웃이 없다).
            guard let layoutEntry = entry.layoutEntry else {
                throw WGPUError.backend("Metal 바인드 그룹에는 레이아웃 항목 정보가 필요하다")
            }
            let resource: WGPUResolvedBinding
            switch entry.resource {
            case .buffer(let buffer, let offset, let boundSize):
                resource = .buffer(buffer.buffer, offset: offset, boundSize: boundSize)
            case .sampler(let sampler):
                resource = .sampler(sampler.sampler)
            case .textureView(let view):
                resource = .texture(view.texture)
            }
            return WGPUMetalBindGroup.Binding(
                binding: entry.binding,
                visibility: layoutEntry.visibility,
                hasDynamicOffset: Self.hasDynamicOffset(layoutEntry),
                resource: resource
            )
        }
        return WGPUMetalBindGroup(bindings: bindings.sorted { $0.binding < $1.binding })
    }

    private static func hasDynamicOffset(_ entry: WGPUBindGroupLayoutEntry) -> Bool {
        if case .buffer(let buffer) = entry.layout { return buffer.hasDynamicOffset }
        return false
    }

    public func makeQuerySet(_ descriptor: WGPUQuerySetDescriptor) throws -> WGPUQuerySetObject {
        try WGPUQuerySetObject(device: device, descriptor: descriptor)
    }

    public func makeRenderPipeline(
        _ descriptor: WGPURenderPipelineDescriptor,
        vertexModule: WGPUShaderModuleObject, fragmentModule: WGPUShaderModuleObject?,
        layout: WGPUResolvedPipelineLayout<WGPUMetalBackend>,
        fieldPath: (String) -> String?
    ) throws -> WGPURenderPipelineCreation<WGPUMetalBackend> {
        var descriptor = descriptor
        // 명세의 "get the entry point" — 이름을 생략하면 그 스테이지의 유일한 진입점을 쓴다.
        // 여기서 한 번 확정해 두면 아래 계층은 전부 결정된 이름만 다룬다.
        descriptor.vertex.entryPoint = try vertexModule.resolveEntryPoint(
            descriptor.vertex.entryPoint, stage: .vertex, path: fieldPath("vertex.entryPoint")
        )
        if let fragmentModule, let fragment = descriptor.fragment {
            descriptor.fragment?.entryPoint = try fragmentModule.resolveEntryPoint(
                fragment.entryPoint, stage: .fragment, path: fieldPath("fragment.entryPoint")
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
        let resolved = try resolveLayout(layout, stages: stages)
        let pipeline = try WGPURenderPipelineObject(
            device: device, descriptor: descriptor, layout: resolved,
            vertexModule: vertexModule, fragmentModule: fragmentModule
        )
        return WGPURenderPipelineCreation(
            pipeline: pipeline,
            info: WGPURenderPipelineInfo(
                requiredGroups: resolved.requiredGroups,
                requiredVertexSlots: pipeline.requiredVertexSlots,
                writesDepth: pipeline.writesDepth,
                writesStencil: pipeline.writesStencil
            )
        )
    }

    public func makeComputePipeline(
        _ descriptor: WGPUComputePipelineDescriptor,
        module: WGPUShaderModuleObject,
        layout: WGPUResolvedPipelineLayout<WGPUMetalBackend>,
        fieldPath: (String) -> String?
    ) throws -> WGPUComputePipelineCreation<WGPUMetalBackend> {
        var descriptor = descriptor
        descriptor.entryPoint = try module.resolveEntryPoint(
            descriptor.entryPoint, stage: .compute, path: fieldPath("compute.entryPoint")
        )
        let resolved = try resolveLayout(layout, stages: [(module, [descriptor.entryPoint!])])
        let pipeline = try WGPUComputePipelineObject(
            device: device, descriptor: descriptor, layout: resolved, module: module
        )
        return WGPUComputePipelineCreation(
            pipeline: pipeline,
            info: WGPUComputePipelineInfo(requiredGroups: resolved.requiredGroups)
        )
    }

    /// 명시적 레이아웃이면 그대로, `"auto"`면 셰이더 선언에서 유도한다.
    private func resolveLayout(
        _ layout: WGPUResolvedPipelineLayout<WGPUMetalBackend>,
        stages: [(module: WGPUShaderModuleObject, entryPoints: [String])]
    ) throws -> WGPUPipelineLayoutObject {
        switch layout {
        case .explicit(let object):
            return object
        case .auto:
            return try WGPUPipelineLayoutObject(groups: WGPUPipelineLayoutResolver.derivedGroups(stages: stages))
        }
    }

    public func bindGroupLayout(
        of pipeline: WGPUResolvedPipeline<WGPUMetalBackend>, index: Int
    ) throws -> WGPUBindGroupLayoutCreation<WGPUMetalBackend>? {
        let layout: WGPUPipelineLayoutObject
        switch pipeline {
        case .render(let render): layout = render.layout
        case .compute(let compute): layout = compute.layout
        }
        guard let group = layout.group(at: index) else { return nil }
        return WGPUBindGroupLayoutCreation(layout: group, entries: group.entries)
    }

    public func makeRenderBundle(
        _ descriptor: WGPURenderBundleDescriptor, commands: [WGPUCommand],
        resolver: WGPUBundleResolver<WGPUMetalBackend>
    ) throws -> Never {
        // capabilities가 record/replay를 선언하므로 엔진은 이 동사를 부르지 않는다.
        throw WGPUError.backend("Metal 백엔드는 네이티브 렌더 번들이 없다 (엔진 record/replay 경로를 쓸 것)")
    }

    // MARK: - 표면

    public func makeLayerSurface(identifier: String, layer: CAMetalLayer) -> WGPUSurfaceCreation<WGPUMetalBackend> {
        let surface = WGPUMetalLayerSurface(identifier: identifier, layer: layer)
        return WGPUSurfaceCreation(surface: surface, pacesFrames: surface.pacesFrames)
    }

    public func makeOffscreenSurface(identifier: String, size: CGSize) throws -> WGPUSurfaceCreation<WGPUMetalBackend> {
        let surface = WGPUOffscreenSurface(identifier: identifier, size: size, device: device)
        return WGPUSurfaceCreation(surface: surface, pacesFrames: surface.pacesFrames)
    }

    public func configureSurface(_ surface: WGPUSurface, configuration: WGPUCanvasConfiguration) throws {
        try surface.configure(configuration, device: device)
    }

    public func resizeSurface(_ surface: WGPUSurface, size: CGSize) {
        surface.updateDrawableSize(size)
    }

    public func surfaceReport(_ surface: WGPUSurface) -> WGPUSurfaceReport {
        WGPUSurfaceReport(
            width: Int(surface.pixelSize.width),
            height: Int(surface.pixelSize.height),
            format: surface.configuredFormat
        )
    }

    public func acquireFrameTexture(_ surface: WGPUSurface) throws -> WGPUAcquiredSurfaceTexture<WGPUMetalBackend>? {
        guard let drawable = surface.nextDrawable() else { return nil }
        // 실제 드로어블 텍스처의 포맷을 쓴다 — 캔버스 설정 반영이 한 프레임 늦을 수 있기 때문.
        let format = WGPUMetalMapping.textureFormat(from: drawable.texture.pixelFormat)
            ?? surface.configuredFormat
        let texture = WGPUTextureObject(drawableTexture: drawable.texture, format: format)
        acquiredDrawables.append(drawable)
        return WGPUAcquiredSurfaceTexture(
            texture: texture,
            format: format,
            width: drawable.texture.width,
            height: drawable.texture.height,
            sampleCount: drawable.texture.sampleCount
        )
    }

    public func readPixels(_ surface: WGPUSurface, identifier: String) throws -> WGPUPixelReadback {
        guard let offscreen = surface as? WGPUOffscreenSurface else {
            throw WGPUError.validation(
                "캔버스 '\(identifier)'은(는) 오프스크린 표면이 아니다 — 픽셀을 읽을 수 없다"
            )
        }
        return try offscreen.readPixels(queue: queue)
    }

    // MARK: - 패스

    public func beginRenderPass(_ pass: WGPUResolvedRenderPass<WGPUMetalBackend>) throws {
        let passDescriptor = MTLRenderPassDescriptor()

        for (index, attachment) in pass.colorAttachments.enumerated() {
            let target = passDescriptor.colorAttachments[index]!
            target.texture = attachment.view.texture
            target.loadAction = WGPUMetalMapping.loadAction(attachment.loadOp)
            target.storeAction = WGPUMetalMapping.storeAction(attachment.storeOp)
            target.clearColor = MTLClearColor(
                red: attachment.clearValue.red,
                green: attachment.clearValue.green,
                blue: attachment.clearValue.blue,
                alpha: attachment.clearValue.alpha
            )
            if let resolve = attachment.resolveTarget {
                target.resolveTexture = resolve.texture
                target.storeAction = .multisampleResolve
            }
        }

        if let depth = pass.depthStencil {
            if depth.format.hasDepth {
                let target = passDescriptor.depthAttachment!
                target.texture = depth.view.texture
                // readOnly면 load/store op이 없다(디코딩에서 막는다) — 내용을 그대로
                // 읽고 그대로 남기는 조합이 된다.
                target.loadAction = WGPUMetalMapping.loadAction(depth.depthLoadOp ?? .load)
                target.storeAction = WGPUMetalMapping.storeAction(depth.depthStoreOp ?? .store)
                target.clearDepth = depth.depthClearValue
            }
            if depth.format.hasStencil {
                let target = passDescriptor.stencilAttachment!
                target.texture = depth.view.texture
                target.loadAction = WGPUMetalMapping.loadAction(depth.stencilLoadOp ?? .load)
                target.storeAction = WGPUMetalMapping.storeAction(depth.stencilStoreOp ?? .store)
                target.clearStencil = UInt32(truncatingIfNeeded: depth.stencilClearValue)
            }
        }

        if let querySet = pass.occlusionQuerySet {
            passDescriptor.visibilityResultBuffer = querySet.visibilityBuffer
        }

        if let writes = pass.timestampWrites {
            guard let counterBuffer = writes.querySet.counterBuffer else {
                throw WGPUError.backend("타임스탬프 샘플 버퍼가 없다")
            }
            let attachment = passDescriptor.sampleBufferAttachments[0]!
            attachment.sampleBuffer = counterBuffer
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
        if let label = pass.label { encoder.label = label }
        renderEncoder = encoder
        currentRenderPipeline = nil
    }

    public func beginComputePass(_ pass: WGPUResolvedComputePass<WGPUMetalBackend>) throws {
        let buffer = try activeCommandBuffer()

        let encoder: MTLComputeCommandEncoder?
        if let writes = pass.timestampWrites {
            guard let counterBuffer = writes.querySet.counterBuffer else {
                throw WGPUError.backend("타임스탬프 샘플 버퍼가 없다")
            }
            let passDescriptor = MTLComputePassDescriptor()
            let attachment = passDescriptor.sampleBufferAttachments[0]!
            attachment.sampleBuffer = counterBuffer
            attachment.startOfEncoderSampleIndex = writes.beginningOfPassWriteIndex ?? MTLCounterDontSample
            attachment.endOfEncoderSampleIndex = writes.endOfPassWriteIndex ?? MTLCounterDontSample
            encoder = buffer.makeComputeCommandEncoder(descriptor: passDescriptor)
        } else {
            encoder = buffer.makeComputeCommandEncoder()
        }
        guard let encoder else {
            throw WGPUError.backend("MTLComputeCommandEncoder 생성 실패")
        }
        if let label = pass.label { encoder.label = label }
        computeEncoder = encoder
        currentComputePipeline = nil
    }

    public func setRenderPipeline(_ pipeline: WGPURenderPipelineObject) {
        guard let encoder = renderEncoder else { return }
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
    }

    public func setComputePipeline(_ pipeline: WGPUComputePipelineObject) {
        guard let encoder = computeEncoder else { return }
        encoder.setComputePipelineState(pipeline.state)
        currentComputePipeline = pipeline
    }

    public func applyBindGroup(_ group: WGPUMetalBindGroup, at groupIndex: Int, dynamicOffsets: [Int]) throws {
        guard let layout = currentRenderPipeline?.layout ?? currentComputePipeline?.layout else { return }
        var offsetCursor = 0
        for binding in group.bindings {
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
                if binding.hasDynamicOffset {
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

    public func applyVertexBuffer(_ buffer: WGPUBufferObject, offset: Int, slot: Int) {
        renderEncoder?.setVertexBuffer(
            buffer.buffer,
            offset: offset,
            index: WGSLMetalLimits.vertexBufferIndex(slot: slot)
        )
    }

    public func setViewport(_ command: WGPUSetViewportCommand) throws {
        renderEncoder?.setViewport(MTLViewport(
            originX: command.x,
            originY: command.y,
            width: command.width,
            height: command.height,
            znear: command.minDepth,
            zfar: command.maxDepth
        ))
    }

    public func setScissorRect(_ command: WGPUSetScissorRectCommand) throws {
        renderEncoder?.setScissorRect(MTLScissorRect(
            x: command.x, y: command.y, width: command.width, height: command.height
        ))
    }

    public func setBlendConstant(_ color: WGPUColor) throws {
        renderEncoder?.setBlendColor(
            red: Float(color.red), green: Float(color.green), blue: Float(color.blue), alpha: Float(color.alpha)
        )
    }

    public func setStencilReference(_ reference: UInt32) throws {
        renderEncoder?.setStencilReferenceValue(reference)
    }

    public func beginOcclusionQuery(index: Int) throws {
        // `.counting`은 통과한 **샘플 수**를 센다 — 명세의 occlusion 결과와 같은 뜻이다.
        renderEncoder?.setVisibilityResultMode(.counting, offset: index * WGPUQuerySetObject.resultStride)
    }

    public func endOcclusionQuery(index: Int) throws {
        renderEncoder?.setVisibilityResultMode(.disabled, offset: index * WGPUQuerySetObject.resultStride)
    }

    public func executeBundles(_ bundles: [Never]) throws {}

    public func pushDebugGroup(_ label: String, scope: WGPUDebugScope) throws {
        switch scope {
        case .pass:
            passEncoder?.pushDebugGroup(label)
        case .frame:
            // 아직 커맨드 버퍼가 없으면 만든다. 그래야 뒤따르는 pop과 짝이 맞는다.
            try activeCommandBuffer().pushDebugGroup(label)
        }
    }

    public func popDebugGroup(scope: WGPUDebugScope) {
        switch scope {
        case .pass:
            passEncoder?.popDebugGroup()
        case .frame:
            commandBuffer?.popDebugGroup()
        }
    }

    public func popFrameDebugGroups(count: Int) {
        guard let commandBuffer else { return }
        for _ in 0..<count { commandBuffer.popDebugGroup() }
    }

    public func insertDebugMarker(_ label: String, scope: WGPUDebugScope) throws {
        switch scope {
        case .pass:
            // 인코더에는 signpost(점 이벤트)가 있다.
            passEncoder?.insertDebugSignpost(label)
        case .frame:
            // 커맨드 버퍼에는 그룹밖에 없어 여닫아 흉내 낸다.
            let buffer = try activeCommandBuffer()
            buffer.pushDebugGroup(label)
            buffer.popDebugGroup()
        }
    }

    // MARK: - 드로우 / 디스패치

    /// `arrayLength()`용 버퍼 크기 표. 88바이트라 setBytes로 매 드로우 올려도 부담이 없다.
    private func uploadBufferSizesIfNeeded() {
        let needsSizes = renderEncoder != nil
            ? (currentRenderPipeline?.needsBufferSizes ?? false)
            : (currentComputePipeline?.needsBufferSizes ?? false)
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

    public func draw(_ command: WGPUDrawCommand) throws {
        guard let encoder = renderEncoder, let pipeline = currentRenderPipeline else {
            throw WGPUError.backend("렌더 패스/파이프라인 없이 draw가 내려왔다")
        }
        uploadBufferSizesIfNeeded()
        encoder.drawPrimitives(
            type: pipeline.primitiveType,
            vertexStart: command.firstVertex,
            vertexCount: command.vertexCount,
            instanceCount: command.instanceCount,
            baseInstance: command.firstInstance
        )
    }

    public func drawIndexed(_ command: WGPUDrawIndexedCommand, index: WGPUResolvedIndexBinding<WGPUMetalBackend>) throws {
        guard let encoder = renderEncoder, let pipeline = currentRenderPipeline else {
            throw WGPUError.backend("렌더 패스/파이프라인 없이 drawIndexed가 내려왔다")
        }
        uploadBufferSizesIfNeeded()
        encoder.drawIndexedPrimitives(
            type: pipeline.primitiveType,
            indexCount: command.indexCount,
            indexType: WGPUMetalMapping.indexType(index.format),
            indexBuffer: index.buffer.buffer,
            indexBufferOffset: index.offset + command.firstIndex * index.stride,
            instanceCount: command.instanceCount,
            baseVertex: command.baseVertex,
            baseInstance: command.firstInstance
        )
    }

    public func drawIndirect(buffer: WGPUBufferObject, offset: Int) throws {
        guard let encoder = renderEncoder, let pipeline = currentRenderPipeline else {
            throw WGPUError.backend("렌더 패스/파이프라인 없이 drawIndirect가 내려왔다")
        }
        uploadBufferSizesIfNeeded()
        encoder.drawPrimitives(
            type: pipeline.primitiveType,
            indirectBuffer: buffer.buffer,
            indirectBufferOffset: offset
        )
    }

    public func drawIndexedIndirect(
        buffer: WGPUBufferObject, offset: Int, index: WGPUResolvedIndexBinding<WGPUMetalBackend>
    ) throws {
        guard let encoder = renderEncoder, let pipeline = currentRenderPipeline else {
            throw WGPUError.backend("렌더 패스/파이프라인 없이 drawIndexedIndirect가 내려왔다")
        }
        uploadBufferSizesIfNeeded()
        encoder.drawIndexedPrimitives(
            type: pipeline.primitiveType,
            indexType: WGPUMetalMapping.indexType(index.format),
            indexBuffer: index.buffer.buffer,
            // 직접 경로(`drawIndexed`)와 달리 `firstIndex`를 여기 더하지 않는다 —
            // 그 값은 인자 버퍼 안에 있고 GPU가 읽는다. 더하면 두 번 세어 조용히 틀린다.
            indexBufferOffset: index.offset,
            indirectBuffer: buffer.buffer,
            indirectBufferOffset: offset
        )
    }

    public func dispatchWorkgroups(_ command: WGPUDispatchWorkgroupsCommand) throws {
        guard let encoder = computeEncoder, let pipeline = currentComputePipeline else {
            throw WGPUError.backend("컴퓨트 패스/파이프라인 없이 dispatchWorkgroups가 내려왔다")
        }
        uploadBufferSizesIfNeeded()
        encoder.dispatchThreadgroups(
            MTLSize(width: command.x, height: command.y, depth: command.z),
            threadsPerThreadgroup: pipeline.threadsPerThreadgroup
        )
    }

    public func dispatchWorkgroupsIndirect(buffer: WGPUBufferObject, offset: Int) throws {
        guard let encoder = computeEncoder, let pipeline = currentComputePipeline else {
            throw WGPUError.backend("컴퓨트 패스/파이프라인 없이 dispatchWorkgroupsIndirect가 내려왔다")
        }
        uploadBufferSizesIfNeeded()
        encoder.dispatchThreadgroups(
            indirectBuffer: buffer.buffer,
            indirectBufferOffset: offset,
            threadsPerThreadgroup: pipeline.threadsPerThreadgroup
        )
    }

    // MARK: - 복사

    public func copyBufferToBuffer(
        source: WGPUBufferObject, sourceOffset: Int,
        destination: WGPUBufferObject, destinationOffset: Int, size: Int
    ) throws {
        let encoder = try activeBlitEncoder()
        encoder.copy(
            from: source.buffer, sourceOffset: sourceOffset,
            to: destination.buffer, destinationOffset: destinationOffset,
            size: size
        )
    }

    public func clearBuffer(_ buffer: WGPUBufferObject, range: Range<Int>) throws {
        let encoder = try activeBlitEncoder()
        encoder.fill(buffer: buffer.buffer, range: range, value: 0)
    }

    public func copyTextureToBuffer(
        texture: WGPUTextureObject, slice: Int, mipLevel: Int, origin: WGPUOrigin3D,
        size: WGPUExtent3D, buffer: WGPUBufferObject, offset: Int,
        bytesPerRow: Int, bytesPerImage: Int
    ) throws {
        let encoder = try activeBlitEncoder()
        encoder.copy(
            from: texture.texture,
            sourceSlice: slice,
            sourceLevel: mipLevel,
            sourceOrigin: MTLOrigin(x: origin.x, y: origin.y, z: 0),
            sourceSize: MTLSize(width: size.width, height: size.height, depth: 1),
            to: buffer.buffer,
            destinationOffset: offset,
            destinationBytesPerRow: bytesPerRow,
            destinationBytesPerImage: bytesPerImage
        )
    }

    public func copyBufferToTexture(
        buffer: WGPUBufferObject, offset: Int, bytesPerRow: Int, bytesPerImage: Int,
        texture: WGPUTextureObject, slice: Int, mipLevel: Int, origin: WGPUOrigin3D,
        size: WGPUExtent3D
    ) throws {
        let encoder = try activeBlitEncoder()
        encoder.copy(
            from: buffer.buffer,
            sourceOffset: offset,
            sourceBytesPerRow: bytesPerRow,
            sourceBytesPerImage: bytesPerImage,
            sourceSize: MTLSize(width: size.width, height: size.height, depth: 1),
            to: texture.texture,
            destinationSlice: slice,
            destinationLevel: mipLevel,
            destinationOrigin: MTLOrigin(x: origin.x, y: origin.y, z: 0)
        )
    }

    public func copyTextureToTexture(
        source: WGPUTextureObject, sourceSlice: Int, sourceMipLevel: Int,
        sourceOrigin: WGPUOrigin3D,
        destination: WGPUTextureObject, destinationSlice: Int, destinationMipLevel: Int,
        destinationOrigin: WGPUOrigin3D, size: WGPUExtent3D
    ) throws {
        let encoder = try activeBlitEncoder()
        encoder.copy(
            from: source.texture,
            sourceSlice: sourceSlice,
            sourceLevel: sourceMipLevel,
            sourceOrigin: MTLOrigin(x: sourceOrigin.x, y: sourceOrigin.y, z: 0),
            sourceSize: MTLSize(width: size.width, height: size.height, depth: 1),
            to: destination.texture,
            destinationSlice: destinationSlice,
            destinationLevel: destinationMipLevel,
            destinationOrigin: MTLOrigin(x: destinationOrigin.x, y: destinationOrigin.y, z: 0)
        )
    }

    /// 쿼리 종류마다 blit 명령이 다르다 (`WGPUQuerySetObject` 문서).
    public func resolveQuerySet(
        _ querySet: WGPUQuerySetObject, first: Int, count: Int,
        destination: WGPUBufferObject, destinationOffset: Int
    ) throws {
        let encoder = try activeBlitEncoder()
        switch querySet.type {
        case .occlusion:
            guard let source = querySet.visibilityBuffer else {
                throw WGPUError.backend("occlusion 쿼리 버퍼가 없다")
            }
            encoder.copy(
                from: source, sourceOffset: first * WGPUQuerySetObject.resultStride,
                to: destination.buffer, destinationOffset: destinationOffset,
                size: count * WGPUQuerySetObject.resultStride
            )
        case .timestamp:
            guard let source = querySet.counterBuffer else {
                throw WGPUError.backend("타임스탬프 샘플 버퍼가 없다")
            }
            // 카운터는 평범한 버퍼가 아니라 전용 저장소라 resolveCounters로만 꺼낼 수 있다.
            encoder.resolveCounters(
                source, range: first..<(first + count),
                destinationBuffer: destination.buffer, destinationOffset: destinationOffset
            )
        }
    }

    // MARK: - 어댑터 정보

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
}

/// Metal 백엔드의 `GPUBindGroup` — 생성 시점에 이미 Metal 객체로 풀린 바인딩 목록.
///
/// 핸들 해석·레이아웃 매칭·`boundSize` 기본값은 엔진이 끝냈고, 여기에는 드로우 직전
/// `applyBindGroup`이 인코더에 올릴 값만 남는다.
public final class WGPUMetalBindGroup {
    struct Binding {
        let binding: Int
        let visibility: WGPUShaderStage
        let hasDynamicOffset: Bool
        let resource: WGPUResolvedBinding
    }

    let bindings: [Binding]

    init(bindings: [Binding]) {
        self.bindings = bindings
    }
}
