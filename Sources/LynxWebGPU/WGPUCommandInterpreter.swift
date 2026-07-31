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
    private var acquiredDrawables: [(handle: WGPUHandle, drawable: WGPUDrawable)] = []
    /// 이번 프레임 업로드에 쓴 스테이징 버퍼 — 커맨드 버퍼 완료 시 풀로 돌아간다.
    private var frameStagingBuffers: [MTLBuffer] = []
    /// 프레임이 끝나면 무효해지는 핸들 (드로어블 텍스처와 그 뷰).
    private var frameScopedHandles: [WGPUHandle] = []
    private var touchedCanvases: [String: WGPUSurface] = [:]
    private var errors: [WGPUError] = []

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
                errors.append(WGPUError(
                    kind: error.kind,
                    message: error.message,
                    path: error.path ?? "commands[\(index)].\(command.optionalString("op") ?? "?")"
                ))
            } catch {
                errors.append(.backend(error.localizedDescription, path: "commands[\(index)]"))
            }
        }

        finish()

        var result: [String: Any] = ["ok": errors.isEmpty, "commandCount": commands.count]
        if !errors.isEmpty {
            result["errors"] = errors.map(\.payload)
        }
        if !touchedCanvases.isEmpty {
            result["canvases"] = touchedCanvases.mapValues { surface in
                ["width": Int(surface.pixelSize.width), "height": Int(surface.pixelSize.height)]
            }
        }
        return result
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
        acquiredDrawables.removeAll()
        frameScopedHandles.removeAll()
        touchedCanvases.removeAll()
        errors.removeAll()
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
            commandBuffer.commit()
            lastCommittedBuffer = commandBuffer
        } else if !frameStagingBuffers.isEmpty {
            // 커밋할 커맨드 버퍼가 없으면 GPU가 이 버퍼들을 참조하지 않는다 — 바로 회수한다.
            stagingPool.recycle(frameStagingBuffers)
        }
        frameStagingBuffers.removeAll()
        // 드로어블 텍스처와 그 뷰는 이번 프레임에서만 유효하다 (WebGPU도 같은 규칙).
        for handle in frameScopedHandles {
            registry.remove(handle)
        }
        commandBuffer = nil
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
        case "createRenderPipeline": try createRenderPipeline(command)
        case "createComputePipeline": try createComputePipeline(command)
        case "getBindGroupLayout": try getBindGroupLayout(command)
        case "destroy": registry.remove(try command.requiredHandle("id"))

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
        case "setStencilReference": try requireRenderEncoder()
            .setStencilReferenceValue(UInt32(command.int("reference", default: 0)))
        case "draw": try draw(command)
        case "drawIndexed": try drawIndexed(command)

        // 컴퓨트 패스
        case "beginComputePass": try beginComputePass()
        case "dispatchWorkgroups": try dispatchWorkgroups(command)

        case "endPass": endActiveEncoders()

        // 복사
        case "copyBufferToBuffer": try copyBufferToBuffer(command)
        case "copyTextureToBuffer": try copyTextureToBuffer(command)
        case "copyBufferToTexture": try copyBufferToTexture(command)
        case "copyTextureToTexture": try copyTextureToTexture(command)

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
        acquiredDrawables.append((handle, drawable))
        frameScopedHandles.append(handle)
    }

    // MARK: - 렌더 패스

    private func beginRenderPass(_ command: WGPUValueReader) throws {
        endActiveEncoders()
        let descriptor = try WGPURenderPassDescriptor(from: command)
        let passDescriptor = MTLRenderPassDescriptor()

        for (index, attachment) in descriptor.colorAttachments.enumerated() {
            let view = try registry.lookup(
                attachment.view, as: WGPUTextureViewObject.self, kind: "GPUTextureView"
            )
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

        if let depth = descriptor.depthStencilAttachment {
            let view = try registry.lookup(depth.view, as: WGPUTextureViewObject.self, kind: "GPUTextureView")
            if view.format.hasDepth {
                let target = passDescriptor.depthAttachment!
                target.texture = view.texture
                target.loadAction = WGPUMetalMapping.loadAction(depth.depthLoadOp ?? .load)
                target.storeAction = WGPUMetalMapping.storeAction(depth.depthStoreOp ?? .store)
                target.clearDepth = depth.depthClearValue
            }
            if view.format.hasStencil {
                let target = passDescriptor.stencilAttachment!
                target.texture = view.texture
                target.loadAction = WGPUMetalMapping.loadAction(depth.stencilLoadOp ?? .load)
                target.storeAction = WGPUMetalMapping.storeAction(depth.stencilStoreOp ?? .store)
                target.clearStencil = UInt32(depth.stencilClearValue)
            }
        }

        guard let encoder = try activeCommandBuffer().makeRenderCommandEncoder(descriptor: passDescriptor) else {
            throw WGPUError.backend("MTLRenderCommandEncoder 생성 실패 — 어태치먼트 설정을 확인할 것")
        }
        if let label = descriptor.label { encoder.label = label }
        renderEncoder = encoder
        currentRenderPipeline = nil
        boundGroups.removeAll()
        dirtyGroups.removeAll()
        indexBinding = nil
    }

    private func setPipeline(_ command: WGPUValueReader) throws {
        let handle = try command.requiredHandle("pipeline")
        if let encoder = renderEncoder {
            let pipeline = try registry.lookup(
                handle, as: WGPURenderPipelineObject.self, kind: "GPURenderPipeline"
            )
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

    private func applyBindGroups() throws {
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

        for groupIndex in dirtyGroups.sorted() {
            guard let bound = boundGroups[groupIndex] else { continue }
            try apply(bound.group, at: groupIndex, dynamicOffsets: bound.offsets, layout: layout)
        }
        dirtyGroups.removeAll()

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
        let encoder = try requireRenderEncoder()
        let slot = try command.requiredInt("slot")
        guard slot >= 0, slot < WGSLMetalLimits.maxVertexBufferSlots else {
            throw WGPUError.validation("정점 버퍼 슬롯은 0~\(WGSLMetalLimits.maxVertexBufferSlots - 1) 범위다")
        }
        let buffer = try registry.lookup(
            try command.requiredHandle("buffer"), as: WGPUBufferObject.self, kind: "GPUBuffer"
        )
        encoder.setVertexBuffer(
            buffer.buffer,
            offset: command.int("offset", default: 0),
            index: WGSLMetalLimits.vertexBufferIndex(slot: slot)
        )
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
        try applyBindGroups()
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
        try applyBindGroups()
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

    // MARK: - 컴퓨트 패스

    private func beginComputePass() throws {
        endActiveEncoders()
        guard let encoder = try activeCommandBuffer().makeComputeCommandEncoder() else {
            throw WGPUError.backend("MTLComputeCommandEncoder 생성 실패")
        }
        computeEncoder = encoder
        currentComputePipeline = nil
        boundGroups.removeAll()
        dirtyGroups.removeAll()
    }

    private func dispatchWorkgroups(_ command: WGPUValueReader) throws {
        let encoder = try requireComputeEncoder()
        try applyBindGroups()
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
