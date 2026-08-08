import Foundation
import CoreGraphics
import QuartzCore
import ImageIO
import WebGPU
import LynxWebGPUCore

/// `WebGPURuntime`의 **Dawn 구현** — `docs/extra/DAWN-BACKEND-REVIEW.md`가 그린 절단면 A의 실물.
///
/// Core가 공유하는 것 위에 선다: 디스패치는 `WGPUCommand`, 오류 스코프는 `WGPUErrorScopeStack`,
/// 응답 모양은 `WGPUBatchResult`, 지연 오류는 `WGPUDeferredErrorQueue`, 프레임 경계는
/// `WGPUFrameBoundary`. 이 파일이 다시 쓰는 것은 정말로 **인코딩**(Core 값 → Dawn C 호출)뿐이다.
///
/// ## Dawn 오류를 와이어 모델로 옮기는 방법
///
/// - 핸들·상태 오류(없는 핸들, 패스 밖 드로우)는 **우리 레지스트리/상태 검사**가 경로 붙은
///   validation으로 만든다 — Metal 런타임과 같은 자리다.
/// - Dawn 자체 검증 오류는 배치 전체를 `wgpuDevicePushErrorScope`(validation·out-of-memory)로
///   감싸 **같은 배치 결과**에 싣는다 (경로는 없다 — 명세 문서가 "선택"으로 두는 자리).
/// - 스코프 밖(uncaptured) 오류는 `WGPUDeferredErrorQueue`로 **다음 배치**에 실린다 —
///   계약 그대로 (`docs/COMMAND-STREAM.md` §2).
///
/// ## 시뮬레이터 주의
///
/// 어댑터 기능을 **하나도 광고하지 않는다.** 간접 드로우는 시뮬레이터 Metal(family 2)에서
/// 단언으로 죽는 경로라 (`CLAUDE.md`), 광고하지 않아 적합성의 간접 검사가 건너뛰게 한다.
final class DawnWebGPURuntime: WebGPURuntime {
    private let instance: WGPUInstance
    private let adapter: WGPUAdapter
    private let device: WGPUDevice
    private let queue: WGPUQueue

    private let registry = WGPUObjectRegistry()
    private let executionLock = NSLock()
    private let deferredErrors = WGPUDeferredErrorQueue()
    private var errorScopes = WGPUErrorScopeStack()

    private var canvases: [String: DawnOffscreenCanvas] = [:]

    // 배치 수명 상태 (execute 하나 동안)
    private var commandEncoder: WGPUCommandEncoder?
    private var renderPass: WGPURenderPassEncoder?
    private var computePass: WGPUComputePassEncoder?
    private var errors: [WGPUError] = []
    private var poppedScopes: [WGPUPoppedErrorScope] = []
    private var touchedCanvases: [String: DawnOffscreenCanvas] = [:]

    // 프레임 수명 상태 (present까지)
    private var frameScopedHandles: [WGPUHandle] = []
    private var acquiredCanvases: Set<String> = []

    init() throws {
        guard let instance = wgpuCreateInstance(nil) else {
            throw DawnBootstrapError("wgpuCreateInstance 실패")
        }
        self.instance = instance
        self.adapter = try DawnBootstrap.requestAdapter(instance: instance)
        let sink = deferredErrors
        self.device = try DawnBootstrap.requestDevice(
            instance: instance, adapter: adapter,
            onUncapturedError: { type, message in
                sink.report(DawnEnum.errorType(type, message: message))
            }
        )
        guard let queue = wgpuDeviceGetQueue(device) else {
            throw DawnBootstrapError("wgpuDeviceGetQueue 실패")
        }
        self.queue = queue
    }

    deinit {
        registry.removeAll()
        canvases.removeAll()
        wgpuQueueRelease(queue)
        wgpuDeviceRelease(device)
        wgpuAdapterRelease(adapter)
        wgpuInstanceRelease(instance)
    }

    // MARK: - 커맨드 스트림

    func execute(_ payload: [String: Any]) -> [String: Any] {
        executionLock.lock()
        defer { executionLock.unlock() }

        let reader = WGPUValueReader(payload)
        let commands: [WGPUValueReader]
        do {
            commands = try reader.requiredObjects("commands")
        } catch let error as WGPUError {
            return WGPUBatchResult.failure([error])
        } catch {
            return WGPUBatchResult.failure([.validation("\(error)")])
        }
        let present = reader.bool("present", default: true)

        resetBatchState()
        for failure in deferredErrors.drain() { record(failure) }

        // Dawn 검증 오류를 이 배치 결과에 싣는 배치 스코프 — pop은 제출 뒤에 한다.
        wgpuDevicePushErrorScope(device, WGPUErrorFilter_OutOfMemory)
        wgpuDevicePushErrorScope(device, WGPUErrorFilter_Validation)

        for (index, commandReader) in commands.enumerated() {
            do {
                try dispatch(try WGPUCommand(from: commandReader))
            } catch let error as WGPUError {
                record(WGPUError(
                    kind: error.kind,
                    message: error.message,
                    path: error.path
                        ?? "commands[\(index)].\(commandReader.optionalString("op") ?? "?")",
                    line: error.line
                ))
            } catch {
                record(.backend("\(error)", path: "commands[\(index)]"))
            }
        }

        endOpenPasses()
        if let encoder = commandEncoder {
            if let commandBuffer = wgpuCommandEncoderFinish(encoder, nil) {
                var submission: WGPUCommandBuffer? = commandBuffer
                wgpuQueueSubmit(queue, 1, &submission)
                wgpuCommandBufferRelease(commandBuffer)
            }
            wgpuCommandEncoderRelease(encoder)
            commandEncoder = nil
        }

        // 배치 스코프 회수 — 안쪽(validation)부터. 동기 펌프라 이 배치 결과에 실린다.
        drainDeviceScope()
        drainDeviceScope()

        // 프레임 경계 — 정책은 Core의 값 타입이 정하고 여기는 적용만 한다.
        let boundary = WGPUFrameBoundary(requestedPresent: present, commandCount: commands.count)
        if boundary.presents, !acquiredCanvases.isEmpty {
            for handle in frameScopedHandles { registry.remove(handle) }
            frameScopedHandles.removeAll()
            acquiredCanvases.removeAll()
        }

        return WGPUBatchResult(
            commandCount: commands.count,
            liveObjectCount: registry.count,
            errors: errors,
            canvases: touchedCanvases.mapValues { canvas in
                WGPUCanvasReport(width: Int(canvas.size.width), height: Int(canvas.size.height))
            },
            poppedScopes: poppedScopes
        ).payload
    }

    private func resetBatchState() {
        commandEncoder = nil
        renderPass = nil
        computePass = nil
        errors.removeAll()
        poppedScopes.removeAll()
        touchedCanvases.removeAll()
        // frameScopedHandles·acquiredCanvases·errorScopes는 비우지 않는다 —
        // 프레임 경계는 배치가 아니라 present다 (Metal 해석기와 같은 규칙).
    }

    private func record(_ error: WGPUError) {
        if errorScopes.capture(error) { return }
        errors.append(error)
    }

    /// 디바이스 스코프 하나를 pop해 잡힌 오류를 이 배치에 기록한다 (동기 펌프).
    private func drainDeviceScope() {
        final class ScopeBox { var done = false; var error: WGPUError? }
        let box = ScopeBox()
        var callbackInfo = WGPUPopErrorScopeCallbackInfo()
        callbackInfo.mode = WGPUCallbackMode_AllowProcessEvents
        callbackInfo.callback = { _, type, message, userdata1, _ in
            let box = Unmanaged<ScopeBox>.fromOpaque(userdata1!).takeRetainedValue()
            if type != WGPUErrorType_NoError {
                box.error = DawnEnum.errorType(type, message: String(wgpu: message))
            }
            box.done = true
        }
        callbackInfo.userdata1 = Unmanaged.passRetained(box).toOpaque()
        _ = wgpuDevicePopErrorScope(device, callbackInfo)
        try? DawnBootstrap.pump(instance: instance, until: { box.done }, what: "popErrorScope")
        if let error = box.error { record(error) }
    }

    // MARK: - 디스패치 (Core의 표에 대한 exhaustive switch)

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
            errorScopes.push(filter)
            if let decodeFailure { throw decodeFailure }
        case .popErrorScope:
            poppedScopes.append(errorScopes.pop())

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
            wgpuRenderPassEncoderSetViewport(
                try requireRenderPass(), Float(c.x), Float(c.y),
                Float(c.width), Float(c.height), Float(c.minDepth), Float(c.maxDepth)
            )
        case .setScissorRect(let c):
            wgpuRenderPassEncoderSetScissorRect(
                try requireRenderPass(), UInt32(c.x), UInt32(c.y), UInt32(c.width), UInt32(c.height)
            )
        case .setBlendConstant(let c):
            var color = DawnEnum.color(c.color)
            wgpuRenderPassEncoderSetBlendConstant(try requireRenderPass(), &color)
        case .setStencilReference(let c):
            wgpuRenderPassEncoderSetStencilReference(try requireRenderPass(), c.reference)
        case .draw(let c):
            wgpuRenderPassEncoderDraw(
                try requireRenderPass(), UInt32(c.vertexCount), UInt32(c.instanceCount),
                UInt32(c.firstVertex), UInt32(c.firstInstance)
            )
        case .drawIndexed(let c):
            wgpuRenderPassEncoderDrawIndexed(
                try requireRenderPass(), UInt32(c.indexCount), UInt32(c.instanceCount),
                UInt32(c.firstIndex), Int32(c.baseVertex), UInt32(c.firstInstance)
            )
        case .drawIndirect(let c):
            let buffer = try lookupBuffer(c.indirectBuffer, path: c.fieldPath("indirectBuffer"))
            wgpuRenderPassEncoderDrawIndirect(
                try requireRenderPass(), buffer.buffer, UInt64(c.indirectOffset)
            )
        case .drawIndexedIndirect(let c):
            let buffer = try lookupBuffer(c.indirectBuffer, path: c.fieldPath("indirectBuffer"))
            wgpuRenderPassEncoderDrawIndexedIndirect(
                try requireRenderPass(), buffer.buffer, UInt64(c.indirectOffset)
            )
        case .executeBundles(let c): try executeBundles(c)
        case .beginOcclusionQuery(let c):
            wgpuRenderPassEncoderBeginOcclusionQuery(try requireRenderPass(), UInt32(c.queryIndex))
        case .endOcclusionQuery:
            wgpuRenderPassEncoderEndOcclusionQuery(try requireRenderPass())

        // 컴퓨트 패스
        case .beginComputePass: try beginComputePass()
        case .dispatchWorkgroups(let c):
            wgpuComputePassEncoderDispatchWorkgroups(
                try requireComputePass(), UInt32(c.x), UInt32(c.y), UInt32(c.z)
            )
        case .dispatchWorkgroupsIndirect(let c):
            let buffer = try lookupBuffer(c.indirectBuffer, path: c.fieldPath("indirectBuffer"))
            wgpuComputePassEncoderDispatchWorkgroupsIndirect(
                try requireComputePass(), buffer.buffer, UInt64(c.indirectOffset)
            )

        case .endPass: endOpenPasses()

        // 복사
        case .copyBufferToBuffer(let c): try copyBufferToBuffer(c)
        case .clearBuffer(let c): try clearBuffer(c)
        case .copyTextureToBuffer(let c): try copyTextureToBuffer(c)
        case .copyBufferToTexture(let c): try copyBufferToTexture(c)
        case .copyTextureToTexture(let c): try copyTextureToTexture(c)

        // 쿼리
        case .resolveQuerySet(let c): try resolveQuerySet(c)

        // 디버그 마커
        case .pushDebugGroup(let c): try pushDebugGroup(c.groupLabel)
        case .popDebugGroup: popDebugGroupOnOpenScope()
        case .insertDebugMarker(let c): try insertDebugMarker(c.markerLabel)
        }
    }

    // MARK: - 인코더 수명

    private func ensureEncoder() -> WGPUCommandEncoder {
        if let commandEncoder { return commandEncoder }
        let encoder = wgpuDeviceCreateCommandEncoder(device, nil)!
        commandEncoder = encoder
        return encoder
    }

    private func requireRenderPass() throws -> WGPURenderPassEncoder {
        guard let renderPass else {
            throw WGPUError.validation("렌더 패스가 시작되지 않았다 (beginRenderPass 먼저)")
        }
        return renderPass
    }

    private func requireComputePass() throws -> WGPUComputePassEncoder {
        guard let computePass else {
            throw WGPUError.validation("컴퓨트 패스가 시작되지 않았다 (beginComputePass 먼저)")
        }
        return computePass
    }

    private func endOpenPasses() {
        if let renderPass {
            wgpuRenderPassEncoderEnd(renderPass)
            wgpuRenderPassEncoderRelease(renderPass)
            self.renderPass = nil
        }
        if let computePass {
            wgpuComputePassEncoderEnd(computePass)
            wgpuComputePassEncoderRelease(computePass)
            self.computePass = nil
        }
    }

    // MARK: - 레지스트리 조회

    private func lookupBuffer(_ handle: WGPUHandle, path: String? = nil) throws -> DawnBufferObject {
        try registry.lookup(handle, as: DawnBufferObject.self, kind: "GPUBuffer", path: path)
    }

    /// 버퍼를 쓰는 큐·복사 명령은 이 경로로 — 매핑 중이면 거부한다 (명세의 unavailable).
    private func unmappedBuffer(_ handle: WGPUHandle, path: String? = nil) throws -> DawnBufferObject {
        let buffer = try lookupBuffer(handle, path: path)
        guard !buffer.isMapped else {
            throw WGPUError.validation(
                "GPUBuffer \(handle)은(는) 매핑 중이라 GPU 작업에 쓸 수 없다 (unmap 먼저)", path: path
            )
        }
        return buffer
    }

    // MARK: - 리소스 생성

    private func createBuffer(_ command: WGPUCreateCommand<LynxWebGPUCore.WGPUBufferDescriptor>) throws {
        let descriptor = command.descriptor
        let arena = DawnArena()
        var dawnDescriptor = WebGPU.WGPUBufferDescriptor()
        dawnDescriptor.label = arena.string(descriptor.label)
        dawnDescriptor.size = UInt64(descriptor.size)
        dawnDescriptor.usage = WGPUBufferUsage(UInt64(descriptor.usage.rawValue))
        let needsInitialData = descriptor.initialData != nil
        dawnDescriptor.mappedAtCreation = (descriptor.mappedAtCreation || needsInitialData) ? 1 : 0

        guard let buffer = wgpuDeviceCreateBuffer(device, &dawnDescriptor) else {
            throw WGPUError.outOfMemory("GPUBuffer 생성 실패", path: command.fieldPath("size"))
        }
        if let data = descriptor.initialData, !data.isEmpty {
            if let mapped = wgpuBufferGetMappedRange(buffer, 0, data.count) {
                data.withUnsafeBytes { source in
                    mapped.copyMemory(from: source.baseAddress!, byteCount: data.count)
                }
            }
        }
        let object = DawnBufferObject(buffer: buffer, size: descriptor.size, usage: descriptor.usage)
        if descriptor.mappedAtCreation {
            object.isMapped = true
            object.dawnMapped = true
        } else if needsInitialData {
            wgpuBufferUnmap(buffer)
        }
        registry.insert(object, at: command.id)
    }

    private func writeBuffer(_ command: WGPUWriteBufferCommand) throws {
        let buffer = try unmappedBuffer(command.buffer, path: command.fieldPath("buffer"))
        command.data.withUnsafeBytes { source in
            wgpuQueueWriteBuffer(
                queue, buffer.buffer, UInt64(command.bufferOffset),
                source.baseAddress, source.count
            )
        }
    }

    private func unmapBuffer(_ command: WGPUUnmapBufferCommand) throws {
        let buffer = try lookupBuffer(command.buffer, path: command.fieldPath("buffer"))
        if buffer.dawnMapped {
            wgpuBufferUnmap(buffer.buffer)
            buffer.dawnMapped = false
        }
        buffer.isMapped = false
    }

    private func createTexture(_ command: WGPUCreateCommand<LynxWebGPUCore.WGPUTextureDescriptor>) throws {
        let descriptor = command.descriptor
        let arena = DawnArena()
        var dawnDescriptor = WebGPU.WGPUTextureDescriptor()
        dawnDescriptor.label = arena.string(descriptor.label)
        dawnDescriptor.usage = WGPUTextureUsage(UInt64(descriptor.usage.rawValue))
        dawnDescriptor.dimension = DawnEnum.textureDimension(descriptor.dimension)
        dawnDescriptor.size = DawnEnum.extent(descriptor.size)
        dawnDescriptor.format = try DawnEnum.textureFormat(descriptor.format)
        dawnDescriptor.mipLevelCount = UInt32(descriptor.mipLevelCount)
        dawnDescriptor.sampleCount = UInt32(descriptor.sampleCount)
        guard let texture = wgpuDeviceCreateTexture(device, &dawnDescriptor) else {
            throw WGPUError.outOfMemory("GPUTexture 생성 실패", path: command.path)
        }
        registry.insert(
            DawnTextureObject(
                texture: texture, format: descriptor.format,
                width: descriptor.size.width, height: descriptor.size.height
            ),
            at: command.id
        )
    }

    private func writeTexture(_ command: WGPUWriteTextureCommand) throws {
        let texture = try registry.lookup(
            command.texture, as: DawnTextureObject.self, kind: "GPUTexture",
            path: command.fieldPath("texture")
        )
        let bytesPerPixel = texture.format.bytesPerPixel
        let bytesPerRow = command.bytesPerRow ?? (command.size.width * bytesPerPixel)
        let rowsPerImage = command.rowsPerImage ?? command.size.height

        var destination = WGPUTexelCopyTextureInfo()
        destination.texture = texture.texture
        destination.mipLevel = UInt32(command.mipLevel)
        destination.origin = DawnEnum.origin(command.origin)
        destination.aspect = WGPUTextureAspect_All

        var layout = WGPUTexelCopyBufferLayout()
        layout.offset = 0
        layout.bytesPerRow = UInt32(bytesPerRow)
        layout.rowsPerImage = UInt32(rowsPerImage)

        var size = DawnEnum.extent(command.size)
        command.data.withUnsafeBytes { source in
            wgpuQueueWriteTexture(
                queue, &destination, source.baseAddress, source.count, &layout, &size
            )
        }
    }

    private func createTextureView(_ command: WGPUCreateTextureViewCommand) throws {
        let texture = try registry.lookup(
            command.texture, as: DawnTextureObject.self, kind: "GPUTexture",
            path: command.fieldPath("texture")
        )
        let descriptor = command.descriptor
        let arena = DawnArena()
        var dawnDescriptor = WebGPU.WGPUTextureViewDescriptor()
        dawnDescriptor.label = arena.string(descriptor.label)
        dawnDescriptor.format = try descriptor.format.map(DawnEnum.textureFormat)
            ?? WGPUTextureFormat_Undefined
        dawnDescriptor.dimension = DawnEnum.viewDimension(descriptor.dimension)
        dawnDescriptor.baseMipLevel = UInt32(descriptor.baseMipLevel)
        dawnDescriptor.mipLevelCount = descriptor.mipLevelCount.map(UInt32.init) ?? UInt32.max
        dawnDescriptor.baseArrayLayer = UInt32(descriptor.baseArrayLayer)
        dawnDescriptor.arrayLayerCount = descriptor.arrayLayerCount.map(UInt32.init) ?? UInt32.max
        dawnDescriptor.aspect = DawnEnum.aspect(descriptor.aspect)
        guard let view = wgpuTextureCreateView(texture.texture, &dawnDescriptor) else {
            throw WGPUError.backend("GPUTextureView 생성 실패", path: command.path)
        }
        registry.insert(DawnTextureViewObject(view: view), at: command.id)
        // 드로어블 텍스처의 뷰는 텍스처와 함께 프레임 스코프다 — 뷰만 살아남으면 present 뒤에도
        // 그 프레임의 스왑체인에 그릴 수 있게 되어 만료 계약이 뚫린다 (Metal 해석기와 같은 규칙).
        if frameScopedHandles.contains(command.texture) {
            frameScopedHandles.append(command.id)
        }
    }

    private func createSampler(_ command: WGPUCreateCommand<LynxWebGPUCore.WGPUSamplerDescriptor>) throws {
        let descriptor = command.descriptor
        let arena = DawnArena()
        var dawnDescriptor = WebGPU.WGPUSamplerDescriptor()
        dawnDescriptor.label = arena.string(descriptor.label)
        dawnDescriptor.addressModeU = DawnEnum.addressMode(descriptor.addressModeU)
        dawnDescriptor.addressModeV = DawnEnum.addressMode(descriptor.addressModeV)
        dawnDescriptor.addressModeW = DawnEnum.addressMode(descriptor.addressModeW)
        dawnDescriptor.magFilter = DawnEnum.filter(descriptor.magFilter)
        dawnDescriptor.minFilter = DawnEnum.filter(descriptor.minFilter)
        dawnDescriptor.mipmapFilter = DawnEnum.mipmapFilter(descriptor.mipmapFilter)
        dawnDescriptor.lodMinClamp = Float(descriptor.lodMinClamp)
        dawnDescriptor.lodMaxClamp = Float(descriptor.lodMaxClamp)
        dawnDescriptor.compare = descriptor.compare.map(DawnEnum.compare) ?? WGPUCompareFunction_Undefined
        dawnDescriptor.maxAnisotropy = UInt16(descriptor.maxAnisotropy)
        guard let sampler = wgpuDeviceCreateSampler(device, &dawnDescriptor) else {
            throw WGPUError.backend("GPUSampler 생성 실패", path: command.path)
        }
        registry.insert(DawnSamplerObject(sampler: sampler), at: command.id)
    }

    private func createShaderModule(
        _ command: WGPUCreateCommand<LynxWebGPUCore.WGPUShaderModuleDescriptor>
    ) throws {
        let descriptor = command.descriptor
        guard descriptor.language == .wgsl else {
            // msl은 Metal 런타임의 탈출구다 — 선택 기능이라 깨끗이 거부한다
            // (`docs/COMMAND-STREAM.md` §4-1, 적합성 `msl-optional`).
            throw WGPUError.unsupported(
                "Dawn 런타임은 language \"\(descriptor.language.rawValue)\"을(를) 지원하지 않는다 (wgsl만)",
                path: command.fieldPath("language")
            )
        }
        let arena = DawnArena()
        var source = WGPUShaderSourceWGSL()
        source.chain.sType = WGPUSType_ShaderSourceWGSL
        source.code = arena.string(descriptor.code)
        let module: WGPUShaderModule? = withUnsafeMutablePointer(to: &source) { sourcePointer in
            var dawnDescriptor = WebGPU.WGPUShaderModuleDescriptor()
            dawnDescriptor.label = arena.string(descriptor.label)
            dawnDescriptor.nextInChain = UnsafeMutableRawPointer(sourcePointer)
                .assumingMemoryBound(to: WGPUChainedStruct.self)
            return wgpuDeviceCreateShaderModule(device, &dawnDescriptor)
        }
        guard let module else {
            throw WGPUError.backend("GPUShaderModule 생성 실패", path: command.path)
        }
        registry.insert(DawnShaderModuleObject(module: module), at: command.id)
    }

    private func createBindGroupLayout(
        _ command: WGPUCreateCommand<LynxWebGPUCore.WGPUBindGroupLayoutDescriptor>
    ) throws {
        let arena = DawnArena()
        let entries = command.descriptor.entries.map { entry -> WebGPU.WGPUBindGroupLayoutEntry in
            var dawnEntry = WebGPU.WGPUBindGroupLayoutEntry()
            dawnEntry.binding = UInt32(entry.binding)
            dawnEntry.visibility = WGPUShaderStage(UInt64(entry.visibility.rawValue))
            switch entry.layout {
            case .buffer(let buffer):
                dawnEntry.buffer.type = DawnEnum.bufferBindingType(buffer.type)
                dawnEntry.buffer.hasDynamicOffset = buffer.hasDynamicOffset ? 1 : 0
                dawnEntry.buffer.minBindingSize = UInt64(buffer.minBindingSize)
            case .sampler(let sampler):
                dawnEntry.sampler.type = DawnEnum.samplerBindingType(sampler.type)
            case .texture(let texture):
                dawnEntry.texture.sampleType = DawnEnum.sampleType(texture.sampleType)
                dawnEntry.texture.viewDimension = DawnEnum.viewDimension(texture.viewDimension)
                dawnEntry.texture.multisampled = texture.multisampled ? 1 : 0
            case .storageTexture(let storage):
                dawnEntry.storageTexture.access = DawnEnum.storageAccess(storage.access)
                dawnEntry.storageTexture.format = (try? DawnEnum.textureFormat(storage.format))
                    ?? WGPUTextureFormat_Undefined
                dawnEntry.storageTexture.viewDimension = DawnEnum.viewDimension(storage.viewDimension)
            }
            return dawnEntry
        }
        var dawnDescriptor = WebGPU.WGPUBindGroupLayoutDescriptor()
        dawnDescriptor.label = arena.string(command.descriptor.label)
        dawnDescriptor.entryCount = entries.count
        dawnDescriptor.entries = arena.array(entries)
        guard let layout = wgpuDeviceCreateBindGroupLayout(device, &dawnDescriptor) else {
            throw WGPUError.backend("GPUBindGroupLayout 생성 실패", path: command.path)
        }
        registry.insert(DawnBindGroupLayoutObject(layout: layout), at: command.id)
    }

    private func createPipelineLayout(
        _ command: WGPUCreateCommand<LynxWebGPUCore.WGPUPipelineLayoutDescriptor>
    ) throws {
        let arena = DawnArena()
        let layouts = try command.descriptor.bindGroupLayouts.map { handle in
            try registry.lookup(
                handle, as: DawnBindGroupLayoutObject.self, kind: "GPUBindGroupLayout",
                path: command.fieldPath("bindGroupLayouts")
            ).layout as WGPUBindGroupLayout?
        }
        var dawnDescriptor = WebGPU.WGPUPipelineLayoutDescriptor()
        dawnDescriptor.label = arena.string(command.descriptor.label)
        dawnDescriptor.bindGroupLayoutCount = layouts.count
        dawnDescriptor.bindGroupLayouts = arena.array(layouts)
        guard let layout = wgpuDeviceCreatePipelineLayout(device, &dawnDescriptor) else {
            throw WGPUError.backend("GPUPipelineLayout 생성 실패", path: command.path)
        }
        registry.insert(DawnPipelineLayoutObject(layout: layout), at: command.id)
    }

    private func createBindGroup(
        _ command: WGPUCreateCommand<LynxWebGPUCore.WGPUBindGroupDescriptor>
    ) throws {
        let layout = try registry.lookup(
            command.descriptor.layout, as: DawnBindGroupLayoutObject.self,
            kind: "GPUBindGroupLayout", path: command.fieldPath("layout")
        )
        let arena = DawnArena()
        let entries = try command.descriptor.entries.map { entry -> WebGPU.WGPUBindGroupEntry in
            var dawnEntry = WebGPU.WGPUBindGroupEntry()
            dawnEntry.binding = UInt32(entry.binding)
            switch entry.resource {
            case .buffer(let handle, let offset, let size):
                let buffer = try lookupBuffer(handle, path: command.fieldPath("entries"))
                dawnEntry.buffer = buffer.buffer
                dawnEntry.offset = UInt64(offset)
                dawnEntry.size = size.map(UInt64.init) ?? UInt64.max
            case .sampler(let handle):
                dawnEntry.sampler = try registry.lookup(
                    handle, as: DawnSamplerObject.self, kind: "GPUSampler",
                    path: command.fieldPath("entries")
                ).sampler
            case .textureView(let handle):
                dawnEntry.textureView = try registry.lookup(
                    handle, as: DawnTextureViewObject.self, kind: "GPUTextureView",
                    path: command.fieldPath("entries")
                ).view
            }
            return dawnEntry
        }
        var dawnDescriptor = WebGPU.WGPUBindGroupDescriptor()
        dawnDescriptor.label = arena.string(command.descriptor.label)
        dawnDescriptor.layout = layout.layout
        dawnDescriptor.entryCount = entries.count
        dawnDescriptor.entries = arena.array(entries)
        guard let group = wgpuDeviceCreateBindGroup(device, &dawnDescriptor) else {
            throw WGPUError.backend("GPUBindGroup 생성 실패", path: command.path)
        }
        registry.insert(DawnBindGroupObject(group: group), at: command.id)
    }

    private func createQuerySet(
        _ command: WGPUCreateCommand<LynxWebGPUCore.WGPUQuerySetDescriptor>
    ) throws {
        let descriptor = command.descriptor
        let arena = DawnArena()
        var dawnDescriptor = WebGPU.WGPUQuerySetDescriptor()
        dawnDescriptor.label = arena.string(descriptor.label)
        dawnDescriptor.type = DawnEnum.queryType(descriptor.type)
        dawnDescriptor.count = UInt32(descriptor.count)
        guard let querySet = wgpuDeviceCreateQuerySet(device, &dawnDescriptor) else {
            throw WGPUError.backend("GPUQuerySet 생성 실패", path: command.path)
        }
        registry.insert(
            DawnQuerySetObject(querySet: querySet, type: descriptor.type, count: descriptor.count),
            at: command.id
        )
    }

    // MARK: - 파이프라인

    private func resolveLayout(
        _ reference: WGPUPipelineLayoutRef, path: String
    ) throws -> WGPUPipelineLayout? {
        switch reference {
        case .auto:
            return nil   // Dawn의 auto 레이아웃 — 명세와 같은 의미다
        case .explicit(let handle):
            return try registry.lookup(
                handle, as: DawnPipelineLayoutObject.self, kind: "GPUPipelineLayout", path: path
            ).layout
        }
    }

    private func constantEntries(
        _ constants: [String: Double], arena: DawnArena
    ) -> [WGPUConstantEntry] {
        constants.map { key, value in
            var entry = WGPUConstantEntry()
            entry.key = arena.string(key)
            entry.value = value
            return entry
        }
    }

    private func createRenderPipeline(
        _ command: WGPUCreateCommand<LynxWebGPUCore.WGPURenderPipelineDescriptor>
    ) throws {
        let descriptor = command.descriptor
        let arena = DawnArena()
        var dawnDescriptor = WebGPU.WGPURenderPipelineDescriptor()
        dawnDescriptor.label = arena.string(descriptor.label)
        dawnDescriptor.layout = try resolveLayout(descriptor.layout, path: command.fieldPath("layout"))

        // vertex
        let vertexModule = try registry.lookup(
            descriptor.vertex.module, as: DawnShaderModuleObject.self, kind: "GPUShaderModule",
            path: command.fieldPath("vertex.module")
        )
        var vertex = WGPUVertexState()
        vertex.module = vertexModule.module
        vertex.entryPoint = arena.string(descriptor.vertex.entryPoint)
        let vertexConstants = constantEntries(descriptor.vertex.constants, arena: arena)
        vertex.constantCount = vertexConstants.count
        vertex.constants = arena.array(vertexConstants)
        let vertexBuffers = try descriptor.vertex.buffers.map { layout -> WebGPU.WGPUVertexBufferLayout in
            var dawnLayout = WebGPU.WGPUVertexBufferLayout()
            dawnLayout.stepMode = DawnEnum.stepMode(layout.stepMode)
            dawnLayout.arrayStride = UInt64(layout.arrayStride)
            let attributes = try layout.attributes.map { attribute -> WebGPU.WGPUVertexAttribute in
                var dawnAttribute = WebGPU.WGPUVertexAttribute()
                dawnAttribute.format = try DawnEnum.vertexFormat(attribute.format)
                dawnAttribute.offset = UInt64(attribute.offset)
                dawnAttribute.shaderLocation = UInt32(attribute.shaderLocation)
                return dawnAttribute
            }
            dawnLayout.attributeCount = attributes.count
            dawnLayout.attributes = arena.array(attributes)
            return dawnLayout
        }
        vertex.bufferCount = vertexBuffers.count
        vertex.buffers = arena.array(vertexBuffers)
        dawnDescriptor.vertex = vertex

        // primitive
        var primitive = WGPUPrimitiveState()
        primitive.topology = DawnEnum.topology(descriptor.primitive.topology)
        primitive.stripIndexFormat = descriptor.primitive.stripIndexFormat.map(DawnEnum.indexFormat)
            ?? WGPUIndexFormat_Undefined
        primitive.frontFace = DawnEnum.frontFace(descriptor.primitive.frontFace)
        primitive.cullMode = DawnEnum.cullMode(descriptor.primitive.cullMode)
        dawnDescriptor.primitive = primitive

        // depthStencil
        if let depthStencil = descriptor.depthStencil {
            var dawnDepthStencil = WebGPU.WGPUDepthStencilState()
            dawnDepthStencil.format = try DawnEnum.textureFormat(depthStencil.format)
            dawnDepthStencil.depthWriteEnabled = depthStencil.depthWriteEnabled
                ? WGPUOptionalBool_True : WGPUOptionalBool_False
            dawnDepthStencil.depthCompare = DawnEnum.compare(depthStencil.depthCompare)
            dawnDepthStencil.stencilFront = stencilFace(depthStencil.stencilFront)
            dawnDepthStencil.stencilBack = stencilFace(depthStencil.stencilBack)
            dawnDepthStencil.stencilReadMask = UInt32(truncatingIfNeeded: depthStencil.stencilReadMask)
            dawnDepthStencil.stencilWriteMask = UInt32(truncatingIfNeeded: depthStencil.stencilWriteMask)
            dawnDepthStencil.depthBias = Int32(depthStencil.depthBias)
            dawnDepthStencil.depthBiasSlopeScale = Float(depthStencil.depthBiasSlopeScale)
            dawnDepthStencil.depthBiasClamp = Float(depthStencil.depthBiasClamp)
            dawnDescriptor.depthStencil = arena.value(dawnDepthStencil)
        }

        // multisample
        var multisample = WGPUMultisampleState()
        multisample.count = UInt32(descriptor.multisample.count)
        multisample.mask = UInt32(truncatingIfNeeded: descriptor.multisample.mask)
        multisample.alphaToCoverageEnabled = descriptor.multisample.alphaToCoverageEnabled ? 1 : 0
        dawnDescriptor.multisample = multisample

        // fragment
        if let fragment = descriptor.fragment {
            let fragmentModule = try registry.lookup(
                fragment.module, as: DawnShaderModuleObject.self, kind: "GPUShaderModule",
                path: command.fieldPath("fragment.module")
            )
            var dawnFragment = WGPUFragmentState()
            dawnFragment.module = fragmentModule.module
            dawnFragment.entryPoint = arena.string(fragment.entryPoint)
            let fragmentConstants = constantEntries(fragment.constants, arena: arena)
            dawnFragment.constantCount = fragmentConstants.count
            dawnFragment.constants = arena.array(fragmentConstants)
            let targets = try fragment.targets.map { target -> WebGPU.WGPUColorTargetState in
                var dawnTarget = WebGPU.WGPUColorTargetState()
                dawnTarget.format = try DawnEnum.textureFormat(target.format)
                dawnTarget.writeMask = WGPUColorWriteMask(UInt64(target.writeMask.rawValue))
                if let blend = target.blend {
                    var dawnBlend = WebGPU.WGPUBlendState()
                    dawnBlend.color = blendComponent(blend.color)
                    dawnBlend.alpha = blendComponent(blend.alpha)
                    dawnTarget.blend = arena.value(dawnBlend)
                }
                return dawnTarget
            }
            dawnFragment.targetCount = targets.count
            dawnFragment.targets = arena.array(targets)
            dawnDescriptor.fragment = arena.value(dawnFragment)
        }

        guard let pipeline = wgpuDeviceCreateRenderPipeline(device, &dawnDescriptor) else {
            throw WGPUError.backend("GPURenderPipeline 생성 실패", path: command.path)
        }
        registry.insert(DawnRenderPipelineObject(pipeline: pipeline), at: command.id)
    }

    private func stencilFace(
        _ face: LynxWebGPUCore.WGPUStencilFaceState
    ) -> WebGPU.WGPUStencilFaceState {
        var dawnFace = WebGPU.WGPUStencilFaceState()
        dawnFace.compare = DawnEnum.compare(face.compare)
        dawnFace.failOp = DawnEnum.stencilOperation(face.failOp)
        dawnFace.depthFailOp = DawnEnum.stencilOperation(face.depthFailOp)
        dawnFace.passOp = DawnEnum.stencilOperation(face.passOp)
        return dawnFace
    }

    private func blendComponent(
        _ component: LynxWebGPUCore.WGPUBlendComponent
    ) -> WebGPU.WGPUBlendComponent {
        var dawnComponent = WebGPU.WGPUBlendComponent()
        dawnComponent.operation = DawnEnum.blendOperation(component.operation)
        dawnComponent.srcFactor = DawnEnum.blendFactor(component.srcFactor)
        dawnComponent.dstFactor = DawnEnum.blendFactor(component.dstFactor)
        return dawnComponent
    }

    private func createComputePipeline(
        _ command: WGPUCreateCommand<LynxWebGPUCore.WGPUComputePipelineDescriptor>
    ) throws {
        let descriptor = command.descriptor
        let module = try registry.lookup(
            descriptor.module, as: DawnShaderModuleObject.self, kind: "GPUShaderModule",
            path: command.fieldPath("compute.module")
        )
        let arena = DawnArena()
        var dawnDescriptor = WebGPU.WGPUComputePipelineDescriptor()
        dawnDescriptor.label = arena.string(descriptor.label)
        dawnDescriptor.layout = try resolveLayout(descriptor.layout, path: command.fieldPath("layout"))
        var compute = WGPUComputeState()
        compute.module = module.module
        compute.entryPoint = arena.string(descriptor.entryPoint)
        let constants = constantEntries(descriptor.constants, arena: arena)
        compute.constantCount = constants.count
        compute.constants = arena.array(constants)
        dawnDescriptor.compute = compute
        guard let pipeline = wgpuDeviceCreateComputePipeline(device, &dawnDescriptor) else {
            throw WGPUError.backend("GPUComputePipeline 생성 실패", path: command.path)
        }
        registry.insert(DawnComputePipelineObject(pipeline: pipeline), at: command.id)
    }

    private func getBindGroupLayout(_ command: WGPUGetBindGroupLayoutCommand) throws {
        let layout: WGPUBindGroupLayout?
        if let render = try? registry.lookup(
            command.pipeline, as: DawnRenderPipelineObject.self, kind: "GPURenderPipeline"
        ) {
            layout = wgpuRenderPipelineGetBindGroupLayout(render.pipeline, UInt32(command.index))
        } else {
            let compute = try registry.lookup(
                command.pipeline, as: DawnComputePipelineObject.self, kind: "GPUPipeline",
                path: command.fieldPath("pipeline")
            )
            layout = wgpuComputePipelineGetBindGroupLayout(compute.pipeline, UInt32(command.index))
        }
        guard let layout else {
            throw WGPUError.validation("getBindGroupLayout 실패", path: command.path)
        }
        registry.insert(DawnBindGroupLayoutObject(layout: layout), at: command.id)
    }

    // MARK: - 렌더 번들

    private func createRenderBundle(_ command: WGPUCreateRenderBundleCommand) throws {
        let descriptor = command.descriptor
        let arena = DawnArena()
        var dawnDescriptor = WGPURenderBundleEncoderDescriptor()
        dawnDescriptor.label = arena.string(descriptor.label)
        let formats = try descriptor.colorFormats.map { format -> WebGPU.WGPUTextureFormat in
            guard let format else { return WGPUTextureFormat_Undefined }   // 빈 슬롯
            return try DawnEnum.textureFormat(format)
        }
        dawnDescriptor.colorFormatCount = formats.count
        dawnDescriptor.colorFormats = arena.array(formats)
        dawnDescriptor.depthStencilFormat = try descriptor.depthStencilFormat
            .map(DawnEnum.textureFormat) ?? WGPUTextureFormat_Undefined
        dawnDescriptor.sampleCount = UInt32(descriptor.sampleCount)
        dawnDescriptor.depthReadOnly = descriptor.depthReadOnly ? 1 : 0
        dawnDescriptor.stencilReadOnly = descriptor.stencilReadOnly ? 1 : 0

        guard let encoder = wgpuDeviceCreateRenderBundleEncoder(device, &dawnDescriptor) else {
            throw WGPUError.backend("GPURenderBundleEncoder 생성 실패", path: command.path)
        }
        defer { wgpuRenderBundleEncoderRelease(encoder) }

        // 저장된 리더를 재생 시점 규칙 그대로 디코딩해 **진짜 Dawn 번들**로 굽는다 —
        // Metal 런타임의 명령 재생과 달리 여기는 명세의 렌더 번들 그 자체다.
        for reader in command.commands {
            let bundleCommand = try WGPUCommand(from: reader)
            switch bundleCommand {
            case .setPipeline(let c):
                let pipeline = try registry.lookup(
                    c.pipeline, as: DawnRenderPipelineObject.self, kind: "GPURenderPipeline",
                    path: c.fieldPath("pipeline")
                )
                wgpuRenderBundleEncoderSetPipeline(encoder, pipeline.pipeline)
            case .setBindGroup(let c):
                let group = try registry.lookup(
                    c.bindGroup, as: DawnBindGroupObject.self, kind: "GPUBindGroup",
                    path: c.fieldPath("bindGroup")
                )
                let offsets = c.dynamicOffsets.map(UInt32.init)
                offsets.withUnsafeBufferPointer { pointer in
                    wgpuRenderBundleEncoderSetBindGroup(
                        encoder, UInt32(c.index), group.group, pointer.count, pointer.baseAddress
                    )
                }
            case .setVertexBuffer(let c):
                let buffer = try unmappedBuffer(c.buffer, path: c.fieldPath("buffer"))
                wgpuRenderBundleEncoderSetVertexBuffer(
                    encoder, UInt32(c.slot), buffer.buffer, UInt64(c.offset), UInt64.max
                )
            case .setIndexBuffer(let c):
                let buffer = try unmappedBuffer(c.buffer, path: c.fieldPath("buffer"))
                wgpuRenderBundleEncoderSetIndexBuffer(
                    encoder, buffer.buffer, DawnEnum.indexFormat(c.format),
                    UInt64(c.offset), UInt64.max
                )
            case .draw(let c):
                wgpuRenderBundleEncoderDraw(
                    encoder, UInt32(c.vertexCount), UInt32(c.instanceCount),
                    UInt32(c.firstVertex), UInt32(c.firstInstance)
                )
            case .drawIndexed(let c):
                wgpuRenderBundleEncoderDrawIndexed(
                    encoder, UInt32(c.indexCount), UInt32(c.instanceCount),
                    UInt32(c.firstIndex), Int32(c.baseVertex), UInt32(c.firstInstance)
                )
            case .drawIndirect(let c):
                let buffer = try lookupBuffer(c.indirectBuffer, path: c.fieldPath("indirectBuffer"))
                wgpuRenderBundleEncoderDrawIndirect(encoder, buffer.buffer, UInt64(c.indirectOffset))
            case .drawIndexedIndirect(let c):
                let buffer = try lookupBuffer(c.indirectBuffer, path: c.fieldPath("indirectBuffer"))
                wgpuRenderBundleEncoderDrawIndexedIndirect(
                    encoder, buffer.buffer, UInt64(c.indirectOffset)
                )
            case .pushDebugGroup(let c):
                let labelArena = DawnArena()
                wgpuRenderBundleEncoderPushDebugGroup(encoder, labelArena.string(c.groupLabel))
            case .popDebugGroup:
                wgpuRenderBundleEncoderPopDebugGroup(encoder)
            case .insertDebugMarker(let c):
                let labelArena = DawnArena()
                wgpuRenderBundleEncoderInsertDebugMarker(encoder, labelArena.string(c.markerLabel))
            default:
                throw WGPUError.validation(
                    "렌더 번들에서 쓸 수 없는 명령 '\(bundleCommand.opName)'", path: reader.path
                )
            }
        }

        guard let bundle = wgpuRenderBundleEncoderFinish(encoder, nil) else {
            throw WGPUError.backend("GPURenderBundle 생성 실패", path: command.path)
        }
        registry.insert(DawnRenderBundleObject(bundle: bundle), at: command.id)
    }

    private func executeBundles(_ command: WGPUExecuteBundlesCommand) throws {
        let pass = try requireRenderPass()
        let bundles = try command.bundles.map { handle in
            try registry.lookup(
                handle, as: DawnRenderBundleObject.self, kind: "GPURenderBundle",
                path: command.fieldPath("bundles")
            ).bundle as WGPURenderBundle?
        }
        bundles.withUnsafeBufferPointer { pointer in
            wgpuRenderPassEncoderExecuteBundles(pass, pointer.count, pointer.baseAddress)
        }
    }

    // MARK: - 캔버스

    private func configureCanvas(_ configuration: LynxWebGPUCore.WGPUCanvasConfiguration) throws {
        guard let canvas = canvases[configuration.canvasId] else {
            throw WGPUError.validation(
                "캔버스 '\(configuration.canvasId)'이(가) 없다 (등록된 것: "
                    + "\(canvases.keys.sorted().joined(separator: ", ")))"
            )
        }
        try canvas.configure(device: device, format: configuration.format)
    }

    private func getCurrentTexture(_ command: WGPUGetCurrentTextureCommand) throws {
        guard let canvas = canvases[command.canvas] else {
            throw WGPUError.validation(
                "캔버스 '\(command.canvas)'이(가) 없다", path: command.fieldPath("canvas")
            )
        }
        guard let texture = canvas.texture else {
            throw WGPUError.validation(
                "캔버스 '\(command.canvas)'이(가) 아직 configure되지 않았다",
                path: command.fieldPath("canvas")
            )
        }
        registry.insert(
            DawnTextureObject(
                texture: texture, format: canvas.format,
                width: Int(canvas.size.width), height: Int(canvas.size.height),
                retain: true
            ),
            at: command.id
        )
        // 드로어블 텍스처는 프레임 스코프다 — present가 프레임을 닫을 때 만료된다.
        frameScopedHandles.append(command.id)
        acquiredCanvases.insert(command.canvas)
        touchedCanvases[command.canvas] = canvas
    }

    // MARK: - 패스

    private func beginRenderPass(_ descriptor: LynxWebGPUCore.WGPURenderPassDescriptor) throws {
        endOpenPasses()
        let arena = DawnArena()
        let colorAttachments = try descriptor.colorAttachments.map { attachment
            -> WebGPU.WGPURenderPassColorAttachment in
            let view = try registry.lookup(
                attachment.view, as: DawnTextureViewObject.self, kind: "GPUTextureView",
                path: "colorAttachments.view"
            )
            var dawnAttachment = WebGPU.WGPURenderPassColorAttachment()
            dawnAttachment.view = view.view
            dawnAttachment.depthSlice = UInt32.max   // WGPU_DEPTH_SLICE_UNDEFINED
            if let resolveTarget = attachment.resolveTarget {
                dawnAttachment.resolveTarget = try registry.lookup(
                    resolveTarget, as: DawnTextureViewObject.self, kind: "GPUTextureView",
                    path: "colorAttachments.resolveTarget"
                ).view
            }
            dawnAttachment.loadOp = DawnEnum.loadOp(attachment.loadOp)
            dawnAttachment.storeOp = DawnEnum.storeOp(attachment.storeOp)
            dawnAttachment.clearValue = DawnEnum.color(attachment.clearValue)
            return dawnAttachment
        }

        var dawnDescriptor = WebGPU.WGPURenderPassDescriptor()
        dawnDescriptor.label = arena.string(descriptor.label)
        dawnDescriptor.colorAttachmentCount = colorAttachments.count
        dawnDescriptor.colorAttachments = arena.array(colorAttachments)

        if let depthStencil = descriptor.depthStencilAttachment {
            let view = try registry.lookup(
                depthStencil.view, as: DawnTextureViewObject.self, kind: "GPUTextureView",
                path: "depthStencilAttachment.view"
            )
            var dawnDepthStencil = WebGPU.WGPURenderPassDepthStencilAttachment()
            dawnDepthStencil.view = view.view
            dawnDepthStencil.depthLoadOp = DawnEnum.loadOp(depthStencil.depthLoadOp)
            dawnDepthStencil.depthStoreOp = DawnEnum.storeOp(depthStencil.depthStoreOp)
            dawnDepthStencil.depthClearValue = Float(depthStencil.depthClearValue)
            dawnDepthStencil.depthReadOnly = depthStencil.depthReadOnly ? 1 : 0
            dawnDepthStencil.stencilLoadOp = DawnEnum.loadOp(depthStencil.stencilLoadOp)
            dawnDepthStencil.stencilStoreOp = DawnEnum.storeOp(depthStencil.stencilStoreOp)
            dawnDepthStencil.stencilClearValue = UInt32(truncatingIfNeeded: depthStencil.stencilClearValue)
            dawnDepthStencil.stencilReadOnly = depthStencil.stencilReadOnly ? 1 : 0
            dawnDescriptor.depthStencilAttachment = arena.value(dawnDepthStencil)
        }

        if let querySet = descriptor.occlusionQuerySet {
            dawnDescriptor.occlusionQuerySet = try registry.lookup(
                querySet, as: DawnQuerySetObject.self, kind: "GPUQuerySet", path: "occlusionQuerySet"
            ).querySet
        }

        guard let pass = wgpuCommandEncoderBeginRenderPass(ensureEncoder(), &dawnDescriptor) else {
            throw WGPUError.backend("렌더 패스 시작 실패")
        }
        renderPass = pass
    }

    private func beginComputePass() throws {
        endOpenPasses()
        guard let pass = wgpuCommandEncoderBeginComputePass(ensureEncoder(), nil) else {
            throw WGPUError.backend("컴퓨트 패스 시작 실패")
        }
        computePass = pass
    }

    private func setPipeline(_ command: WGPUSetPipelineCommand) throws {
        if let renderPass {
            let pipeline = try registry.lookup(
                command.pipeline, as: DawnRenderPipelineObject.self, kind: "GPURenderPipeline",
                path: command.fieldPath("pipeline")
            )
            wgpuRenderPassEncoderSetPipeline(renderPass, pipeline.pipeline)
        } else if let computePass {
            let pipeline = try registry.lookup(
                command.pipeline, as: DawnComputePipelineObject.self, kind: "GPUComputePipeline",
                path: command.fieldPath("pipeline")
            )
            wgpuComputePassEncoderSetPipeline(computePass, pipeline.pipeline)
        } else {
            throw WGPUError.validation("패스가 시작되지 않았다 (beginRenderPass/beginComputePass 먼저)")
        }
    }

    private func setBindGroup(_ command: WGPUSetBindGroupCommand) throws {
        let group = try registry.lookup(
            command.bindGroup, as: DawnBindGroupObject.self, kind: "GPUBindGroup",
            path: command.fieldPath("bindGroup")
        )
        let offsets = command.dynamicOffsets.map(UInt32.init)
        try offsets.withUnsafeBufferPointer { pointer in
            if let renderPass {
                wgpuRenderPassEncoderSetBindGroup(
                    renderPass, UInt32(command.index), group.group, pointer.count, pointer.baseAddress
                )
            } else if let computePass {
                wgpuComputePassEncoderSetBindGroup(
                    computePass, UInt32(command.index), group.group, pointer.count, pointer.baseAddress
                )
            } else {
                throw WGPUError.validation("패스가 시작되지 않았다 (setBindGroup)")
            }
        }
    }

    private func setVertexBuffer(_ command: WGPUSetVertexBufferCommand) throws {
        let buffer = try unmappedBuffer(command.buffer, path: command.fieldPath("buffer"))
        wgpuRenderPassEncoderSetVertexBuffer(
            try requireRenderPass(), UInt32(command.slot), buffer.buffer,
            UInt64(command.offset), UInt64.max
        )
    }

    private func setIndexBuffer(_ command: WGPUSetIndexBufferCommand) throws {
        let buffer = try unmappedBuffer(command.buffer, path: command.fieldPath("buffer"))
        wgpuRenderPassEncoderSetIndexBuffer(
            try requireRenderPass(), buffer.buffer, DawnEnum.indexFormat(command.format),
            UInt64(command.offset), UInt64.max
        )
    }

    // MARK: - 복사

    private func copyBufferToBuffer(_ command: WGPUCopyBufferToBufferCommand) throws {
        let source = try unmappedBuffer(command.source, path: command.fieldPath("source"))
        let destination = try unmappedBuffer(command.destination, path: command.fieldPath("destination"))
        // size 생략 = 원본의 남은 전부 (명세의 짧은 오버로드 — 레지스트리를 봐야 해서 여기서 정한다)
        let size = command.size ?? (source.size - command.sourceOffset)
        guard size >= 0 else {
            throw WGPUError.validation("복사 크기가 음수다", path: command.fieldPath("size"))
        }
        wgpuCommandEncoderCopyBufferToBuffer(
            ensureEncoder(), source.buffer, UInt64(command.sourceOffset),
            destination.buffer, UInt64(command.destinationOffset), UInt64(size)
        )
    }

    private func clearBuffer(_ command: WGPUClearBufferCommand) throws {
        let buffer = try unmappedBuffer(command.buffer, path: command.fieldPath("buffer"))
        let size = command.size ?? (buffer.size - command.offset)
        guard size >= 0 else {
            throw WGPUError.validation("clearBuffer 크기가 음수다", path: command.fieldPath("size"))
        }
        wgpuCommandEncoderClearBuffer(
            ensureEncoder(), buffer.buffer, UInt64(command.offset), UInt64(size)
        )
    }

    private func texelCopyTexture(
        _ info: WGPUTexelCopyTextureInfo_Core, path: String
    ) throws -> (WebGPU.WGPUTexelCopyTextureInfo, DawnTextureObject) {
        let texture = try registry.lookup(
            info.texture, as: DawnTextureObject.self, kind: "GPUTexture", path: path
        )
        var dawnInfo = WebGPU.WGPUTexelCopyTextureInfo()
        dawnInfo.texture = texture.texture
        dawnInfo.mipLevel = UInt32(info.mipLevel)
        dawnInfo.origin = DawnEnum.origin(info.origin)
        dawnInfo.aspect = WGPUTextureAspect_All
        return (dawnInfo, texture)
    }

    private func texelCopyBuffer(
        _ info: WGPUTexelCopyBufferInfo_Core,
        copyHeight: Int,
        format: LynxWebGPUCore.WGPUTextureFormat,
        copyWidth: Int,
        path: String
    ) throws -> WebGPU.WGPUTexelCopyBufferInfo {
        let buffer = try unmappedBuffer(info.buffer, path: path)
        var dawnInfo = WebGPU.WGPUTexelCopyBufferInfo()
        dawnInfo.buffer = buffer.buffer
        dawnInfo.layout.offset = UInt64(info.offset)
        let bytesPerRow = info.bytesPerRow ?? (copyWidth * format.bytesPerPixel)
        dawnInfo.layout.bytesPerRow = UInt32(bytesPerRow)
        dawnInfo.layout.rowsPerImage = UInt32(info.rowsPerImage ?? copyHeight)
        return dawnInfo
    }

    private func copyTextureToBuffer(_ command: WGPUCopyTextureToBufferCommand) throws {
        var (source, texture) = try texelCopyTexture(
            command.source, path: command.fieldPath("source.texture")
        )
        var destination = try texelCopyBuffer(
            command.destination, copyHeight: command.copySize.height,
            format: texture.format, copyWidth: command.copySize.width,
            path: command.fieldPath("destination.buffer")
        )
        var size = DawnEnum.extent(command.copySize)
        wgpuCommandEncoderCopyTextureToBuffer(ensureEncoder(), &source, &destination, &size)
    }

    private func copyBufferToTexture(_ command: WGPUCopyBufferToTextureCommand) throws {
        var (destination, texture) = try texelCopyTexture(
            command.destination, path: command.fieldPath("destination.texture")
        )
        var source = try texelCopyBuffer(
            command.source, copyHeight: command.copySize.height,
            format: texture.format, copyWidth: command.copySize.width,
            path: command.fieldPath("source.buffer")
        )
        var size = DawnEnum.extent(command.copySize)
        wgpuCommandEncoderCopyBufferToTexture(ensureEncoder(), &source, &destination, &size)
    }

    private func copyTextureToTexture(_ command: WGPUCopyTextureToTextureCommand) throws {
        var (source, _) = try texelCopyTexture(
            command.source, path: command.fieldPath("source.texture")
        )
        var (destination, _) = try texelCopyTexture(
            command.destination, path: command.fieldPath("destination.texture")
        )
        var size = DawnEnum.extent(command.copySize)
        wgpuCommandEncoderCopyTextureToTexture(ensureEncoder(), &source, &destination, &size)
    }

    private func resolveQuerySet(_ command: WGPUResolveQuerySetCommand) throws {
        let querySet = try registry.lookup(
            command.querySet, as: DawnQuerySetObject.self, kind: "GPUQuerySet",
            path: command.fieldPath("querySet")
        )
        let destination = try unmappedBuffer(
            command.destination, path: command.fieldPath("destination")
        )
        let queryCount = command.queryCount ?? (querySet.count - command.firstQuery)
        wgpuCommandEncoderResolveQuerySet(
            ensureEncoder(), querySet.querySet, UInt32(command.firstQuery),
            UInt32(max(queryCount, 0)), destination.buffer, UInt64(command.destinationOffset)
        )
    }

    // MARK: - 디버그 마커

    private func pushDebugGroup(_ label: String) throws {
        let arena = DawnArena()
        if let renderPass {
            wgpuRenderPassEncoderPushDebugGroup(renderPass, arena.string(label))
        } else if let computePass {
            wgpuComputePassEncoderPushDebugGroup(computePass, arena.string(label))
        } else {
            wgpuCommandEncoderPushDebugGroup(ensureEncoder(), arena.string(label))
        }
    }

    private func popDebugGroupOnOpenScope() {
        if let renderPass {
            wgpuRenderPassEncoderPopDebugGroup(renderPass)
        } else if let computePass {
            wgpuComputePassEncoderPopDebugGroup(computePass)
        } else {
            wgpuCommandEncoderPopDebugGroup(ensureEncoder())
        }
    }

    private func insertDebugMarker(_ label: String) throws {
        let arena = DawnArena()
        if let renderPass {
            wgpuRenderPassEncoderInsertDebugMarker(renderPass, arena.string(label))
        } else if let computePass {
            wgpuComputePassEncoderInsertDebugMarker(computePass, arena.string(label))
        } else {
            wgpuCommandEncoderInsertDebugMarker(ensureEncoder(), arena.string(label))
        }
    }

    // MARK: - 조회 (WebGPURuntime)

    func adapterInfo() -> [String: Any] {
        var limits = WGPULimits()
        _ = wgpuAdapterGetLimits(adapter, &limits)
        var info = WGPUAdapterInfo()
        _ = wgpuAdapterGetInfo(adapter, &info)

        let limitsPayload: [String: Any] = [
            "maxTextureDimension1D": Int(limits.maxTextureDimension1D),
            "maxTextureDimension2D": Int(limits.maxTextureDimension2D),
            "maxTextureDimension3D": Int(limits.maxTextureDimension3D),
            "maxTextureArrayLayers": Int(limits.maxTextureArrayLayers),
            "maxBindGroups": Int(limits.maxBindGroups),
            "maxBindGroupsPlusVertexBuffers": Int(limits.maxBindGroupsPlusVertexBuffers),
            "maxBindingsPerBindGroup": Int(limits.maxBindingsPerBindGroup),
            "maxSampledTexturesPerShaderStage": Int(limits.maxSampledTexturesPerShaderStage),
            "maxSamplersPerShaderStage": Int(limits.maxSamplersPerShaderStage),
            "maxStorageBuffersPerShaderStage": Int(limits.maxStorageBuffersPerShaderStage),
            "maxStorageTexturesPerShaderStage": Int(limits.maxStorageTexturesPerShaderStage),
            "maxUniformBuffersPerShaderStage": Int(limits.maxUniformBuffersPerShaderStage),
            "maxDynamicUniformBuffersPerPipelineLayout":
                Int(limits.maxDynamicUniformBuffersPerPipelineLayout),
            "maxDynamicStorageBuffersPerPipelineLayout":
                Int(limits.maxDynamicStorageBuffersPerPipelineLayout),
            "maxBufferSize": Int(clamping: limits.maxBufferSize),
            "maxUniformBufferBindingSize": Int(clamping: limits.maxUniformBufferBindingSize),
            "maxStorageBufferBindingSize": Int(clamping: limits.maxStorageBufferBindingSize),
            "minUniformBufferOffsetAlignment": Int(limits.minUniformBufferOffsetAlignment),
            "minStorageBufferOffsetAlignment": Int(limits.minStorageBufferOffsetAlignment),
            "maxVertexBuffers": Int(limits.maxVertexBuffers),
            "maxVertexAttributes": Int(limits.maxVertexAttributes),
            "maxVertexBufferArrayStride": Int(limits.maxVertexBufferArrayStride),
            "maxInterStageShaderVariables": Int(limits.maxInterStageShaderVariables),
            "maxColorAttachments": Int(limits.maxColorAttachments),
            "maxColorAttachmentBytesPerSample": Int(limits.maxColorAttachmentBytesPerSample),
            "maxComputeWorkgroupStorageSize": Int(limits.maxComputeWorkgroupStorageSize),
            "maxComputeInvocationsPerWorkgroup": Int(limits.maxComputeInvocationsPerWorkgroup),
            "maxComputeWorkgroupSizeX": Int(limits.maxComputeWorkgroupSizeX),
            "maxComputeWorkgroupSizeY": Int(limits.maxComputeWorkgroupSizeY),
            "maxComputeWorkgroupSizeZ": Int(limits.maxComputeWorkgroupSizeZ),
            "maxComputeWorkgroupsPerDimension": Int(limits.maxComputeWorkgroupsPerDimension),
        ]

        let adapterPayload: [String: Any] = [
            "vendor": String(wgpu: info.vendor),
            "architecture": String(wgpu: info.architecture),
            "device": String(wgpu: info.device),
            "description": String(wgpu: info.description),
            "isFallbackAdapter": info.adapterType == WGPUAdapterType_CPU,
            "subgroupMinSize": Int(info.subgroupMinSize),
            "subgroupMaxSize": Int(info.subgroupMaxSize),
        ]
        let name = String(wgpu: info.description)
        wgpuAdapterInfoFreeMembers(info)

        return [
            "ok": true,
            "info": adapterPayload,
            "name": name,
            "backend": "dawn-metal",
            "preferredCanvasFormat": LynxWebGPUCore.WGPUTextureFormat.bgra8unorm.rawValue,
            "limits": limitsPayload,
            // 기능은 광고하지 않는다 — 시뮬레이터의 간접 드로우가 Metal 단언으로 죽는 경로라
            // (`CLAUDE.md`), 간접 검사가 건너뛰게 두는 쪽이 픽스처로서 정직하다.
            "features": [String](),
        ]
    }

    func shaderCompilationInfo(handle: Int) -> [String: Any] {
        executionLock.lock()
        defer { executionLock.unlock() }
        guard let module = try? registry.lookup(
            WGPUHandle(handle), as: DawnShaderModuleObject.self, kind: "GPUShaderModule"
        ) else {
            return ["ok": false, "errors": [
                WGPUError.validation("GPUShaderModule #\(handle)이(가) 없다").payload,
            ]]
        }

        final class InfoBox { var done = false; var messages: [[String: Any]] = [] }
        let box = InfoBox()
        var callbackInfo = WGPUCompilationInfoCallbackInfo()
        callbackInfo.mode = WGPUCallbackMode_AllowProcessEvents
        callbackInfo.callback = { _, info, userdata1, _ in
            let box = Unmanaged<InfoBox>.fromOpaque(userdata1!).takeRetainedValue()
            if let info = info?.pointee, let messages = info.messages {
                for index in 0..<info.messageCount {
                    let message = messages[index]
                    let type: String
                    if message.type == WGPUCompilationMessageType_Warning { type = "warning" }
                    else if message.type == WGPUCompilationMessageType_Info { type = "info" }
                    else { type = "error" }
                    box.messages.append([
                        "message": String(wgpu: message.message),
                        "type": type,
                        "lineNum": Int(message.lineNum),
                        "linePos": Int(message.linePos),
                        "offset": Int(message.offset),
                        "length": Int(message.length),
                    ])
                }
            }
            box.done = true
        }
        callbackInfo.userdata1 = Unmanaged.passRetained(box).toOpaque()
        _ = wgpuShaderModuleGetCompilationInfo(module.module, callbackInfo)
        do {
            try DawnBootstrap.pump(instance: instance, until: { box.done }, what: "getCompilationInfo")
        } catch {
            return ["ok": false, "errors": [WGPUError.backend("\(error)").payload]]
        }
        return ["ok": true, "messages": box.messages]
    }

    func canvasInfo(identifier: String) -> [String: Any] {
        executionLock.lock()
        defer { executionLock.unlock() }
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

    // MARK: - 비동기 (WebGPURuntime)

    func readBuffer(
        handle: Int, offset: Int, size: Int?,
        completion: @escaping ([String: Any]) -> Void
    ) {
        executionLock.lock()
        let target: DawnBufferObject
        do {
            target = try lookupBuffer(WGPUHandle(handle))
        } catch let error as WGPUError {
            executionLock.unlock()
            return completion(["ok": false, "errors": [error.payload]])
        } catch {
            executionLock.unlock()
            return completion(["ok": false, "errors": [WGPUError.backend("\(error)").payload]])
        }
        guard !target.isMapped else {
            executionLock.unlock()
            return completion(["ok": false, "errors": [WGPUError.validation(
                "GPUBuffer \(WGPUHandle(handle))은(는) 이미 매핑 중이다 (unmap()을 먼저 부를 것)"
            ).payload]])
        }
        target.isMapped = true   // 명세의 unavailable — unmapBuffer op이 푼다
        let length = size ?? (target.size - offset)

        // MAP_READ가 있으면 직접 매핑, 없으면 스테이징 복사 (COPY_SRC 필요).
        if target.usage.contains(.mapRead) {
            mapAndDeliver(
                buffer: target.buffer, mapOffset: 0, mapSize: target.size,
                readOffset: offset, readLength: length,
                markDawnMapped: target, completion: completion
            )
        } else if target.usage.contains(.copySrc) {
            var descriptor = WebGPU.WGPUBufferDescriptor()
            let alignedLength = (max(length, 0) + 3) / 4 * 4
            descriptor.size = UInt64(alignedLength)
            descriptor.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead
            guard let staging = wgpuDeviceCreateBuffer(device, &descriptor),
                  let encoder = wgpuDeviceCreateCommandEncoder(device, nil) else {
                target.isMapped = false
                executionLock.unlock()
                return completion(["ok": false, "errors": [
                    WGPUError.outOfMemory("리드백 스테이징 생성 실패").payload,
                ]])
            }
            wgpuCommandEncoderCopyBufferToBuffer(
                encoder, target.buffer, UInt64(offset), staging, 0, UInt64(alignedLength)
            )
            if let commandBuffer = wgpuCommandEncoderFinish(encoder, nil) {
                var submission: WGPUCommandBuffer? = commandBuffer
                wgpuQueueSubmit(queue, 1, &submission)
                wgpuCommandBufferRelease(commandBuffer)
            }
            wgpuCommandEncoderRelease(encoder)
            mapAndDeliver(
                buffer: staging, mapOffset: 0, mapSize: alignedLength,
                readOffset: 0, readLength: length,
                releaseAfter: staging, completion: completion
            )
        } else {
            target.isMapped = false
            executionLock.unlock()
            return completion(["ok": false, "errors": [WGPUError.validation(
                "readBuffer에는 MAP_READ 또는 COPY_SRC usage가 필요하다"
            ).payload]])
        }
        executionLock.unlock()
    }

    /// mapAsync를 걸고, 콜백(processEvents 펌프에서 온다)에서 바이트를 떠 completion을 부른다.
    private func mapAndDeliver(
        buffer: WGPUBuffer,
        mapOffset: Int,
        mapSize: Int,
        readOffset: Int,
        readLength: Int,
        markDawnMapped: DawnBufferObject? = nil,
        releaseAfter: WGPUBuffer? = nil,
        completion: @escaping ([String: Any]) -> Void
    ) {
        final class MapContext {
            let buffer: WGPUBuffer
            let readOffset: Int
            let readLength: Int
            let markDawnMapped: DawnBufferObject?
            let releaseAfter: WGPUBuffer?
            let completion: ([String: Any]) -> Void
            init(buffer: WGPUBuffer, readOffset: Int, readLength: Int,
                 markDawnMapped: DawnBufferObject?, releaseAfter: WGPUBuffer?,
                 completion: @escaping ([String: Any]) -> Void) {
                self.buffer = buffer
                self.readOffset = readOffset
                self.readLength = readLength
                self.markDawnMapped = markDawnMapped
                self.releaseAfter = releaseAfter
                self.completion = completion
            }
        }
        let context = MapContext(
            buffer: buffer, readOffset: readOffset, readLength: readLength,
            markDawnMapped: markDawnMapped, releaseAfter: releaseAfter, completion: completion
        )
        var callbackInfo = WGPUBufferMapCallbackInfo()
        callbackInfo.mode = WGPUCallbackMode_AllowProcessEvents
        callbackInfo.callback = { status, message, userdata1, _ in
            let context = Unmanaged<MapContext>.fromOpaque(userdata1!).takeRetainedValue()
            defer { if let staging = context.releaseAfter { wgpuBufferRelease(staging) } }
            guard status == WGPUMapAsyncStatus_Success else {
                context.markDawnMapped?.isMapped = false
                context.completion(["ok": false, "errors": [WGPUError.backend(
                    "mapAsync 실패 — \(String(wgpu: message))"
                ).payload]])
                return
            }
            let data: Data
            if let pointer = wgpuBufferGetConstMappedRange(
                context.buffer, context.readOffset, context.readLength
            ) {
                data = Data(bytes: pointer, count: context.readLength)
            } else {
                data = Data()
            }
            wgpuBufferUnmap(context.buffer)
            context.completion(["ok": true, "data": data, "byteLength": data.count])
        }
        callbackInfo.userdata1 = Unmanaged.passRetained(context).toOpaque()
        _ = wgpuBufferMapAsync(
            buffer, WGPUMapMode_Read, mapOffset, mapSize, callbackInfo
        )
    }

    func decodeImage(
        handle: Int, data: Data?, name: String?, options: WGPUImageDecodeOptions,
        provider: WGPUAssetProvider?, completion: @escaping ([String: Any]) -> Void
    ) {
        let fail = { (error: WGPUError) in
            completion(["ok": false, "errors": [error.payload]])
        }
        let decode = { [weak self] (bytes: Data) in
            guard let self else { return }
            do {
                let bitmap = try Self.decodeRGBA(bytes, options: options)
                self.executionLock.lock()
                self.registry.insert(bitmap, at: WGPUHandle(handle))
                self.executionLock.unlock()
                completion(["ok": true, "width": bitmap.width, "height": bitmap.height])
            } catch let error as WGPUError {
                fail(error)
            } catch {
                fail(.backend("\(error)"))
            }
        }
        if let data {
            decode(data)
        } else if let name, let provider {
            provider.loadAsset(named: name) { result in
                switch result {
                case .success(let bytes): decode(bytes)
                case .failure(let error): fail(error)
                }
            }
        } else {
            fail(.validation("createImageBitmap에는 이미지 바이트나 애셋 이름이 필요하다"))
        }
    }

    /// ImageIO로 RGBA8 비트맵을 만든다 — Metal 런타임의 `WGPUImageDecoder`에 해당하는 자리.
    /// (알파는 곱해진 상태로 나온다 — CGContext RGBA8의 제약. 불투명 이미지에서는 차이가 없다.)
    private static func decodeRGBA(
        _ bytes: Data, options: WGPUImageDecodeOptions
    ) throws -> DawnImageBitmapObject {
        guard let source = CGImageSourceCreateWithData(bytes as CFData, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw WGPUError.validation("이미지를 디코딩할 수 없다")
        }
        let width = options.resize?.width ?? image.width
        let height = options.resize?.height ?? image.height
        var pixels = Data(count: width * height * 4)
        try pixels.withUnsafeMutableBytes { buffer in
            guard let context = CGContext(
                data: buffer.baseAddress,
                width: width, height: height,
                bitsPerComponent: 8, bytesPerRow: width * 4,
                space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            ) else {
                throw WGPUError.backend("디코딩 컨텍스트 생성 실패")
            }
            if options.flipY {
                context.translateBy(x: 0, y: CGFloat(height))
                context.scaleBy(x: 1, y: -1)
            }
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        }
        return DawnImageBitmapObject(data: pixels, width: width, height: height)
    }

    private func copyExternalImageToTexture(_ command: WGPUCopyExternalImageCommand) throws {
        let bitmap = try registry.lookup(
            command.source.image, as: DawnImageBitmapObject.self, kind: "ImageBitmap",
            path: command.source.fieldPath("source")
        )
        let (destination, texture) = try texelCopyTexture(
            command.destination, path: command.fieldPath("destination.texture")
        )
        _ = texture
        let copyWidth = command.copySize?.width ?? (bitmap.width - command.source.origin.x)
        let copyHeight = command.copySize?.height ?? (bitmap.height - command.source.origin.y)
        guard copyWidth > 0, copyHeight > 0 else { return }

        // 원본에서 (origin, flipY)를 반영한 행들을 촘촘히 이어 붙인다.
        var upload = Data(capacity: copyWidth * copyHeight * 4)
        for row in 0..<copyHeight {
            let sourceRow = command.source.flipY
                ? (command.source.origin.y + (copyHeight - 1 - row))
                : (command.source.origin.y + row)
            let start = (sourceRow * bitmap.width + command.source.origin.x) * 4
            upload.append(bitmap.data.subdata(in: start..<(start + copyWidth * 4)))
        }

        var dawnDestination = destination
        var layout = WGPUTexelCopyBufferLayout()
        layout.offset = 0
        layout.bytesPerRow = UInt32(copyWidth * 4)
        layout.rowsPerImage = UInt32(copyHeight)
        var size = WebGPU.WGPUExtent3D(
            width: UInt32(copyWidth), height: UInt32(copyHeight), depthOrArrayLayers: 1
        )
        upload.withUnsafeBytes { source in
            wgpuQueueWriteTexture(
                queue, &dawnDestination, source.baseAddress, source.count, &layout, &size
            )
        }
    }

    // MARK: - 캔버스 (WebGPURuntime)

    func attachCanvas(identifier: String, layer: CAMetalLayer) {
        // 이 픽스처는 화면 표면을 다루지 않는다 — Dawn의 WGPUSurfaceSourceMetalLayer 배선은
        // 실제 백엔드 저장소(§3-6)의 몫이다. 조용히 무시하지 않고 로그만 남긴다.
        print("DawnCheck: attachCanvas(\(identifier))는 이 픽스처 범위 밖이다 (오프스크린 전용)")
    }

    func attachOffscreenCanvas(identifier: String, size: CGSize) throws {
        executionLock.lock()
        defer { executionLock.unlock() }
        canvases[identifier] = DawnOffscreenCanvas(identifier: identifier, size: size)
    }

    func resizeCanvas(identifier: String, drawableSize: CGSize) {
        executionLock.lock()
        defer { executionLock.unlock() }
        canvases[identifier]?.updateSize(drawableSize, device: device)
    }

    func detachCanvas(identifier: String) {
        executionLock.lock()
        defer { executionLock.unlock() }
        canvases.removeValue(forKey: identifier)
    }

    func readCanvasPixels(identifier: String) throws -> WGPUPixelReadback {
        executionLock.lock()
        defer { executionLock.unlock() }
        guard let canvas = canvases[identifier], let texture = canvas.texture else {
            throw WGPUError.validation("캔버스 '\(identifier)'이(가) 없거나 configure 전이다")
        }
        let width = Int(canvas.size.width)
        let height = Int(canvas.size.height)
        let bytesPerPixel = canvas.format.bytesPerPixel
        let bytesPerRow = max(256, (width * bytesPerPixel + 255) / 256 * 256)
        let total = bytesPerRow * height

        var descriptor = WebGPU.WGPUBufferDescriptor()
        descriptor.size = UInt64(total)
        descriptor.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead
        guard let staging = wgpuDeviceCreateBuffer(device, &descriptor),
              let encoder = wgpuDeviceCreateCommandEncoder(device, nil) else {
            throw WGPUError.outOfMemory("픽셀 읽기 스테이징 생성 실패")
        }
        defer { wgpuBufferRelease(staging) }

        var source = WebGPU.WGPUTexelCopyTextureInfo()
        source.texture = texture
        source.mipLevel = 0
        source.origin = WebGPU.WGPUOrigin3D(x: 0, y: 0, z: 0)
        source.aspect = WGPUTextureAspect_All
        var destination = WebGPU.WGPUTexelCopyBufferInfo()
        destination.buffer = staging
        destination.layout.offset = 0
        destination.layout.bytesPerRow = UInt32(bytesPerRow)
        destination.layout.rowsPerImage = UInt32(height)
        var size = WebGPU.WGPUExtent3D(
            width: UInt32(width), height: UInt32(height), depthOrArrayLayers: 1
        )
        wgpuCommandEncoderCopyTextureToBuffer(encoder, &source, &destination, &size)
        if let commandBuffer = wgpuCommandEncoderFinish(encoder, nil) {
            var submission: WGPUCommandBuffer? = commandBuffer
            wgpuQueueSubmit(queue, 1, &submission)
            wgpuCommandBufferRelease(commandBuffer)
        }
        wgpuCommandEncoderRelease(encoder)

        // 동기 매핑 — 계약이 "GPU 완료 후 읽기"다 (mapAsync가 완료를 기다린다).
        final class MapBox { var done = false; var message = "" }
        let box = MapBox()
        var callbackInfo = WGPUBufferMapCallbackInfo()
        callbackInfo.mode = WGPUCallbackMode_AllowProcessEvents
        callbackInfo.callback = { status, message, userdata1, _ in
            let box = Unmanaged<MapBox>.fromOpaque(userdata1!).takeRetainedValue()
            if status != WGPUMapAsyncStatus_Success { box.message = String(wgpu: message) }
            box.done = true
        }
        callbackInfo.userdata1 = Unmanaged.passRetained(box).toOpaque()
        _ = wgpuBufferMapAsync(staging, WGPUMapMode_Read, 0, total, callbackInfo)
        try DawnBootstrap.pump(instance: instance, until: { box.done }, what: "readCanvasPixels 매핑")
        guard let pointer = wgpuBufferGetConstMappedRange(staging, 0, total) else {
            throw WGPUError.backend("픽셀 매핑 실패 — \(box.message)")
        }
        let data = Data(bytes: pointer, count: total)
        wgpuBufferUnmap(staging)
        return WGPUPixelReadback(
            data: data, format: canvas.format, width: width, height: height, bytesPerRow: bytesPerRow
        )
    }

    // MARK: - 프레임 · 수명 (WebGPURuntime)

    var isReadyForNextFrame: Bool { true }   // 오프스크린 전용 — 페이싱 대상 표면이 없다

    func processEvents() {
        wgpuInstanceProcessEvents(instance)
    }

    func reset() {
        executionLock.lock()
        defer { executionLock.unlock() }
        registry.removeAll()
        errorScopes.discardAll()
        frameScopedHandles.removeAll()
        acquiredCanvases.removeAll()
        _ = deferredErrors.drain()
    }
}

// Core 타입 별칭 — 이 파일 안에서 Dawn C 타입과 이름이 겹치는 것들.
private typealias WGPUTexelCopyTextureInfo_Core = LynxWebGPUCore.WGPUTexelCopyTextureInfo
private typealias WGPUTexelCopyBufferInfo_Core = LynxWebGPUCore.WGPUTexelCopyBufferInfo
