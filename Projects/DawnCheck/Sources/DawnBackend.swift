import Foundation
import CoreGraphics
import QuartzCore
import WebGPU
import LynxWebGPUCore

/// `WGPUBackend`의 **Dawn 구현** — `docs/extra/DAWN-BACKEND-REVIEW.md`가 그린 절단면의 실물.
///
/// 오케스트레이션(디코딩·검증·오류 스코프·프레임 수명·매핑 게이트·직렬화)은 전부
/// `WGPUBackendEngine`(Core)이 한다 — Metal 백엔드와 **같은 엔진**이다. 이 파일이 쓰는 것은
/// 정말로 인코딩뿐이다: 해석이 끝난 Core 값 → Dawn C 호출 (`DawnEnum`·`DawnArena`),
/// JS 유래 정수의 안전 변환(`dawnU32` 계열 — 트랩 대신 validation).
///
/// ## Dawn 오류를 와이어 모델로 옮기는 방법
///
/// - 핸들·상태·범위 오류는 **엔진**이 경로 붙은 validation으로 만든다 (Metal과 같은 문구).
/// - Dawn 자체 검증 오류는 배치 전체를 디바이스 오류 스코프(validation·out-of-memory)로 감싸
///   `collectBatchDiagnostics()`에서 회수한다 — 엔진이 같은 배치 결과에 싣는다.
/// - 스코프 밖(uncaptured) 오류도 같은 통로로 모아 두었다가 다음 회수 때 넘긴다 —
///   엔진의 지연 오류 큐가 다음 배치에 실어 보낸다 (`docs/COMMAND-STREAM.md` §2).
///
/// ## 시뮬레이터 주의
///
/// 간접 드로우는 시뮬레이터 Metal(family 2)에서 단언으로 죽는 경로라
/// `ensureIndirectSupported()`가 막고, `indirect-first-instance` 광고도 뺀다 (`CLAUDE.md`).
final class DawnBackend: WGPUBackend {
    typealias Buffer = DawnBufferObject
    typealias Texture = DawnTextureObject
    typealias TextureView = DawnTextureViewObject
    typealias Sampler = DawnSamplerObject
    typealias ShaderModule = DawnShaderModuleObject
    typealias BindGroupLayout = DawnBindGroupLayoutObject
    typealias PipelineLayout = DawnPipelineLayoutObject
    typealias BindGroup = DawnBindGroupObject
    typealias RenderPipeline = DawnRenderPipelineObject
    typealias ComputePipeline = DawnComputePipelineObject
    typealias QuerySet = DawnQuerySetObject
    typealias RenderBundle = DawnRenderBundleObject
    typealias Surface = DawnCanvas

    private let instance: WGPUInstance
    private let adapter: WGPUAdapter
    private let device: WGPUDevice
    private let queue: WGPUQueue
    /// 광고 가능한 기능의 명세 철자 — 압축 지원 질의와 adapterInfo가 같은 목록을 쓴다.
    private let featureLabels: Set<String>

    /// 스코프 밖(uncaptured) 디바이스 오류 — `collectBatchDiagnostics`가 비운다.
    private let uncaptured = WGPUDeferredErrorQueue()

    // 배치 수명 상태 (beginBatch ~ submit)
    private var commandEncoder: WGPUCommandEncoder?
    private var renderPass: WGPURenderPassEncoder?
    private var computePass: WGPUComputePassEncoder?
    /// 이번 프레임에 텍스처를 내준 캔버스 — `submit(present: true)`가 화면으로 보낸다.
    private var presentTargets: [String: DawnCanvas] = [:]

    init() throws {
        guard let instance = wgpuCreateInstance(nil) else {
            throw DawnBootstrapError("wgpuCreateInstance 실패")
        }
        self.instance = instance
        self.adapter = try DawnBootstrap.requestAdapter(instance: instance)
        let sink = uncaptured
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
        self.featureLabels = Set(
            DawnBootstrap.supportedFeatures(adapter: adapter).compactMap(DawnEnum.featureLabel)
        )
    }

    deinit {
        endOpenPasses()
        if let commandEncoder { wgpuCommandEncoderRelease(commandEncoder) }
        presentTargets.removeAll()
        wgpuQueueRelease(queue)
        wgpuDeviceRelease(device)
        wgpuAdapterRelease(adapter)
        wgpuInstanceRelease(instance)
    }

    // MARK: - 능력

    var capabilities: WGPUBackendCapabilities {
        WGPUBackendCapabilities(
            supportsNativeRenderBundles: true,
            // 명세 기본값 — 어댑터 한계(maxVertexBuffers)와 같은 값이다.
            maxVertexBufferSlots: 8,
            // mapAsync 완료가 wgpuInstanceProcessEvents에서만 나온다 — 프레임 티커가 없는
            // 씬을 위해 엔진이 자가 펌프를 돌린다 (`WGPUBackendCapabilities` 문서).
            needsEventPump: true
        )
    }

    func supportsTextureCompression(_ format: LynxWebGPUCore.WGPUTextureFormat) -> Bool {
        guard let feature = format.compressionFamily.featureName else { return true }
        return featureLabels.contains(feature)
    }

    /// 시뮬레이터는 간접 인자를 지원하지 않는다 (Apple GPU family 2 — Metal은 3 이상 요구).
    /// Dawn도 결국 Metal 단언으로 죽는 경로라 여기서 막는다 (`CLAUDE.md`).
    func ensureIndirectSupported() throws {
        #if targetEnvironment(simulator)
        throw WGPUError.unsupported(
            "iOS 시뮬레이터는 간접 드로우·디스패치를 지원하지 않는다 (실기기 A12 이상에서는 동작)"
        )
        #endif
    }

    func pumpEvents() {
        wgpuInstanceProcessEvents(instance)
    }

    func reset() {
        endOpenPasses()
        if let commandEncoder {
            wgpuCommandEncoderRelease(commandEncoder)
            self.commandEncoder = nil
        }
        presentTargets.removeAll()
        _ = uncaptured.drain()
    }

    // MARK: - 배치 수명

    func beginBatch() {
        // 앞 배치가 제출 없이 끝났다면 남은 인코더를 정리한다 (도달할 일은 없어야 한다).
        endOpenPasses()
        if let commandEncoder {
            wgpuCommandEncoderRelease(commandEncoder)
            self.commandEncoder = nil
        }
        // Dawn 검증 오류를 이 배치 결과에 싣는 배치 스코프 — pop은 제출 뒤에 한다.
        wgpuDevicePushErrorScope(device, WGPUErrorFilter_OutOfMemory)
        wgpuDevicePushErrorScope(device, WGPUErrorFilter_Validation)
    }

    func collectBatchDiagnostics() -> [WGPUError] {
        var diagnostics: [WGPUError] = []
        // 안쪽(validation)부터 — push의 역순이다. 동기 펌프라 이 배치 결과에 실린다.
        if let error = drainDeviceScope() { diagnostics.append(error) }
        if let error = drainDeviceScope() { diagnostics.append(error) }
        diagnostics.append(contentsOf: uncaptured.drain())
        return diagnostics
    }

    /// 디바이스 스코프 하나를 pop해 잡힌 오류를 돌려준다 (동기 펌프).
    private func drainDeviceScope() -> WGPUError? {
        final class ScopeBox { var done = false; var error: WGPUError? }
        let box = ScopeBox()
        var callbackInfo = WGPUPopErrorScopeCallbackInfo()
        callbackInfo.mode = WGPUCallbackMode_AllowProcessEvents
        callbackInfo.callback = { _, type, message, userdata1, _ in
            guard let userdata1 else { return }
            let box = Unmanaged<ScopeBox>.fromOpaque(userdata1).takeRetainedValue()
            if type != WGPUErrorType_NoError {
                box.error = DawnEnum.errorType(type, message: String(wgpu: message))
            }
            box.done = true
        }
        callbackInfo.userdata1 = Unmanaged.passRetained(box).toOpaque()
        _ = wgpuDevicePopErrorScope(device, callbackInfo)
        try? DawnBootstrap.pump(instance: instance, until: { box.done }, what: "popErrorScope")
        return box.error
    }

    var hasPendingWork: Bool { commandEncoder != nil }

    func ensureSubmittableWork() {
        _ = try? ensureEncoder()
    }

    func submit(present: Bool, onCompleted: @escaping (WGPUError?) -> Void) {
        endOpenPasses()
        guard let encoder = commandEncoder else { return }
        if let commandBuffer = wgpuCommandEncoderFinish(encoder, nil) {
            var submission: WGPUCommandBuffer? = commandBuffer
            wgpuQueueSubmit(queue, 1, &submission)
            wgpuCommandBufferRelease(commandBuffer)
        }
        wgpuCommandEncoderRelease(encoder)
        commandEncoder = nil

        // present는 제출 **뒤**다 (Metal이 commit 전 present인 것과 반대 — 백엔드 규칙).
        if present {
            for (_, canvas) in presentTargets { canvas.present() }
            presentTargets.removeAll()
        }
        // GPU 실패는 uncaptured/디바이스 로스트 통로로 온다 — 완료 통지는 성공으로 보낸다.
        // 콜백은 processEvents 펌프(호스트 틱)에서 돌아온다.
        var callbackInfo = WGPUQueueWorkDoneCallbackInfo()
        callbackInfo.mode = WGPUCallbackMode_AllowProcessEvents
        callbackInfo.callback = { _, _, userdata1, _ in
            guard let userdata1 else { return }
            Unmanaged<WorkDoneBox>.fromOpaque(userdata1).takeRetainedValue().body(nil)
        }
        callbackInfo.userdata1 = Unmanaged.passRetained(WorkDoneBox(onCompleted)).toOpaque()
        _ = wgpuQueueOnSubmittedWorkDone(queue, callbackInfo)
    }

    private final class WorkDoneBox {
        let body: (WGPUError?) -> Void
        init(_ body: @escaping (WGPUError?) -> Void) { self.body = body }
    }

    // MARK: - 인코더 수명

    private func ensureEncoder() throws -> WGPUCommandEncoder {
        if let commandEncoder { return commandEncoder }
        // 디바이스 로스트 등으로 nil이 올 수 있다 — 강제 언랩은 곧 크래시다.
        guard let encoder = wgpuDeviceCreateCommandEncoder(device, nil) else {
            throw WGPUError.backend("커맨드 인코더 생성 실패 (디바이스 로스트?)")
        }
        commandEncoder = encoder
        return encoder
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

    func endPass() {
        endOpenPasses()
    }

    // MARK: - 버퍼

    func makeBuffer(_ descriptor: LynxWebGPUCore.WGPUBufferDescriptor) throws -> DawnBufferObject {
        let arena = DawnArena()
        var dawnDescriptor = WebGPU.WGPUBufferDescriptor()
        dawnDescriptor.label = arena.string(descriptor.label)
        dawnDescriptor.size = try dawnU64(descriptor.size, "size")
        dawnDescriptor.usage = WGPUBufferUsage(UInt64(truncatingIfNeeded: descriptor.usage.rawValue))
        let needsInitialData = descriptor.initialData != nil
        dawnDescriptor.mappedAtCreation = (descriptor.mappedAtCreation || needsInitialData) ? 1 : 0

        guard let buffer = wgpuDeviceCreateBuffer(device, &dawnDescriptor) else {
            throw WGPUError.outOfMemory("GPUBuffer 생성 실패")
        }
        if let data = descriptor.initialData, !data.isEmpty {
            if let mapped = wgpuBufferGetMappedRange(buffer, 0, data.count) {
                data.withUnsafeBytes { source in
                    guard let base = source.baseAddress else { return }
                    mapped.copyMemory(from: base, byteCount: data.count)
                }
            }
        }
        let object = DawnBufferObject(buffer: buffer, size: descriptor.size, usage: descriptor.usage)
        if descriptor.mappedAtCreation {
            object.dawnMapped = true
        } else if needsInitialData {
            wgpuBufferUnmap(buffer)
        }
        return object
    }

    func writeBuffer(_ buffer: DawnBufferObject, offset: Int, data: Data) throws {
        // offset은 엔진이 범위를 확인했다 (음수 없음).
        data.withUnsafeBytes { source in
            wgpuQueueWriteBuffer(queue, buffer.buffer, UInt64(offset), source.baseAddress, source.count)
        }
    }

    func unmapBuffer(_ buffer: DawnBufferObject) {
        if buffer.dawnMapped {
            wgpuBufferUnmap(buffer.buffer)
            buffer.dawnMapped = false
        }
    }

    func readBuffer(
        _ buffer: DawnBufferObject, offset: Int, length: Int,
        deliver: @escaping (Result<Data, WGPUError>) -> Void
    ) {
        // MAP_READ가 있으면 직접 매핑, 없으면 스테이징 복사 (COPY_SRC 필요).
        if buffer.usage.contains(.mapRead) {
            mapAndDeliver(
                buffer: buffer.buffer, mapSize: buffer.size,
                readOffset: offset, readLength: length,
                releaseAfter: nil, deliver: deliver
            )
        } else if buffer.usage.contains(.copySrc) {
            var descriptor = WebGPU.WGPUBufferDescriptor()
            let alignedLength = (max(length, 0) + 3) / 4 * 4
            descriptor.size = UInt64(alignedLength)
            descriptor.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead
            guard let staging = wgpuDeviceCreateBuffer(device, &descriptor),
                  let encoder = wgpuDeviceCreateCommandEncoder(device, nil) else {
                deliver(.failure(WGPUError.outOfMemory("리드백 스테이징 생성 실패")))
                return
            }
            wgpuCommandEncoderCopyBufferToBuffer(
                encoder, buffer.buffer, UInt64(offset), staging, 0, UInt64(alignedLength)
            )
            if let commandBuffer = wgpuCommandEncoderFinish(encoder, nil) {
                var submission: WGPUCommandBuffer? = commandBuffer
                wgpuQueueSubmit(queue, 1, &submission)
                wgpuCommandBufferRelease(commandBuffer)
            }
            wgpuCommandEncoderRelease(encoder)
            mapAndDeliver(
                buffer: staging, mapSize: alignedLength,
                readOffset: 0, readLength: length,
                releaseAfter: staging, deliver: deliver
            )
        } else {
            deliver(.failure(WGPUError.validation(
                "readBuffer에는 MAP_READ 또는 COPY_SRC usage가 필요하다"
            )))
        }
    }

    /// mapAsync를 걸고, 콜백(processEvents 펌프에서 온다)에서 바이트를 떠 deliver를 부른다.
    private func mapAndDeliver(
        buffer: WGPUBuffer,
        mapSize: Int,
        readOffset: Int,
        readLength: Int,
        releaseAfter: WGPUBuffer?,
        deliver: @escaping (Result<Data, WGPUError>) -> Void
    ) {
        final class MapContext {
            let buffer: WGPUBuffer
            let readOffset: Int
            let readLength: Int
            let releaseAfter: WGPUBuffer?
            let deliver: (Result<Data, WGPUError>) -> Void
            init(buffer: WGPUBuffer, readOffset: Int, readLength: Int,
                 releaseAfter: WGPUBuffer?, deliver: @escaping (Result<Data, WGPUError>) -> Void) {
                self.buffer = buffer
                self.readOffset = readOffset
                self.readLength = readLength
                self.releaseAfter = releaseAfter
                self.deliver = deliver
            }
        }
        let context = MapContext(
            buffer: buffer, readOffset: readOffset, readLength: readLength,
            releaseAfter: releaseAfter, deliver: deliver
        )
        var callbackInfo = WGPUBufferMapCallbackInfo()
        callbackInfo.mode = WGPUCallbackMode_AllowProcessEvents
        callbackInfo.callback = { status, message, userdata1, _ in
            guard let userdata1 else { return }
            let context = Unmanaged<MapContext>.fromOpaque(userdata1).takeRetainedValue()
            defer { if let staging = context.releaseAfter { wgpuBufferRelease(staging) } }
            guard status == WGPUMapAsyncStatus_Success else {
                context.deliver(.failure(WGPUError.backend(
                    "mapAsync 실패 — \(String(wgpu: message))"
                )))
                return
            }
            guard context.readLength >= 0, let pointer = wgpuBufferGetConstMappedRange(
                context.buffer, context.readOffset, context.readLength
            ) else {
                wgpuBufferUnmap(context.buffer)
                context.deliver(.failure(WGPUError.backend(
                    "매핑 범위를 얻지 못했다 (offset \(context.readOffset), length \(context.readLength))"
                )))
                return
            }
            let data = Data(bytes: pointer, count: context.readLength)
            wgpuBufferUnmap(context.buffer)
            context.deliver(.success(data))
        }
        callbackInfo.userdata1 = Unmanaged.passRetained(context).toOpaque()
        _ = wgpuBufferMapAsync(buffer, WGPUMapMode_Read, 0, mapSize, callbackInfo)
    }

    // MARK: - 텍스처

    func makeTexture(_ descriptor: LynxWebGPUCore.WGPUTextureDescriptor) throws -> DawnTextureObject {
        let arena = DawnArena()
        var dawnDescriptor = WebGPU.WGPUTextureDescriptor()
        dawnDescriptor.label = arena.string(descriptor.label)
        dawnDescriptor.usage = WGPUTextureUsage(UInt64(truncatingIfNeeded: descriptor.usage.rawValue))
        dawnDescriptor.dimension = DawnEnum.textureDimension(descriptor.dimension)
        dawnDescriptor.size = try DawnEnum.extent(descriptor.size, field: "size")
        dawnDescriptor.format = try DawnEnum.textureFormat(descriptor.format)
        dawnDescriptor.mipLevelCount = try dawnU32(descriptor.mipLevelCount, "mipLevelCount")
        dawnDescriptor.sampleCount = try dawnU32(descriptor.sampleCount, "sampleCount")
        guard let texture = wgpuDeviceCreateTexture(device, &dawnDescriptor) else {
            throw WGPUError.outOfMemory("GPUTexture 생성 실패")
        }
        return DawnTextureObject(
            texture: texture, format: descriptor.format,
            width: descriptor.size.width, height: descriptor.size.height
        )
    }

    func writeTexture(
        _ texture: DawnTextureObject, data: Data, origin: LynxWebGPUCore.WGPUOrigin3D,
        size: LynxWebGPUCore.WGPUExtent3D, mipLevel: Int, bytesPerRow: Int, rowsPerImage: Int
    ) throws {
        var destination = WGPUTexelCopyTextureInfo()
        destination.texture = texture.texture
        destination.mipLevel = try dawnU32(mipLevel, "mipLevel")
        destination.origin = try DawnEnum.origin(origin, field: "origin")
        destination.aspect = WGPUTextureAspect_All

        var layout = WGPUTexelCopyBufferLayout()
        layout.offset = 0
        layout.bytesPerRow = try dawnU32(bytesPerRow, "bytesPerRow")
        layout.rowsPerImage = try dawnU32(rowsPerImage, "rowsPerImage")

        var dawnSize = try DawnEnum.extent(size, field: "size")
        data.withUnsafeBytes { source in
            wgpuQueueWriteTexture(
                queue, &destination, source.baseAddress, source.count, &layout, &dawnSize
            )
        }
    }

    func makeTextureView(
        _ texture: DawnTextureObject, descriptor: LynxWebGPUCore.WGPUTextureViewDescriptor,
        format: LynxWebGPUCore.WGPUTextureFormat
    ) throws -> DawnTextureViewObject {
        let arena = DawnArena()
        var dawnDescriptor = WebGPU.WGPUTextureViewDescriptor()
        dawnDescriptor.label = arena.string(descriptor.label)
        dawnDescriptor.format = try descriptor.format.map(DawnEnum.textureFormat)
            ?? WGPUTextureFormat_Undefined
        dawnDescriptor.dimension = DawnEnum.viewDimension(descriptor.dimension)
        dawnDescriptor.baseMipLevel = try dawnU32(descriptor.baseMipLevel, "baseMipLevel")
        dawnDescriptor.mipLevelCount = try descriptor.mipLevelCount
            .map { try dawnU32($0, "mipLevelCount") } ?? UInt32.max
        dawnDescriptor.baseArrayLayer = try dawnU32(descriptor.baseArrayLayer, "baseArrayLayer")
        dawnDescriptor.arrayLayerCount = try descriptor.arrayLayerCount
            .map { try dawnU32($0, "arrayLayerCount") } ?? UInt32.max
        dawnDescriptor.aspect = DawnEnum.aspect(descriptor.aspect)
        guard let view = wgpuTextureCreateView(texture.texture, &dawnDescriptor) else {
            throw WGPUError.backend("GPUTextureView 생성 실패")
        }
        return DawnTextureViewObject(view: view)
    }

    func makeSampler(_ descriptor: LynxWebGPUCore.WGPUSamplerDescriptor) throws -> DawnSamplerObject {
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
        dawnDescriptor.maxAnisotropy = try dawnU16(descriptor.maxAnisotropy, "maxAnisotropy")
        guard let sampler = wgpuDeviceCreateSampler(device, &dawnDescriptor) else {
            throw WGPUError.backend("GPUSampler 생성 실패")
        }
        return DawnSamplerObject(sampler: sampler)
    }

    // MARK: - 셰이더 · 레이아웃

    func makeShaderModule(
        _ descriptor: LynxWebGPUCore.WGPUShaderModuleDescriptor, fieldPath: (String) -> String?
    ) throws -> WGPUShaderModuleCreation<DawnBackend> {
        guard descriptor.language == .wgsl else {
            // msl은 Metal 런타임의 탈출구다 — 선택 기능이라 깨끗이 거부한다
            // (`docs/COMMAND-STREAM.md` §4-1, 적합성 `msl-optional`).
            throw WGPUError.unsupported(
                "Dawn 런타임은 language \"\(descriptor.language.rawValue)\"을(를) 지원하지 않는다 (wgsl만)",
                path: fieldPath("language")
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
            throw WGPUError.backend("GPUShaderModule 생성 실패")
        }
        // 컴파일 진단은 디바이스 스코프(collectBatchDiagnostics)와 getCompilationInfo로 나온다.
        return WGPUShaderModuleCreation(module: DawnShaderModuleObject(module: module), failure: nil)
    }

    func compilationMessages(of module: DawnShaderModuleObject) -> [LynxWebGPUCore.WGPUCompilationMessage] {
        final class InfoBox { var done = false; var messages: [LynxWebGPUCore.WGPUCompilationMessage] = [] }
        let box = InfoBox()
        var callbackInfo = WGPUCompilationInfoCallbackInfo()
        callbackInfo.mode = WGPUCallbackMode_AllowProcessEvents
        callbackInfo.callback = { _, info, userdata1, _ in
            guard let userdata1 else { return }
            let box = Unmanaged<InfoBox>.fromOpaque(userdata1).takeRetainedValue()
            if let info = info?.pointee, let messages = info.messages {
                for index in 0..<info.messageCount {
                    let message = messages[index]
                    let type: String
                    if message.type == WGPUCompilationMessageType_Warning { type = "warning" }
                    else if message.type == WGPUCompilationMessageType_Info { type = "info" }
                    else { type = "error" }
                    box.messages.append(LynxWebGPUCore.WGPUCompilationMessage(
                        message: String(wgpu: message.message),
                        type: type,
                        lineNum: Int(clamping: message.lineNum),
                        linePos: Int(clamping: message.linePos),
                        offset: Int(clamping: message.offset),
                        length: Int(clamping: message.length)
                    ))
                }
            }
            box.done = true
        }
        callbackInfo.userdata1 = Unmanaged.passRetained(box).toOpaque()
        _ = wgpuShaderModuleGetCompilationInfo(module.module, callbackInfo)
        try? DawnBootstrap.pump(instance: instance, until: { box.done }, what: "getCompilationInfo")
        return box.messages
    }

    func makeBindGroupLayout(
        _ entries: [LynxWebGPUCore.WGPUBindGroupLayoutEntry]
    ) throws -> DawnBindGroupLayoutObject {
        let arena = DawnArena()
        let dawnEntries = try entries.map { entry -> WebGPU.WGPUBindGroupLayoutEntry in
            var dawnEntry = WebGPU.WGPUBindGroupLayoutEntry()
            dawnEntry.binding = try dawnU32(entry.binding, "entries.binding")
            dawnEntry.visibility = WGPUShaderStage(UInt64(truncatingIfNeeded: entry.visibility.rawValue))
            switch entry.layout {
            case .buffer(let buffer):
                dawnEntry.buffer.type = DawnEnum.bufferBindingType(buffer.type)
                dawnEntry.buffer.hasDynamicOffset = buffer.hasDynamicOffset ? 1 : 0
                dawnEntry.buffer.minBindingSize = try dawnU64(buffer.minBindingSize, "entries.buffer.minBindingSize")
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
        dawnDescriptor.entryCount = dawnEntries.count
        dawnDescriptor.entries = arena.array(dawnEntries)
        guard let layout = wgpuDeviceCreateBindGroupLayout(device, &dawnDescriptor) else {
            throw WGPUError.backend("GPUBindGroupLayout 생성 실패")
        }
        return DawnBindGroupLayoutObject(layout: layout)
    }

    func makePipelineLayout(_ groups: [DawnBindGroupLayoutObject]) throws -> DawnPipelineLayoutObject {
        let arena = DawnArena()
        let layouts = groups.map { $0.layout as WGPUBindGroupLayout? }
        var dawnDescriptor = WebGPU.WGPUPipelineLayoutDescriptor()
        dawnDescriptor.bindGroupLayoutCount = layouts.count
        dawnDescriptor.bindGroupLayouts = arena.array(layouts)
        guard let layout = wgpuDeviceCreatePipelineLayout(device, &dawnDescriptor) else {
            throw WGPUError.backend("GPUPipelineLayout 생성 실패")
        }
        return DawnPipelineLayoutObject(layout: layout)
    }

    func makeBindGroup(
        layout: DawnBindGroupLayoutObject,
        entries: [WGPUResolvedBindGroupEntry<DawnBackend>]
    ) throws -> DawnBindGroupObject {
        let arena = DawnArena()
        let dawnEntries = try entries.map { entry -> WebGPU.WGPUBindGroupEntry in
            var dawnEntry = WebGPU.WGPUBindGroupEntry()
            dawnEntry.binding = try dawnU32(entry.binding, "entries.binding")
            switch entry.resource {
            case .buffer(let buffer, let offset, let boundSize):
                dawnEntry.buffer = buffer.buffer
                dawnEntry.offset = try dawnU64(offset, "entries.offset")
                dawnEntry.size = try dawnU64(boundSize, "entries.size")
            case .sampler(let sampler):
                dawnEntry.sampler = sampler.sampler
            case .textureView(let view):
                dawnEntry.textureView = view.view
            }
            return dawnEntry
        }
        var dawnDescriptor = WebGPU.WGPUBindGroupDescriptor()
        dawnDescriptor.layout = layout.layout
        dawnDescriptor.entryCount = dawnEntries.count
        dawnDescriptor.entries = arena.array(dawnEntries)
        guard let group = wgpuDeviceCreateBindGroup(device, &dawnDescriptor) else {
            throw WGPUError.backend("GPUBindGroup 생성 실패")
        }
        return DawnBindGroupObject(group: group)
    }

    func makeQuerySet(_ descriptor: LynxWebGPUCore.WGPUQuerySetDescriptor) throws -> DawnQuerySetObject {
        let arena = DawnArena()
        var dawnDescriptor = WebGPU.WGPUQuerySetDescriptor()
        dawnDescriptor.label = arena.string(descriptor.label)
        dawnDescriptor.type = DawnEnum.queryType(descriptor.type)
        dawnDescriptor.count = try dawnU32(descriptor.count, "count")
        guard let querySet = wgpuDeviceCreateQuerySet(device, &dawnDescriptor) else {
            throw WGPUError.backend("GPUQuerySet 생성 실패")
        }
        return DawnQuerySetObject(querySet: querySet, type: descriptor.type, count: descriptor.count)
    }

    // MARK: - 파이프라인

    private func resolveLayout(_ layout: WGPUResolvedPipelineLayout<DawnBackend>) -> WGPUPipelineLayout? {
        switch layout {
        case .auto: return nil   // Dawn의 auto 레이아웃 — 명세와 같은 의미다
        case .explicit(let object): return object.layout
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

    func makeRenderPipeline(
        _ descriptor: LynxWebGPUCore.WGPURenderPipelineDescriptor,
        vertexModule: DawnShaderModuleObject, fragmentModule: DawnShaderModuleObject?,
        layout: WGPUResolvedPipelineLayout<DawnBackend>,
        fieldPath: (String) -> String?
    ) throws -> WGPURenderPipelineCreation<DawnBackend> {
        let arena = DawnArena()
        var dawnDescriptor = WebGPU.WGPURenderPipelineDescriptor()
        dawnDescriptor.label = arena.string(descriptor.label)
        dawnDescriptor.layout = resolveLayout(layout)

        // vertex — 진입점 생략은 Dawn의 네이티브 기본 규칙(유일 진입점)에 맡긴다.
        var vertex = WGPUVertexState()
        vertex.module = vertexModule.module
        vertex.entryPoint = arena.string(descriptor.vertex.entryPoint)
        let vertexConstants = constantEntries(descriptor.vertex.constants, arena: arena)
        vertex.constantCount = vertexConstants.count
        vertex.constants = arena.array(vertexConstants)
        let vertexBuffers = try descriptor.vertex.buffers.map { layout -> WebGPU.WGPUVertexBufferLayout in
            var dawnLayout = WebGPU.WGPUVertexBufferLayout()
            dawnLayout.stepMode = DawnEnum.stepMode(layout.stepMode)
            dawnLayout.arrayStride = try dawnU64(
                layout.arrayStride, fieldPath("vertex.buffers.arrayStride") ?? "arrayStride"
            )
            let attributes = try layout.attributes.map { attribute -> WebGPU.WGPUVertexAttribute in
                var dawnAttribute = WebGPU.WGPUVertexAttribute()
                dawnAttribute.format = try DawnEnum.vertexFormat(attribute.format)
                dawnAttribute.offset = try dawnU64(
                    attribute.offset, fieldPath("vertex.buffers.attributes.offset") ?? "offset"
                )
                dawnAttribute.shaderLocation = try dawnU32(
                    attribute.shaderLocation,
                    fieldPath("vertex.buffers.attributes.shaderLocation") ?? "shaderLocation"
                )
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
            dawnDepthStencil.depthBias = try dawnI32(
                depthStencil.depthBias, fieldPath("depthStencil.depthBias") ?? "depthBias"
            )
            dawnDepthStencil.depthBiasSlopeScale = Float(depthStencil.depthBiasSlopeScale)
            dawnDepthStencil.depthBiasClamp = Float(depthStencil.depthBiasClamp)
            dawnDescriptor.depthStencil = arena.value(dawnDepthStencil)
        }

        // multisample
        var multisample = WGPUMultisampleState()
        multisample.count = try dawnU32(
            descriptor.multisample.count, fieldPath("multisample.count") ?? "count"
        )
        multisample.mask = UInt32(truncatingIfNeeded: descriptor.multisample.mask)
        multisample.alphaToCoverageEnabled = descriptor.multisample.alphaToCoverageEnabled ? 1 : 0
        dawnDescriptor.multisample = multisample

        // fragment
        if let fragment = descriptor.fragment, let fragmentModule {
            var dawnFragment = WGPUFragmentState()
            dawnFragment.module = fragmentModule.module
            dawnFragment.entryPoint = arena.string(fragment.entryPoint)
            let fragmentConstants = constantEntries(fragment.constants, arena: arena)
            dawnFragment.constantCount = fragmentConstants.count
            dawnFragment.constants = arena.array(fragmentConstants)
            let targets = try fragment.targets.map { target -> WebGPU.WGPUColorTargetState in
                var dawnTarget = WebGPU.WGPUColorTargetState()
                dawnTarget.format = try DawnEnum.textureFormat(target.format)
                dawnTarget.writeMask = WGPUColorWriteMask(UInt64(truncatingIfNeeded: target.writeMask.rawValue))
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
            throw WGPUError.backend("GPURenderPipeline 생성 실패")
        }
        // 드로우 전 검사 메타데이터는 주지 않는다 — Dawn이 네이티브로 검증하고,
        // 오류는 디바이스 스코프로 이 배치 결과에 실린다.
        return WGPURenderPipelineCreation(
            pipeline: DawnRenderPipelineObject(pipeline: pipeline),
            info: WGPURenderPipelineInfo()
        )
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

    func makeComputePipeline(
        _ descriptor: LynxWebGPUCore.WGPUComputePipelineDescriptor,
        module: DawnShaderModuleObject,
        layout: WGPUResolvedPipelineLayout<DawnBackend>,
        fieldPath: (String) -> String?
    ) throws -> WGPUComputePipelineCreation<DawnBackend> {
        let arena = DawnArena()
        var dawnDescriptor = WebGPU.WGPUComputePipelineDescriptor()
        dawnDescriptor.label = arena.string(descriptor.label)
        dawnDescriptor.layout = resolveLayout(layout)
        var compute = WGPUComputeState()
        compute.module = module.module
        compute.entryPoint = arena.string(descriptor.entryPoint)
        let constants = constantEntries(descriptor.constants, arena: arena)
        compute.constantCount = constants.count
        compute.constants = arena.array(constants)
        dawnDescriptor.compute = compute
        guard let pipeline = wgpuDeviceCreateComputePipeline(device, &dawnDescriptor) else {
            throw WGPUError.backend("GPUComputePipeline 생성 실패")
        }
        return WGPUComputePipelineCreation(
            pipeline: DawnComputePipelineObject(pipeline: pipeline),
            info: WGPUComputePipelineInfo()
        )
    }

    func bindGroupLayout(
        of pipeline: WGPUResolvedPipeline<DawnBackend>, index: Int
    ) throws -> WGPUBindGroupLayoutCreation<DawnBackend>? {
        let dawnIndex = try dawnU32(index, "index")
        let layout: WGPUBindGroupLayout?
        switch pipeline {
        case .render(let render):
            layout = wgpuRenderPipelineGetBindGroupLayout(render.pipeline, dawnIndex)
        case .compute(let compute):
            layout = wgpuComputePipelineGetBindGroupLayout(compute.pipeline, dawnIndex)
        }
        guard let layout else { return nil }
        // 항목은 모른다 (Dawn은 파생 레이아웃의 구성을 되묻는 API가 없다) —
        // 바인드 그룹 항목 매칭은 Dawn의 네이티브 검증에 맡겨진다.
        return WGPUBindGroupLayoutCreation(
            layout: DawnBindGroupLayoutObject(layout: layout), entries: nil
        )
    }

    // MARK: - 렌더 번들 (네이티브)

    func makeRenderBundle(
        _ descriptor: LynxWebGPUCore.WGPURenderBundleDescriptor, commands: [WGPUCommand],
        resolver: WGPUBundleResolver<DawnBackend>
    ) throws -> DawnRenderBundleObject {
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
        dawnDescriptor.sampleCount = try dawnU32(descriptor.sampleCount, "sampleCount")
        dawnDescriptor.depthReadOnly = descriptor.depthReadOnly ? 1 : 0
        dawnDescriptor.stencilReadOnly = descriptor.stencilReadOnly ? 1 : 0

        guard let encoder = wgpuDeviceCreateRenderBundleEncoder(device, &dawnDescriptor) else {
            throw WGPUError.backend("GPURenderBundleEncoder 생성 실패")
        }
        defer { wgpuRenderBundleEncoderRelease(encoder) }

        // 엔진이 디코딩·op 검증을 끝낸 명령을 **진짜 Dawn 번들**로 굽는다 —
        // Metal 백엔드의 record/replay와 달리 여기는 명세의 렌더 번들 그 자체다.
        for bundleCommand in commands {
            switch bundleCommand {
            case .setPipeline(let c):
                let pipeline = try resolver.renderPipeline(c.pipeline, c.fieldPath("pipeline"))
                wgpuRenderBundleEncoderSetPipeline(encoder, pipeline.pipeline)
            case .setBindGroup(let c):
                let group = try resolver.bindGroup(c.bindGroup, c.fieldPath("bindGroup"))
                let offsets = try c.dynamicOffsets.map {
                    try dawnU32($0, c.fieldPath("dynamicOffsets") ?? "dynamicOffsets")
                }
                let groupIndex = try dawnU32(c.index, c.fieldPath("index") ?? "index")
                offsets.withUnsafeBufferPointer { pointer in
                    wgpuRenderBundleEncoderSetBindGroup(
                        encoder, groupIndex, group.group, pointer.count, pointer.baseAddress
                    )
                }
            case .setVertexBuffer(let c):
                let buffer = try resolver.buffer(c.buffer, c.fieldPath("buffer"))
                wgpuRenderBundleEncoderSetVertexBuffer(
                    encoder, try dawnU32(c.slot, c.fieldPath("slot") ?? "slot"), buffer.buffer,
                    try dawnU64(c.offset, c.fieldPath("offset") ?? "offset"), UInt64.max
                )
            case .setIndexBuffer(let c):
                let buffer = try resolver.buffer(c.buffer, c.fieldPath("buffer"))
                wgpuRenderBundleEncoderSetIndexBuffer(
                    encoder, buffer.buffer, DawnEnum.indexFormat(c.format),
                    try dawnU64(c.offset, c.fieldPath("offset") ?? "offset"), UInt64.max
                )
            case .draw(let c):
                wgpuRenderBundleEncoderDraw(
                    encoder,
                    try dawnU32(c.vertexCount, c.fieldPath("vertexCount") ?? "vertexCount"),
                    try dawnU32(c.instanceCount, c.fieldPath("instanceCount") ?? "instanceCount"),
                    try dawnU32(c.firstVertex, c.fieldPath("firstVertex") ?? "firstVertex"),
                    try dawnU32(c.firstInstance, c.fieldPath("firstInstance") ?? "firstInstance")
                )
            case .drawIndexed(let c):
                wgpuRenderBundleEncoderDrawIndexed(
                    encoder,
                    try dawnU32(c.indexCount, c.fieldPath("indexCount") ?? "indexCount"),
                    try dawnU32(c.instanceCount, c.fieldPath("instanceCount") ?? "instanceCount"),
                    try dawnU32(c.firstIndex, c.fieldPath("firstIndex") ?? "firstIndex"),
                    try dawnI32(c.baseVertex, c.fieldPath("baseVertex") ?? "baseVertex"),
                    try dawnU32(c.firstInstance, c.fieldPath("firstInstance") ?? "firstInstance")
                )
            case .drawIndirect(let c):
                try ensureIndirectSupported()
                let buffer = try resolver.buffer(c.indirectBuffer, c.fieldPath("indirectBuffer"))
                wgpuRenderBundleEncoderDrawIndirect(
                    encoder, buffer.buffer,
                    try dawnU64(c.indirectOffset, c.fieldPath("indirectOffset") ?? "indirectOffset")
                )
            case .drawIndexedIndirect(let c):
                try ensureIndirectSupported()
                let buffer = try resolver.buffer(c.indirectBuffer, c.fieldPath("indirectBuffer"))
                wgpuRenderBundleEncoderDrawIndexedIndirect(
                    encoder, buffer.buffer,
                    try dawnU64(c.indirectOffset, c.fieldPath("indirectOffset") ?? "indirectOffset")
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
                // 엔진의 allowedOps 검증이 먼저 걸러 도달할 수 없다.
                throw WGPUError.validation(
                    "렌더 번들에서 쓸 수 없는 명령 '\(bundleCommand.opName)'"
                )
            }
        }

        guard let bundle = wgpuRenderBundleEncoderFinish(encoder, nil) else {
            throw WGPUError.backend("GPURenderBundle 생성 실패")
        }
        return DawnRenderBundleObject(bundle: bundle)
    }

    func executeBundles(_ bundles: [DawnRenderBundleObject]) throws {
        guard let renderPass else { return }
        let handles = bundles.map { $0.bundle as WGPURenderBundle? }
        handles.withUnsafeBufferPointer { pointer in
            wgpuRenderPassEncoderExecuteBundles(renderPass, pointer.count, pointer.baseAddress)
        }
    }

    // MARK: - 표면

    func makeLayerSurface(identifier: String, layer: CAMetalLayer) -> WGPUSurfaceCreation<DawnBackend> {
        WGPUSurfaceCreation(
            surface: DawnLayerCanvas(identifier: identifier, layer: layer, instance: instance),
            pacesFrames: true
        )
    }

    func makeOffscreenSurface(identifier: String, size: CGSize) throws -> WGPUSurfaceCreation<DawnBackend> {
        WGPUSurfaceCreation(
            surface: DawnOffscreenCanvas(identifier: identifier, size: size),
            pacesFrames: false
        )
    }

    func configureSurface(
        _ surface: DawnCanvas, configuration: LynxWebGPUCore.WGPUCanvasConfiguration
    ) throws {
        try surface.configure(device: device, format: configuration.format)
    }

    func resizeSurface(_ surface: DawnCanvas, size: CGSize) {
        // NaN·음수·비유한 크기는 이후 모든 UInt32 변환의 트랩 씨앗이다 — 입구에서 거른다.
        guard size.width.isFinite, size.height.isFinite,
              size.width >= 0, size.height >= 0 else { return }
        surface.updateSize(size, device: device)
    }

    func surfaceReport(_ surface: DawnCanvas) -> WGPUSurfaceReport {
        WGPUSurfaceReport(
            width: Int(surface.size.width),
            height: Int(surface.size.height),
            format: surface.format
        )
    }

    func acquireFrameTexture(_ surface: DawnCanvas) throws -> WGPUAcquiredSurfaceTexture<DawnBackend>? {
        let (texture, owned) = try surface.acquireTexture(device: device)
        presentTargets[surface.identifier] = surface
        return WGPUAcquiredSurfaceTexture(
            texture: DawnTextureObject(
                texture: texture, format: surface.format,
                width: Int(surface.size.width), height: Int(surface.size.height),
                retain: !owned   // 표면 텍스처는 이미 +1로 나온다 — 빌린 것만 retain
            ),
            format: surface.format,
            width: Int(surface.size.width),
            height: Int(surface.size.height)
        )
    }

    func readPixels(_ surface: DawnCanvas, identifier: String) throws -> WGPUPixelReadback {
        // 픽셀 읽기는 오프스크린 통로다 — 화면 표면 텍스처는 present와 함께 사라진다.
        guard let canvas = surface as? DawnOffscreenCanvas, let texture = canvas.texture else {
            throw WGPUError.validation(
                "캔버스 '\(identifier)'은(는) 오프스크린 표면이 아니거나 configure 전이다"
            )
        }
        let dimensions = try dawnTextureDimensions(canvas.size, "캔버스 '\(identifier)'")
        let width = Int(dimensions.width)
        let height = Int(dimensions.height)
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

        var source = WGPUTexelCopyTextureInfo()
        source.texture = texture
        source.mipLevel = 0
        source.origin = WebGPU.WGPUOrigin3D(x: 0, y: 0, z: 0)
        source.aspect = WGPUTextureAspect_All
        var destination = WGPUTexelCopyBufferInfo()
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
            guard let userdata1 else { return }
            let box = Unmanaged<MapBox>.fromOpaque(userdata1).takeRetainedValue()
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

    // MARK: - 패스

    func beginRenderPass(_ pass: WGPUResolvedRenderPass<DawnBackend>) throws {
        let arena = DawnArena()
        let colorAttachments = pass.colorAttachments.map { attachment
            -> WebGPU.WGPURenderPassColorAttachment in
            var dawnAttachment = WebGPU.WGPURenderPassColorAttachment()
            dawnAttachment.view = attachment.view.view
            dawnAttachment.depthSlice = UInt32.max   // WGPU_DEPTH_SLICE_UNDEFINED
            if let resolveTarget = attachment.resolveTarget {
                dawnAttachment.resolveTarget = resolveTarget.view
            }
            dawnAttachment.loadOp = DawnEnum.loadOp(attachment.loadOp)
            dawnAttachment.storeOp = DawnEnum.storeOp(attachment.storeOp)
            dawnAttachment.clearValue = DawnEnum.color(attachment.clearValue)
            return dawnAttachment
        }

        var dawnDescriptor = WebGPU.WGPURenderPassDescriptor()
        dawnDescriptor.label = arena.string(pass.label)
        dawnDescriptor.colorAttachmentCount = colorAttachments.count
        dawnDescriptor.colorAttachments = arena.array(colorAttachments)

        if let depthStencil = pass.depthStencil {
            var dawnDepthStencil = WebGPU.WGPURenderPassDepthStencilAttachment()
            dawnDepthStencil.view = depthStencil.view.view
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

        if let occlusion = pass.occlusionQuerySet {
            dawnDescriptor.occlusionQuerySet = occlusion.querySet
        }

        if let writes = pass.timestampWrites {
            var dawnWrites = WebGPU.WGPUPassTimestampWrites()
            dawnWrites.querySet = writes.querySet.querySet
            // 생략된 자리는 WGPU_QUERY_SET_INDEX_UNDEFINED(=UInt32.max)다.
            dawnWrites.beginningOfPassWriteIndex = writes.beginningOfPassWriteIndex
                .map { UInt32($0) } ?? UInt32.max
            dawnWrites.endOfPassWriteIndex = writes.endOfPassWriteIndex
                .map { UInt32($0) } ?? UInt32.max
            dawnDescriptor.timestampWrites = arena.value(dawnWrites)
        }

        guard let encoder = wgpuCommandEncoderBeginRenderPass(try ensureEncoder(), &dawnDescriptor) else {
            throw WGPUError.backend("렌더 패스 시작 실패")
        }
        renderPass = encoder
    }

    func beginComputePass(_ pass: WGPUResolvedComputePass<DawnBackend>) throws {
        let arena = DawnArena()
        var dawnDescriptor = WebGPU.WGPUComputePassDescriptor()
        dawnDescriptor.label = arena.string(pass.label)
        if let writes = pass.timestampWrites {
            var dawnWrites = WebGPU.WGPUPassTimestampWrites()
            dawnWrites.querySet = writes.querySet.querySet
            dawnWrites.beginningOfPassWriteIndex = writes.beginningOfPassWriteIndex
                .map { UInt32($0) } ?? UInt32.max
            dawnWrites.endOfPassWriteIndex = writes.endOfPassWriteIndex
                .map { UInt32($0) } ?? UInt32.max
            dawnDescriptor.timestampWrites = arena.value(dawnWrites)
        }
        guard let encoder = wgpuCommandEncoderBeginComputePass(try ensureEncoder(), &dawnDescriptor) else {
            throw WGPUError.backend("컴퓨트 패스 시작 실패")
        }
        computePass = encoder
    }

    func setRenderPipeline(_ pipeline: DawnRenderPipelineObject) {
        guard let renderPass else { return }
        wgpuRenderPassEncoderSetPipeline(renderPass, pipeline.pipeline)
    }

    func setComputePipeline(_ pipeline: DawnComputePipelineObject) {
        guard let computePass else { return }
        wgpuComputePassEncoderSetPipeline(computePass, pipeline.pipeline)
    }

    func applyBindGroup(_ group: DawnBindGroupObject, at index: Int, dynamicOffsets: [Int]) throws {
        let offsets = try dynamicOffsets.map { try dawnU32($0, "dynamicOffsets") }
        let groupIndex = try dawnU32(index, "index")
        offsets.withUnsafeBufferPointer { pointer in
            if let renderPass {
                wgpuRenderPassEncoderSetBindGroup(
                    renderPass, groupIndex, group.group, pointer.count, pointer.baseAddress
                )
            } else if let computePass {
                wgpuComputePassEncoderSetBindGroup(
                    computePass, groupIndex, group.group, pointer.count, pointer.baseAddress
                )
            }
        }
    }

    func applyVertexBuffer(_ buffer: DawnBufferObject, offset: Int, slot: Int) {
        guard let renderPass else { return }
        // slot·offset은 엔진이 범위를 확인했다 (0 ≤ slot < 8, 0 ≤ offset ≤ size).
        wgpuRenderPassEncoderSetVertexBuffer(
            renderPass, UInt32(slot), buffer.buffer, UInt64(offset), UInt64.max
        )
    }

    func setViewport(_ command: WGPUSetViewportCommand) throws {
        guard let renderPass else { return }
        wgpuRenderPassEncoderSetViewport(
            renderPass, Float(command.x), Float(command.y),
            Float(command.width), Float(command.height),
            Float(command.minDepth), Float(command.maxDepth)
        )
    }

    func setScissorRect(_ command: WGPUSetScissorRectCommand) throws {
        guard let renderPass else { return }
        wgpuRenderPassEncoderSetScissorRect(
            renderPass,
            try dawnU32(command.x, command.fieldPath("x") ?? "x"),
            try dawnU32(command.y, command.fieldPath("y") ?? "y"),
            try dawnU32(command.width, command.fieldPath("width") ?? "width"),
            try dawnU32(command.height, command.fieldPath("height") ?? "height")
        )
    }

    func setBlendConstant(_ color: LynxWebGPUCore.WGPUColor) throws {
        guard let renderPass else { return }
        var dawnColor = DawnEnum.color(color)
        wgpuRenderPassEncoderSetBlendConstant(renderPass, &dawnColor)
    }

    func setStencilReference(_ reference: UInt32) throws {
        guard let renderPass else { return }
        wgpuRenderPassEncoderSetStencilReference(renderPass, reference)
    }

    func beginOcclusionQuery(index: Int) throws {
        guard let renderPass else { return }
        // index는 엔진이 범위·중첩·재사용을 확인했다.
        wgpuRenderPassEncoderBeginOcclusionQuery(renderPass, UInt32(index))
    }

    func endOcclusionQuery(index: Int) throws {
        guard let renderPass else { return }
        wgpuRenderPassEncoderEndOcclusionQuery(renderPass)
    }

    func pushDebugGroup(_ label: String, scope: WGPUDebugScope) throws {
        let arena = DawnArena()
        switch scope {
        case .pass:
            if let renderPass {
                wgpuRenderPassEncoderPushDebugGroup(renderPass, arena.string(label))
            } else if let computePass {
                wgpuComputePassEncoderPushDebugGroup(computePass, arena.string(label))
            }
        case .frame:
            wgpuCommandEncoderPushDebugGroup(try ensureEncoder(), arena.string(label))
        }
    }

    func popDebugGroup(scope: WGPUDebugScope) {
        switch scope {
        case .pass:
            if let renderPass {
                wgpuRenderPassEncoderPopDebugGroup(renderPass)
            } else if let computePass {
                wgpuComputePassEncoderPopDebugGroup(computePass)
            }
        case .frame:
            if let commandEncoder {
                wgpuCommandEncoderPopDebugGroup(commandEncoder)
            }
        }
    }

    func popFrameDebugGroups(count: Int) {
        guard let commandEncoder else { return }
        for _ in 0..<count { wgpuCommandEncoderPopDebugGroup(commandEncoder) }
    }

    func insertDebugMarker(_ label: String, scope: WGPUDebugScope) throws {
        let arena = DawnArena()
        switch scope {
        case .pass:
            if let renderPass {
                wgpuRenderPassEncoderInsertDebugMarker(renderPass, arena.string(label))
            } else if let computePass {
                wgpuComputePassEncoderInsertDebugMarker(computePass, arena.string(label))
            }
        case .frame:
            wgpuCommandEncoderInsertDebugMarker(try ensureEncoder(), arena.string(label))
        }
    }

    // MARK: - 드로우 / 디스패치

    func draw(_ command: WGPUDrawCommand) throws {
        guard let renderPass else { return }
        wgpuRenderPassEncoderDraw(
            renderPass,
            try dawnU32(command.vertexCount, command.fieldPath("vertexCount") ?? "vertexCount"),
            try dawnU32(command.instanceCount, command.fieldPath("instanceCount") ?? "instanceCount"),
            try dawnU32(command.firstVertex, command.fieldPath("firstVertex") ?? "firstVertex"),
            try dawnU32(command.firstInstance, command.fieldPath("firstInstance") ?? "firstInstance")
        )
    }

    func drawIndexed(
        _ command: WGPUDrawIndexedCommand, index: WGPUResolvedIndexBinding<DawnBackend>
    ) throws {
        guard let renderPass else { return }
        // 인덱스 바인딩은 엔진 그림자 상태에서 온다 — 드로우마다 다시 세팅해도
        // Dawn은 상태 변경으로만 처리하므로 비용이 없다.
        wgpuRenderPassEncoderSetIndexBuffer(
            renderPass, index.buffer.buffer, DawnEnum.indexFormat(index.format),
            UInt64(index.offset), UInt64.max
        )
        wgpuRenderPassEncoderDrawIndexed(
            renderPass,
            try dawnU32(command.indexCount, command.fieldPath("indexCount") ?? "indexCount"),
            try dawnU32(command.instanceCount, command.fieldPath("instanceCount") ?? "instanceCount"),
            try dawnU32(command.firstIndex, command.fieldPath("firstIndex") ?? "firstIndex"),
            try dawnI32(command.baseVertex, command.fieldPath("baseVertex") ?? "baseVertex"),
            try dawnU32(command.firstInstance, command.fieldPath("firstInstance") ?? "firstInstance")
        )
    }

    func drawIndirect(buffer: DawnBufferObject, offset: Int) throws {
        guard let renderPass else { return }
        // offset은 엔진이 정렬·범위를 확인했다.
        wgpuRenderPassEncoderDrawIndirect(renderPass, buffer.buffer, UInt64(offset))
    }

    func drawIndexedIndirect(
        buffer: DawnBufferObject, offset: Int, index: WGPUResolvedIndexBinding<DawnBackend>
    ) throws {
        guard let renderPass else { return }
        wgpuRenderPassEncoderSetIndexBuffer(
            renderPass, index.buffer.buffer, DawnEnum.indexFormat(index.format),
            UInt64(index.offset), UInt64.max
        )
        wgpuRenderPassEncoderDrawIndexedIndirect(renderPass, buffer.buffer, UInt64(offset))
    }

    func dispatchWorkgroups(_ command: WGPUDispatchWorkgroupsCommand) throws {
        guard let computePass else { return }
        wgpuComputePassEncoderDispatchWorkgroups(
            computePass,
            try dawnU32(command.x, command.fieldPath("x") ?? "x"),
            try dawnU32(command.y, command.fieldPath("y") ?? "y"),
            try dawnU32(command.z, command.fieldPath("z") ?? "z")
        )
    }

    func dispatchWorkgroupsIndirect(buffer: DawnBufferObject, offset: Int) throws {
        guard let computePass else { return }
        wgpuComputePassEncoderDispatchWorkgroupsIndirect(computePass, buffer.buffer, UInt64(offset))
    }

    // MARK: - 복사

    func copyBufferToBuffer(
        source: DawnBufferObject, sourceOffset: Int,
        destination: DawnBufferObject, destinationOffset: Int, size: Int
    ) throws {
        // 범위·부호는 엔진이 확인했다.
        wgpuCommandEncoderCopyBufferToBuffer(
            try ensureEncoder(), source.buffer, UInt64(sourceOffset),
            destination.buffer, UInt64(destinationOffset), UInt64(size)
        )
    }

    func clearBuffer(_ buffer: DawnBufferObject, range: Range<Int>) throws {
        wgpuCommandEncoderClearBuffer(
            try ensureEncoder(), buffer.buffer,
            UInt64(range.lowerBound), UInt64(range.count)
        )
    }

    func copyTextureToBuffer(
        texture: DawnTextureObject, slice: Int, mipLevel: Int, origin: LynxWebGPUCore.WGPUOrigin3D,
        size: LynxWebGPUCore.WGPUExtent3D, buffer: DawnBufferObject, offset: Int,
        bytesPerRow: Int, bytesPerImage: Int
    ) throws {
        var source = WGPUTexelCopyTextureInfo()
        source.texture = texture.texture
        source.mipLevel = try dawnU32(mipLevel, "mipLevel")
        source.origin = try DawnEnum.origin(origin, field: "origin")
        source.aspect = WGPUTextureAspect_All

        var destination = WGPUTexelCopyBufferInfo()
        destination.buffer = buffer.buffer
        destination.layout.offset = try dawnU64(offset, "offset")
        destination.layout.bytesPerRow = try dawnU32(bytesPerRow, "bytesPerRow")
        destination.layout.rowsPerImage = try dawnU32(
            bytesPerRow > 0 ? bytesPerImage / bytesPerRow : 0, "rowsPerImage"
        )

        var dawnSize = try DawnEnum.extent(size, field: "copySize")
        wgpuCommandEncoderCopyTextureToBuffer(try ensureEncoder(), &source, &destination, &dawnSize)
    }

    func copyBufferToTexture(
        buffer: DawnBufferObject, offset: Int, bytesPerRow: Int, bytesPerImage: Int,
        texture: DawnTextureObject, slice: Int, mipLevel: Int, origin: LynxWebGPUCore.WGPUOrigin3D,
        size: LynxWebGPUCore.WGPUExtent3D
    ) throws {
        var source = WGPUTexelCopyBufferInfo()
        source.buffer = buffer.buffer
        source.layout.offset = try dawnU64(offset, "offset")
        source.layout.bytesPerRow = try dawnU32(bytesPerRow, "bytesPerRow")
        source.layout.rowsPerImage = try dawnU32(
            bytesPerRow > 0 ? bytesPerImage / bytesPerRow : 0, "rowsPerImage"
        )

        var destination = WGPUTexelCopyTextureInfo()
        destination.texture = texture.texture
        destination.mipLevel = try dawnU32(mipLevel, "mipLevel")
        destination.origin = try DawnEnum.origin(origin, field: "origin")
        destination.aspect = WGPUTextureAspect_All

        var dawnSize = try DawnEnum.extent(size, field: "copySize")
        wgpuCommandEncoderCopyBufferToTexture(try ensureEncoder(), &source, &destination, &dawnSize)
    }

    func copyTextureToTexture(
        source: DawnTextureObject, sourceSlice: Int, sourceMipLevel: Int,
        sourceOrigin: LynxWebGPUCore.WGPUOrigin3D,
        destination: DawnTextureObject, destinationSlice: Int, destinationMipLevel: Int,
        destinationOrigin: LynxWebGPUCore.WGPUOrigin3D, size: LynxWebGPUCore.WGPUExtent3D
    ) throws {
        var dawnSource = WGPUTexelCopyTextureInfo()
        dawnSource.texture = source.texture
        dawnSource.mipLevel = try dawnU32(sourceMipLevel, "source.mipLevel")
        dawnSource.origin = try DawnEnum.origin(sourceOrigin, field: "source.origin")
        dawnSource.aspect = WGPUTextureAspect_All

        var dawnDestination = WGPUTexelCopyTextureInfo()
        dawnDestination.texture = destination.texture
        dawnDestination.mipLevel = try dawnU32(destinationMipLevel, "destination.mipLevel")
        dawnDestination.origin = try DawnEnum.origin(destinationOrigin, field: "destination.origin")
        dawnDestination.aspect = WGPUTextureAspect_All

        var dawnSize = try DawnEnum.extent(size, field: "copySize")
        wgpuCommandEncoderCopyTextureToTexture(
            try ensureEncoder(), &dawnSource, &dawnDestination, &dawnSize
        )
    }

    func resolveQuerySet(
        _ querySet: DawnQuerySetObject, first: Int, count: Int,
        destination: DawnBufferObject, destinationOffset: Int
    ) throws {
        // 범위·정렬·usage는 엔진이 확인했다.
        wgpuCommandEncoderResolveQuerySet(
            try ensureEncoder(), querySet.querySet,
            UInt32(first), UInt32(count), destination.buffer, UInt64(destinationOffset)
        )
    }

    // MARK: - 어댑터 정보

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
            // 요청한 것만 광고한다 — 목록의 출처가 requestDevice와 같아야 광고와 실제가
            // 어긋나지 않는다 (시뮬레이터에서는 간접 기능이 목록에서 빠진다).
            "features": Array(featureLabels).sorted(),
        ]
    }
}

// MARK: - 런타임 조립

/// Dawn 백엔드를 공유 엔진에 얹은 **런타임** — 예전 이름을 그대로 쓴다.
/// 오케스트레이션은 전부 엔진 몫이므로, 이 별칭이 곧 "Dawn을 임포트해서 쓰는 방법"의 전부다.
typealias DawnWebGPURuntime = WGPUBackendEngine<DawnBackend>

extension WGPUBackendEngine where B == DawnBackend {
    convenience init() throws {
        self.init(backend: try DawnBackend())
    }
}
