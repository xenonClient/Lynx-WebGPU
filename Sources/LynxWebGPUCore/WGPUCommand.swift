import Foundation

/// 커맨드 스트림 op 하나의 **디코딩된 형태** — op 이름과 인자를 케이스 하나로 묶은 것.
///
/// ## 왜 열거형인가
///
/// 개별 인자 파싱은 `WGPUCommands.swift`가 하지만, "어떤 op 이름이 어떤 구조체로 디코딩되는가"
/// 라는 **분기표 자체**는 오랫동안 Metal 해석기 안에 있었다. 그 표가 백엔드 안에 있으면
/// 두 가지가 무너진다:
///
/// 1. 다른 런타임(Dawn 등)을 만드는 쪽이 51개 분기를 처음부터 다시 쓴다 — 순수 기계적 중복이다.
/// 2. op을 하나 더할 때 백엔드가 분기를 빠뜨려도 컴파일러가 모른다 (`default`로 샌다).
///
/// 여기로 올리면 백엔드는 이 열거형에 대한 **exhaustive switch**(default 없이)만 쓴다 —
/// 케이스가 늘면 모든 백엔드의 컴파일이 깨져서, 누락이 조용히 지나가지 않는다.
/// op 목록의 사양은 `docs/COMMAND-STREAM.md` §4다.
///
/// ## 예외 두 케이스
///
/// - `pushErrorScope` — 필터 디코딩이 **실패해도 케이스가 만들어진다** (`filter: nil` +
///   `decodeFailure`). 스코프 스택 깊이는 실패와 무관하게 맞아야 하기 때문이다
///   (`WGPUErrorScopeStack.push` 문서). 백엔드는 push부터 하고 `decodeFailure`를 던질 것.
/// - `createRenderBundle` — 명령 목록을 리더 그대로 운반한다 (`WGPUCreateRenderBundleCommand`
///   문서). 재생 시점에 이 이니셜라이저로 다시 디코딩한다.
public enum WGPUCommand {
    // MARK: 리소스
    case createBuffer(WGPUCreateCommand<WGPUBufferDescriptor>)
    case writeBuffer(WGPUWriteBufferCommand)
    case unmapBuffer(WGPUUnmapBufferCommand)
    case createTexture(WGPUCreateCommand<WGPUTextureDescriptor>)
    case writeTexture(WGPUWriteTextureCommand)
    case copyExternalImageToTexture(WGPUCopyExternalImageCommand)
    case createTextureView(WGPUCreateTextureViewCommand)
    case createSampler(WGPUCreateCommand<WGPUSamplerDescriptor>)
    case createShaderModule(WGPUCreateCommand<WGPUShaderModuleDescriptor>)
    case createBindGroupLayout(WGPUCreateCommand<WGPUBindGroupLayoutDescriptor>)
    case createPipelineLayout(WGPUCreateCommand<WGPUPipelineLayoutDescriptor>)
    case createBindGroup(WGPUCreateCommand<WGPUBindGroupDescriptor>)
    case createQuerySet(WGPUCreateCommand<WGPUQuerySetDescriptor>)
    case createRenderBundle(WGPUCreateRenderBundleCommand)
    case createRenderPipeline(WGPUCreateCommand<WGPURenderPipelineDescriptor>)
    case createComputePipeline(WGPUCreateCommand<WGPUComputePipelineDescriptor>)
    case getBindGroupLayout(WGPUGetBindGroupLayoutCommand)
    case destroy(WGPUDestroyCommand)

    // MARK: 오류 스코프
    case pushErrorScope(filter: WGPUErrorFilter?, decodeFailure: WGPUError?)
    case popErrorScope

    // MARK: 캔버스
    case configureCanvas(WGPUCanvasConfiguration)
    case getCurrentTexture(WGPUGetCurrentTextureCommand)

    // MARK: 렌더 패스
    case beginRenderPass(WGPURenderPassDescriptor)
    case setPipeline(WGPUSetPipelineCommand)
    case setBindGroup(WGPUSetBindGroupCommand)
    case setVertexBuffer(WGPUSetVertexBufferCommand)
    case setIndexBuffer(WGPUSetIndexBufferCommand)
    case setViewport(WGPUSetViewportCommand)
    case setScissorRect(WGPUSetScissorRectCommand)
    case setBlendConstant(WGPUSetBlendConstantCommand)
    case setStencilReference(WGPUSetStencilReferenceCommand)
    case draw(WGPUDrawCommand)
    case drawIndexed(WGPUDrawIndexedCommand)
    case drawIndirect(WGPUIndirectCommand)
    case drawIndexedIndirect(WGPUIndirectCommand)
    case executeBundles(WGPUExecuteBundlesCommand)
    case beginOcclusionQuery(WGPUBeginOcclusionQueryCommand)
    case endOcclusionQuery

    // MARK: 컴퓨트 패스
    case beginComputePass(WGPUComputePassDescriptor)
    case dispatchWorkgroups(WGPUDispatchWorkgroupsCommand)
    case dispatchWorkgroupsIndirect(WGPUIndirectCommand)

    // MARK: 패스 공통
    case endPass

    // MARK: 복사
    case copyBufferToBuffer(WGPUCopyBufferToBufferCommand)
    case clearBuffer(WGPUClearBufferCommand)
    case copyTextureToBuffer(WGPUCopyTextureToBufferCommand)
    case copyBufferToTexture(WGPUCopyBufferToTextureCommand)
    case copyTextureToTexture(WGPUCopyTextureToTextureCommand)

    // MARK: 쿼리
    case resolveQuerySet(WGPUResolveQuerySetCommand)

    // MARK: 디버그 마커
    case pushDebugGroup(WGPUPushDebugGroupCommand)
    case popDebugGroup
    case insertDebugMarker(WGPUInsertDebugMarkerCommand)

    /// op 이름 문자열 → 케이스. 미지 op은 `unsupported`다 — "여기 없는 op은 다른 런타임에서
    /// 구현되지 않는다"는 계약(`docs/COMMAND-STREAM.md` §7)의 코드 쪽 끝이다.
    public init(from reader: WGPUValueReader) throws {
        let op = try reader.requiredString("op")
        switch op {
        // 리소스
        case "createBuffer": self = .createBuffer(try WGPUCreateCommand(from: reader))
        case "writeBuffer": self = .writeBuffer(try WGPUWriteBufferCommand(from: reader))
        case "unmapBuffer": self = .unmapBuffer(try WGPUUnmapBufferCommand(from: reader))
        case "createTexture": self = .createTexture(try WGPUCreateCommand(from: reader))
        case "writeTexture": self = .writeTexture(try WGPUWriteTextureCommand(from: reader))
        case "copyExternalImageToTexture":
            self = .copyExternalImageToTexture(try WGPUCopyExternalImageCommand(from: reader))
        case "createTextureView":
            self = .createTextureView(try WGPUCreateTextureViewCommand(from: reader))
        case "createSampler": self = .createSampler(try WGPUCreateCommand(from: reader))
        case "createShaderModule": self = .createShaderModule(try WGPUCreateCommand(from: reader))
        case "createBindGroupLayout":
            self = .createBindGroupLayout(try WGPUCreateCommand(from: reader))
        case "createPipelineLayout":
            self = .createPipelineLayout(try WGPUCreateCommand(from: reader))
        case "createBindGroup": self = .createBindGroup(try WGPUCreateCommand(from: reader))
        case "createQuerySet": self = .createQuerySet(try WGPUCreateCommand(from: reader))
        case "createRenderBundle":
            self = .createRenderBundle(try WGPUCreateRenderBundleCommand(from: reader))
        case "createRenderPipeline":
            self = .createRenderPipeline(try WGPUCreateCommand(from: reader))
        case "createComputePipeline":
            self = .createComputePipeline(try WGPUCreateCommand(from: reader))
        case "getBindGroupLayout":
            self = .getBindGroupLayout(try WGPUGetBindGroupLayoutCommand(from: reader))
        case "destroy": self = .destroy(try WGPUDestroyCommand(from: reader))

        // 오류 스코프
        case "pushErrorScope":
            // 실패해도 케이스는 만든다 — 깊이 유지 계약 (타입 문서 참고).
            do {
                self = .pushErrorScope(
                    filter: try WGPUPushErrorScopeCommand(from: reader).filter,
                    decodeFailure: nil
                )
            } catch let error as WGPUError {
                self = .pushErrorScope(filter: nil, decodeFailure: error)
            } catch {
                self = .pushErrorScope(filter: nil, decodeFailure: .backend("\(error)"))
            }
        case "popErrorScope": self = .popErrorScope

        // 캔버스
        case "configureCanvas": self = .configureCanvas(try WGPUCanvasConfiguration(from: reader))
        case "getCurrentTexture":
            self = .getCurrentTexture(try WGPUGetCurrentTextureCommand(from: reader))

        // 렌더 패스
        case "beginRenderPass": self = .beginRenderPass(try WGPURenderPassDescriptor(from: reader))
        case "setPipeline": self = .setPipeline(try WGPUSetPipelineCommand(from: reader))
        case "setBindGroup": self = .setBindGroup(try WGPUSetBindGroupCommand(from: reader))
        case "setVertexBuffer": self = .setVertexBuffer(try WGPUSetVertexBufferCommand(from: reader))
        case "setIndexBuffer": self = .setIndexBuffer(try WGPUSetIndexBufferCommand(from: reader))
        case "setViewport": self = .setViewport(try WGPUSetViewportCommand(from: reader))
        case "setScissorRect": self = .setScissorRect(try WGPUSetScissorRectCommand(from: reader))
        case "setBlendConstant":
            self = .setBlendConstant(try WGPUSetBlendConstantCommand(from: reader))
        case "setStencilReference":
            self = .setStencilReference(try WGPUSetStencilReferenceCommand(from: reader))
        case "draw": self = .draw(try WGPUDrawCommand(from: reader))
        case "drawIndexed": self = .drawIndexed(try WGPUDrawIndexedCommand(from: reader))
        case "drawIndirect": self = .drawIndirect(try WGPUIndirectCommand(from: reader))
        case "drawIndexedIndirect":
            self = .drawIndexedIndirect(try WGPUIndirectCommand(from: reader))
        case "executeBundles": self = .executeBundles(try WGPUExecuteBundlesCommand(from: reader))
        case "beginOcclusionQuery":
            self = .beginOcclusionQuery(try WGPUBeginOcclusionQueryCommand(from: reader))
        case "endOcclusionQuery": self = .endOcclusionQuery

        // 컴퓨트 패스
        case "beginComputePass":
            self = .beginComputePass(try WGPUComputePassDescriptor(from: reader))
        case "dispatchWorkgroups":
            self = .dispatchWorkgroups(try WGPUDispatchWorkgroupsCommand(from: reader))
        case "dispatchWorkgroupsIndirect":
            self = .dispatchWorkgroupsIndirect(try WGPUIndirectCommand(from: reader))

        case "endPass": self = .endPass

        // 복사
        case "copyBufferToBuffer":
            self = .copyBufferToBuffer(try WGPUCopyBufferToBufferCommand(from: reader))
        case "clearBuffer": self = .clearBuffer(try WGPUClearBufferCommand(from: reader))
        case "copyTextureToBuffer":
            self = .copyTextureToBuffer(try WGPUCopyTextureToBufferCommand(from: reader))
        case "copyBufferToTexture":
            self = .copyBufferToTexture(try WGPUCopyBufferToTextureCommand(from: reader))
        case "copyTextureToTexture":
            self = .copyTextureToTexture(try WGPUCopyTextureToTextureCommand(from: reader))

        // 쿼리
        case "resolveQuerySet":
            self = .resolveQuerySet(try WGPUResolveQuerySetCommand(from: reader))

        // 디버그 마커
        case "pushDebugGroup": self = .pushDebugGroup(try WGPUPushDebugGroupCommand(from: reader))
        case "popDebugGroup": self = .popDebugGroup
        case "insertDebugMarker":
            self = .insertDebugMarker(try WGPUInsertDebugMarkerCommand(from: reader))

        default:
            throw WGPUError.unsupported("알 수 없는 명령 '\(op)'", path: reader.fieldPath("op"))
        }
    }

    /// 케이스 → op 이름 문자열 (와이어 철자 그대로).
    ///
    /// **일부러 exhaustive switch다** — 케이스를 더하고 여기(그리고 위 디코더 표)를 빠뜨리면
    /// 컴파일이 깨진다. 진단 메시지·문서 대조·번들 허용 목록 비교에 쓴다.
    public var opName: String {
        switch self {
        case .createBuffer: return "createBuffer"
        case .writeBuffer: return "writeBuffer"
        case .unmapBuffer: return "unmapBuffer"
        case .createTexture: return "createTexture"
        case .writeTexture: return "writeTexture"
        case .copyExternalImageToTexture: return "copyExternalImageToTexture"
        case .createTextureView: return "createTextureView"
        case .createSampler: return "createSampler"
        case .createShaderModule: return "createShaderModule"
        case .createBindGroupLayout: return "createBindGroupLayout"
        case .createPipelineLayout: return "createPipelineLayout"
        case .createBindGroup: return "createBindGroup"
        case .createQuerySet: return "createQuerySet"
        case .createRenderBundle: return "createRenderBundle"
        case .createRenderPipeline: return "createRenderPipeline"
        case .createComputePipeline: return "createComputePipeline"
        case .getBindGroupLayout: return "getBindGroupLayout"
        case .destroy: return "destroy"
        case .pushErrorScope: return "pushErrorScope"
        case .popErrorScope: return "popErrorScope"
        case .configureCanvas: return "configureCanvas"
        case .getCurrentTexture: return "getCurrentTexture"
        case .beginRenderPass: return "beginRenderPass"
        case .setPipeline: return "setPipeline"
        case .setBindGroup: return "setBindGroup"
        case .setVertexBuffer: return "setVertexBuffer"
        case .setIndexBuffer: return "setIndexBuffer"
        case .setViewport: return "setViewport"
        case .setScissorRect: return "setScissorRect"
        case .setBlendConstant: return "setBlendConstant"
        case .setStencilReference: return "setStencilReference"
        case .draw: return "draw"
        case .drawIndexed: return "drawIndexed"
        case .drawIndirect: return "drawIndirect"
        case .drawIndexedIndirect: return "drawIndexedIndirect"
        case .executeBundles: return "executeBundles"
        case .beginOcclusionQuery: return "beginOcclusionQuery"
        case .endOcclusionQuery: return "endOcclusionQuery"
        case .beginComputePass: return "beginComputePass"
        case .dispatchWorkgroups: return "dispatchWorkgroups"
        case .dispatchWorkgroupsIndirect: return "dispatchWorkgroupsIndirect"
        case .endPass: return "endPass"
        case .copyBufferToBuffer: return "copyBufferToBuffer"
        case .clearBuffer: return "clearBuffer"
        case .copyTextureToBuffer: return "copyTextureToBuffer"
        case .copyBufferToTexture: return "copyBufferToTexture"
        case .copyTextureToTexture: return "copyTextureToTexture"
        case .resolveQuerySet: return "resolveQuerySet"
        case .pushDebugGroup: return "pushDebugGroup"
        case .popDebugGroup: return "popDebugGroup"
        case .insertDebugMarker: return "insertDebugMarker"
        }
    }
}
