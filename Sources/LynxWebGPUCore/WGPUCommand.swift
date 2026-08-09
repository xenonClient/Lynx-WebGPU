import Foundation

/// One command-stream op in **decoded form** — op name and arguments bound into a single case.
///
/// ## Why an enum
///
/// Parsing individual arguments is `WGPUCommands.swift`'s job, but the **dispatch table itself** —
/// which op name decodes into which struct — lived inside the Metal interpreter for a long time.
/// With that table inside a backend, two things break down:
///
/// 1. Anyone building another runtime (Dawn, say) rewrites all 51 branches from scratch — purely
///    mechanical duplication.
/// 2. Adding an op and forgetting a branch goes unnoticed by the compiler (it leaks into `default`).
///
/// Lifted here, a backend writes only an **exhaustive switch** over this enum (no `default`) —
/// adding a case breaks every backend's compile, so an omission cannot slip through quietly.
/// The op list is specified in `docs/COMMAND-STREAM.md` §4.
///
/// ## Two exceptional cases
///
/// - `pushErrorScope` — the case is built **even when filter decoding fails** (`filter: nil` plus
///   `decodeFailure`), because the scope stack depth has to stay correct regardless of failure
///   (see `WGPUErrorScopeStack.push`). A backend pushes first, then throws `decodeFailure`.
/// - `createRenderBundle` — carries the command list as readers (see
///   `WGPUCreateRenderBundleCommand`). Replay decodes them again through this initializer.
public enum WGPUCommand {
    // MARK: Resources
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

    // MARK: Error scopes
    case pushErrorScope(filter: WGPUErrorFilter?, decodeFailure: WGPUError?)
    case popErrorScope

    // MARK: Canvas
    case configureCanvas(WGPUCanvasConfiguration)
    case getCurrentTexture(WGPUGetCurrentTextureCommand)

    // MARK: Render pass
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

    // MARK: Compute pass
    case beginComputePass(WGPUComputePassDescriptor)
    case dispatchWorkgroups(WGPUDispatchWorkgroupsCommand)
    case dispatchWorkgroupsIndirect(WGPUIndirectCommand)

    // MARK: Shared by passes
    case endPass

    // MARK: Copies
    case copyBufferToBuffer(WGPUCopyBufferToBufferCommand)
    case clearBuffer(WGPUClearBufferCommand)
    case copyTextureToBuffer(WGPUCopyTextureToBufferCommand)
    case copyBufferToTexture(WGPUCopyBufferToTextureCommand)
    case copyTextureToTexture(WGPUCopyTextureToTextureCommand)

    // MARK: Queries
    case resolveQuerySet(WGPUResolveQuerySetCommand)

    // MARK: Debug markers
    case pushDebugGroup(WGPUPushDebugGroupCommand)
    case popDebugGroup
    case insertDebugMarker(WGPUInsertDebugMarkerCommand)

    /// Op name string → case. An unknown op is `unsupported` — the code-side end of the contract
    /// "an op absent from here is implemented by no other runtime" (`docs/COMMAND-STREAM.md` §7).
    public init(from reader: WGPUValueReader) throws {
        let op = try reader.requiredString("op")
        switch op {
        // Resources
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

        // Error scopes
        case "pushErrorScope":
            // Build the case even on failure — the depth-keeping contract (see the type docs).
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

        // Canvas
        case "configureCanvas": self = .configureCanvas(try WGPUCanvasConfiguration(from: reader))
        case "getCurrentTexture":
            self = .getCurrentTexture(try WGPUGetCurrentTextureCommand(from: reader))

        // Render pass
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

        // Compute pass
        case "beginComputePass":
            self = .beginComputePass(try WGPUComputePassDescriptor(from: reader))
        case "dispatchWorkgroups":
            self = .dispatchWorkgroups(try WGPUDispatchWorkgroupsCommand(from: reader))
        case "dispatchWorkgroupsIndirect":
            self = .dispatchWorkgroupsIndirect(try WGPUIndirectCommand(from: reader))

        case "endPass": self = .endPass

        // Copies
        case "copyBufferToBuffer":
            self = .copyBufferToBuffer(try WGPUCopyBufferToBufferCommand(from: reader))
        case "clearBuffer": self = .clearBuffer(try WGPUClearBufferCommand(from: reader))
        case "copyTextureToBuffer":
            self = .copyTextureToBuffer(try WGPUCopyTextureToBufferCommand(from: reader))
        case "copyBufferToTexture":
            self = .copyBufferToTexture(try WGPUCopyBufferToTextureCommand(from: reader))
        case "copyTextureToTexture":
            self = .copyTextureToTexture(try WGPUCopyTextureToTextureCommand(from: reader))

        // Queries
        case "resolveQuerySet":
            self = .resolveQuerySet(try WGPUResolveQuerySetCommand(from: reader))

        // Debug markers
        case "pushDebugGroup": self = .pushDebugGroup(try WGPUPushDebugGroupCommand(from: reader))
        case "popDebugGroup": self = .popDebugGroup
        case "insertDebugMarker":
            self = .insertDebugMarker(try WGPUInsertDebugMarkerCommand(from: reader))

        default:
            throw WGPUError.unsupported("unknown command '\(op)'", path: reader.fieldPath("op"))
        }
    }

    /// Case → op name string (the wire spelling exactly).
    ///
    /// **Deliberately an exhaustive switch** — add a case and forget this (or the decoder table
    /// above) and the compile breaks. Used for diagnostics, doc cross-checks and bundle allow-lists.
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
