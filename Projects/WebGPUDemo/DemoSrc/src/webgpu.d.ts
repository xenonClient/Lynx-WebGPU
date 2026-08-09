/**
 * Lynx-WebGPU — a WebGPU-shaped JS client.
 *
 * It keeps the same object graph as browser WebGPU on the JS side, but actual calls are **only recorded as
 * commands** and sent to native in one go at `queue.submit()`. Handles (ids) are issued by JS, so object
 * creation does not wait for a native round trip — the bridge crossings per frame are fixed at one.
 *
 * See docs/ARCHITECTURE.md §3 for the design and docs/WEBGPU-API.md for the supported surface.
 */
export declare const GPUBufferUsage: {
    MAP_READ: number;
    MAP_WRITE: number;
    COPY_SRC: number;
    COPY_DST: number;
    INDEX: number;
    VERTEX: number;
    UNIFORM: number;
    STORAGE: number;
    INDIRECT: number;
    QUERY_RESOLVE: number;
};
export declare const GPUTextureUsage: {
    COPY_SRC: number;
    COPY_DST: number;
    TEXTURE_BINDING: number;
    STORAGE_BINDING: number;
    RENDER_ATTACHMENT: number;
};
export declare const GPUShaderStage: {
    VERTEX: number;
    FRAGMENT: number;
    COMPUTE: number;
};
export declare const GPUColorWrite: {
    RED: number;
    GREEN: number;
    BLUE: number;
    ALPHA: number;
    ALL: number;
};
export declare const GPUMapMode: {
    READ: number;
    WRITE: number;
};
export type WGPUError = WGPUErrorPayload;
export type GPUExtent3DDict = {
    width: number;
    height?: number;
    depthOrArrayLayers?: number;
};
export type GPUExtent3D = GPUExtent3DDict | number[];
export type GPUOrigin3DDict = {
    x?: number;
    y?: number;
    z?: number;
};
export type GPUColorDict = {
    r: number;
    g: number;
    b: number;
    a: number;
};
export type GPUColor = GPUColorDict | number[];
export type GPUDataSource = ArrayBuffer | ArrayBufferView | number[];
export type GPUCommand = Record<string, any>;
export type GPUBufferDescriptor = {
    size: number;
    usage: number;
    mappedAtCreation?: boolean;
    label?: string;
};
export type GPUTextureDescriptor = {
    size: GPUExtent3D;
    format: string;
    usage: number;
    dimension?: string;
    mipLevelCount?: number;
    sampleCount?: number;
    label?: string;
};
export type GPUShaderModuleDescriptor = {
    code: string;
    language?: 'wgsl' | 'msl';
    label?: string;
};
export type GPUBindGroupLayoutDescriptor = {
    entries: Record<string, any>[];
    label?: string;
};
export type GPUPipelineLayoutDescriptor = {
    bindGroupLayouts: GPUPipelineLayoutSource[];
    label?: string;
};
export type GPUBindGroupDescriptor = {
    layout: GPUBindGroupLayout;
    entries: {
        binding: number;
        resource: any;
    }[];
    label?: string;
};
export type GPUCanvasConfiguration = {
    device: GPUDevice;
    format?: string;
    usage?: number;
    alphaMode?: 'opaque' | 'premultiplied';
    colorSpace?: 'srgb' | 'display-p3';
    toneMapping?: {
        mode: 'standard' | 'extended';
    };
};
export type GPUStencilFaceState = {
    compare?: GPUCompareFunction;
    failOp?: GPUStencilOperation;
    depthFailOp?: GPUStencilOperation;
    passOp?: GPUStencilOperation;
};
export type GPUCompareFunction = 'never' | 'less' | 'equal' | 'less-equal' | 'greater' | 'not-equal' | 'greater-equal' | 'always';
export type GPUStencilOperation = 'keep' | 'zero' | 'replace' | 'invert' | 'increment-clamp' | 'decrement-clamp' | 'increment-wrap' | 'decrement-wrap';
export type GPUDepthStencilState = {
    format: string;
    depthWriteEnabled?: boolean;
    depthCompare?: GPUCompareFunction;
    depthBias?: number;
    depthBiasSlopeScale?: number;
    depthBiasClamp?: number;
    stencilFront?: GPUStencilFaceState;
    stencilBack?: GPUStencilFaceState;
    stencilReadMask?: number;
    stencilWriteMask?: number;
};
export type GPUQuerySetDescriptor = {
    type: 'occlusion' | 'timestamp';
    count: number;
    label?: string;
};
export type GPUAdapterInfoView = {
    vendor: string;
    architecture: string;
    device: string;
    description: string;
    isFallbackAdapter: boolean;
    subgroupMinSize: number;
    subgroupMaxSize: number;
};
export type GPUCompilationMessage = {
    message: string;
    type: 'error' | 'warning' | 'info';
    lineNum: number;
    linePos: number;
    offset: number;
    length: number;
};
export type GPUDeviceLostInfo = {
    reason: 'unknown' | 'destroyed';
    message: string;
};
export type GPUPassTimestampWrites = {
    querySet: GPUQuerySet;
    beginningOfPassWriteIndex?: number;
    endOfPassWriteIndex?: number;
};
export type GPURenderBundleEncoderDescriptor = {
    colorFormats: (string | null)[];
    depthStencilFormat?: string;
    sampleCount?: number;
    depthReadOnly?: boolean;
    stencilReadOnly?: boolean;
    label?: string;
};
export type GPUPipelineLayoutSource = {
    id: number;
};
export type GPUTextureInit = {
    size?: GPUExtent3D;
    format?: string;
    usage?: number;
    dimension?: string;
    mipLevelCount?: number;
    sampleCount?: number;
    textureBindingViewDimension?: string;
    label?: string;
    frameScoped?: boolean;
};
/**
 * The spec's `GPUError` hierarchy — the object the `uncapturederror` event carries.
 *
 * The spec only requires `message`, but `kind` and `path` are carried along. An error from the command
 * stream knows "which field of which command", and throwing that away makes diagnosis much worse.
 * Web code reads the kind with `instanceof` (or `constructor.name`) — hence the subclasses.
 */
declare class GPUError {
    message: string;
    /** Extra information this implementation attaches — not in the spec. */
    kind: "backend" | "out-of-memory" | "unsupported" | "validation";
    path: string | undefined;
    /** @param {WGPUError} payload */
    constructor(payload: WGPUError);
}
declare class GPUValidationError extends GPUError {
}
declare class GPUOutOfMemoryError extends GPUError {
}
declare class GPUInternalError extends GPUError {
}
declare class Recorder {
    /** @type {GPUCommand[]} */
    pending: GPUCommand[];
    /** @type {((error: WGPUError, text: string) => void)[]} */
    errorHandlers: ((error: WGPUError, text: string) => void)[];
    /**
     * A hook that also lets errors uncaught by a scope out through the spec's `uncapturederror` path.
     * The device plugs itself in. It returns `true` if at least one listener took it.
     * @type {((error: WGPUError) => boolean) | null}
     */
    uncapturedDispatch: ((error: WGPUError) => boolean) | null;
    /**
     * The result functions of the Promises `popErrorScope()` returned — **in the order they were popped**.
     * Native returns the `errorScopes` array in the same order, so they are paired by index.
     * @type {{resolve: (error: WGPUError | null) => void, reject: (reason: Error) => void}[]}
     */
    pendingErrorScopes: {
        resolve: (error: WGPUError | null) => void;
        reject: (reason: Error) => void;
    }[];
    constructor();
    /**
     * A new handle id. It comes from the **module-wide counter** — counting per recorder would collide
     * when there are two devices (see the `allocateHandle` comment).
     * @returns {number}
     */
    allocate(): number;
    /**
     * Stacks a command **frozen at its record-time value** (see `snapshotValue`) — for device/queue ops
     * this is the call site itself, so reusing a descriptor after the call cannot pollute the stream.
     * @param {GPUCommand} command
     * @returns {GPUCommand}
     */
    push(command: GPUCommand): GPUCommand;
    /**
     * Hands the accumulated commands to native. If there is nothing to run, it does nothing.
     *
     * `present: false` marks **an internal mid-frame submission** (a batch flushed early so `popErrorScope`
     * or `mapAsync` can get a result). Native commits the batch but defers the drawable present and the
     * swapchain handle expiry until a real frame submission (`queue.submit`) — otherwise an acquired canvas
     * texture is presented before it is drawn and the remaining passes are rejected wholesale.
     *
     * **Inside a frame loop callback the present is deferred** (`frameTickDepth`) — the same place the
     * browser presents at the end of a task. A deferred present goes out from `endFrameTick()` at tick end.
     *
     * @param {boolean} [present] whether this batch is a frame submission (true by default)
     * @param {{presentOnly?: boolean}} [options] `presentOnly` is the wrap-up call at tick end —
     *   it sends a batch even with no commands, to **push the drawable out.**
     * @returns {WGPUExecuteResult}
     */
    flush(present?: boolean, options?: {
        presentOnly?: boolean;
    }): WGPUExecuteResult;
    /**
     * Stacks a `popErrorScope` command and returns the result Promise — **without flushing.**
     *
     * Used when closing several scopes in one batch (asynchronous pipeline creation does this).
     * Native returns the results in pop order, so the call order is the pairing order.
     *
     * @returns {Promise<WGPUError | null>}
     */
    recordPop(): Promise<WGPUError | null>;
    /**
     * Resolves the `popErrorScope()` Promises that were waiting.
     *
     * It relies on the contract that native does not shift indices (a failed pop still leaves its slot), so
     * all that is needed here is pairing them up in order. If the response is short on results, the value is
     * `null` — better than a Promise that never resolves.
     *
     * A slot of `{rejected: true}` means it was unpaired with a `push`. The spec says to **reject** with an
     * `OperationError` in that case (rather than producing an error), which is followed here — that is what
     * lets an app tell "the scope was clean (null)" apart from "it was unpaired".
     *
     * @param {(WGPUError | {rejected: true} | null)[]} popped
     * @returns {void}
     */
    settleErrorScopes(popped: (WGPUError | {
        rejected: true;
    } | null)[]): void;
    /**
     * Sends an error uncaught by a scope down every registered channel.
     *
     * There are two channels and they receive it **together** — this implementation's `onError` (which even
     * gives the path-tagged text) and the spec's `uncapturederror` (the name web code knows). It falls back
     * to the console only when nobody is listening: no error may vanish silently, and logging to the console while someone listens would double the log.
     *
     * @param {WGPUError[]} errors
     */
    report(errors: WGPUError[]): void;
}
declare class GPUObjectBase {
    _device: GPUDevice;
    _recorder: Recorder;
    id: number;
    label: string;
    /**
     * @param {GPUDevice} device
     * @param {number} id the handle JS issued
     * @param {string} [label]
     * @param {boolean} [frameScoped] true if it is a handle native reclaims at the end of the frame
     */
    constructor(device: GPUDevice, id: number, label?: string, frameScoped?: boolean);
    destroy(): void;
}
declare class GPUBuffer extends GPUObjectBase {
    size: number;
    usage: number;
    /** @type {ArrayBuffer | null} */
    _mapped: ArrayBuffer | null;
    /** Whether this is `mappedAtCreation` initial data (whether unmap must record the creation command). */
    _mappedAtCreation: boolean;
    /** Whether `mapAsync` is still waiting for a result (the spec's `"pending"` state). */
    _mapPending: boolean;
    /**
     * The ranges handed out by `getMappedRange()` — used for the overlap check and `unmap()`'s write-back.
     * @type {{offset: number, length: number, view: ArrayBuffer | null}[]}
     */
    _mappedRanges: {
        offset: number;
        length: number;
        view: ArrayBuffer | null;
    }[];
    /**
     * @param {GPUDevice} device
     * @param {number} id
     * @param {GPUBufferDescriptor} descriptor
     */
    constructor(device: GPUDevice, id: number, descriptor: GPUBufferDescriptor);
    /**
     * The spec's `GPUBufferMapState` — `'unmapped'` · `'pending'` · `'mapped'`.
     *
     * A buffer being mapped is rejected by queue operations, so code that wants to reuse it must be able to
     * learn the state **without asking**. Without this it sees `undefined` and misreads it as "not mapped".
     *
     * @returns {'unmapped' | 'pending' | 'mapped'}
     */
    get mapState(): 'unmapped' | 'pending' | 'mapped';
    /** The initial data region of a buffer created with `mappedAtCreation: true`. */
    /**
     * Obtains the mapped range as an `ArrayBuffer` (after `mappedAtCreation` or `mapAsync`).
     *
     * **In JS an `ArrayBuffer` cannot point into part of another `ArrayBuffer`.** A browser gives you a view
     * straight onto the mapping memory; that is impossible here, so the range is handed over as a copy and
     * **written back in `unmap()`.** Nothing you wrote is lost — but the returned buffer only has meaning
     * **until `unmap()`** (the same point at which a browser detaches it).
     *
     * Requesting the whole range first returns the mapping itself with no copy — saving one copy on the
     * common path of filling a large vertex buffer with `mappedAtCreation`.
     *
     * Spec rules: `offset` must be a multiple of 8, `size` a multiple of 4, and ranges **may not overlap**.
     *
     * @param {number} [offset] the byte offset (a multiple of 8)
     * @param {number} [size] the number of bytes (a multiple of 4). Omitted, it runs to the end
     * @returns {ArrayBuffer}
     */
    getMappedRange(offset?: number, size?: number): ArrayBuffer;
    /** Writes what was written into the range copies back into the mapping (just before `unmap`). */
    _flushMappedRanges(): void;
    /**
     * Unmaps.
     *
     * For `mappedAtCreation`, this is where the actual creation command (initial data included) is recorded;
     * for something mapped by `mapAsync` it tells native "queue operations may use it now" —
     * a buffer being mapped is rejected by queue operations per the spec, so **this must be called once you have read.**
     *
     * @returns {void}
     */
    unmap(): void;
    /**
     * Reads the buffer's contents. It is WebGPU's `mapAsync` + `getMappedRange` merged into one.
     *
     * While reading, this buffer becomes **"unavailable"** per the spec and is rejected by queue operations
     * (writes, copies, resolves, draw bindings). Otherwise the next frame's write would overlap the same
     * memory while the readback waits for GPU completion, and which frame the received value belongs to would not be guaranteed.
     * **Call `unmap()` once you are done reading.**
     *
     * @param {number} [_mode] for spec compatibility — this implementation does not look at it
     * @param {number} [offset] the byte offset
     * @param {number} [size] the number of bytes to read. Omitted, it runs to the end
     * @returns {Promise<ArrayBuffer>}
     */
    mapAsync(_mode?: number, offset?: number, size?: number): Promise<ArrayBuffer>;
}
declare class GPUTexture extends GPUObjectBase {
    _frameScoped: boolean;
    width: any;
    height: any;
    depthOrArrayLayers: any;
    mipLevelCount: number;
    sampleCount: number;
    dimension: string;
    format: string | undefined;
    usage: number;
    /**
     * The default view dimension when binding this texture. The spec allows omitting it, in which case it
     * is derived from `dimension` and the layer count (2d with 2 or more layers gives `2d-array`).
     * @type {string}
     */
    textureBindingViewDimension: string;
    /**
     * @param {GPUDevice} device
     * @param {number} id
     * @param {GPUTextureInit} [descriptor]
     */
    constructor(device: GPUDevice, id: number, descriptor?: GPUTextureInit);
    /**
     * @param {Record<string, any>} [descriptor]
     * @returns {GPUTextureView}
     */
    createView(descriptor?: Record<string, any>): GPUTextureView;
}
declare class GPUTextureView extends GPUObjectBase {
}
declare class GPUSampler extends GPUObjectBase {
}
declare class GPUShaderModule extends GPUObjectBase {
    /**
     * This module's compilation diagnostics (the spec's `GPUCompilationInfo`).
     *
     * A shader module **is created even when compilation fails** (the spec model) — the failure surfaces here
     * and as a pipeline creation failure. So there is something to check even after `createShaderModule()` succeeds.
     *
     * `messages[].lineNum` is the WGSL source line number (1-based). `linePos`, `offset` and `length` are
     * **left at 0** because this implementation does not know them — inventing values would make an editor underline the wrong place.
     *
     * It costs one round trip, so use it on diagnostic paths only.
     *
     * @returns {Promise<{messages: GPUCompilationMessage[]}>}
     */
    getCompilationInfo(): Promise<{
        messages: GPUCompilationMessage[];
    }>;
}
declare class GPUBindGroupLayout extends GPUObjectBase {
}
declare class GPUPipelineLayout extends GPUObjectBase {
}
declare class GPUBindGroup extends GPUObjectBase {
}
declare class GPUPipelineBase extends GPUObjectBase {
    /**
     * Pulls out a bind group layout derived by a `layout: 'auto'` pipeline.
     * @param {number} index
     * @returns {GPUBindGroupLayout}
     */
    getBindGroupLayout(index: number): GPUBindGroupLayout;
}
declare class GPURenderPipeline extends GPUPipelineBase {
}
declare class GPUComputePipeline extends GPUPipelineBase {
}
/**
 * The reusable bundle of draws `bundleEncoder.finish()` returns.
 *
 * It is the only structure whose recorded commands **outlive the wrappers**, so it holds on to the resource
 * wrappers it uses (`_retained`). Otherwise, when an initialization function returns only the bundle and
 * drops the pipelines and buffers, GC slips in a `destroy` and the bundle **quietly draws nothing** (`docs/JS-AUTHORING.md` §8).
 */
declare class GPURenderBundle extends GPUObjectBase {
    /** @type {object[]} the resource wrappers this bundle references — their lifetimes are tied together. */
    _retained: object[];
    /**
     * @param {GPUDevice} device
     * @param {number} id
     * @param {string} [label]
     */
    constructor(device: GPUDevice, id: number, label?: string);
}
/** The query store `device.createQuerySet()` returns. */
declare class GPUQuerySet extends GPUObjectBase {
    type: "occlusion" | "timestamp";
    count: number;
    /**
     * @param {GPUDevice} device
     * @param {number} id
     * @param {GPUQuerySetDescriptor} descriptor
     */
    constructor(device: GPUDevice, id: number, descriptor: GPUQuerySetDescriptor);
}
/** An encoder gathers its own commands separately, then they merge into the stream at `finish()` → `submit()`. */
declare class GPUCommandBuffer {
    commands: GPUCommand[];
    /** @param {GPUCommand[]} commands */
    constructor(commands: GPUCommand[]);
}
declare class GPUPassEncoderBase {
    _commands: GPUCommand[];
    /**
     * The resource wrappers met while recording — **only the bundle encoder** fills this.
     *
     * A bundle is the only structure whose recorded commands outlive the wrappers, so on an engine with
     * automatic release (GC), an initialization function returning only the bundle and dropping the
     * pipeline and buffer wrappers makes the bundle quietly draw nothing. In the spec model a bundle
     * **owns** the objects it uses, so the same ownership is built here.
     * @type {object[] | null}
     */
    _retained: object[] | null;
    /** @param {GPUCommand[]} commands */
    constructor(commands: GPUCommand[]);
    /**
     * @param {object} resource
     * @returns {void}
     */
    _retain(resource: object): void;
    /**
     * Opens a debug group — **the range name shows up as is in an Xcode GPU capture** (Metal `pushDebugGroup`).
     *
     * Without it a capture is a list of nameless draws and there is no telling which pass is which. It is the
     * first thing you miss when looking at performance, so it is better put in early while laying out the frame.
     *
     * **It must be paired with `popDebugGroup()`.** An unpaired one is reported by native as a validation
     * error (Metal kills the process with an assertion in that situation, so it is blocked there).
     *
     * @param {string} groupLabel
     * @returns {void}
     */
    pushDebugGroup(groupLabel: string): void;
    /** @returns {void} */
    popDebugGroup(): void;
    /**
     * Leaves a marker at a single point (Metal `insertDebugSignpost`) — a point event, not a range.
     * @param {string} markerLabel
     * @returns {void}
     */
    insertDebugMarker(markerLabel: string): void;
    /**
     * @param {GPURenderPipeline | GPUComputePipeline} pipeline
     * @returns {void}
     */
    setPipeline(pipeline: GPURenderPipeline | GPUComputePipeline): void;
    /**
     * @param {number} index
     * @param {GPUBindGroup} bindGroup
     * @param {number[]} [dynamicOffsets]
     * @returns {void}
     */
    setBindGroup(index: number, bindGroup: GPUBindGroup, dynamicOffsets?: number[]): void;
}
/**
 * The commands a render pass and a render bundle can **both** use.
 *
 * This boundary is the spec's — a bundle cannot carry a viewport, scissor, blend constant, stencil
 * reference or a nested bundle. Keeping those on `GPURenderPassEncoder` alone means the bundle encoder
 * never has those methods, so code using them wrongly never reaches native.
 */
declare class GPURenderCommandsBase extends GPUPassEncoderBase {
    /**
     * @param {number} slot
     * @param {GPUBuffer} buffer
     * @param {number} [offset]
     * @returns {void}
     */
    setVertexBuffer(slot: number, buffer: GPUBuffer, offset?: number): void;
    /**
     * @param {GPUBuffer} buffer
     * @param {'uint16' | 'uint32'} format
     * @param {number} [offset]
     * @returns {void}
     */
    setIndexBuffer(buffer: GPUBuffer, format: 'uint16' | 'uint32', offset?: number): void;
    /**
     * @param {number} vertexCount
     * @param {number} [instanceCount]
     * @param {number} [firstVertex]
     * @param {number} [firstInstance]
     * @returns {void}
     */
    draw(vertexCount: number, instanceCount?: number, firstVertex?: number, firstInstance?: number): void;
    /**
     * @param {number} indexCount
     * @param {number} [instanceCount]
     * @param {number} [firstIndex]
     * @param {number} [baseVertex]
     * @param {number} [firstInstance]
     * @returns {void}
     */
    drawIndexed(indexCount: number, instanceCount?: number, firstIndex?: number, baseVertex?: number, firstInstance?: number): void;
    /**
     * Draws by reading the draw arguments from a GPU buffer — used to draw as many as a compute pass produced.
     *
     * The buffer must contain 4 `u32`s in this order:
     * `vertexCount, instanceCount, firstVertex, firstInstance`.
     * The buffer must be created with `GPUBufferUsage.INDIRECT`, and `indirectOffset` must be a multiple of 4.
     *
     * A non-zero `firstInstance` requires the `indirect-first-instance` feature. This implementation always
     * reports that feature, but in a browser **the draw becomes a wholesale no-op if it is not requested**.
     *
     * @param {GPUBuffer} indirectBuffer
     * @param {number} [indirectOffset]
     * @returns {void}
     */
    drawIndirect(indirectBuffer: GPUBuffer, indirectOffset?: number): void;
    /**
     * Draws indexed by reading the draw arguments from a GPU buffer.
     *
     * The buffer must contain 5 `u32`s in this order:
     * `indexCount, instanceCount, firstIndex, baseVertex (a signed i32), firstInstance`.
     * `firstIndex` lives inside the argument buffer, so it is applied **separately rather than added to**
     * the offset of `setIndexBuffer(buffer, format, offset)`.
     *
     * A non-zero `firstInstance` requires the `indirect-first-instance` feature. This implementation always
     * reports that feature, but in a browser **the draw becomes a wholesale no-op if it is not requested**.
     *
     * @param {GPUBuffer} indirectBuffer
     * @param {number} [indirectOffset]
     * @returns {void}
     */
    drawIndexedIndirect(indirectBuffer: GPUBuffer, indirectOffset?: number): void;
}
/** Pass-only commands — these four and `executeBundles` cannot go in a bundle (per the spec). */
declare class GPURenderPassEncoder extends GPURenderCommandsBase {
    /**
     * @param {number} x
     * @param {number} y
     * @param {number} width
     * @param {number} height
     * @param {number} [minDepth]
     * @param {number} [maxDepth]
     * @returns {void}
     */
    setViewport(x: number, y: number, width: number, height: number, minDepth?: number, maxDepth?: number): void;
    /**
     * @param {number} x
     * @param {number} y
     * @param {number} width
     * @param {number} height
     * @returns {void}
     */
    setScissorRect(x: number, y: number, width: number, height: number): void;
    /**
     * @param {GPUColor} color
     * @returns {void}
     */
    setBlendConstant(color: GPUColor): void;
    /**
     * @param {number} reference
     * @returns {void}
     */
    setStencilReference(reference: number): void;
    /**
     * Begins counting the samples these draws let through.
     *
     * It can only be used in a pass given an `occlusionQuerySet` in `beginRenderPass`, and cannot nest.
     * The same index cannot be used twice in one pass, and it must be closed with `endOcclusionQuery`
     * before the pass is closed.
     *
     * @param {number} queryIndex
     * @returns {void}
     */
    beginOcclusionQuery(queryIndex: number): void;
    /** @returns {void} */
    endOcclusionQuery(): void;
    /**
     * Replays previously recorded bundles into this pass.
     *
     * A bundle **does not inherit** pass state, and when it finishes the pass's pipeline, bind groups and
     * vertex/index buffer bindings are **invalidated** (not restored to their previous values — the spec's
     * contract). To keep drawing you must redo `setPipeline`, `setBindGroup` and `setVertexBuffer`; omitting
     * one gets that draw rejected. The viewport, scissor, blend constant and stencil reference stay as they are.
     *
     * @param {GPURenderBundle[]} bundles
     * @returns {void}
     */
    executeBundles(bundles: GPURenderBundle[]): void;
    /** @returns {void} */
    end(): void;
}
/**
 * Records a bundle of draws to reuse across many frames (`device.createRenderBundleEncoder`).
 *
 * A bundle's benefit here differs from a browser's — a browser builds driver commands ahead of time,
 * whereas here the benefit is **that JS does not have to rebuild the same command array every frame**.
 * The command that runs a bundle is a single handle, and the replay is native's job.
 */
declare class GPURenderBundleEncoder extends GPURenderCommandsBase {
    /** @type {object[]} the wrappers met while recording — the finished bundle inherits and holds them. */
    _retained: object[];
    _finished: boolean;
    _device: GPUDevice;
    _descriptor: any;
    /**
     * @param {GPUDevice} device
     * @param {GPURenderBundleEncoderDescriptor} descriptor
     */
    constructor(device: GPUDevice, descriptor: GPURenderBundleEncoderDescriptor);
    /**
     * Ends recording and produces a reusable bundle.
     *
     * **It can only be called once.** A second call raises an error per the spec and returns **an invalid
     * bundle** — quietly handing back an empty bundle would make `executeBundles` draw nothing with no
     * error, and the cause hard to find.
     *
     * @param {{label?: string}} [descriptor]
     * @returns {GPURenderBundle}
     */
    finish(descriptor?: {
        label?: string;
    }): GPURenderBundle;
}
declare class GPUComputePassEncoder extends GPUPassEncoderBase {
    /**
     * @param {number} x
     * @param {number} [y]
     * @param {number} [z]
     * @returns {void}
     */
    dispatchWorkgroups(x: number, y?: number, z?: number): void;
    /**
     * Dispatches by reading the workgroup counts from a GPU buffer.
     *
     * The buffer must contain 3 `u32`s (`x, y, z`). It must be created with `GPUBufferUsage.INDIRECT`,
     * and `indirectOffset` must be a multiple of 4.
     *
     * @param {GPUBuffer} indirectBuffer
     * @param {number} [indirectOffset]
     * @returns {void}
     */
    dispatchWorkgroupsIndirect(indirectBuffer: GPUBuffer, indirectOffset?: number): void;
    /** @returns {void} */
    end(): void;
}
declare class GPUCommandEncoder {
    _device: GPUDevice;
    /** @type {GPUCommand[]} */
    _commands: GPUCommand[];
    /** @param {GPUDevice} device */
    constructor(device: GPUDevice);
    /**
     * @param {Record<string, any>} descriptor
     * @returns {GPURenderPassEncoder}
     */
    beginRenderPass(descriptor: Record<string, any>): GPURenderPassEncoder;
    /**
     * @param {{label?: string, timestampWrites?: GPUPassTimestampWrites}} [descriptor]
     * @returns {GPUComputePassEncoder}
     */
    beginComputePass(descriptor?: {
        label?: string;
        timestampWrites?: GPUPassTimestampWrites;
    }): GPUComputePassEncoder;
    /**
     * Resolves query results into a buffer. One result is a `u64` (8 bytes).
     *
     * The destination buffer must be created with `GPUBufferUsage.QUERY_RESOLVE`, and
     * `destinationOffset` must be **a multiple of 256** (a spec requirement).
     *
     * @param {GPUQuerySet} querySet
     * @param {number} firstQuery
     * @param {number} queryCount
     * @param {GPUBuffer} destination
     * @param {number} destinationOffset
     * @returns {void}
     */
    resolveQuerySet(querySet: GPUQuerySet, firstQuery: number, queryCount: number, destination: GPUBuffer, destinationOffset: number): void;
    /**
     * A buffer → buffer copy. Per the spec it accepts **two forms**:
     *
     * ```js
     * encoder.copyBufferToBuffer(src, dst)             // all of it (the size comes from src)
     * encoder.copyBufferToBuffer(src, dst, size)       // size bytes from the front
     * encoder.copyBufferToBuffer(src, 16, dst, 0, 64)  // offsets given too
     * ```
     *
     * The short form is told apart by whether the second argument is a `GPUBuffer` — the same criterion as the spec's overload resolution.
     * Omitting `size` means all the remaining bytes of the source.
     *
     * @param {GPUBuffer} source
     * @param {GPUBuffer | number} destinationOrSourceOffset
     * @param {GPUBuffer | number} [sizeOrDestination]
     * @param {number} [destinationOffset]
     * @param {number} [size]
     * @returns {void}
     */
    copyBufferToBuffer(source: GPUBuffer, destinationOrSourceOffset: GPUBuffer | number, sizeOrDestination?: GPUBuffer | number, destinationOffset?: number, size?: number): void;
    /**
     * Fills a range of a buffer with zeros.
     *
     * The result is the same as pushing an array of zeros through `writeBuffer`, but **it does not build
     * that array on the CPU and carry it across the bridge** — a big difference on a compute path that clears a large storage buffer every frame.
     *
     * `offset` and `size` must be multiples of 4, and the buffer must be created with `COPY_DST` (spec rules).
     * Omitting `size` means to the end of the buffer.
     *
     * @param {GPUBuffer} buffer
     * @param {number} [offset]
     * @param {number} [size]
     * @returns {void}
     */
    clearBuffer(buffer: GPUBuffer, offset?: number, size?: number): void;
    /**
     * @param {{texture: GPUTexture} & Record<string, any>} source
     * @param {{buffer: GPUBuffer} & Record<string, any>} destination
     * @param {GPUExtent3D} copySize
     * @returns {void}
     */
    copyTextureToBuffer(source: {
        texture: GPUTexture;
    } & Record<string, any>, destination: {
        buffer: GPUBuffer;
    } & Record<string, any>, copySize: GPUExtent3D): void;
    /**
     * @param {{buffer: GPUBuffer} & Record<string, any>} source
     * @param {{texture: GPUTexture} & Record<string, any>} destination
     * @param {GPUExtent3D} copySize
     * @returns {void}
     */
    copyBufferToTexture(source: {
        buffer: GPUBuffer;
    } & Record<string, any>, destination: {
        texture: GPUTexture;
    } & Record<string, any>, copySize: GPUExtent3D): void;
    /**
     * @param {{texture: GPUTexture} & Record<string, any>} source
     * @param {{texture: GPUTexture} & Record<string, any>} destination
     * @param {GPUExtent3D} copySize
     * @returns {void}
     */
    copyTextureToTexture(source: {
        texture: GPUTexture;
    } & Record<string, any>, destination: {
        texture: GPUTexture;
    } & Record<string, any>, copySize: GPUExtent3D): void;
    /**
     * Opens a debug group — **the range name shows up as is in an Xcode GPU capture** (Metal `pushDebugGroup`).
     *
     * Without it a capture is a list of nameless draws and there is no telling which pass is which. It is the
     * first thing you miss when looking at performance, so it is better put in early while laying out the frame.
     *
     * **It must be paired with `popDebugGroup()`.** An unpaired one is reported by native as a validation
     * error (Metal kills the process with an assertion in that situation, so it is blocked there).
     *
     * @param {string} groupLabel
     * @returns {void}
     */
    pushDebugGroup(groupLabel: string): void;
    /** @returns {void} */
    popDebugGroup(): void;
    /**
     * Leaves a marker at a single point (Metal `insertDebugSignpost`) — a point event, not a range.
     * @param {string} markerLabel
     * @returns {void}
     */
    insertDebugMarker(markerLabel: string): void;
    /** @returns {GPUCommandBuffer} */
    finish(): GPUCommandBuffer;
}
declare class GPUQueue {
    _device: GPUDevice;
    _recorder: Recorder;
    /** @param {GPUDevice} device */
    constructor(device: GPUDevice);
    /**
     * @param {GPUBuffer} buffer
     * @param {number} bufferOffset
     * @param {GPUDataSource} data
     * @param {number} [dataOffset] the start position in elements
     * @param {number} [size] the number of elements
     * @returns {void}
     */
    writeBuffer(buffer: GPUBuffer, bufferOffset: number, data: GPUDataSource, dataOffset?: number, size?: number): void;
    /**
     * @param {{texture: GPUTexture, mipLevel?: number, origin?: GPUOrigin3DDict}} destination
     * @param {GPUDataSource} data
     * @param {{bytesPerRow: number, rowsPerImage?: number}} dataLayout
     * @param {GPUExtent3D} size
     * @returns {void}
     */
    writeTexture(destination: {
        texture: GPUTexture;
        mipLevel?: number;
        origin?: GPUOrigin3DDict;
    }, data: GPUDataSource, dataLayout: {
        bytesPerRow: number;
        rowsPerImage?: number;
    }, size: GPUExtent3D): void;
    /**
     * Uploads an already-decoded image (`createImageBitmap`) into a texture.
     *
     * On the web `<img>`, `<canvas>` and `VideoFrame` are sources too, but Lynx has no such elements —
     * `GPUImageBitmap` takes that place. The pixels stay native, so **all that crosses the bridge is a
     * single handle** (uploading with `writeTexture` moves the whole image).
     *
     * `source.flipY` flips top to bottom **at copy time** (`createImageBitmap`'s `flipY` is at decode time
     * and separate — turning both on flips twice and lands back where it started). Web libraries use this
     * one: three.js's `Texture.flipY` defaults to `true`.
     *
     * @param {{source: GPUImageBitmap, origin?: GPUOrigin3DDict, flipY?: boolean}} source
     * @param {{texture: GPUTexture, mipLevel?: number, origin?: GPUOrigin3DDict,
     *          premultipliedAlpha?: boolean, colorSpace?: string}} destination
     * @param {GPUExtent3D} [copySize] omitted, the whole image
     * @returns {void}
     */
    copyExternalImageToTexture(source: {
        source: GPUImageBitmap;
        origin?: GPUOrigin3DDict;
        flipY?: boolean;
    }, destination: {
        texture: GPUTexture;
        mipLevel?: number;
        origin?: GPUOrigin3DDict;
        premultipliedAlpha?: boolean;
        colorSpace?: string;
    }, copySize?: GPUExtent3D): void;
    /**
     * Merges the commands the encoders gathered into the stream and sends them to native **in one go**.
     * @param {GPUCommandBuffer[]} commandBuffers
     * @returns {WGPUExecuteResult}
     */
    submit(commandBuffers: GPUCommandBuffer[]): WGPUExecuteResult;
    /** @returns {Promise<void>} */
    onSubmittedWorkDone(): Promise<void>;
}
declare class GPUDevice {
    adapter: GPUAdapter;
    limits: Record<string, number>;
    /** The spec's `GPUDevice.adapterInfo` — it looks straight at the adapter's. */
    adapterInfo: GPUAdapterInfoView;
    /**
     * The features enabled on this device — per the spec it holds **only what was requested** (even if the
     * adapter supports it, `has()` is false unless it was asked for via `requiredFeatures`). Web code such
     * as three.js uses the pattern of requesting exactly what it picked from the adapter and re-checking here.
     */
    features: {
        has: (name: string) => boolean;
        size: number;
        values: () => string[];
    };
    /**
     * Device loss notification (the counterpart of `device.lost.then(...)`).
     *
     * This implementation does not report loss, so it stays **pending forever** — the scenarios where a
     * Metal device disappears (an eGPU being unplugged, say) do not exist on iOS, and when the process
     * dies JS dies with it. The property has to exist for bootstraps like `WebGPUBackend.init()` to pass without a TypeError.
     * @type {Promise<GPUDeviceLostInfo>}
     */
    lost: Promise<GPUDeviceLostInfo>;
    /**
     * Errors uncaught by a scope (the spec's `GPUDevice.onuncapturederror`).
     *
     * The name web code knows, so it is kept as is — three.js assigns to it and forwards to
     * `renderer.onError`. The value received is `{type, error}`, where `error` is of the `GPUValidationError` family.
     * @type {((event: {type: string, error: GPUError}) => void) | null}
     */
    onuncapturederror: ((event: {
        type: string;
        error: GPUError;
    }) => void) | null;
    /** @type {((event: {type: string, error: GPUError}) => void)[]} */
    _uncapturedListeners: ((event: {
        type: string;
        error: GPUError;
    }) => void)[];
    _recorder: Recorder;
    queue: GPUQueue;
    /**
     * @param {GPUAdapter} adapter
     * @param {string[]} [requiredFeatures] the feature names already validated in `requestDevice()`
     */
    constructor(adapter: GPUAdapter, requiredFeatures?: string[]);
    /**
     * The command execution error handler (`{kind, message, path}`). Unregistered, errors go to console.error.
     *
     * It works **together with** the spec's `onuncapturederror` — register both and both receive.
     * This one even gives the finished text with the path attached, which is handier for diagnosis.
     *
     * @param {(error: WGPUError, text: string) => void} handler
     * @returns {void}
     */
    onError(handler: (error: WGPUError, text: string) => void): void;
    /**
     * Registers an `uncapturederror` listener (the spec's `EventTarget` slot — this event only).
     * @param {string} type
     * @param {(event: {type: string, error: GPUError}) => void} listener
     * @returns {void}
     */
    addEventListener(type: string, listener: (event: {
        type: string;
        error: GPUError;
    }) => void): void;
    /**
     * @param {string} type
     * @param {(event: {type: string, error: GPUError}) => void} listener
     * @returns {void}
     */
    removeEventListener(type: string, listener: (event: {
        type: string;
        error: GPUError;
    }) => void): void;
    /**
     * Sends one error down the `uncapturederror` channel. `true` if at least one place took it.
     *
     * **The remaining listeners and the next errors keep going** even if a listener throws — one mistake
     * swallowing the whole report would make the very error that caused it disappear, and diagnosis impossible.
     *
     * @param {WGPUError} payload
     * @returns {boolean}
     */
    _dispatchUncaptured(payload: WGPUError): boolean;
    /**
     * Intercepts the errors raised between here and `popErrorScope()` that **match the filter**.
     *
     * An intercepted error does not go to the global handler (`onError`) — it has already been claimed.
     * Scopes can nest, and an error is taken by **the innermost matching scope**.
     *
     * It only records, so it adds no round trip. Use it freely inside a frame.
     *
     * An unknown `filter` is a **synchronous `TypeError`** at the same place as in a browser (the WebIDL enum conversion).
     * Without blocking it here the native scope stack shifts and a later `pop` takes an outer scope —
     * the scope opened to diagnose becomes the source of a misdiagnosis.
     *
     * @param {'validation' | 'out-of-memory' | 'internal'} filter
     * @returns {void}
     */
    pushErrorScope(filter: 'validation' | 'out-of-memory' | 'internal'): void;
    /**
     * Closes the innermost scope and returns **the first error caught there** (`null` if none).
     *
     * It **submits immediately**, like `mapAsync` — otherwise the Promise would not resolve until the next
     * `submit()`. So calling it inside a frame loop adds a round trip.
     * It is an API meant for initialization and diagnostic paths (`docs/JS-AUTHORING.md` §5).
     *
     * @returns {Promise<WGPUError | null>}
     */
    popErrorScope(): Promise<WGPUError | null>;
    /**
     * @param {GPUBufferDescriptor} descriptor
     * @returns {GPUBuffer}
     */
    createBuffer(descriptor: GPUBufferDescriptor): GPUBuffer;
    /**
     * @param {GPUTextureDescriptor} descriptor
     * @returns {GPUTexture}
     */
    createTexture(descriptor: GPUTextureDescriptor): GPUTexture;
    /**
     * @param {Record<string, any>} [descriptor]
     * @returns {GPUSampler}
     */
    createSampler(descriptor?: Record<string, any>): GPUSampler;
    /**
     * @param {GPUShaderModuleDescriptor} descriptor
     * @returns {GPUShaderModule}
     */
    createShaderModule(descriptor: GPUShaderModuleDescriptor): GPUShaderModule;
    /**
     * @param {GPUBindGroupLayoutDescriptor} descriptor
     * @returns {GPUBindGroupLayout}
     */
    createBindGroupLayout(descriptor: GPUBindGroupLayoutDescriptor): GPUBindGroupLayout;
    /**
     * @param {GPUPipelineLayoutDescriptor} descriptor
     * @returns {GPUPipelineLayout}
     */
    createPipelineLayout(descriptor: GPUPipelineLayoutDescriptor): GPUPipelineLayout;
    /**
     * @param {GPUBindGroupDescriptor} descriptor
     * @returns {GPUBindGroup}
     */
    createBindGroup(descriptor: GPUBindGroupDescriptor): GPUBindGroup;
    /**
     * @param {Record<string, any>} descriptor
     * @returns {GPURenderPipeline}
     */
    createRenderPipeline(descriptor: Record<string, any>): GPURenderPipeline;
    /**
     * @param {Record<string, any>} descriptor
     * @returns {GPUComputePipeline}
     */
    createComputePipeline(descriptor: Record<string, any>): GPUComputePipeline;
    /**
     * The asynchronous form of `createRenderPipeline` — it reports failure **as a rejection rather than an exception**.
     *
     * The synchronous form only records a command, so a failure arrives late, in the next `submit()`'s error
     * array. This one wraps the creation command in error scopes, submits immediately, and resolves the
     * Promise from the result. So **you learn right there whether this pipeline is usable** (shader translation failures included).
     *
     * The price is one round trip — an API meant for initialization paths (`docs/JS-AUTHORING.md` §5).
     *
     * @param {Record<string, any>} descriptor
     * @returns {Promise<GPURenderPipeline>}
     */
    createRenderPipelineAsync(descriptor: Record<string, any>): Promise<GPURenderPipeline>;
    /**
     * The asynchronous form of `createComputePipeline` (the same contract as `createRenderPipelineAsync`).
     * @param {Record<string, any>} descriptor
     * @returns {Promise<GPUComputePipeline>}
     */
    createComputePipelineAsync(descriptor: Record<string, any>): Promise<GPUComputePipeline>;
    /**
     * Wraps pipeline creation in error scopes, submits immediately, and resolves the Promise from the result.
     *
     * Why the scopes are **two deep**: pipeline creation fails in two ways. A descriptor problem is
     * `validation` (plus `unsupported`), while a WGSL→MSL translation or Metal compilation failure is
     * `backend` and only the `internal` filter catches it. Striking one layer only lets the other half slip
     * past the scope and leak to the global handler, while the Promise resolves as success and **you hold an unusable pipeline.**
     *
     * Both pops ride in one batch, so the round trips stay at one.
     *
     * @template {GPURenderPipeline | GPUComputePipeline} T
     * @param {() => T} record a function that records the creation command and returns the handle
     * @returns {Promise<T>}
     */
    _createPipelineAsync<T extends GPURenderPipeline | GPUComputePipeline>(record: () => T): Promise<T>;
    /** @returns {GPUCommandEncoder} */
    createCommandEncoder(): GPUCommandEncoder;
    /**
     * Begins recording a bundle of draws to reuse across many frames.
     *
     * `colorFormats` (and `depthStencilFormat`/`sampleCount` if present) are **the shape of the pass this
     * bundle will run in**. A mismatch with the actual pass raises an error at `executeBundles`.
     *
     * @param {GPURenderBundleEncoderDescriptor} descriptor
     * @returns {GPURenderBundleEncoder}
     */
    createRenderBundleEncoder(descriptor: GPURenderBundleEncoderDescriptor): GPURenderBundleEncoder;
    /**
     * Creates a query store.
     *
     * `'occlusion'` counts the samples the draws let through — being deterministic, the value can be trusted.
     * `'timestamp'` is a GPU clock, so the value differs every time even on the same input. On some devices
     * it cannot be created at all (`adapter.features.has('timestamp-query')`), so handle failure.
     *
     * `count` is between 1 and 4096 (the spec's cap).
     *
     * @param {GPUQuerySetDescriptor} descriptor
     * @returns {GPUQuerySet}
     */
    createQuerySet(descriptor: GPUQuerySetDescriptor): GPUQuerySet;
    /** Throws away every GPU object (called when leaving the page). @returns {void} */
    destroy(): void;
}
declare class GPUCanvasContext {
    canvasId: string;
    /** @type {GPUDevice | null} */
    _device: GPUDevice | null;
    format: string;
    /**
     * The last configuration `getConfiguration()` will return — `null` until there is one (as in the spec).
     * @type {GPUCanvasConfiguration | null}
     */
    _configuration: GPUCanvasConfiguration | null;
    /**
     * The spec's `[[currentTexture]]` — **the same object** `getCurrentTexture()` returns until it expires.
     * @type {GPUTexture | null}
     */
    _currentTexture: GPUTexture | null;
    /** The canvas size at the time that texture was taken — used to detect resize expiry. */
    _currentSize: {
        width: any;
        height: any;
    } | null;
    /** @param {string} canvasId the value of `<webgpu-canvas canvas-id="…">` */
    constructor(canvasId: string);
    /**
     * @param {GPUCanvasConfiguration} configuration
     * @returns {void}
     */
    configure(configuration: GPUCanvasConfiguration): void;
    /**
     * Unconfigures — nothing can be drawn through this context until `configure()` is called again.
     *
     * This is what code that reconfigures with a different format (an HDR toggle, say) steps through. After
     * it, `getCurrentTexture()` rejects with "call configure() first".
     *
     * **It does not erase a frame already on screen.** A browser clears the canvas to transparent black, but
     * doing that here would mean clearing and presenting the surface once — a call that unconfigures
     * consuming a frame seems more surprising, so it is not done (`docs/WEBGPU-API.md` §2).
     *
     * @returns {void}
     */
    unconfigure(): void;
    /**
     * The last configuration given (`null` if there is none yet or after `unconfigure()`).
     * @returns {GPUCanvasConfiguration | null}
     */
    getConfiguration(): GPUCanvasConfiguration | null;
    /**
     * This frame's swapchain texture. It becomes invalid once the frame ends (the same rule as a browser).
     * @returns {GPUTexture}
     */
    getCurrentTexture(): GPUTexture;
    /**
     * The spec's "Expire the current texture" — it empties `[[currentTexture]]`.
     *
     * Where the spec calls it: **present**, `configure()`, and a canvas resize. The next call takes a new drawable.
     */
    _expireCurrentTexture(): void;
    /**
     * The canvas's current pixel size.
     *
     * It reads a cache refreshed by the submission (`submit`) response, so calling it inside a frame costs no
     * round trip. Right after a resize with no submission yet, it may be one frame stale — if you need
     * immediacy, use `<webgpu-canvas>`'s `bindcanvasresize` event.
     *
     * @returns {{width: number, height: number}}
     */
    getSize(): {
        width: number;
        height: number;
    };
    /**
     * A synchronous native query — used only when the cache is empty.
     * @returns {{width: number, height: number}}
     */
    _fetchSize(): {
        width: number;
        height: number;
    };
}
declare class GPUAdapter {
    /**
     * The spec's `GPUAdapterInfo`. The `name`, `backend` and `hasUnifiedMemory` below are **this
     * implementation's additions**, present since before the spec had names, and kept as is (existing code uses them).
     * Keys outside the spec are **optional** — another runtime (Dawn, say) may not fill them, and they are
     * filled in harmlessly here (the grade table in `docs/COMMAND-STREAM.md` §5).
     */
    info: GPUAdapterInfoView;
    name: string;
    backend: string;
    limits: Record<string, number>;
    hasUnifiedMemory: boolean;
    /**
     * Device-dependent features — only `has` is imitated so that the same branch as on the web
     * (`adapter.features.has('timestamp-query')`) works (the engine may not have `Set`).
     */
    features: {
        has: (name: string) => boolean;
        size: number;
        values: () => string[];
    };
    /** @param {WGPUAdapterInfo} info */
    constructor(info: WGPUAdapterInfo);
    /**
     * Requiring a feature the adapter does not support **rejects** per the spec — quietly dropping it and
     * creating anyway would blow up much further away, at a later call like `createQuerySet`.
     * @param {{label?: string, requiredFeatures?: string[], requiredLimits?: Record<string, number>}} [descriptor]
     * @returns {Promise<GPUDevice>}
     */
    requestDevice(descriptor?: {
        label?: string;
        requiredFeatures?: string[];
        requiredLimits?: Record<string, number>;
    }): Promise<GPUDevice>;
}
export declare const gpu: {
    /**
     * `navigator.gpu.requestAdapter()`.
     * @returns {Promise<GPUAdapter | null>}
     */
    requestAdapter(): Promise<GPUAdapter | null>;
    /**
     * The format that best fits a canvas surface. The same as `<webgpu-canvas>`'s default.
     * @returns {string}
     */
    getPreferredCanvasFormat(): string;
    /**
     * Attaches to the surface of `<webgpu-canvas canvas-id="…">` (the counterpart of `canvas.getContext('webgpu')`).
     *
     * The **same object** is returned for the same `canvasId` — as with a browser's `getContext('webgpu')`.
     * Making a new one would split the configuration state when `configure()` is called on one handle and
     * `unconfigure()` on another, so the context you thought you were drawing into is really the unconfigured one.
     *
     * @param {string} canvasId
     * @returns {GPUCanvasContext}
     */
    getCanvasContext(canvasId: string): GPUCanvasContext;
};
/**
 * Calls the callback in step with the display refresh (the counterpart of `requestAnimationFrame`).
 *
 * A native CADisplayLink drives it, so frames are more even than with `setInterval`.
 * Calling the returned function stops the loop — it must be called when leaving the page.
 *
 * @param {(frame: {timestamp: number, delta: number}) => void} handler
 * @param {{fps?: number}} [options]
 * @returns {() => void} the stop function
 */
export declare function startFrameLoop(handler: (frame: {
    timestamp: number;
    delta: number;
}) => void, options?: {
    fps?: number;
}): () => void;
/**
 * Installs `requestAnimationFrame` / `cancelAnimationFrame` on the global — **for porting web libraries**.
 *
 * A library that runs its own frame loop, such as three.js, assumes rAF exists. PrimJS has none, so
 * putting it up as is leaves `renderer.init()` **permanently stalled** with no error (the loop never starts).
 *
 * Code written directly here does not need it — `startFrameLoop` is more precise (rAF itself is a thin
 * layer on top of it) and its stop point is clearer. Use this function **only when someone else's code calls rAF**.
 *
 * The display link only runs while callbacks remain — with nobody scheduling the next frame it stops by
 * itself. So when the library ends its loop, the battery is let go too.
 *
 * ```js
 * const uninstall = installAnimationFrame()   // before importing three
 * // …
 * uninstall()                                 // when leaving the page
 * ```
 *
 * In an environment that already has rAF (a browser, some test runners) it **does not overwrite.**
 *
 * @param {{fps?: number}} [options]
 * @returns {() => void} the function that undoes it (restoring the globals)
 */
export declare function installAnimationFrame(options?: {
    fps?: number;
}): () => void;
/**
 * Reads an asset as an `ArrayBuffer` — the slot the browser's `fetch()` takes.
 *
 * It is the channel for data too large to embed in JS source, such as pixels destined for a texture. When
 * native returns `Data`, Lynx converts it to an `ArrayBuffer`, so there is nothing to decode.
 *
 * Name resolution is decided by the host's `assetProvider`. The default provider tries, in order:
 *   1. A name the host registered with `register(_:for:)` — the channel for things that arrive as data
 *      rather than a file, like an image picker's output.
 *   2. An absolute path or a `file://` URL — a file URL from a picker or a download is passed through as is.
 *   3. A path relative to the app bundle (`'hdr-sample.bin'`, `'LUTs/neutral.cube'`)
 *
 * If the host narrowed the accessible scope (`allowedRoots`), a path outside it is rejected
 * (see `WGPUAssetProvider`).
 *
 * @param {string} name the asset name — a registered name, a file path/URL, or a bundle-relative path
 * @returns {Promise<ArrayBuffer>}
 */
export declare function loadAsset(name: string): Promise<ArrayBuffer>;
/**
 * A decoded image — the slot the spec's `ImageBitmap` takes.
 *
 * The pixels **stay native.** All JS knows is the handle and the size, so no data crosses the bridge even
 * for a large image.
 */
declare class GPUImageBitmap {
    id: number;
    width: number;
    height: number;
    _recorder: Recorder | null;
    _closed: boolean;
    /**
     * @param {number} id
     * @param {number} width
     * @param {number} height
     * @param {Recorder | null} recorder
     */
    constructor(id: number, width: number, height: number, recorder: Recorder | null);
    /** Throws away the native pixels (the spec's `ImageBitmap.close()`). */
    close(): void;
}
/**
 * Decodes an encoded image (PNG, JPEG, HEIC …) and readies it to be uploaded into a texture — the slot the
 * web's `createImageBitmap()` takes.
 *
 * The decoding is done natively (ImageIO). It is far faster than unpacking a PNG by hand in JS, and it
 * opens formats such as HEIC that have no JS decoder.
 *
 * ```js
 * const bitmap = await createImageBitmap(await loadAsset('photo.jpg'))
 * const texture = device.createTexture({
 *   size: [bitmap.width, bitmap.height], format: 'rgba8unorm',
 *   usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
 * })
 * device.queue.copyExternalImageToTexture({ source: bitmap }, { texture },
 *   [bitmap.width, bitmap.height])
 * bitmap.close()
 * ```
 *
 * @param {ArrayBuffer | ArrayBufferView | string} source the image bytes, or an asset name
 * @param {{flipY?: boolean, premultiplyAlpha?: 'none' | 'premultiply' | 'default',
 *          resizeWidth?: number, resizeHeight?: number}} [options]
 * @returns {Promise<GPUImageBitmap>}
 */
export declare function createImageBitmap(source: ArrayBuffer | ArrayBufferView | string, options?: {
    flipY?: boolean;
    premultiplyAlpha?: 'none' | 'premultiply' | 'default';
    resizeWidth?: number;
    resizeHeight?: number;
}): Promise<GPUImageBitmap>;
export { GPUImageBitmap, GPUBuffer, GPUTexture, GPUTextureView, GPUSampler, GPUDevice, GPUCanvasContext, GPURenderBundle, GPURenderBundleEncoder, GPUQuerySet, GPUError, GPUValidationError, GPUOutOfMemoryError, GPUInternalError, };
export default gpu;
