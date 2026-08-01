/**
 * Lynx-WebGPU — WebGPU 모양의 JS 클라이언트.
 *
 * 브라우저 WebGPU와 같은 객체 그래프를 JS 쪽에 그대로 두되, 실제 호출은 **명령으로 기록만** 하고
 * `queue.submit()` 시점에 한 번에 네이티브로 보낸다. 핸들(id)은 JS가 발급하므로 객체 생성이
 * 네이티브 왕복을 기다리지 않는다 — 프레임당 브리지 왕복이 1회로 고정된다.
 *
 * 자세한 설계는 docs/ARCHITECTURE.md §3, 지원 범위는 docs/WEBGPU-API.md 참고.
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
};
export type GPUPipelineLayoutSource = {
    id: number;
};
export type GPUTextureInit = {
    size?: GPUExtent3D;
    format?: string;
    usage?: number;
    label?: string;
    frameScoped?: boolean;
};
declare class Recorder {
    /** @type {GPUCommand[]} */
    pending: GPUCommand[];
    nextId: number;
    /** @type {((error: WGPUError, text: string) => void)[]} */
    errorHandlers: ((error: WGPUError, text: string) => void)[];
    constructor();
    /** @returns {number} 새 핸들 id */
    allocate(): number;
    /**
     * @param {GPUCommand} command
     * @returns {GPUCommand}
     */
    push(command: GPUCommand): GPUCommand;
    /**
     * 모아 둔 명령을 네이티브로 넘긴다. 실행할 것이 없으면 아무것도 하지 않는다.
     * @returns {WGPUExecuteResult}
     */
    flush(): WGPUExecuteResult;
    /** @param {WGPUError[]} errors */
    report(errors: WGPUError[]): void;
}
declare class GPUObjectBase {
    _device: GPUDevice;
    _recorder: Recorder;
    id: number;
    label: string;
    /**
     * @param {GPUDevice} device
     * @param {number} id JS가 발급한 핸들
     * @param {string} [label]
     * @param {boolean} [frameScoped] 프레임 끝에 네이티브가 회수하는 핸들이면 true
     */
    constructor(device: GPUDevice, id: number, label?: string, frameScoped?: boolean);
    destroy(): void;
}
declare class GPUBuffer extends GPUObjectBase {
    size: number;
    usage: number;
    /** @type {ArrayBuffer | null} */
    _mapped: ArrayBuffer | null;
    /** @type {ArrayBuffer | null} */
    _mappedRange: ArrayBuffer | null;
    /**
     * @param {GPUDevice} device
     * @param {number} id
     * @param {GPUBufferDescriptor} descriptor
     */
    constructor(device: GPUDevice, id: number, descriptor: GPUBufferDescriptor);
    /** `mappedAtCreation: true`로 만든 버퍼의 초기 데이터 영역. */
    getMappedRange(): ArrayBuffer;
    /** 매핑을 풀면서 실제 생성 명령(초기 데이터 포함)을 기록한다. @returns {void} */
    unmap(): void;
    /**
     * 버퍼 내용을 읽는다. WebGPU의 `mapAsync` + `getMappedRange`를 하나로 합친 형태다.
     *
     * @param {number} [_mode] 스펙 호환용 — 이 구현은 보지 않는다
     * @param {number} [offset] 바이트 오프셋
     * @param {number} [size] 읽을 바이트 수. 생략하면 끝까지
     * @returns {Promise<ArrayBuffer>}
     */
    mapAsync(_mode?: number, offset?: number, size?: number): Promise<ArrayBuffer>;
}
declare class GPUTexture extends GPUObjectBase {
    _frameScoped: boolean;
    width: any;
    height: any;
    format: string | undefined;
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
}
declare class GPUBindGroupLayout extends GPUObjectBase {
}
declare class GPUPipelineLayout extends GPUObjectBase {
}
declare class GPUBindGroup extends GPUObjectBase {
}
declare class GPUPipelineBase extends GPUObjectBase {
    /**
     * `layout: 'auto'` 파이프라인이 유도한 바인드 그룹 레이아웃을 꺼낸다.
     * @param {number} index
     * @returns {GPUBindGroupLayout}
     */
    getBindGroupLayout(index: number): GPUBindGroupLayout;
}
declare class GPURenderPipeline extends GPUPipelineBase {
}
declare class GPUComputePipeline extends GPUPipelineBase {
}
/** 인코더는 자기 명령을 따로 모았다가 `finish()` → `submit()`에서 스트림에 합쳐진다. */
declare class GPUCommandBuffer {
    commands: GPUCommand[];
    /** @param {GPUCommand[]} commands */
    constructor(commands: GPUCommand[]);
}
declare class GPUPassEncoderBase {
    _commands: GPUCommand[];
    /** @param {GPUCommand[]} commands */
    constructor(commands: GPUCommand[]);
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
    /** @returns {void} */
    end(): void;
}
declare class GPURenderPassEncoder extends GPUPassEncoderBase {
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
}
declare class GPUComputePassEncoder extends GPUPassEncoderBase {
    /**
     * @param {number} x
     * @param {number} [y]
     * @param {number} [z]
     * @returns {void}
     */
    dispatchWorkgroups(x: number, y?: number, z?: number): void;
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
    /** @returns {GPUComputePassEncoder} */
    beginComputePass(): GPUComputePassEncoder;
    /**
     * @param {GPUBuffer} source
     * @param {number} sourceOffset
     * @param {GPUBuffer} destination
     * @param {number} destinationOffset
     * @param {number} size
     * @returns {void}
     */
    copyBufferToBuffer(source: GPUBuffer, sourceOffset: number, destination: GPUBuffer, destinationOffset: number, size: number): void;
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
     * @param {number} [dataOffset] 원소 단위 시작 위치
     * @param {number} [size] 원소 개수
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
     * 인코더가 모은 명령을 스트림에 합쳐 **한 번에** 네이티브로 보낸다.
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
    _recorder: Recorder;
    queue: GPUQueue;
    /** @param {GPUAdapter} adapter */
    constructor(adapter: GPUAdapter);
    /**
     * 커맨드 실행 오류 핸들러 (`{kind, message, path}`). 등록하지 않으면 console.error로 나간다.
     * @param {(error: WGPUError, text: string) => void} handler
     * @returns {void}
     */
    onError(handler: (error: WGPUError, text: string) => void): void;
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
    /** @returns {GPUCommandEncoder} */
    createCommandEncoder(): GPUCommandEncoder;
    /** 모든 GPU 객체를 버린다 (페이지 이탈 시 호출). @returns {void} */
    destroy(): void;
}
declare class GPUCanvasContext {
    canvasId: string;
    /** @type {GPUDevice | null} */
    _device: GPUDevice | null;
    format: string;
    /** @param {string} canvasId `<webgpu-canvas canvas-id="…">` 의 값 */
    constructor(canvasId: string);
    /**
     * @param {GPUCanvasConfiguration} configuration
     * @returns {void}
     */
    configure(configuration: GPUCanvasConfiguration): void;
    /**
     * 이번 프레임의 스왑체인 텍스처. 프레임이 끝나면 무효해진다 (브라우저와 같은 규칙).
     * @returns {GPUTexture}
     */
    getCurrentTexture(): GPUTexture;
    /**
     * 캔버스의 현재 픽셀 크기.
     *
     * 제출(`submit`) 응답으로 갱신되는 캐시를 읽으므로 프레임 안에서 불러도 왕복이 없다.
     * 리사이즈 직후 아직 제출이 없었다면 한 프레임 이전 값일 수 있다 — 즉시성이 필요하면
     * `<webgpu-canvas>`의 `bindcanvasresize` 이벤트를 쓸 것.
     *
     * @returns {{width: number, height: number}}
     */
    getSize(): {
        width: number;
        height: number;
    };
    /**
     * 동기 네이티브 조회 — 캐시가 비어 있을 때만 쓴다.
     * @returns {{width: number, height: number}}
     */
    _fetchSize(): {
        width: number;
        height: number;
    };
}
declare class GPUAdapter {
    name: string;
    backend: string;
    limits: Record<string, number>;
    hasUnifiedMemory: boolean | undefined;
    /** @param {WGPUAdapterInfo} info */
    constructor(info: WGPUAdapterInfo);
    /** @returns {Promise<GPUDevice>} */
    requestDevice(): Promise<GPUDevice>;
}
export declare const gpu: {
    /**
     * `navigator.gpu.requestAdapter()`.
     * @returns {Promise<GPUAdapter | null>}
     */
    requestAdapter(): Promise<GPUAdapter | null>;
    /**
     * 캔버스 표면에 가장 잘 맞는 포맷. `<webgpu-canvas>`의 기본값과 같다.
     * @returns {string}
     */
    getPreferredCanvasFormat(): string;
    /**
     * `<webgpu-canvas canvas-id="…">` 의 표면에 붙는다 (`canvas.getContext('webgpu')` 대응).
     * @param {string} canvasId
     * @returns {GPUCanvasContext}
     */
    getCanvasContext(canvasId: string): GPUCanvasContext;
};
/**
 * 화면 갱신 주기에 맞춰 콜백을 부른다 (`requestAnimationFrame` 대응).
 *
 * 네이티브 CADisplayLink가 몰아 주므로 `setInterval`보다 프레임이 고르다.
 * 반환된 함수를 호출하면 루프가 멈춘다 — 페이지를 떠날 때 반드시 호출할 것.
 *
 * @param {(frame: {timestamp: number, delta: number}) => void} handler
 * @param {{fps?: number}} [options]
 * @returns {() => void} 정지 함수
 */
export declare function startFrameLoop(handler: (frame: {
    timestamp: number;
    delta: number;
}) => void, options?: {
    fps?: number;
}): () => void;
export { GPUBuffer, GPUTexture, GPUTextureView, GPUSampler, GPUDevice, GPUCanvasContext };
export default gpu;
