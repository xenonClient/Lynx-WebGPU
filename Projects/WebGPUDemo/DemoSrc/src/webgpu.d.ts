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
 * 명세의 `GPUError` 계층 — `uncapturederror` 이벤트가 실어 나르는 객체.
 *
 * 명세는 `message` 하나만 요구하지만 `kind`·`path`를 함께 둔다. 커맨드 스트림에서 온 오류는
 * "몇 번째 명령의 어느 필드"까지 알고 있고, 그걸 버리면 진단이 크게 나빠지기 때문이다.
 * 웹 코드가 종류를 볼 때는 `instanceof`(또는 `constructor.name`)를 쓴다 — 그래서 하위 클래스로 나눈다.
 */
declare class GPUError {
    message: string;
    /** 이 구현이 붙이는 추가 정보 — 명세에는 없다. */
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
    nextId: number;
    /** @type {((error: WGPUError, text: string) => void)[]} */
    errorHandlers: ((error: WGPUError, text: string) => void)[];
    /**
     * 스코프에 안 잡힌 오류를 명세의 `uncapturederror` 경로로도 흘려보내는 훅.
     * 디바이스가 자기 자신을 꽂는다. 하나라도 받아 갔으면 `true`를 돌려준다.
     * @type {((error: WGPUError) => boolean) | null}
     */
    uncapturedDispatch: ((error: WGPUError) => boolean) | null;
    /**
     * `popErrorScope()`가 돌려준 Promise의 결과 함수들 — **pop한 순서 그대로**다.
     * 네이티브가 같은 순서로 `errorScopes` 배열을 돌려주므로 인덱스로 짝을 맞춘다.
     * @type {{resolve: (error: WGPUError | null) => void, reject: (reason: Error) => void}[]}
     */
    pendingErrorScopes: {
        resolve: (error: WGPUError | null) => void;
        reject: (reason: Error) => void;
    }[];
    constructor();
    /** @returns {number} 새 핸들 id */
    allocate(): number;
    /**
     * 명령을 **기록 시점 값으로 고정해** 쌓는다 (`snapshotValue` 참고) — 디바이스/큐 op은
     * 여기가 곧 호출 시점이라, 호출 뒤 디스크립터 재사용이 스트림을 오염시키지 못한다.
     * @param {GPUCommand} command
     * @returns {GPUCommand}
     */
    push(command: GPUCommand): GPUCommand;
    /**
     * 모아 둔 명령을 네이티브로 넘긴다. 실행할 것이 없으면 아무것도 하지 않는다.
     *
     * `present: false`는 **프레임 중간의 내부 제출**이라는 표시다 (`popErrorScope`·`mapAsync`가
     * 결과를 받으려고 미리 흘려보내는 배치). 네이티브는 이 배치를 커밋하되 드로어블 present와
     * 스왑체인 핸들 만료를 진짜 프레임 제출(`queue.submit`)까지 미룬다 — 안 그러면 획득해 둔
     * 캔버스 텍스처가 그리기도 전에 present되어 남은 패스가 통째로 거부된다.
     *
     * @param {boolean} [present] 이 배치가 프레임 제출인가 (기본 true)
     * @returns {WGPUExecuteResult}
     */
    flush(present?: boolean): WGPUExecuteResult;
    /**
     * `popErrorScope` 명령을 쌓고 결과 Promise를 돌려준다 — **flush하지 않는다.**
     *
     * 스코프 여러 개를 한 배치에 닫을 때 쓴다 (비동기 파이프라인 생성이 그렇다).
     * 네이티브가 pop 순서 그대로 결과를 돌려주므로, 부른 순서가 곧 짝짓는 순서다.
     *
     * @returns {Promise<WGPUError | null>}
     */
    recordPop(): Promise<WGPUError | null>;
    /**
     * 기다리고 있던 `popErrorScope()` Promise들을 푼다.
     *
     * 네이티브가 인덱스를 밀지 않는다는 계약에 기대므로(pop이 실패해도 자리를 남긴다),
     * 여기서는 순서대로 짝지어 주기만 하면 된다. 응답에 결과가 모자라면 `null`이다 —
     * Promise가 영원히 안 풀리는 것보다 낫다.
     *
     * 슬롯이 `{rejected: true}`면 `push`와 짝이 맞지 않았다는 뜻이다. 명세는 그 경우
     * `OperationError`로 **reject**하라고 정하므로(오류를 만들지 않는다) 그대로 따른다 —
     * 그래야 앱이 "스코프가 깨끗했다(null)"와 "짝이 안 맞았다"를 구분할 수 있다.
     *
     * @param {(WGPUError | {rejected: true} | null)[]} popped
     * @returns {void}
     */
    settleErrorScopes(popped: (WGPUError | {
        rejected: true;
    } | null)[]): void;
    /**
     * 스코프에 안 잡힌 오류를 등록된 모든 통로로 보낸다.
     *
     * 통로는 둘이고 **함께** 받는다 — 이 구현의 `onError`(경로가 붙은 텍스트까지 준다)와
     * 명세의 `uncapturederror`(웹 코드가 아는 이름). 아무도 안 듣고 있을 때만 콘솔로 떨어뜨린다:
     * 조용히 사라지는 오류가 없어야 하고, 듣고 있는데 콘솔에도 찍히면 로그가 두 번 남는다.
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
    /** `mappedAtCreation`의 초기 데이터인가 (unmap이 생성 명령을 기록해야 하는가). */
    _mappedAtCreation: boolean;
    /** `mapAsync`가 아직 결과를 기다리는 중인가 (명세의 `"pending"` 상태). */
    _mapPending: boolean;
    /**
     * @param {GPUDevice} device
     * @param {number} id
     * @param {GPUBufferDescriptor} descriptor
     */
    constructor(device: GPUDevice, id: number, descriptor: GPUBufferDescriptor);
    /**
     * 명세의 `GPUBufferMapState` — `'unmapped'` · `'pending'` · `'mapped'`.
     *
     * 매핑 중인 버퍼는 큐 작업에서 거부되므로, 재사용하려는 코드가 **묻지 않고도** 상태를
     * 알 수 있어야 한다. 없으면 `undefined`를 보고 "매핑 안 됐다"로 오해한다.
     *
     * @returns {'unmapped' | 'pending' | 'mapped'}
     */
    get mapState(): 'unmapped' | 'pending' | 'mapped';
    /** `mappedAtCreation: true`로 만든 버퍼의 초기 데이터 영역. */
    getMappedRange(): ArrayBuffer;
    /**
     * 매핑을 푼다.
     *
     * `mappedAtCreation`이면 여기서 실제 생성 명령(초기 데이터 포함)이 기록되고,
     * `mapAsync`로 매핑한 것이면 네이티브에 "이제 큐 작업에 써도 된다"를 알린다 —
     * 매핑 중인 버퍼는 명세대로 큐 작업에서 거부되므로 **읽고 나면 반드시 불러야 한다.**
     *
     * @returns {void}
     */
    unmap(): void;
    /**
     * 버퍼 내용을 읽는다. WebGPU의 `mapAsync` + `getMappedRange`를 하나로 합친 형태다.
     *
     * 읽는 동안 이 버퍼는 명세대로 **"unavailable"**이 되어 큐 작업(쓰기·복사·resolve·드로우
     * 바인딩)에서 거부된다. 그러지 않으면 리드백이 GPU 완료를 기다리는 사이 다음 프레임의
     * 쓰기가 같은 메모리에 겹쳐, 받은 값이 어느 프레임 것인지 보장되지 않는다.
     * **다 읽었으면 `unmap()`을 부를 것.**
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
    depthOrArrayLayers: any;
    mipLevelCount: number;
    sampleCount: number;
    dimension: string;
    format: string | undefined;
    usage: number;
    /**
     * 이 텍스처를 바인딩할 때의 기본 뷰 차원. 명세는 생략을 허용하고, 그때는 `dimension`과
     * 레이어 수에서 정해진다 (2d + 레이어 2 이상이면 `2d-array`).
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
     * 이 모듈의 컴파일 진단 (명세 `GPUCompilationInfo`).
     *
     * 셰이더 모듈은 **컴파일에 실패해도 만들어진다** (명세 모델) — 실패는 여기와 파이프라인
     * 생성 실패로 드러난다. 그래서 `createShaderModule()`이 성공한 뒤에도 확인할 값이 있다.
     *
     * `messages[].lineNum`은 WGSL 소스의 줄 번호(1부터)다. `linePos`·`offset`·`length`는
     * 이 구현이 알지 못해 **0으로 둔다** — 모르는 값을 지어내면 편집기가 엉뚱한 곳에 밑줄을 긋는다.
     *
     * 왕복이 하나 붙으므로 진단 경로에서만 쓸 것.
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
/**
 * `bundleEncoder.finish()`가 돌려주는 재사용 가능한 드로우 묶음.
 *
 * 기록한 명령이 **래퍼보다 오래 사는** 유일한 구조라, 자기가 쓰는 리소스 래퍼를 붙잡는다
 * (`_retained`). 그러지 않으면 초기화 함수가 번들만 반환하고 파이프라인·버퍼를 버렸을 때
 * GC가 `destroy`를 끼워 넣어 번들이 **조용히 안 그려진다** (`docs/JS-AUTHORING.md` §8).
 */
declare class GPURenderBundle extends GPUObjectBase {
    /** @type {object[]} 이 번들이 참조하는 리소스 래퍼 — 수명을 함께 묶는다. */
    _retained: object[];
    /**
     * @param {GPUDevice} device
     * @param {number} id
     * @param {string} [label]
     */
    constructor(device: GPUDevice, id: number, label?: string);
}
/** `device.createQuerySet()`이 돌려주는 쿼리 저장소. */
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
/** 인코더는 자기 명령을 따로 모았다가 `finish()` → `submit()`에서 스트림에 합쳐진다. */
declare class GPUCommandBuffer {
    commands: GPUCommand[];
    /** @param {GPUCommand[]} commands */
    constructor(commands: GPUCommand[]);
}
declare class GPUPassEncoderBase {
    _commands: GPUCommand[];
    /**
     * 기록 중 만난 리소스 래퍼 — **번들 인코더만** 채운다.
     *
     * 번들은 기록한 명령이 래퍼보다 오래 사는 유일한 구조라서, 자동 해제(GC)가 붙은 엔진에서는
     * 초기화 함수가 번들만 반환하고 파이프라인·버퍼 래퍼를 버리면 번들이 조용히 안 그려진다.
     * 명세 모델에서 번들은 자기가 쓰는 객체를 **소유**하므로, 여기서도 같은 소유 관계를 만든다.
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
     * 디버그 그룹을 연다 — **Xcode GPU 캡처에 구간 이름이 그대로 뜬다** (Metal `pushDebugGroup`).
     *
     * 없으면 캡처가 이름 없는 드로우 나열이 되어 어느 패스가 무엇인지 알 수 없다. 성능을 볼 때
     * 가장 먼저 아쉬워지는 부분이라, 프레임 구조를 잡을 때 미리 넣어 두는 편이 낫다.
     *
     * `popDebugGroup()`과 **반드시 짝을 맞출 것.** 짝이 안 맞으면 네이티브가 validation 오류로
     * 알려 준다 (Metal은 그 상황에서 단언으로 프로세스를 죽이므로 거기서 막는다).
     *
     * @param {string} groupLabel
     * @returns {void}
     */
    pushDebugGroup(groupLabel: string): void;
    /** @returns {void} */
    popDebugGroup(): void;
    /**
     * 한 지점에 표식을 남긴다 (Metal `insertDebugSignpost`) — 구간이 아니라 점 이벤트다.
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
 * 렌더 패스와 렌더 번들이 **함께** 쓸 수 있는 명령.
 *
 * 이 경계는 명세가 정한 것이다 — 번들에는 뷰포트·시저·블렌드 상수·스텐실 참조·중첩 번들을
 * 담을 수 없다. 그것들을 `GPURenderPassEncoder`에만 두면 번들 인코더에는 애초에 그 메서드가
 * 없으므로, 잘못 쓰는 코드가 네이티브까지 가지 않는다.
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
     * 드로우 인자를 GPU 버퍼에서 읽어 그린다 — 컴퓨트가 만든 수만큼 그릴 때 쓴다.
     *
     * 버퍼에는 `u32` 4개가 이 순서로 들어 있어야 한다:
     * `vertexCount, instanceCount, firstVertex, firstInstance`.
     * 버퍼는 `GPUBufferUsage.INDIRECT`로 만들고, `indirectOffset`은 4의 배수여야 한다.
     *
     * `firstInstance`가 0이 아니면 `indirect-first-instance` 기능이 필요하다. 이 구현은 그 기능을
     * 항상 보고하지만, 브라우저에서는 **요청하지 않으면 드로우가 통째로 no-op**이 된다.
     *
     * @param {GPUBuffer} indirectBuffer
     * @param {number} [indirectOffset]
     * @returns {void}
     */
    drawIndirect(indirectBuffer: GPUBuffer, indirectOffset?: number): void;
    /**
     * 인덱스 드로우 인자를 GPU 버퍼에서 읽어 그린다.
     *
     * 버퍼에는 `u32` 5개가 이 순서로 들어 있어야 한다:
     * `indexCount, instanceCount, firstIndex, baseVertex(부호 있는 i32), firstInstance`.
     * `firstIndex`는 인자 버퍼 안에 있으므로 `setIndexBuffer(buffer, format, offset)`의
     * 오프셋과 **더해지지 않고 따로** 적용된다.
     *
     * `firstInstance`가 0이 아니면 `indirect-first-instance` 기능이 필요하다. 이 구현은 그 기능을
     * 항상 보고하지만, 브라우저에서는 **요청하지 않으면 드로우가 통째로 no-op**이 된다.
     *
     * @param {GPUBuffer} indirectBuffer
     * @param {number} [indirectOffset]
     * @returns {void}
     */
    drawIndexedIndirect(indirectBuffer: GPUBuffer, indirectOffset?: number): void;
}
/** 패스 전용 명령 — 아래 넷과 `executeBundles`는 번들에 담을 수 없다 (명세). */
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
     * 이 드로우들이 통과시킨 샘플 수를 세기 시작한다.
     *
     * `beginRenderPass`에 `occlusionQuerySet`을 준 패스에서만 쓸 수 있고, 중첩할 수 없다.
     * 같은 인덱스를 한 패스에서 두 번 쓸 수 없고, 패스를 닫기 전에 `endOcclusionQuery`로
     * 반드시 닫아야 한다.
     *
     * @param {number} queryIndex
     * @returns {void}
     */
    beginOcclusionQuery(queryIndex: number): void;
    /** @returns {void} */
    endOcclusionQuery(): void;
    /**
     * 미리 기록해 둔 번들들을 이 패스에 되풀이한다.
     *
     * 번들은 패스 상태를 **물려받지 않고**, 실행이 끝나면 패스의 파이프라인·바인드 그룹·
     * 정점/인덱스 버퍼 바인딩이 **무효화된다** (이전 값으로 복원되는 것이 아니다 — 명세 계약).
     * 이어서 그리려면 `setPipeline`·`setBindGroup`·`setVertexBuffer`를 다시 해야 하고,
     * 빠뜨리면 그 드로우가 거부된다. 뷰포트·시저·블렌드 상수·스텐실 참조는 그대로 남는다.
     *
     * @param {GPURenderBundle[]} bundles
     * @returns {void}
     */
    executeBundles(bundles: GPURenderBundle[]): void;
    /** @returns {void} */
    end(): void;
}
/**
 * 여러 프레임에 걸쳐 다시 쓸 드로우 묶음을 기록한다 (`device.createRenderBundleEncoder`).
 *
 * 이 구현에서 번들의 이득은 브라우저와 다르다 — 브라우저는 드라이버 명령을 미리 만들어 두지만,
 * 여기서는 **JS가 매 프레임 같은 명령 배열을 다시 만들지 않아도 되는 것**이 이득이다.
 * 번들을 실행하는 명령은 핸들 하나뿐이고, 되풀이는 네이티브가 한다.
 */
declare class GPURenderBundleEncoder extends GPURenderCommandsBase {
    /** @type {object[]} 기록 중 만난 래퍼 — 만든 번들이 이어받아 붙잡는다. */
    _retained: object[];
    _finished: boolean;
    _device: GPUDevice;
    _descriptor: GPURenderBundleEncoderDescriptor;
    /**
     * @param {GPUDevice} device
     * @param {GPURenderBundleEncoderDescriptor} descriptor
     */
    constructor(device: GPUDevice, descriptor: GPURenderBundleEncoderDescriptor);
    /**
     * 기록을 끝내고 재사용 가능한 번들을 만든다.
     *
     * **한 번만 부를 수 있다.** 두 번째 호출은 명세대로 오류를 내고 **무효한 번들**을 돌려준다 —
     * 조용히 빈 번들을 주면 `executeBundles`가 아무것도 그리지 않고 오류도 없어서 원인을
     * 찾기 어렵다.
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
     * 워크그룹 수를 GPU 버퍼에서 읽어 디스패치한다.
     *
     * 버퍼에는 `u32` 3개(`x, y, z`)가 들어 있어야 한다. 버퍼는 `GPUBufferUsage.INDIRECT`로
     * 만들고, `indirectOffset`은 4의 배수여야 한다.
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
     * 쿼리 결과를 버퍼로 내린다. 결과 하나는 `u64`(8바이트)다.
     *
     * 목적지 버퍼는 `GPUBufferUsage.QUERY_RESOLVE`로 만들어야 하고,
     * `destinationOffset`은 **256의 배수**여야 한다 (명세 요구).
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
     * 버퍼 → 버퍼 복사. 명세대로 **두 가지 형태**를 받는다:
     *
     * ```js
     * encoder.copyBufferToBuffer(src, dst)             // 전부 (크기는 src에서)
     * encoder.copyBufferToBuffer(src, dst, size)       // 앞에서 size 바이트
     * encoder.copyBufferToBuffer(src, 16, dst, 0, 64)  // 오프셋까지 지정
     * ```
     *
     * 짧은 형태는 두 번째 인자가 `GPUBuffer`인지로 가른다 — 명세의 오버로드 해소와 같은 기준이다.
     * `size`를 생략하면 원본의 남은 바이트 전부다.
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
     * 버퍼의 한 구간을 0으로 채운다.
     *
     * `writeBuffer`로 0 배열을 밀어 넣는 것과 결과는 같지만 **CPU에서 그 배열을 만들어 브리지로
     * 실어 보내지 않는다** — 큰 스토리지 버퍼를 프레임마다 초기화하는 컴퓨트 경로에서 차이가 크다.
     *
     * `offset`·`size`는 4의 배수여야 하고, 버퍼는 `COPY_DST`로 만들어야 한다 (명세 규칙).
     * `size`를 생략하면 버퍼 끝까지다.
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
     * 디버그 그룹을 연다 — **Xcode GPU 캡처에 구간 이름이 그대로 뜬다** (Metal `pushDebugGroup`).
     *
     * 없으면 캡처가 이름 없는 드로우 나열이 되어 어느 패스가 무엇인지 알 수 없다. 성능을 볼 때
     * 가장 먼저 아쉬워지는 부분이라, 프레임 구조를 잡을 때 미리 넣어 두는 편이 낫다.
     *
     * `popDebugGroup()`과 **반드시 짝을 맞출 것.** 짝이 안 맞으면 네이티브가 validation 오류로
     * 알려 준다 (Metal은 그 상황에서 단언으로 프로세스를 죽이므로 거기서 막는다).
     *
     * @param {string} groupLabel
     * @returns {void}
     */
    pushDebugGroup(groupLabel: string): void;
    /** @returns {void} */
    popDebugGroup(): void;
    /**
     * 한 지점에 표식을 남긴다 (Metal `insertDebugSignpost`) — 구간이 아니라 점 이벤트다.
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
    /** 명세 `GPUDevice.adapterInfo` — 어댑터의 것을 그대로 본다. */
    adapterInfo: GPUAdapterInfoView;
    /**
     * 이 디바이스에 활성화된 기능 — 명세대로 **요청한 것만** 들어 있다 (어댑터가 지원해도
     * `requiredFeatures`로 요청하지 않았으면 `has()`는 false다). Three.js 등 웹 코드가
     * 어댑터에서 고른 기능을 그대로 요청하고 여기서 다시 확인하는 패턴을 쓴다.
     */
    features: {
        has: (name: string) => boolean;
        size: number;
        values: () => string[];
    };
    /**
     * 디바이스 유실 통지 (`device.lost.then(...)` 대응).
     *
     * 이 구현은 유실을 보고하지 않으므로 **영원히 pending**이다 — Metal 디바이스가 사라지는
     * 시나리오(eGPU 분리 등)가 iOS에는 없고, 프로세스가 죽는 경우는 JS도 함께 죽는다.
     * 속성 자체는 있어야 `WebGPUBackend.init()`류의 부트스트랩이 TypeError 없이 지나간다.
     * @type {Promise<GPUDeviceLostInfo>}
     */
    lost: Promise<GPUDeviceLostInfo>;
    /**
     * 스코프에 안 잡힌 오류 (명세 `GPUDevice.onuncapturederror`).
     *
     * 웹 코드가 아는 이름이라 그대로 둔다 — Three.js가 여기에 대입해 `renderer.onError`로
     * 넘긴다. 받는 값은 `{type, error}`이고 `error`는 `GPUValidationError` 계열이다.
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
     * @param {string[]} [requiredFeatures] `requestDevice()`에서 검증을 마친 기능 이름들
     */
    constructor(adapter: GPUAdapter, requiredFeatures?: string[]);
    /**
     * 커맨드 실행 오류 핸들러 (`{kind, message, path}`). 등록하지 않으면 console.error로 나간다.
     *
     * 명세의 `onuncapturederror`와 **함께** 동작한다 — 둘 다 등록하면 둘 다 받는다.
     * 이쪽은 경로가 붙은 완성된 텍스트까지 주므로 진단에는 더 편하다.
     *
     * @param {(error: WGPUError, text: string) => void} handler
     * @returns {void}
     */
    onError(handler: (error: WGPUError, text: string) => void): void;
    /**
     * `uncapturederror` 리스너 등록 (명세의 `EventTarget` 자리 — 이 이벤트만 받는다).
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
     * 오류 하나를 `uncapturederror` 통로로 흘린다. 받아 간 곳이 하나라도 있으면 `true`.
     *
     * 리스너가 던져도 **나머지 리스너와 다음 오류는 계속 간다** — 하나의 실수가 전체 보고를
     * 삼키면, 정작 원인인 오류가 사라져 진단이 불가능해진다.
     *
     * @param {WGPUError} payload
     * @returns {boolean}
     */
    _dispatchUncaptured(payload: WGPUError): boolean;
    /**
     * 여기서부터 `popErrorScope()`까지 사이에 난 오류 중 **필터에 맞는 것**을 가로챈다.
     *
     * 가로챈 오류는 전역 핸들러(`onError`)로 가지 않는다 — 이미 처리하기로 한 것이기 때문이다.
     * 스코프는 중첩할 수 있고, 오류는 **가장 안쪽의 맞는 스코프**가 가져간다.
     *
     * 기록만 하므로 왕복이 늘지 않는다. 프레임 안에서 마음껏 써도 된다.
     *
     * 알 수 없는 `filter`는 브라우저(WebIDL enum 변환)와 같은 자리에서 **동기 `TypeError`**다.
     * 여기서 막지 않으면 네이티브 스코프 스택이 밀려 이후 `pop`이 바깥 스코프를 가져간다 —
     * 진단하려고 연 스코프가 오히려 오진을 만든다.
     *
     * @param {'validation' | 'out-of-memory' | 'internal'} filter
     * @returns {void}
     */
    pushErrorScope(filter: 'validation' | 'out-of-memory' | 'internal'): void;
    /**
     * 가장 안쪽 스코프를 닫고 **거기서 처음 잡힌 오류**를 돌려준다 (없으면 `null`).
     *
     * `mapAsync`처럼 **즉시 제출한다** — 그러지 않으면 다음 `submit()`이 올 때까지 Promise가
     * 풀리지 않기 때문이다. 그래서 프레임 루프 안에서 부르면 왕복이 하나 늘어난다.
     * 초기화나 진단 경로에서 쓰는 것을 전제로 한 API다 (`docs/JS-AUTHORING.md` §5).
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
     * `createRenderPipeline`의 비동기 판 — 실패를 **예외가 아니라 거부로** 알린다.
     *
     * 동기 판은 명령만 기록하므로 실패가 다음 `submit()`의 오류 배열로 늦게 온다. 이쪽은
     * 생성 명령을 오류 스코프로 감싸 즉시 제출하고, 결과를 보고 Promise를 푼다. 그래서
     * **"이 파이프라인이 쓸 수 있는가"를 그 자리에서 알 수 있다** (셰이더 번역 실패 포함).
     *
     * 대가는 왕복 하나다 — 초기화 경로에서 쓰는 것을 전제로 한 API다 (`docs/JS-AUTHORING.md` §5).
     *
     * @param {Record<string, any>} descriptor
     * @returns {Promise<GPURenderPipeline>}
     */
    createRenderPipelineAsync(descriptor: Record<string, any>): Promise<GPURenderPipeline>;
    /**
     * `createComputePipeline`의 비동기 판 (`createRenderPipelineAsync`와 같은 계약).
     * @param {Record<string, any>} descriptor
     * @returns {Promise<GPUComputePipeline>}
     */
    createComputePipelineAsync(descriptor: Record<string, any>): Promise<GPUComputePipeline>;
    /**
     * 파이프라인 생성을 오류 스코프로 감싸 즉시 제출하고 결과로 Promise를 푼다.
     *
     * 스코프가 **두 겹**인 이유: 파이프라인 생성은 두 종류로 실패한다. 디스크립터 문제는
     * `validation`(+`unsupported`)이고, WGSL→MSL 번역·Metal 컴파일 실패는 `backend`라
     * `internal` 필터로만 잡힌다. 한 겹만 치면 나머지 절반이 스코프를 빠져나가 전역
     * 핸들러로 새고, Promise는 성공으로 풀려 **못 쓰는 파이프라인을 손에 쥔다.**
     *
     * 두 pop을 한 배치에 실어 왕복은 하나로 유지한다.
     *
     * @template {GPURenderPipeline | GPUComputePipeline} T
     * @param {() => T} record 생성 명령을 기록하고 핸들을 돌려주는 함수
     * @returns {Promise<T>}
     */
    _createPipelineAsync<T extends GPURenderPipeline | GPUComputePipeline>(record: () => T): Promise<T>;
    /** @returns {GPUCommandEncoder} */
    createCommandEncoder(): GPUCommandEncoder;
    /**
     * 여러 프레임에 걸쳐 다시 쓸 드로우 묶음을 기록하기 시작한다.
     *
     * `colorFormats`(와 있다면 `depthStencilFormat`·`sampleCount`)는 이 번들을 **실행할 패스의
     * 모양**이다. 실제 패스와 어긋나면 `executeBundles`에서 오류가 난다.
     *
     * @param {GPURenderBundleEncoderDescriptor} descriptor
     * @returns {GPURenderBundleEncoder}
     */
    createRenderBundleEncoder(descriptor: GPURenderBundleEncoderDescriptor): GPURenderBundleEncoder;
    /**
     * 쿼리 저장소를 만든다.
     *
     * `'occlusion'`은 드로우가 통과시킨 샘플 수를 센다 — 결정적이라 값을 믿을 수 있다.
     * `'timestamp'`는 GPU 시계라 같은 입력에도 값이 매번 다르다. 기기에 따라 아예 만들 수
     * 없으므로(`adapter.features.has('timestamp-query')`) 실패를 처리할 것.
     *
     * `count`는 1 이상 4096 이하다 (명세 상한).
     *
     * @param {GPUQuerySetDescriptor} descriptor
     * @returns {GPUQuerySet}
     */
    createQuerySet(descriptor: GPUQuerySetDescriptor): GPUQuerySet;
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
    /**
     * 명세 `GPUAdapterInfo`. 아래 `name`·`backend`·`hasUnifiedMemory`는 **이 구현의 추가**로,
     * 명세 이름이 없던 시절부터 있던 것이라 그대로 둔다 (기존 코드가 쓴다).
     */
    info: GPUAdapterInfoView;
    name: string;
    backend: string;
    limits: Record<string, number>;
    hasUnifiedMemory: boolean | undefined;
    /**
     * 기기마다 갈리는 기능 — 웹과 같은 분기(`adapter.features.has('timestamp-query')`)가
     * 동작하도록 `has`만 흉내 낸다 (엔진에 `Set`이 없을 수 있다).
     */
    features: {
        has: (name: string) => boolean;
        size: number;
        values: () => string[];
    };
    /** @param {WGPUAdapterInfo} info */
    constructor(info: WGPUAdapterInfo);
    /**
     * 어댑터가 지원하지 않는 기능을 요구하면 명세대로 **거부**한다 — 조용히 빼고 만들면
     * 이후 `createQuerySet` 같은 호출이 훨씬 먼 곳에서 터진다.
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
/**
 * `requestAnimationFrame` / `cancelAnimationFrame`을 전역에 깐다 — **웹 라이브러리 이식용**.
 *
 * Three.js처럼 자체 프레임 루프를 도는 라이브러리는 rAF가 있다고 전제한다. PrimJS에는 없으므로
 * 그대로 올리면 `renderer.init()`이 오류 없이 **영구 정지**한다 (루프가 시작되지 않는다).
 *
 * 직접 쓰는 코드에는 필요 없다 — `startFrameLoop`가 더 정확하고(rAF 자체가 그 위의 얇은 층이다)
 * 정지 시점도 분명하다. 이 함수는 **남의 코드가 rAF를 부를 때만** 쓴다.
 *
 * 콜백이 남아 있는 동안만 디스플레이 링크가 돈다 — 아무도 다음 프레임을 예약하지 않으면
 * 스스로 멈춘다. 그래서 라이브러리가 루프를 끝내면 배터리도 따라서 놓인다.
 *
 * ```js
 * const uninstall = installAnimationFrame()   // three를 import 하기 전에
 * // …
 * uninstall()                                 // 페이지를 떠날 때
 * ```
 *
 * 이미 rAF가 있는 환경(브라우저·일부 테스트 러너)에서는 **덮지 않는다.**
 *
 * @param {{fps?: number}} [options]
 * @returns {() => void} 되돌리는 함수 (전역을 원래대로)
 */
export declare function installAnimationFrame(options?: {
    fps?: number;
}): () => void;
/**
 * 애셋을 `ArrayBuffer`로 읽는다 — 브라우저의 `fetch()` 자리다.
 *
 * 텍스처로 올릴 픽셀처럼 JS 소스에 박기엔 큰 데이터를 가져오는 통로다. 네이티브가
 * `Data`로 돌려주면 Lynx가 `ArrayBuffer`로 바꿔 주므로 디코딩할 것이 없다.
 *
 * 이름 해석은 호스트의 `assetProvider`가 정한다. 기본 공급자는 순서대로:
 *   1. 호스트가 `register(_:for:)`로 등록한 이름 — 이미지 피커처럼 파일이 아니라
 *      데이터로 오는 것의 통로다.
 *   2. 절대 경로 또는 `file://` URL — 피커·다운로드가 준 파일 URL을 그대로 넘긴다.
 *   3. 앱 번들 상대 경로 (`'hdr-sample.bin'`, `'LUTs/neutral.cube'`)
 *
 * 호스트가 접근 범위를 좁혀 두었다면(`allowedRoots`) 그 밖의 경로는 거부된다
 * (`WGPUAssetProvider` 참고).
 *
 * @param {string} name 애셋 이름 — 등록 이름, 파일 경로/URL, 또는 번들 상대 경로
 * @returns {Promise<ArrayBuffer>}
 */
export declare function loadAsset(name: string): Promise<ArrayBuffer>;
export { GPUBuffer, GPUTexture, GPUTextureView, GPUSampler, GPUDevice, GPUCanvasContext, GPURenderBundle, GPURenderBundleEncoder, GPUQuerySet, GPUError, GPUValidationError, GPUOutOfMemoryError, GPUInternalError, };
export default gpu;
