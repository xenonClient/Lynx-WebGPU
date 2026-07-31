/**
 * Lynx-WebGPU 클라이언트 타입 선언.
 *
 * 브라우저 WebGPU 타입(`@webgpu/types`)의 **부분집합**이다 — 이 구현이 지원하는 범위만 담았다
 * (docs/WEBGPU-API.md). 브라우저와 코드를 공유하려면 `@webgpu/types`를 쓰고 이 파일은
 * `gpu`/`startFrameLoop` 진입점 용도로만 참고할 것.
 */

export declare const GPUBufferUsage: {
  MAP_READ: number; MAP_WRITE: number; COPY_SRC: number; COPY_DST: number;
  INDEX: number; VERTEX: number; UNIFORM: number; STORAGE: number;
  INDIRECT: number; QUERY_RESOLVE: number;
};

export declare const GPUTextureUsage: {
  COPY_SRC: number; COPY_DST: number; TEXTURE_BINDING: number;
  STORAGE_BINDING: number; RENDER_ATTACHMENT: number;
};

export declare const GPUShaderStage: { VERTEX: number; FRAGMENT: number; COMPUTE: number };
export declare const GPUColorWrite: { RED: number; GREEN: number; BLUE: number; ALPHA: number; ALL: number };
export declare const GPUMapMode: { READ: number; WRITE: number };

export interface WGPUError {
  kind: 'validation' | 'out-of-memory' | 'unsupported' | 'backend';
  message: string;
  path?: string;
}

export interface GPUExtent3DDict { width: number; height?: number; depthOrArrayLayers?: number }
export type GPUExtent3D = GPUExtent3DDict | number[];
export interface GPUOrigin3DDict { x?: number; y?: number; z?: number }
export interface GPUColorDict { r: number; g: number; b: number; a: number }
export type GPUColor = GPUColorDict | number[];

export declare class GPUBuffer {
  readonly id: number;
  readonly size: number;
  readonly usage: number;
  label: string;
  getMappedRange(): ArrayBuffer;
  unmap(): void;
  mapAsync(mode: number, offset?: number, size?: number): Promise<ArrayBuffer>;
  destroy(): void;
}

export declare class GPUTextureView {
  readonly id: number;
  destroy(): void;
}

export declare class GPUTexture {
  readonly id: number;
  readonly width: number;
  readonly height: number;
  createView(descriptor?: Record<string, unknown>): GPUTextureView;
  destroy(): void;
}

export declare class GPUSampler {
  readonly id: number;
  destroy(): void;
}

export declare class GPUShaderModule {
  readonly id: number;
  destroy(): void;
}

export declare class GPUBindGroupLayout { readonly id: number }
export declare class GPUPipelineLayout { readonly id: number }
export declare class GPUBindGroup { readonly id: number }

export declare class GPURenderPipeline {
  readonly id: number;
  getBindGroupLayout(index: number): GPUBindGroupLayout;
}

export declare class GPUComputePipeline {
  readonly id: number;
  getBindGroupLayout(index: number): GPUBindGroupLayout;
}

export declare class GPURenderPassEncoder {
  setPipeline(pipeline: GPURenderPipeline): void;
  setBindGroup(index: number, bindGroup: GPUBindGroup, dynamicOffsets?: number[]): void;
  setVertexBuffer(slot: number, buffer: GPUBuffer, offset?: number): void;
  setIndexBuffer(buffer: GPUBuffer, format: 'uint16' | 'uint32', offset?: number): void;
  setViewport(x: number, y: number, width: number, height: number, minDepth?: number, maxDepth?: number): void;
  setScissorRect(x: number, y: number, width: number, height: number): void;
  setBlendConstant(color: GPUColor): void;
  setStencilReference(reference: number): void;
  draw(vertexCount: number, instanceCount?: number, firstVertex?: number, firstInstance?: number): void;
  drawIndexed(
    indexCount: number, instanceCount?: number, firstIndex?: number,
    baseVertex?: number, firstInstance?: number
  ): void;
  end(): void;
}

export declare class GPUComputePassEncoder {
  setPipeline(pipeline: GPUComputePipeline): void;
  setBindGroup(index: number, bindGroup: GPUBindGroup, dynamicOffsets?: number[]): void;
  dispatchWorkgroups(x: number, y?: number, z?: number): void;
  end(): void;
}

export declare class GPUCommandBuffer {}

export declare class GPUCommandEncoder {
  beginRenderPass(descriptor: Record<string, any>): GPURenderPassEncoder;
  beginComputePass(): GPUComputePassEncoder;
  copyBufferToBuffer(
    source: GPUBuffer, sourceOffset: number,
    destination: GPUBuffer, destinationOffset: number, size: number
  ): void;
  copyTextureToBuffer(source: Record<string, any>, destination: Record<string, any>, copySize: GPUExtent3D): void;
  copyBufferToTexture(source: Record<string, any>, destination: Record<string, any>, copySize: GPUExtent3D): void;
  copyTextureToTexture(source: Record<string, any>, destination: Record<string, any>, copySize: GPUExtent3D): void;
  finish(): GPUCommandBuffer;
}

export declare class GPUQueue {
  writeBuffer(
    buffer: GPUBuffer, bufferOffset: number,
    data: ArrayBuffer | ArrayBufferView | number[], dataOffset?: number, size?: number
  ): void;
  writeTexture(
    destination: { texture: GPUTexture; mipLevel?: number; origin?: GPUOrigin3DDict },
    data: ArrayBuffer | ArrayBufferView | number[],
    dataLayout: { bytesPerRow: number; rowsPerImage?: number },
    size: GPUExtent3D
  ): void;
  submit(commandBuffers: GPUCommandBuffer[]): {
    ok: boolean;
    commandCount?: number;
    errors?: WGPUError[];
    /** 이번 제출이 건드린 캔버스의 픽셀 크기 — `context.getSize()` 캐시가 이걸로 갱신된다. */
    canvases?: Record<string, { width: number; height: number }>;
    /** 네이티브에 살아 있는 GPU 객체 수 — 프레임마다 늘면 destroy 누락이다 (docs/JS-AUTHORING.md §8). */
    objects?: number;
  };
  onSubmittedWorkDone(): Promise<void>;
}

export declare class GPUDevice {
  readonly queue: GPUQueue;
  readonly limits: Record<string, number>;
  onError(handler: (error: WGPUError, text: string) => void): void;
  createBuffer(descriptor: { size: number; usage: number; mappedAtCreation?: boolean; label?: string }): GPUBuffer;
  createTexture(descriptor: {
    size: GPUExtent3D; format: string; usage: number;
    dimension?: string; mipLevelCount?: number; sampleCount?: number; label?: string;
  }): GPUTexture;
  createSampler(descriptor?: Record<string, unknown>): GPUSampler;
  createShaderModule(descriptor: { code: string; language?: 'wgsl' | 'msl'; label?: string }): GPUShaderModule;
  createBindGroupLayout(descriptor: { entries: Record<string, any>[]; label?: string }): GPUBindGroupLayout;
  createPipelineLayout(descriptor: { bindGroupLayouts: GPUBindGroupLayout[]; label?: string }): GPUPipelineLayout;
  createBindGroup(descriptor: {
    layout: GPUBindGroupLayout;
    entries: { binding: number; resource: any }[];
    label?: string;
  }): GPUBindGroup;
  createRenderPipeline(descriptor: Record<string, any>): GPURenderPipeline;
  createComputePipeline(descriptor: Record<string, any>): GPUComputePipeline;
  createCommandEncoder(): GPUCommandEncoder;
  destroy(): void;
}

export declare class GPUCanvasContext {
  readonly canvasId: string;
  readonly format: string;
  configure(configuration: {
    device: GPUDevice; format?: string; usage?: number; alphaMode?: 'opaque' | 'premultiplied';
  }): void;
  getCurrentTexture(): GPUTexture;
  getSize(): { width: number; height: number };
}

export declare class GPUAdapter {
  readonly name: string;
  readonly backend: string;
  readonly limits: Record<string, number>;
  requestDevice(): Promise<GPUDevice>;
}

export declare const gpu: {
  requestAdapter(): Promise<GPUAdapter | null>;
  getPreferredCanvasFormat(): string;
  getCanvasContext(canvasId: string): GPUCanvasContext;
};

/** 화면 갱신 주기에 맞춰 콜백을 부른다. 반환값을 호출하면 멈춘다. */
export declare function startFrameLoop(
  handler: (frame: { timestamp: number; delta: number }) => void,
  options?: { fps?: number }
): () => void;

export default gpu;
