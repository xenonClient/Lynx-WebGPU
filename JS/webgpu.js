/**
 * Lynx-WebGPU — WebGPU 모양의 JS 클라이언트.
 *
 * 브라우저 WebGPU와 같은 객체 그래프를 JS 쪽에 그대로 두되, 실제 호출은 **명령으로 기록만** 하고
 * `queue.submit()` 시점에 한 번에 네이티브로 보낸다. 핸들(id)은 JS가 발급하므로 객체 생성이
 * 네이티브 왕복을 기다리지 않는다 — 프레임당 브리지 왕복이 1회로 고정된다.
 *
 * 자세한 설계는 docs/ARCHITECTURE.md §3, 지원 범위는 docs/WEBGPU-API.md 참고.
 */

/* eslint-disable no-bitwise */

// ---------------------------------------------------------------------------
// WebGPU 상수 (네이티브의 OptionSet 값과 반드시 일치해야 한다)
// ---------------------------------------------------------------------------

export const GPUBufferUsage = {
  MAP_READ: 0x0001,
  MAP_WRITE: 0x0002,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  INDEX: 0x0010,
  VERTEX: 0x0020,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
  INDIRECT: 0x0100,
  QUERY_RESOLVE: 0x0200,
};

export const GPUTextureUsage = {
  COPY_SRC: 0x01,
  COPY_DST: 0x02,
  TEXTURE_BINDING: 0x04,
  STORAGE_BINDING: 0x08,
  RENDER_ATTACHMENT: 0x10,
};

export const GPUShaderStage = { VERTEX: 0x1, FRAGMENT: 0x2, COMPUTE: 0x4 };

export const GPUColorWrite = { RED: 0x1, GREEN: 0x2, BLUE: 0x4, ALPHA: 0x8, ALL: 0xf };

export const GPUMapMode = { READ: 0x1, WRITE: 0x2 };

// ---------------------------------------------------------------------------
// 공용 타입 — 여기 정의가 `npm run types`로 webgpu.d.ts에 그대로 나간다
// ---------------------------------------------------------------------------

/**
 * 커맨드 실행 중 발생한 오류.
 * @typedef {WGPUErrorPayload} WGPUError
 */

/** @typedef {{width: number, height?: number, depthOrArrayLayers?: number}} GPUExtent3DDict */
/** @typedef {GPUExtent3DDict | number[]} GPUExtent3D */
/** @typedef {{x?: number, y?: number, z?: number}} GPUOrigin3DDict */
/** @typedef {{r: number, g: number, b: number, a: number}} GPUColorDict */
/** @typedef {GPUColorDict | number[]} GPUColor */

/** `writeBuffer`/`writeTexture`가 받는 바이트열. */
/** @typedef {ArrayBuffer | ArrayBufferView | number[]} GPUDataSource */

/**
 * 커맨드 스트림의 한 항목.
 *
 * 필드는 네이티브의 `WGPUCommandInterpreter`가 **문자열 키로** 읽는다. 이름이 어긋나도
 * 여기서는 잡히지 않으므로, op를 추가·수정할 때는 반드시 양쪽을 함께 고칠 것
 * (`.claude/skills/webgpu-command/SKILL.md`).
 * @typedef {Record<string, any>} GPUCommand
 */

// --- 디스크립터 -------------------------------------------------------------
// 파이프라인처럼 필드가 깊고 넓은 것은 Record<string, any>로 둔다. 여기서 스펙 전체를
// 다시 쓰는 것보다, 네이티브의 디스크립터 디코더가 경로까지 붙여 오류를 내는 편이 낫다.

/** @typedef {{size: number, usage: number, mappedAtCreation?: boolean, label?: string}} GPUBufferDescriptor */
/** @typedef {{size: GPUExtent3D, format: string, usage: number, dimension?: string, mipLevelCount?: number, sampleCount?: number, label?: string}} GPUTextureDescriptor */
/** @typedef {{code: string, language?: 'wgsl' | 'msl', label?: string}} GPUShaderModuleDescriptor */
/** @typedef {{entries: Record<string, any>[], label?: string}} GPUBindGroupLayoutDescriptor */
/** @typedef {{bindGroupLayouts: GPUPipelineLayoutSource[], label?: string}} GPUPipelineLayoutDescriptor */
/** @typedef {{layout: GPUBindGroupLayout, entries: {binding: number, resource: any}[], label?: string}} GPUBindGroupDescriptor */
/**
 * `toneMapping.mode: 'extended'`는 1.0을 넘는 값을 디스플레이의 여유 밝기(EDR)로 내보낸다.
 * 이때 `format`은 `'rgba16float'`여야 하고, 셰이더는 sRGB 인코딩 없이 **선형** 값을 써야 한다.
 */
/** @typedef {{device: GPUDevice, format?: string, usage?: number, alphaMode?: 'opaque' | 'premultiplied', colorSpace?: 'srgb' | 'display-p3', toneMapping?: {mode: 'standard' | 'extended'}}} GPUCanvasConfiguration */

/**
 * 한 면(앞/뒤)의 스텐실 동작. 명세 기본값은 "아무것도 하지 않음"이다 —
 * `compare: 'always'` + 세 연산 모두 `'keep'`.
 */
/** @typedef {{compare?: GPUCompareFunction, failOp?: GPUStencilOperation, depthFailOp?: GPUStencilOperation, passOp?: GPUStencilOperation}} GPUStencilFaceState */
/** @typedef {'never' | 'less' | 'equal' | 'less-equal' | 'greater' | 'not-equal' | 'greater-equal' | 'always'} GPUCompareFunction */
/** @typedef {'keep' | 'zero' | 'replace' | 'invert' | 'increment-clamp' | 'decrement-clamp' | 'increment-wrap' | 'decrement-wrap'} GPUStencilOperation */

/**
 * `createRenderPipeline`의 `depthStencil`. 마스크 기본값은 둘 다 `0xFFFFFFFF`이며,
 * 비교는 `(reference & readMask)`와 `(저장된 값 & readMask)` 사이에서 일어난다.
 */
/** @typedef {{format: string, depthWriteEnabled?: boolean, depthCompare?: GPUCompareFunction, depthBias?: number, depthBiasSlopeScale?: number, depthBiasClamp?: number, stencilFront?: GPUStencilFaceState, stencilBack?: GPUStencilFaceState, stencilReadMask?: number, stencilWriteMask?: number}} GPUDepthStencilState */

/**
 * `createRenderBundleEncoder`의 디스크립터 — 이 번들을 **실행할 패스의 모양**이다.
 * `colorFormats`의 `null`은 "그 슬롯은 비어 있다"는 뜻이다.
 */
/** @typedef {{colorFormats: (string | null)[], depthStencilFormat?: string, sampleCount?: number, label?: string}} GPURenderBundleEncoderDescriptor */

/** `createPipelineLayout`이 받는 레이아웃 — id만 있으면 된다. */
/** @typedef {{id: number}} GPUPipelineLayoutSource */

/** `GPUTexture` 생성자 내부용 — 스왑체인 텍스처는 usage/format이 없을 수 있다. */
/** @typedef {{size?: GPUExtent3D, format?: string, usage?: number, label?: string, frameScoped?: boolean}} GPUTextureInit */

// ---------------------------------------------------------------------------
// 바이너리 유틸
// ---------------------------------------------------------------------------

/**
 * TypedArray / ArrayBuffer / 숫자 배열을 커맨드에 실을 `ArrayBuffer`로 바꾼다.
 *
 * **뷰(TypedArray)를 그대로 실으면 안 된다.** Lynx의 값 변환기는 진짜 `ArrayBuffer`만
 * 알아보고(`isArrayBuffer`), 뷰는 평범한 객체로 취급해 `{"0":1,"1":2,…}` 로 만들어 버린다 —
 * 오류 없이 조용히 깨지는 종류다. 바이트열은 **반드시 여기를 거쳐** 커맨드에 실을 것.
 *
 * 백킹 버퍼 전체를 덮는 뷰는 복사 없이 그대로 넘어가고, 일부만 덮는 뷰(또는 오프셋·개수가
 * 지정된 경우)는 그 구간만 잘라 낸다 — `view.buffer`는 뷰가 아니라 버퍼 전체이기 때문이다.
 *
 * @param {GPUDataSource} source
 * @param {number} [elementOffset] 원소 단위 시작 위치 (DataView는 바이트 단위)
 * @param {number} [elementCount] 원소 개수. 생략하면 끝까지
 * @returns {ArrayBuffer}
 */
function toArrayBuffer(source, elementOffset, elementCount) {
  // 항상 **호출 시점에 복사**한다 — WebGPU 명세의 계약이다 ("the contents of data are
  // copied"). 커맨드는 submit까지 큐에 머무르므로, 참조로 실으면 호출자가 같은 배열을
  // 재사용할 때(유니폼 배열 하나로 버퍼 여러 개를 쓰는 흔한 패턴) 마지막 값이 전부를
  // 덮어쓴다. 복사 비용은 브리지 왕복에 비해 무시할 수준이다 (docs/ARCHITECTURE.md §3).
  if (source instanceof ArrayBuffer) return source.slice(0);
  if (ArrayBuffer.isView(source)) {
    // DataView에는 BYTES_PER_ELEMENT가 없다 — 그 경우 오프셋을 바이트로 해석한다.
    const elementSize = /** @type {{BYTES_PER_ELEMENT?: number}} */ (source).BYTES_PER_ELEMENT || 1;
    const start = source.byteOffset + (elementOffset || 0) * elementSize;
    const length =
      elementCount === undefined ? source.byteLength - (elementOffset || 0) * elementSize
        : elementCount * elementSize;
    const backing = /** @type {ArrayBuffer} */ (source.buffer);
    return backing.slice(start, start + length);
  }
  if (Array.isArray(source)) return /** @type {ArrayBuffer} */ (new Uint8Array(source).buffer);
  throw new TypeError('데이터는 TypedArray · ArrayBuffer · 숫자 배열이어야 한다');
}

// ---------------------------------------------------------------------------
// 네이티브 접점
// ---------------------------------------------------------------------------

function nativeModule() {
  const modules =
    typeof NativeModules !== 'undefined' ? NativeModules
      : typeof lynx !== 'undefined' ? lynx.NativeModules : undefined;
  if (!modules || !modules.WebGPU) {
    throw new Error(
      'NativeModules.WebGPU 를 찾을 수 없다 — 호스트가 LynxWebGPU.register(in:host:)를 호출했는지 확인할 것'
    );
  }
  return modules.WebGPU;
}

// ---------------------------------------------------------------------------
// 캔버스 크기 캐시
// ---------------------------------------------------------------------------

/**
 * canvasId → `{width, height}` (픽셀).
 *
 * `execute` 응답의 `canvases`가 **제출할 때마다** 갱신하므로, 프레임 안에서 크기를 읽어도
 * 동기 네이티브 왕복이 생기지 않는다. 동기 조회(`canvasInfo`)는 캐시가 비어 있을 때
 * (= `configure` 직후 첫 조회) 한 번만 일어난다.
 */
const canvasSizeCache = new Map();

// ---------------------------------------------------------------------------
// 커맨드 레코더
// ---------------------------------------------------------------------------

class Recorder {
  constructor() {
    /** @type {GPUCommand[]} */
    this.pending = [];
    this.nextId = 1;
    /** @type {((error: WGPUError, text: string) => void)[]} */
    this.errorHandlers = [];
    /**
     * `popErrorScope()`가 돌려준 Promise의 resolve 함수들 — **pop한 순서 그대로**다.
     * 네이티브가 같은 순서로 `errorScopes` 배열을 돌려주므로 인덱스로 짝을 맞춘다.
     * @type {((error: WGPUError | null) => void)[]}
     */
    this.pendingErrorScopes = [];
  }

  /** @returns {number} 새 핸들 id */
  allocate() {
    return this.nextId++;
  }

  /**
   * @param {GPUCommand} command
   * @returns {GPUCommand}
   */
  push(command) {
    this.pending.push(command);
    return command;
  }

  /**
   * 모아 둔 명령을 네이티브로 넘긴다. 실행할 것이 없으면 아무것도 하지 않는다.
   * @returns {WGPUExecuteResult}
   */
  flush() {
    if (this.pending.length === 0) {
      this.settleErrorScopes([]);
      return { ok: true, commandCount: 0 };
    }
    const commands = this.pending;
    this.pending = [];
    const result = /** @type {WGPUExecuteResult} */ (nativeModule().execute({ commands }) || {});
    this.settleErrorScopes(result.errorScopes || []);
    if (result.canvases) {
      for (const canvasId in result.canvases) {
        const info = result.canvases[canvasId];
        if (info && typeof info.width === 'number') {
          canvasSizeCache.set(canvasId, { width: info.width, height: info.height });
        }
      }
    }
    if (result.ok === false) this.report(result.errors || []);
    return result;
  }

  /**
   * 기다리고 있던 `popErrorScope()` Promise들을 푼다.
   *
   * 네이티브가 인덱스를 밀지 않는다는 계약에 기대므로(pop이 실패해도 자리를 남긴다),
   * 여기서는 순서대로 짝지어 주기만 하면 된다. 응답에 결과가 모자라면 `null`이다 —
   * Promise가 영원히 안 풀리는 것보다 낫다.
   *
   * @param {(WGPUError | null)[]} popped
   * @returns {void}
   */
  settleErrorScopes(popped) {
    if (this.pendingErrorScopes.length === 0) return;
    const waiting = this.pendingErrorScopes;
    this.pendingErrorScopes = [];
    waiting.forEach((resolve, index) => resolve(popped[index] || null));
  }

  /** @param {WGPUError[]} errors */
  report(errors) {
    for (const error of errors) {
      const text = `[WebGPU:${error.kind}] ${error.path ? error.path + ' — ' : ''}${error.message}`;
      if (this.errorHandlers.length === 0) console.error(text);
      else for (const handler of this.errorHandlers) handler(error, text);
    }
  }
}

// ---------------------------------------------------------------------------
// 리소스 객체
// ---------------------------------------------------------------------------

/**
 * GC 연동 자동 해제 — 엔진이 FinalizationRegistry를 지원할 때만.
 *
 * 핸들은 정수라 JS GC가 네이티브 객체의 수명을 모른다. 래퍼가 GC로 사라지면 destroy 명령을
 * 다음 제출에 끼워 넣어 네이티브 쪽도 따라 해제한다. 사용 명령은 래퍼가 살아 있는 동안에만
 * 기록될 수 있으므로, 뒤늦게 붙는 destroy가 앞선 사용보다 먼저 실행될 일은 없다.
 *
 * PrimJS처럼 지원이 없는 엔진에서는 조용히 꺼진다 — **명시적 destroy()가 여전히 정답**이고,
 * 이 장치는 놓친 것을 주워 담는 안전망이다 (docs/JS-AUTHORING.md §8).
 */
/** @type {FinalizationRegistry<{recorder: Recorder, id: number}> | null} */
const autoReleasePool =
  typeof FinalizationRegistry === 'function'
    ? new FinalizationRegistry((held) => {
        held.recorder.push({ op: 'destroy', id: held.id });
      })
    : null;

class GPUObjectBase {
  /**
   * @param {GPUDevice} device
   * @param {number} id JS가 발급한 핸들
   * @param {string} [label]
   * @param {boolean} [frameScoped] 프레임 끝에 네이티브가 회수하는 핸들이면 true
   */
  constructor(device, id, label, frameScoped) {
    this._device = device;
    // 생성 경로가 모두 GPUDevice를 넘기므로 실제로는 항상 Recorder다. 방어 분기는 그대로 둔다.
    this._recorder = /** @type {Recorder} */ (device ? device._recorder : null);
    this.id = id;
    this.label = label || '';
    // 프레임 스코프 핸들(스왑체인 텍스처와 그 뷰)은 네이티브가 프레임 끝에 회수한다 — 등록 제외.
    if (autoReleasePool && this._recorder && !frameScoped) {
      autoReleasePool.register(this, { recorder: this._recorder, id }, this);
    }
  }

  destroy() {
    if (!this._recorder) return;
    if (autoReleasePool) autoReleasePool.unregister(this);
    this._recorder.push({ op: 'destroy', id: this.id });
  }
}

class GPUBuffer extends GPUObjectBase {
  /**
   * @param {GPUDevice} device
   * @param {number} id
   * @param {GPUBufferDescriptor} descriptor
   */
  constructor(device, id, descriptor) {
    super(device, id, descriptor.label);
    this.size = descriptor.size;
    this.usage = descriptor.usage;
    /** @type {ArrayBuffer | null} */
    this._mapped = null;
    /** @type {ArrayBuffer | null} */
    this._mappedRange = null;
  }

  /** `mappedAtCreation: true`로 만든 버퍼의 초기 데이터 영역. */
  getMappedRange() {
    if (!this._mapped) {
      throw new Error('getMappedRange는 mappedAtCreation 또는 mapAsync 이후에만 쓸 수 있다');
    }
    return this._mapped;
  }

  /** 매핑을 풀면서 실제 생성 명령(초기 데이터 포함)을 기록한다. @returns {void} */
  unmap() {
    if (!this._mapped) return;
    this._recorder.push({
      op: 'createBuffer',
      id: this.id,
      size: this.size,
      usage: this.usage,
      label: this.label,
      data: this._mapped,   // 이미 ArrayBuffer다
    });
    this._mapped = null;
  }

  /**
   * 버퍼 내용을 읽는다. WebGPU의 `mapAsync` + `getMappedRange`를 하나로 합친 형태다.
   *
   * @param {number} [_mode] 스펙 호환용 — 이 구현은 보지 않는다
   * @param {number} [offset] 바이트 오프셋
   * @param {number} [size] 읽을 바이트 수. 생략하면 끝까지
   * @returns {Promise<ArrayBuffer>}
   */
  async mapAsync(_mode, offset, size) {
    this._recorder.flush();
    const result = await /** @type {Promise<WGPUReadBufferResult | undefined>} */ (
      new Promise((resolve) => {
        nativeModule().readBuffer({ buffer: this.id, offset: offset || 0, size }, resolve);
      })
    );
    if (!result || result.ok === false) {
      this._recorder.report((result && result.errors) || []);
      throw new Error('버퍼 읽기 실패');
    }
    // 네이티브가 `Data`로 돌려주면 Lynx가 ArrayBuffer로 바꿔 준다 — 디코딩할 것이 없다.
    const mapped = result.data;
    this._mapped = mapped;
    return mapped;
  }
}

class GPUTexture extends GPUObjectBase {
  /**
   * @param {GPUDevice} device
   * @param {number} id
   * @param {GPUTextureInit} [descriptor]
   */
  constructor(device, id, descriptor) {
    super(device, id, descriptor && descriptor.label, descriptor && descriptor.frameScoped);
    this._frameScoped = !!(descriptor && descriptor.frameScoped);
    // size는 dict({width,…})와 배열([w,h,…])을 모두 받는다 — 덕 타이핑 그대로 읽는다.
    /** @type {any} */
    const size = descriptor && descriptor.size;
    this.width = size ? size.width || size[0] : 0;
    this.height = size ? size.height || size[1] || 1 : 0;
    this.format = descriptor && descriptor.format;
  }

  /**
   * @param {Record<string, any>} [descriptor]
   * @returns {GPUTextureView}
   */
  createView(descriptor) {
    const id = this._recorder.allocate();
    this._recorder.push({ op: 'createTextureView', id, texture: this.id, ...(descriptor || {}) });
    // 스왑체인 텍스처의 뷰도 프레임 스코프다 — 네이티브가 프레임 끝에 함께 회수한다.
    return new GPUTextureView(this._device, id, descriptor && descriptor.label, this._frameScoped);
  }
}

class GPUTextureView extends GPUObjectBase {}
class GPUSampler extends GPUObjectBase {}
class GPUShaderModule extends GPUObjectBase {}
class GPUBindGroupLayout extends GPUObjectBase {}
class GPUPipelineLayout extends GPUObjectBase {}
class GPUBindGroup extends GPUObjectBase {}

class GPUPipelineBase extends GPUObjectBase {
  /**
   * `layout: 'auto'` 파이프라인이 유도한 바인드 그룹 레이아웃을 꺼낸다.
   * @param {number} index
   * @returns {GPUBindGroupLayout}
   */
  getBindGroupLayout(index) {
    const id = this._recorder.allocate();
    this._recorder.push({ op: 'getBindGroupLayout', id, pipeline: this.id, index });
    return new GPUBindGroupLayout(this._device, id);
  }
}

class GPURenderPipeline extends GPUPipelineBase {}
class GPUComputePipeline extends GPUPipelineBase {}

/** `bundleEncoder.finish()`가 돌려주는 재사용 가능한 드로우 묶음. */
class GPURenderBundle extends GPUObjectBase {}

// ---------------------------------------------------------------------------
// 커맨드 인코더
// ---------------------------------------------------------------------------

/** 인코더는 자기 명령을 따로 모았다가 `finish()` → `submit()`에서 스트림에 합쳐진다. */
class GPUCommandBuffer {
  /** @param {GPUCommand[]} commands */
  constructor(commands) {
    this.commands = commands;
  }
}

class GPUPassEncoderBase {
  /** @param {GPUCommand[]} commands */
  constructor(commands) {
    this._commands = commands;
  }

  /**
   * @param {GPURenderPipeline | GPUComputePipeline} pipeline
   * @returns {void}
   */
  setPipeline(pipeline) {
    this._commands.push({ op: 'setPipeline', pipeline: pipeline.id });
  }

  /**
   * @param {number} index
   * @param {GPUBindGroup} bindGroup
   * @param {number[]} [dynamicOffsets]
   * @returns {void}
   */
  setBindGroup(index, bindGroup, dynamicOffsets) {
    /** @type {GPUCommand} */
    const command = { op: 'setBindGroup', index, bindGroup: bindGroup.id };
    if (dynamicOffsets && dynamicOffsets.length) command.dynamicOffsets = Array.from(dynamicOffsets);
    this._commands.push(command);
  }
}

/**
 * 렌더 패스와 렌더 번들이 **함께** 쓸 수 있는 명령.
 *
 * 이 경계는 명세가 정한 것이다 — 번들에는 뷰포트·시저·블렌드 상수·스텐실 참조·중첩 번들을
 * 담을 수 없다. 그것들을 `GPURenderPassEncoder`에만 두면 번들 인코더에는 애초에 그 메서드가
 * 없으므로, 잘못 쓰는 코드가 네이티브까지 가지 않는다.
 */
class GPURenderCommandsBase extends GPUPassEncoderBase {
  /**
   * @param {number} slot
   * @param {GPUBuffer} buffer
   * @param {number} [offset]
   * @returns {void}
   */
  setVertexBuffer(slot, buffer, offset) {
    this._commands.push({ op: 'setVertexBuffer', slot, buffer: buffer.id, offset: offset || 0 });
  }

  /**
   * @param {GPUBuffer} buffer
   * @param {'uint16' | 'uint32'} format
   * @param {number} [offset]
   * @returns {void}
   */
  setIndexBuffer(buffer, format, offset) {
    this._commands.push({ op: 'setIndexBuffer', buffer: buffer.id, format, offset: offset || 0 });
  }

  /**
   * @param {number} vertexCount
   * @param {number} [instanceCount]
   * @param {number} [firstVertex]
   * @param {number} [firstInstance]
   * @returns {void}
   */
  draw(vertexCount, instanceCount, firstVertex, firstInstance) {
    this._commands.push({
      op: 'draw', vertexCount,
      instanceCount: instanceCount === undefined ? 1 : instanceCount,
      firstVertex: firstVertex || 0,
      firstInstance: firstInstance || 0,
    });
  }

  /**
   * @param {number} indexCount
   * @param {number} [instanceCount]
   * @param {number} [firstIndex]
   * @param {number} [baseVertex]
   * @param {number} [firstInstance]
   * @returns {void}
   */
  drawIndexed(indexCount, instanceCount, firstIndex, baseVertex, firstInstance) {
    this._commands.push({
      op: 'drawIndexed', indexCount,
      instanceCount: instanceCount === undefined ? 1 : instanceCount,
      firstIndex: firstIndex || 0,
      baseVertex: baseVertex || 0,
      firstInstance: firstInstance || 0,
    });
  }

  /**
   * 드로우 인자를 GPU 버퍼에서 읽어 그린다 — 컴퓨트가 만든 수만큼 그릴 때 쓴다.
   *
   * 버퍼에는 `u32` 4개가 이 순서로 들어 있어야 한다:
   * `vertexCount, instanceCount, firstVertex, firstInstance`.
   * 버퍼는 `GPUBufferUsage.INDIRECT`로 만들고, `indirectOffset`은 4의 배수여야 한다.
   *
   * @param {GPUBuffer} indirectBuffer
   * @param {number} [indirectOffset]
   * @returns {void}
   */
  drawIndirect(indirectBuffer, indirectOffset) {
    this._commands.push({
      op: 'drawIndirect', indirectBuffer: indirectBuffer.id, indirectOffset: indirectOffset || 0,
    });
  }

  /**
   * 인덱스 드로우 인자를 GPU 버퍼에서 읽어 그린다.
   *
   * 버퍼에는 `u32` 5개가 이 순서로 들어 있어야 한다:
   * `indexCount, instanceCount, firstIndex, baseVertex(부호 있는 i32), firstInstance`.
   * `firstIndex`는 인자 버퍼 안에 있으므로 `setIndexBuffer(buffer, format, offset)`의
   * 오프셋과 **더해지지 않고 따로** 적용된다.
   *
   * @param {GPUBuffer} indirectBuffer
   * @param {number} [indirectOffset]
   * @returns {void}
   */
  drawIndexedIndirect(indirectBuffer, indirectOffset) {
    this._commands.push({
      op: 'drawIndexedIndirect',
      indirectBuffer: indirectBuffer.id,
      indirectOffset: indirectOffset || 0,
    });
  }
}

/** 패스 전용 명령 — 아래 넷과 `executeBundles`는 번들에 담을 수 없다 (명세). */
class GPURenderPassEncoder extends GPURenderCommandsBase {
  /**
   * @param {number} x
   * @param {number} y
   * @param {number} width
   * @param {number} height
   * @param {number} [minDepth]
   * @param {number} [maxDepth]
   * @returns {void}
   */
  setViewport(x, y, width, height, minDepth, maxDepth) {
    this._commands.push({
      op: 'setViewport', x, y, width, height,
      minDepth: minDepth === undefined ? 0 : minDepth,
      maxDepth: maxDepth === undefined ? 1 : maxDepth,
    });
  }

  /**
   * @param {number} x
   * @param {number} y
   * @param {number} width
   * @param {number} height
   * @returns {void}
   */
  setScissorRect(x, y, width, height) {
    this._commands.push({ op: 'setScissorRect', x, y, width, height });
  }

  /**
   * @param {GPUColor} color
   * @returns {void}
   */
  setBlendConstant(color) {
    this._commands.push({ op: 'setBlendConstant', color });
  }

  /**
   * @param {number} reference
   * @returns {void}
   */
  setStencilReference(reference) {
    this._commands.push({ op: 'setStencilReference', reference });
  }

  /**
   * 미리 기록해 둔 번들들을 이 패스에 되풀이한다.
   *
   * 번들은 패스 상태를 **물려받지 않고**, 실행이 끝나면 패스의 파이프라인·바인드 그룹·
   * 정점/인덱스 버퍼 바인딩이 **무효화된다** (이전 값으로 복원되는 것이 아니다 — 명세 계약).
   * 이어서 그리려면 `setPipeline`부터 다시 해야 한다. 뷰포트·시저·블렌드 상수·스텐실 참조는
   * 그대로 남는다.
   *
   * @param {GPURenderBundle[]} bundles
   * @returns {void}
   */
  executeBundles(bundles) {
    this._commands.push({
      op: 'executeBundles',
      bundles: Array.from(bundles || []).map((bundle) => bundle.id),
    });
  }

  /** @returns {void} */
  end() {
    this._commands.push({ op: 'endPass' });
  }
}

/**
 * 여러 프레임에 걸쳐 다시 쓸 드로우 묶음을 기록한다 (`device.createRenderBundleEncoder`).
 *
 * 이 구현에서 번들의 이득은 브라우저와 다르다 — 브라우저는 드라이버 명령을 미리 만들어 두지만,
 * 여기서는 **JS가 매 프레임 같은 명령 배열을 다시 만들지 않아도 되는 것**이 이득이다.
 * 번들을 실행하는 명령은 핸들 하나뿐이고, 되풀이는 네이티브가 한다.
 */
class GPURenderBundleEncoder extends GPURenderCommandsBase {
  /**
   * @param {GPUDevice} device
   * @param {GPURenderBundleEncoderDescriptor} descriptor
   */
  constructor(device, descriptor) {
    super([]);
    /** @type {GPUCommand[]} 번들 인코더는 자기 배열에만 모은다 (패스 스트림과 섞이지 않는다). */
    this._commands = [];
    this._device = device;
    this._descriptor = descriptor || { colorFormats: [] };
  }

  /**
   * 기록을 끝내고 재사용 가능한 번들을 만든다.
   * @param {{label?: string}} [descriptor]
   * @returns {GPURenderBundle}
   */
  finish(descriptor) {
    const recorder = this._device._recorder;
    const id = recorder.allocate();
    const label = (descriptor && descriptor.label) || this._descriptor.label;
    recorder.push({
      op: 'createRenderBundle', id,
      commands: this._commands,
      colorFormats: this._descriptor.colorFormats || [],
      depthStencilFormat: this._descriptor.depthStencilFormat,
      sampleCount: this._descriptor.sampleCount,
      label,
    });
    // 인코더는 한 번만 finish할 수 있다 — 남겨 두면 다음 finish에 같은 명령이 또 실린다.
    this._commands = [];
    return new GPURenderBundle(this._device, id, label);
  }
}

class GPUComputePassEncoder extends GPUPassEncoderBase {
  /**
   * @param {number} x
   * @param {number} [y]
   * @param {number} [z]
   * @returns {void}
   */
  dispatchWorkgroups(x, y, z) {
    this._commands.push({ op: 'dispatchWorkgroups', x: x || 1, y: y || 1, z: z || 1 });
  }

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
  dispatchWorkgroupsIndirect(indirectBuffer, indirectOffset) {
    this._commands.push({
      op: 'dispatchWorkgroupsIndirect',
      indirectBuffer: indirectBuffer.id,
      indirectOffset: indirectOffset || 0,
    });
  }

  /** @returns {void} */
  end() {
    this._commands.push({ op: 'endPass' });
  }
}

class GPUCommandEncoder {
  /** @param {GPUDevice} device */
  constructor(device) {
    this._device = device;
    /** @type {GPUCommand[]} */
    this._commands = [];
  }

  /**
   * @param {Record<string, any>} descriptor
   * @returns {GPURenderPassEncoder}
   */
  beginRenderPass(descriptor) {
    const colorAttachments = (descriptor.colorAttachments || []).map(
      /** @param {Record<string, any>} attachment */
      (attachment) => ({
        view: attachment.view.id,
        resolveTarget: attachment.resolveTarget ? attachment.resolveTarget.id : undefined,
        loadOp: attachment.loadOp || 'clear',
        storeOp: attachment.storeOp || 'store',
        clearValue: attachment.clearValue,
      })
    );
    /** @type {GPUCommand} */
    const command = { op: 'beginRenderPass', colorAttachments, label: descriptor.label };
    if (descriptor.depthStencilAttachment) {
      const depth = descriptor.depthStencilAttachment;
      command.depthStencilAttachment = { ...depth, view: depth.view.id };
    }
    this._commands.push(command);
    return new GPURenderPassEncoder(this._commands);
  }

  /** @returns {GPUComputePassEncoder} */
  beginComputePass() {
    this._commands.push({ op: 'beginComputePass' });
    return new GPUComputePassEncoder(this._commands);
  }

  /**
   * @param {GPUBuffer} source
   * @param {number} sourceOffset
   * @param {GPUBuffer} destination
   * @param {number} destinationOffset
   * @param {number} size
   * @returns {void}
   */
  copyBufferToBuffer(source, sourceOffset, destination, destinationOffset, size) {
    this._commands.push({
      op: 'copyBufferToBuffer',
      source: source.id, sourceOffset, destination: destination.id, destinationOffset, size,
    });
  }

  /**
   * @param {{texture: GPUTexture} & Record<string, any>} source
   * @param {{buffer: GPUBuffer} & Record<string, any>} destination
   * @param {GPUExtent3D} copySize
   * @returns {void}
   */
  copyTextureToBuffer(source, destination, copySize) {
    this._commands.push({
      op: 'copyTextureToBuffer',
      source: { ...source, texture: source.texture.id },
      destination: { ...destination, buffer: destination.buffer.id },
      copySize,
    });
  }

  /**
   * @param {{buffer: GPUBuffer} & Record<string, any>} source
   * @param {{texture: GPUTexture} & Record<string, any>} destination
   * @param {GPUExtent3D} copySize
   * @returns {void}
   */
  copyBufferToTexture(source, destination, copySize) {
    this._commands.push({
      op: 'copyBufferToTexture',
      source: { ...source, buffer: source.buffer.id },
      destination: { ...destination, texture: destination.texture.id },
      copySize,
    });
  }

  /**
   * @param {{texture: GPUTexture} & Record<string, any>} source
   * @param {{texture: GPUTexture} & Record<string, any>} destination
   * @param {GPUExtent3D} copySize
   * @returns {void}
   */
  copyTextureToTexture(source, destination, copySize) {
    this._commands.push({
      op: 'copyTextureToTexture',
      source: { ...source, texture: source.texture.id },
      destination: { ...destination, texture: destination.texture.id },
      copySize,
    });
  }

  /** @returns {GPUCommandBuffer} */
  finish() {
    const commands = this._commands;
    this._commands = [];
    return new GPUCommandBuffer(commands);
  }
}

// ---------------------------------------------------------------------------
// 큐 / 디바이스
// ---------------------------------------------------------------------------

class GPUQueue {
  /** @param {GPUDevice} device */
  constructor(device) {
    this._device = device;
    this._recorder = device._recorder;
  }

  /**
   * @param {GPUBuffer} buffer
   * @param {number} bufferOffset
   * @param {GPUDataSource} data
   * @param {number} [dataOffset] 원소 단위 시작 위치
   * @param {number} [size] 원소 개수
   * @returns {void}
   */
  writeBuffer(buffer, bufferOffset, data, dataOffset, size) {
    this._recorder.push({
      op: 'writeBuffer',
      buffer: buffer.id,
      bufferOffset: bufferOffset || 0,
      data: toArrayBuffer(data, dataOffset, size),
    });
  }

  /**
   * @param {{texture: GPUTexture, mipLevel?: number, origin?: GPUOrigin3DDict}} destination
   * @param {GPUDataSource} data
   * @param {{bytesPerRow: number, rowsPerImage?: number}} dataLayout
   * @param {GPUExtent3D} size
   * @returns {void}
   */
  writeTexture(destination, data, dataLayout, size) {
    this._recorder.push({
      op: 'writeTexture',
      texture: destination.texture.id,
      mipLevel: destination.mipLevel || 0,
      origin: destination.origin,
      data: toArrayBuffer(data),
      bytesPerRow: dataLayout.bytesPerRow,
      rowsPerImage: dataLayout.rowsPerImage,
      size,
    });
  }

  /**
   * 인코더가 모은 명령을 스트림에 합쳐 **한 번에** 네이티브로 보낸다.
   * @param {GPUCommandBuffer[]} commandBuffers
   * @returns {WGPUExecuteResult}
   */
  submit(commandBuffers) {
    for (const commandBuffer of commandBuffers || []) {
      for (const command of commandBuffer.commands) this._recorder.push(command);
    }
    return this._recorder.flush();
  }

  /** @returns {Promise<void>} */
  onSubmittedWorkDone() {
    this._recorder.flush();
    return Promise.resolve();
  }
}

class GPUDevice {
  /** @param {GPUAdapter} adapter */
  constructor(adapter) {
    this.adapter = adapter;
    this.limits = adapter.limits;
    this._recorder = new Recorder();
    this.queue = new GPUQueue(this);
  }

  /**
   * 커맨드 실행 오류 핸들러 (`{kind, message, path}`). 등록하지 않으면 console.error로 나간다.
   * @param {(error: WGPUError, text: string) => void} handler
   * @returns {void}
   */
  onError(handler) {
    this._recorder.errorHandlers.push(handler);
  }

  /**
   * 여기서부터 `popErrorScope()`까지 사이에 난 오류 중 **필터에 맞는 것**을 가로챈다.
   *
   * 가로챈 오류는 전역 핸들러(`onError`)로 가지 않는다 — 이미 처리하기로 한 것이기 때문이다.
   * 스코프는 중첩할 수 있고, 오류는 **가장 안쪽의 맞는 스코프**가 가져간다.
   *
   * 기록만 하므로 왕복이 늘지 않는다. 프레임 안에서 마음껏 써도 된다.
   *
   * @param {'validation' | 'out-of-memory' | 'internal'} filter
   * @returns {void}
   */
  pushErrorScope(filter) {
    this._recorder.push({ op: 'pushErrorScope', filter });
  }

  /**
   * 가장 안쪽 스코프를 닫고 **거기서 처음 잡힌 오류**를 돌려준다 (없으면 `null`).
   *
   * `mapAsync`처럼 **즉시 제출한다** — 그러지 않으면 다음 `submit()`이 올 때까지 Promise가
   * 풀리지 않기 때문이다. 그래서 프레임 루프 안에서 부르면 왕복이 하나 늘어난다.
   * 초기화나 진단 경로에서 쓰는 것을 전제로 한 API다 (`docs/JS-AUTHORING.md` §5).
   *
   * @returns {Promise<WGPUError | null>}
   */
  popErrorScope() {
    this._recorder.push({ op: 'popErrorScope' });
    const promise = /** @type {Promise<WGPUError | null>} */ (
      new Promise((resolve) => {
        this._recorder.pendingErrorScopes.push(resolve);
      })
    );
    this._recorder.flush();
    return promise;
  }

  /**
   * @param {GPUBufferDescriptor} descriptor
   * @returns {GPUBuffer}
   */
  createBuffer(descriptor) {
    const id = this._recorder.allocate();
    const buffer = new GPUBuffer(this, id, descriptor);
    if (descriptor.mappedAtCreation) {
      // 생성 명령은 unmap()에서 초기 데이터와 함께 기록된다.
      buffer._mapped = new ArrayBuffer(descriptor.size);
    } else {
      this._recorder.push({
        op: 'createBuffer', id,
        size: descriptor.size, usage: descriptor.usage, label: descriptor.label,
      });
    }
    return buffer;
  }

  /**
   * @param {GPUTextureDescriptor} descriptor
   * @returns {GPUTexture}
   */
  createTexture(descriptor) {
    const id = this._recorder.allocate();
    this._recorder.push({ op: 'createTexture', id, ...descriptor });
    return new GPUTexture(this, id, descriptor);
  }

  /**
   * @param {Record<string, any>} [descriptor]
   * @returns {GPUSampler}
   */
  createSampler(descriptor) {
    const id = this._recorder.allocate();
    this._recorder.push({ op: 'createSampler', id, ...(descriptor || {}) });
    return new GPUSampler(this, id, descriptor && descriptor.label);
  }

  /**
   * @param {GPUShaderModuleDescriptor} descriptor
   * @returns {GPUShaderModule}
   */
  createShaderModule(descriptor) {
    const id = this._recorder.allocate();
    this._recorder.push({
      op: 'createShaderModule', id,
      code: descriptor.code, language: descriptor.language || 'wgsl', label: descriptor.label,
    });
    return new GPUShaderModule(this, id, descriptor.label);
  }

  /**
   * @param {GPUBindGroupLayoutDescriptor} descriptor
   * @returns {GPUBindGroupLayout}
   */
  createBindGroupLayout(descriptor) {
    const id = this._recorder.allocate();
    this._recorder.push({ op: 'createBindGroupLayout', id, ...descriptor });
    return new GPUBindGroupLayout(this, id, descriptor.label);
  }

  /**
   * @param {GPUPipelineLayoutDescriptor} descriptor
   * @returns {GPUPipelineLayout}
   */
  createPipelineLayout(descriptor) {
    const id = this._recorder.allocate();
    this._recorder.push({
      op: 'createPipelineLayout', id,
      bindGroupLayouts: descriptor.bindGroupLayouts.map((layout) => layout.id),
      label: descriptor.label,
    });
    return new GPUPipelineLayout(this, id, descriptor.label);
  }

  /**
   * @param {GPUBindGroupDescriptor} descriptor
   * @returns {GPUBindGroup}
   */
  createBindGroup(descriptor) {
    const id = this._recorder.allocate();
    this._recorder.push({
      op: 'createBindGroup', id,
      layout: descriptor.layout.id,
      label: descriptor.label,
      entries: descriptor.entries.map((entry) => ({
        binding: entry.binding,
        resource: serializeBindingResource(entry.resource),
      })),
    });
    return new GPUBindGroup(this, id, descriptor.label);
  }

  /**
   * @param {Record<string, any>} descriptor
   * @returns {GPURenderPipeline}
   */
  createRenderPipeline(descriptor) {
    const id = this._recorder.allocate();
    /** @type {GPUCommand} */
    const command = {
      op: 'createRenderPipeline', id,
      layout: descriptor.layout && descriptor.layout.id !== undefined ? descriptor.layout.id : 'auto',
      label: descriptor.label,
      vertex: { ...descriptor.vertex, module: descriptor.vertex.module.id },
      primitive: descriptor.primitive,
      depthStencil: descriptor.depthStencil,
      multisample: descriptor.multisample,
    };
    if (descriptor.fragment) {
      command.fragment = { ...descriptor.fragment, module: descriptor.fragment.module.id };
    }
    this._recorder.push(command);
    return new GPURenderPipeline(this, id, descriptor.label);
  }

  /**
   * @param {Record<string, any>} descriptor
   * @returns {GPUComputePipeline}
   */
  createComputePipeline(descriptor) {
    const id = this._recorder.allocate();
    this._recorder.push({
      op: 'createComputePipeline', id,
      layout: descriptor.layout && descriptor.layout.id !== undefined ? descriptor.layout.id : 'auto',
      label: descriptor.label,
      compute: { ...descriptor.compute, module: descriptor.compute.module.id },
    });
    return new GPUComputePipeline(this, id, descriptor.label);
  }

  /** @returns {GPUCommandEncoder} */
  createCommandEncoder() {
    return new GPUCommandEncoder(this);
  }

  /**
   * 여러 프레임에 걸쳐 다시 쓸 드로우 묶음을 기록하기 시작한다.
   *
   * `colorFormats`(와 있다면 `depthStencilFormat`·`sampleCount`)는 이 번들을 **실행할 패스의
   * 모양**이다. 실제 패스와 어긋나면 `executeBundles`에서 오류가 난다.
   *
   * @param {GPURenderBundleEncoderDescriptor} descriptor
   * @returns {GPURenderBundleEncoder}
   */
  createRenderBundleEncoder(descriptor) {
    return new GPURenderBundleEncoder(this, descriptor);
  }

  /** 모든 GPU 객체를 버린다 (페이지 이탈 시 호출). @returns {void} */
  destroy() {
    this._recorder.pending = [];
    // 기다리던 popErrorScope Promise를 그냥 두면 영원히 안 풀린다 — null로 닫는다.
    this._recorder.settleErrorScopes([]);
    canvasSizeCache.clear();
    nativeModule().reset();
  }
}

/**
 * @param {any} resource GPUBuffer · GPUSampler · GPUTextureView 또는 `{buffer, offset?, size?}`
 * @returns {Record<string, any>}
 */
function serializeBindingResource(resource) {
  if (resource instanceof GPUSampler) return { sampler: resource.id };
  if (resource instanceof GPUTextureView) return { textureView: resource.id };
  if (resource && resource.buffer) {
    return { buffer: resource.buffer.id, offset: resource.offset || 0, size: resource.size };
  }
  if (resource instanceof GPUBuffer) return { buffer: resource.id };
  throw new TypeError('바인딩 리소스는 GPUBuffer · GPUSampler · GPUTextureView여야 한다');
}

// ---------------------------------------------------------------------------
// 캔버스 컨텍스트
// ---------------------------------------------------------------------------

class GPUCanvasContext {
  /** @param {string} canvasId `<webgpu-canvas canvas-id="…">` 의 값 */
  constructor(canvasId) {
    this.canvasId = canvasId;
    /** @type {GPUDevice | null} */
    this._device = null;
    this.format = 'bgra8unorm';
  }

  /**
   * @param {GPUCanvasConfiguration} configuration
   * @returns {void}
   */
  configure(configuration) {
    this._device = configuration.device;
    this.format = configuration.format || 'bgra8unorm';
    this._device._recorder.push({
      op: 'configureCanvas',
      canvas: this.canvasId,
      format: this.format,
      usage: configuration.usage,
      alphaMode: configuration.alphaMode,
      colorSpace: configuration.colorSpace,
      toneMapping: configuration.toneMapping,
    });
    // 크기를 미리 캐시한다 — 이후에는 제출 응답이 갱신하므로 동기 조회는 사실상 이 1회뿐이다.
    this._fetchSize();
  }

  /**
   * 이번 프레임의 스왑체인 텍스처. 프레임이 끝나면 무효해진다 (브라우저와 같은 규칙).
   * @returns {GPUTexture}
   */
  getCurrentTexture() {
    if (!this._device) throw new Error('configure()를 먼저 호출해야 한다');
    const id = this._device._recorder.allocate();
    this._device._recorder.push({ op: 'getCurrentTexture', id, canvas: this.canvasId });
    const info = canvasSizeCache.get(this.canvasId) || this._fetchSize();
    return new GPUTexture(this._device, id, {
      size: { width: info.width, height: info.height },
      format: this.format,
      frameScoped: true,   // 네이티브가 프레임 끝에 회수 — GC 자동 해제 대상이 아니다
    });
  }

  /**
   * 캔버스의 현재 픽셀 크기.
   *
   * 제출(`submit`) 응답으로 갱신되는 캐시를 읽으므로 프레임 안에서 불러도 왕복이 없다.
   * 리사이즈 직후 아직 제출이 없었다면 한 프레임 이전 값일 수 있다 — 즉시성이 필요하면
   * `<webgpu-canvas>`의 `bindcanvasresize` 이벤트를 쓸 것.
   *
   * @returns {{width: number, height: number}}
   */
  getSize() {
    const cached = canvasSizeCache.get(this.canvasId);
    if (cached) return { width: cached.width, height: cached.height };
    return this._fetchSize();
  }

  /**
   * 동기 네이티브 조회 — 캐시가 비어 있을 때만 쓴다.
   * @returns {{width: number, height: number}}
   */
  _fetchSize() {
    const info = nativeModule().canvasInfo({ canvas: this.canvasId }) || {};
    const size = { width: info.width || 0, height: info.height || 0 };
    // 표면이 아직 등록 전이면(크기 0) 캐시하지 않는다 — 다음 조회가 다시 시도한다.
    if (info.ok !== false && size.width > 0 && size.height > 0) {
      canvasSizeCache.set(this.canvasId, { width: size.width, height: size.height });
    }
    return size;
  }
}

// ---------------------------------------------------------------------------
// 진입점 (navigator.gpu 대응)
// ---------------------------------------------------------------------------

class GPUAdapter {
  /** @param {WGPUAdapterInfo} info */
  constructor(info) {
    this.name = info.name;
    this.backend = info.backend;
    this.limits = info.limits || {};
    this.hasUnifiedMemory = info.hasUnifiedMemory;
  }

  /** @returns {Promise<GPUDevice>} */
  async requestDevice() {
    return new GPUDevice(this);
  }
}

export const gpu = {
  /**
   * `navigator.gpu.requestAdapter()`.
   * @returns {Promise<GPUAdapter | null>}
   */
  async requestAdapter() {
    const info = nativeModule().adapterInfo();
    if (!info || info.ok === false) return null;
    return new GPUAdapter(info);
  },

  /**
   * 캔버스 표면에 가장 잘 맞는 포맷. `<webgpu-canvas>`의 기본값과 같다.
   * @returns {string}
   */
  getPreferredCanvasFormat() {
    return 'bgra8unorm';
  },

  /**
   * `<webgpu-canvas canvas-id="…">` 의 표면에 붙는다 (`canvas.getContext('webgpu')` 대응).
   * @param {string} canvasId
   * @returns {GPUCanvasContext}
   */
  getCanvasContext(canvasId) {
    return new GPUCanvasContext(canvasId);
  },
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
export function startFrameLoop(handler, options) {
  const fps = (options && options.fps) || 60;
  const module = nativeModule();

  /** @type {LynxGlobalEventEmitter | null} */
  let emitter = null;
  try {
    // 전역 자체가 없을 수 있다 — `nativeModule()`과 같은 방식으로 확인한다.
    // (getJSModule 자체가 던지는 경우도 있어 try는 그대로 둔다.)
    emitter = typeof lynx !== 'undefined' && lynx ? lynx.getJSModule('GlobalEventEmitter') : null;
  } catch (error) {
    emitter = null;
  }

  if (emitter) {
    /** @param {{timestamp: number, delta: number} | undefined} frame */
    const listener = (frame) => handler(frame || { timestamp: 0, delta: 16 });
    emitter.addListener('webgpu:frame', listener);
    module.startFrameLoop({ fps });
    return () => {
      module.stopFrameLoop();
      emitter.removeListener('webgpu:frame', listener);
    };
  }

  // GlobalEventEmitter가 없는 환경(테스트 러너 등)용 폴백.
  let previous = Date.now();
  const timer = setInterval(() => {
    const now = Date.now();
    handler({ timestamp: now, delta: now - previous });
    previous = now;
  }, Math.max(1, Math.round(1000 / fps)));
  return () => clearInterval(timer);
}

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
export async function loadAsset(name) {
  const result = await /** @type {Promise<WGPULoadAssetResult | undefined>} */ (
    new Promise((resolve) => {
      nativeModule().loadAsset({ name }, resolve);
    })
  );
  if (!result || result.ok === false) {
    const errors = (result && result.errors) || [];
    throw new Error(
      errors.length ? errors[0].message : `애셋 '${name}'을(를) 읽지 못했다`
    );
  }
  return result.data;
}

export {
  GPUBuffer, GPUTexture, GPUTextureView, GPUSampler, GPUDevice, GPUCanvasContext,
  GPURenderBundle, GPURenderBundleEncoder,
};
export default gpu;
