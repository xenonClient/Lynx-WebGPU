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
// 바이너리 유틸 — PrimJS에는 btoa/atob가 없다
// ---------------------------------------------------------------------------

const BASE64_ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
const BASE64_PAD = 61; // '='

/** 6비트 값 → 문자 코드. */
const BASE64_CODES = (() => {
  const codes = new Uint8Array(64);
  for (let index = 0; index < 64; index += 1) codes[index] = BASE64_ALPHABET.charCodeAt(index);
  return codes;
})();

/** 문자 코드 → 6비트 값 (유효하지 않으면 0xff). */
const BASE64_VALUES = (() => {
  const values = new Uint8Array(128).fill(0xff);
  for (let index = 0; index < 64; index += 1) values[BASE64_ALPHABET.charCodeAt(index)] = index;
  return values;
})();

// 텍스처 업로드처럼 수십 KB~수 MB가 지나가는 경로다. 문자열 누적(`+=`)과 문자당
// `indexOf` 스캔은 여기서 바로 병목이 되므로, 룩업 테이블로 문자 코드를 만든 뒤
// `String.fromCharCode`를 청크 단위로 호출해 O(n)으로 처리한다.
function encodeBase64(bytes) {
  const length = bytes.length;
  const parts = [];
  const codes = [];
  let index = 0;
  const tripleEnd = length - (length % 3);
  while (index < tripleEnd) {
    const a = bytes[index];
    const b = bytes[index + 1];
    const c = bytes[index + 2];
    index += 3;
    codes.push(
      BASE64_CODES[a >> 2],
      BASE64_CODES[((a & 0x03) << 4) | (b >> 4)],
      BASE64_CODES[((b & 0x0f) << 2) | (c >> 6)],
      BASE64_CODES[c & 0x3f]
    );
    // apply의 인자 개수 제한을 넘지 않도록 잘라서 문자열로 바꾼다.
    if (codes.length >= 4096) {
      parts.push(String.fromCharCode.apply(null, codes));
      codes.length = 0;
    }
  }
  const remainder = length - index;
  if (remainder === 1) {
    const a = bytes[index];
    codes.push(BASE64_CODES[a >> 2], BASE64_CODES[(a & 0x03) << 4], BASE64_PAD, BASE64_PAD);
  } else if (remainder === 2) {
    const a = bytes[index];
    const b = bytes[index + 1];
    codes.push(
      BASE64_CODES[a >> 2],
      BASE64_CODES[((a & 0x03) << 4) | (b >> 4)],
      BASE64_CODES[(b & 0x0f) << 2],
      BASE64_PAD
    );
  }
  if (codes.length > 0) parts.push(String.fromCharCode.apply(null, codes));
  return parts.join('');
}

function decodeBase64(text) {
  const length = text.length;
  // 유효 문자 수를 먼저 세어 정확한 크기로 할당한다 (패딩·공백·개행은 건너뛴다).
  let effective = 0;
  for (let index = 0; index < length; index += 1) {
    const code = text.charCodeAt(index);
    if (code < 128 && BASE64_VALUES[code] !== 0xff) effective += 1;
  }
  const bytes = new Uint8Array(Math.floor((effective * 3) / 4));
  let accumulator = 0;
  let bits = 0;
  let byteIndex = 0;
  for (let index = 0; index < length; index += 1) {
    const code = text.charCodeAt(index);
    if (code >= 128) continue;
    const value = BASE64_VALUES[code];
    if (value === 0xff) continue;
    accumulator = (accumulator << 6) | value;
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      if (byteIndex < bytes.length) bytes[byteIndex++] = (accumulator >> bits) & 0xff;
    }
  }
  return bytes;
}

/** TypedArray / ArrayBuffer / 숫자 배열을 커맨드에 실을 base64로 바꾼다. */
function toBase64(source, elementOffset, elementCount) {
  let bytes;
  if (source instanceof ArrayBuffer) {
    bytes = new Uint8Array(source);
  } else if (ArrayBuffer.isView(source)) {
    const elementSize = source.BYTES_PER_ELEMENT || 1;
    const start = source.byteOffset + (elementOffset || 0) * elementSize;
    const length =
      elementCount === undefined ? source.byteLength - (elementOffset || 0) * elementSize
        : elementCount * elementSize;
    bytes = new Uint8Array(source.buffer, start, length);
  } else if (Array.isArray(source)) {
    bytes = new Uint8Array(source);
  } else {
    throw new TypeError('데이터는 TypedArray · ArrayBuffer · 숫자 배열이어야 한다');
  }
  return encodeBase64(bytes);
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
    this.pending = [];
    this.nextId = 1;
    this.errorHandlers = [];
  }

  allocate() {
    return this.nextId++;
  }

  push(command) {
    this.pending.push(command);
    return command;
  }

  /** 모아 둔 명령을 네이티브로 넘긴다. 실행할 것이 없으면 아무것도 하지 않는다. */
  flush() {
    if (this.pending.length === 0) return { ok: true, commandCount: 0 };
    const commands = this.pending;
    this.pending = [];
    const result = nativeModule().execute({ commands }) || {};
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
const autoReleasePool =
  typeof FinalizationRegistry === 'function'
    ? new FinalizationRegistry((held) => {
        held.recorder.push({ op: 'destroy', id: held.id });
      })
    : null;

class GPUObjectBase {
  constructor(device, id, label, frameScoped) {
    this._device = device;
    this._recorder = device ? device._recorder : null;
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
  constructor(device, id, descriptor) {
    super(device, id, descriptor.label);
    this.size = descriptor.size;
    this.usage = descriptor.usage;
    this._mapped = null;
    this._mappedRange = null;
  }

  /** `mappedAtCreation: true`로 만든 버퍼의 초기 데이터 영역. */
  getMappedRange() {
    if (!this._mapped) {
      throw new Error('getMappedRange는 mappedAtCreation 또는 mapAsync 이후에만 쓸 수 있다');
    }
    return this._mapped;
  }

  /** 매핑을 풀면서 실제 생성 명령(초기 데이터 포함)을 기록한다. */
  unmap() {
    if (!this._mapped) return;
    this._recorder.push({
      op: 'createBuffer',
      id: this.id,
      size: this.size,
      usage: this.usage,
      label: this.label,
      data: toBase64(this._mapped),
    });
    this._mapped = null;
  }

  /**
   * 버퍼 내용을 읽는다. WebGPU의 `mapAsync` + `getMappedRange`를 하나로 합친 형태다.
   * @returns {Promise<ArrayBuffer>}
   */
  async mapAsync(_mode, offset, size) {
    this._recorder.flush();
    const result = await new Promise((resolve) => {
      nativeModule().readBuffer({ buffer: this.id, offset: offset || 0, size }, resolve);
    });
    if (!result || result.ok === false) {
      this._recorder.report((result && result.errors) || []);
      throw new Error('버퍼 읽기 실패');
    }
    this._mapped = decodeBase64(result.data).buffer;
    return this._mapped;
  }
}

class GPUTexture extends GPUObjectBase {
  constructor(device, id, descriptor) {
    super(device, id, descriptor && descriptor.label, descriptor && descriptor.frameScoped);
    this._frameScoped = !!(descriptor && descriptor.frameScoped);
    this.width = descriptor && descriptor.size ? descriptor.size.width || descriptor.size[0] : 0;
    this.height = descriptor && descriptor.size ? descriptor.size.height || descriptor.size[1] || 1 : 0;
    this.format = descriptor && descriptor.format;
  }

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
  /** `layout: 'auto'` 파이프라인이 유도한 바인드 그룹 레이아웃을 꺼낸다. */
  getBindGroupLayout(index) {
    const id = this._recorder.allocate();
    this._recorder.push({ op: 'getBindGroupLayout', id, pipeline: this.id, index });
    return new GPUBindGroupLayout(this._device, id);
  }
}

class GPURenderPipeline extends GPUPipelineBase {}
class GPUComputePipeline extends GPUPipelineBase {}

// ---------------------------------------------------------------------------
// 커맨드 인코더
// ---------------------------------------------------------------------------

/** 인코더는 자기 명령을 따로 모았다가 `finish()` → `submit()`에서 스트림에 합쳐진다. */
class GPUCommandBuffer {
  constructor(commands) {
    this.commands = commands;
  }
}

class GPUPassEncoderBase {
  constructor(commands) {
    this._commands = commands;
  }

  setPipeline(pipeline) {
    this._commands.push({ op: 'setPipeline', pipeline: pipeline.id });
  }

  setBindGroup(index, bindGroup, dynamicOffsets) {
    const command = { op: 'setBindGroup', index, bindGroup: bindGroup.id };
    if (dynamicOffsets && dynamicOffsets.length) command.dynamicOffsets = Array.from(dynamicOffsets);
    this._commands.push(command);
  }

  end() {
    this._commands.push({ op: 'endPass' });
  }
}

class GPURenderPassEncoder extends GPUPassEncoderBase {
  setVertexBuffer(slot, buffer, offset) {
    this._commands.push({ op: 'setVertexBuffer', slot, buffer: buffer.id, offset: offset || 0 });
  }

  setIndexBuffer(buffer, format, offset) {
    this._commands.push({ op: 'setIndexBuffer', buffer: buffer.id, format, offset: offset || 0 });
  }

  setViewport(x, y, width, height, minDepth, maxDepth) {
    this._commands.push({
      op: 'setViewport', x, y, width, height,
      minDepth: minDepth === undefined ? 0 : minDepth,
      maxDepth: maxDepth === undefined ? 1 : maxDepth,
    });
  }

  setScissorRect(x, y, width, height) {
    this._commands.push({ op: 'setScissorRect', x, y, width, height });
  }

  setBlendConstant(color) {
    this._commands.push({ op: 'setBlendConstant', color });
  }

  setStencilReference(reference) {
    this._commands.push({ op: 'setStencilReference', reference });
  }

  draw(vertexCount, instanceCount, firstVertex, firstInstance) {
    this._commands.push({
      op: 'draw', vertexCount,
      instanceCount: instanceCount === undefined ? 1 : instanceCount,
      firstVertex: firstVertex || 0,
      firstInstance: firstInstance || 0,
    });
  }

  drawIndexed(indexCount, instanceCount, firstIndex, baseVertex, firstInstance) {
    this._commands.push({
      op: 'drawIndexed', indexCount,
      instanceCount: instanceCount === undefined ? 1 : instanceCount,
      firstIndex: firstIndex || 0,
      baseVertex: baseVertex || 0,
      firstInstance: firstInstance || 0,
    });
  }
}

class GPUComputePassEncoder extends GPUPassEncoderBase {
  dispatchWorkgroups(x, y, z) {
    this._commands.push({ op: 'dispatchWorkgroups', x: x || 1, y: y || 1, z: z || 1 });
  }
}

class GPUCommandEncoder {
  constructor(device) {
    this._device = device;
    this._commands = [];
  }

  beginRenderPass(descriptor) {
    const colorAttachments = (descriptor.colorAttachments || []).map((attachment) => ({
      view: attachment.view.id,
      resolveTarget: attachment.resolveTarget ? attachment.resolveTarget.id : undefined,
      loadOp: attachment.loadOp || 'clear',
      storeOp: attachment.storeOp || 'store',
      clearValue: attachment.clearValue,
    }));
    const command = { op: 'beginRenderPass', colorAttachments, label: descriptor.label };
    if (descriptor.depthStencilAttachment) {
      const depth = descriptor.depthStencilAttachment;
      command.depthStencilAttachment = { ...depth, view: depth.view.id };
    }
    this._commands.push(command);
    return new GPURenderPassEncoder(this._commands);
  }

  beginComputePass() {
    this._commands.push({ op: 'beginComputePass' });
    return new GPUComputePassEncoder(this._commands);
  }

  copyBufferToBuffer(source, sourceOffset, destination, destinationOffset, size) {
    this._commands.push({
      op: 'copyBufferToBuffer',
      source: source.id, sourceOffset, destination: destination.id, destinationOffset, size,
    });
  }

  copyTextureToBuffer(source, destination, copySize) {
    this._commands.push({
      op: 'copyTextureToBuffer',
      source: { ...source, texture: source.texture.id },
      destination: { ...destination, buffer: destination.buffer.id },
      copySize,
    });
  }

  copyBufferToTexture(source, destination, copySize) {
    this._commands.push({
      op: 'copyBufferToTexture',
      source: { ...source, buffer: source.buffer.id },
      destination: { ...destination, texture: destination.texture.id },
      copySize,
    });
  }

  copyTextureToTexture(source, destination, copySize) {
    this._commands.push({
      op: 'copyTextureToTexture',
      source: { ...source, texture: source.texture.id },
      destination: { ...destination, texture: destination.texture.id },
      copySize,
    });
  }

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
  constructor(device) {
    this._device = device;
    this._recorder = device._recorder;
  }

  writeBuffer(buffer, bufferOffset, data, dataOffset, size) {
    this._recorder.push({
      op: 'writeBuffer',
      buffer: buffer.id,
      bufferOffset: bufferOffset || 0,
      data: toBase64(data, dataOffset, size),
    });
  }

  writeTexture(destination, data, dataLayout, size) {
    this._recorder.push({
      op: 'writeTexture',
      texture: destination.texture.id,
      mipLevel: destination.mipLevel || 0,
      origin: destination.origin,
      data: toBase64(data),
      bytesPerRow: dataLayout.bytesPerRow,
      rowsPerImage: dataLayout.rowsPerImage,
      size,
    });
  }

  /** 인코더가 모은 명령을 스트림에 합쳐 **한 번에** 네이티브로 보낸다. */
  submit(commandBuffers) {
    for (const commandBuffer of commandBuffers || []) {
      for (const command of commandBuffer.commands) this._recorder.push(command);
    }
    return this._recorder.flush();
  }

  onSubmittedWorkDone() {
    this._recorder.flush();
    return Promise.resolve();
  }
}

class GPUDevice {
  constructor(adapter) {
    this.adapter = adapter;
    this.limits = adapter.limits;
    this._recorder = new Recorder();
    this.queue = new GPUQueue(this);
  }

  /** 커맨드 실행 오류 핸들러 (`{kind, message, path}`). 등록하지 않으면 console.error로 나간다. */
  onError(handler) {
    this._recorder.errorHandlers.push(handler);
  }

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

  createTexture(descriptor) {
    const id = this._recorder.allocate();
    this._recorder.push({ op: 'createTexture', id, ...descriptor });
    return new GPUTexture(this, id, descriptor);
  }

  createSampler(descriptor) {
    const id = this._recorder.allocate();
    this._recorder.push({ op: 'createSampler', id, ...(descriptor || {}) });
    return new GPUSampler(this, id, descriptor && descriptor.label);
  }

  createShaderModule(descriptor) {
    const id = this._recorder.allocate();
    this._recorder.push({
      op: 'createShaderModule', id,
      code: descriptor.code, language: descriptor.language || 'wgsl', label: descriptor.label,
    });
    return new GPUShaderModule(this, id, descriptor.label);
  }

  createBindGroupLayout(descriptor) {
    const id = this._recorder.allocate();
    this._recorder.push({ op: 'createBindGroupLayout', id, ...descriptor });
    return new GPUBindGroupLayout(this, id, descriptor.label);
  }

  createPipelineLayout(descriptor) {
    const id = this._recorder.allocate();
    this._recorder.push({
      op: 'createPipelineLayout', id,
      bindGroupLayouts: descriptor.bindGroupLayouts.map((layout) => layout.id),
      label: descriptor.label,
    });
    return new GPUPipelineLayout(this, id, descriptor.label);
  }

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

  createRenderPipeline(descriptor) {
    const id = this._recorder.allocate();
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

  createCommandEncoder() {
    return new GPUCommandEncoder(this);
  }

  /** 모든 GPU 객체를 버린다 (페이지 이탈 시 호출). */
  destroy() {
    this._recorder.pending = [];
    canvasSizeCache.clear();
    nativeModule().reset();
  }
}

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
  constructor(canvasId) {
    this.canvasId = canvasId;
    this._device = null;
    this.format = 'bgra8unorm';
  }

  configure(configuration) {
    this._device = configuration.device;
    this.format = configuration.format || 'bgra8unorm';
    this._device._recorder.push({
      op: 'configureCanvas',
      canvas: this.canvasId,
      format: this.format,
      usage: configuration.usage,
      alphaMode: configuration.alphaMode,
    });
    // 크기를 미리 캐시한다 — 이후에는 제출 응답이 갱신하므로 동기 조회는 사실상 이 1회뿐이다.
    this._fetchSize();
  }

  /** 이번 프레임의 스왑체인 텍스처. 프레임이 끝나면 무효해진다 (브라우저와 같은 규칙). */
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
   */
  getSize() {
    const cached = canvasSizeCache.get(this.canvasId);
    if (cached) return { width: cached.width, height: cached.height };
    return this._fetchSize();
  }

  /** 동기 네이티브 조회 — 캐시가 비어 있을 때만 쓴다. */
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
  constructor(info) {
    this.name = info.name;
    this.backend = info.backend;
    this.limits = info.limits || {};
    this.hasUnifiedMemory = info.hasUnifiedMemory;
  }

  async requestDevice() {
    return new GPUDevice(this);
  }
}

export const gpu = {
  /** `navigator.gpu.requestAdapter()`. */
  async requestAdapter() {
    const info = nativeModule().adapterInfo();
    if (!info || info.ok === false) return null;
    return new GPUAdapter(info);
  },

  /** 캔버스 표면에 가장 잘 맞는 포맷. `<webgpu-canvas>`의 기본값과 같다. */
  getPreferredCanvasFormat() {
    return 'bgra8unorm';
  },

  /** `<webgpu-canvas canvas-id="…">` 의 표면에 붙는다 (`canvas.getContext('webgpu')` 대응). */
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

  let emitter = null;
  try {
    emitter = lynx.getJSModule('GlobalEventEmitter');
  } catch (error) {
    emitter = null;
  }

  if (emitter) {
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

export { GPUBuffer, GPUTexture, GPUTextureView, GPUSampler, GPUDevice, GPUCanvasContext };
export default gpu;
