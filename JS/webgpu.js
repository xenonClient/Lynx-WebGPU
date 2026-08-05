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

/** @typedef {{type: 'occlusion' | 'timestamp', count: number, label?: string}} GPUQuerySetDescriptor */

/** 명세 `GPUAdapterInfo`. */
/** @typedef {{vendor: string, architecture: string, device: string, description: string, isFallbackAdapter: boolean, subgroupMinSize: number, subgroupMaxSize: number}} GPUAdapterInfoView */

/** 셰이더 컴파일 진단 하나 (명세 `GPUCompilationMessage`). */
/** @typedef {{message: string, type: 'error' | 'warning' | 'info', lineNum: number, linePos: number, offset: number, length: number}} GPUCompilationMessage */

/** `device.lost`가 (유실을 보고하는 구현에서) 풀리는 값 — 이 구현은 영원히 pending이다. */
/** @typedef {{reason: 'unknown' | 'destroyed', message: string}} GPUDeviceLostInfo */

/**
 * 패스 경계에서 타임스탬프를 찍을 자리. 두 인덱스는 각각 생략할 수 있다.
 */
/** @typedef {{querySet: GPUQuerySet, beginningOfPassWriteIndex?: number, endOfPassWriteIndex?: number}} GPUPassTimestampWrites */

/**
 * `createRenderBundleEncoder`의 디스크립터 — 이 번들을 **실행할 패스의 모양**이다.
 * `colorFormats`의 `null`은 "그 슬롯은 비어 있다"는 뜻이고, 후행 `null`은 비교에서 무시된다.
 * `depthReadOnly`/`stencilReadOnly`는 "이 번들은 깊이/스텐실을 쓰지 않는다"는 선언이다 —
 * 같은 이름으로 열린 패스에서 실행하려면 번들도 `true`여야 한다.
 */
/** @typedef {{colorFormats: (string | null)[], depthStencilFormat?: string, sampleCount?: number, depthReadOnly?: boolean, stencilReadOnly?: boolean, label?: string}} GPURenderBundleEncoderDescriptor */

/** `createPipelineLayout`이 받는 레이아웃 — id만 있으면 된다. */
/** @typedef {{id: number}} GPUPipelineLayoutSource */

/** `GPUTexture` 생성자 내부용 — 스왑체인 텍스처는 usage/format이 없을 수 있다. */
/** @typedef {{size?: GPUExtent3D, format?: string, usage?: number, dimension?: string, mipLevelCount?: number, sampleCount?: number, textureBindingViewDimension?: string, label?: string, frameScoped?: boolean}} GPUTextureInit */

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
/**
 * GPU 객체 핸들 발급기 — **모듈 전체가 하나를 쓴다.**
 *
 * 네이티브 레지스트리는 `LynxWebGPUContext`당 하나이고 **핸들 정수만으로** 객체를 찾는다
 * (디바이스별 칸이 없다). 그래서 카운터를 디바이스마다 두면 두 번째 디바이스가 1번부터 다시
 * 발급해 첫 디바이스의 객체를 **조용히 덮어쓴다** — 오류 없이 남의 버퍼에 그리게 된다.
 *
 * 번호는 되돌리지 않는다. `device.destroy()`가 레지스트리를 비워도 재사용하지 않는 편이,
 * 그 번호를 아직 들고 있는 JS 객체가 나중에 **남의 자리**를 가리키는 일을 막는다.
 */
/**
 * 지금 **프레임 루프 콜백 안**인가 (중첩까지 세는 깊이).
 *
 * 브라우저의 present 시점은 `queue.submit()`이 아니라 **태스크의 끝**이다 — 한 프레임 안에서
 * submit을 몇 번 하든 캔버스는 콜백이 끝난 뒤에 한 번 나간다. 그래서 웹 라이브러리는 한
 * 프레임에 여러 번 submit하면서 드로어블 텍스처 뷰를 그 프레임 내내 재사용한다
 * (three.js의 `PostProcessing`이 그렇다: 씬 패스 → bloom 밉 체인 → 출력 패스).
 *
 * submit마다 present하면 첫 submit이 드로어블을 내보내고 그 뷰를 만료시켜, 같은 프레임의
 * 남은 패스가 "없는 핸들"로 통째로 거부된다. 그래서 **틱 안의 flush는 present를 미루고**,
 * 틱이 끝날 때 한 번만 present한다 (`endFrameTick`).
 */
let frameTickDepth = 0;

/**
 * 지금 프레임 티커를 구독한 수. 네이티브 티커는 하나뿐이라, 마지막 구독자가 놓을 때만
 * 멈춘다 — 안 그러면 한 쪽의 `stop()`이 다른 쪽(특히 rAF 펌프)까지 끈다.
 */
let frameLoopSubscribers = 0;

/**
 * 이번 틱에서 present를 빚진 레코더들. 틱이 끝나면 여기 있는 것만 present한다 —
 * GPU 작업이 없던 틱은 브리지를 건너지 않는다.
 * @type {Set<Recorder>}
 */
const framePresentDebt = new Set();

/**
 * 프레임 루프 콜백 하나를 감싼다. **콜백이 던져도 present는 나간다** —
 * 안 그러면 화면이 그 프레임에서 멈춘 채로 남는다.
 * @param {(frame: {timestamp: number, delta: number}) => void} handler
 * @param {{timestamp: number, delta: number}} frame
 */
function runFrameTick(handler, frame) {
  frameTickDepth += 1;
  try {
    handler(frame);
  } finally {
    frameTickDepth -= 1;
    if (frameTickDepth === 0) endFrameTick();
  }
}

/** 틱의 끝 — 미뤄 둔 present를 여기서 한 번에 낸다. */
function endFrameTick() {
  if (framePresentDebt.size === 0) return;
  const owing = Array.from(framePresentDebt);
  framePresentDebt.clear();
  for (const recorder of owing) recorder.flush(true, { presentOnly: true });
}

let nextHandle = 1;

/** @returns {number} 아무도 쓰지 않은 새 핸들 */
function allocateHandle() {
  return nextHandle++;
}

/**
 * 마지막으로 만든 디바이스의 레코더 — 전역 함수인 `createImageBitmap()`이 `close()`의
 * `destroy` 명령을 **어느 스트림에 실을지** 정하는 데만 쓴다. 핸들 번호와는 무관하다
 * (그건 위의 `allocateHandle()`이 낸다).
 *
 * 이미지는 디바이스가 아니라 컨텍스트의 것이라 어느 스트림에 실려도 같은 곳에 닿는다.
 * @type {Recorder | null}
 */
let activeRecorder = null;

const canvasSizeCache = new Map();

/**
 * canvasId → `GPUCanvasContext`.
 *
 * 브라우저의 `canvas.getContext('webgpu')`가 늘 같은 객체를 주는 것과 맞춘다 — 매번 새로
 * 만들면 설정 상태(`configure`/`unconfigure`)가 핸들마다 갈라진다.
 * @type {Map<string, GPUCanvasContext>}
 */
const canvasContexts = new Map();

/** `pushErrorScope`가 받는 필터 (명세 `GPUErrorFilter` 철자 그대로). */
const ERROR_FILTERS = ['validation', 'out-of-memory', 'internal'];

/**
 * 명세의 `GPUError` 계층 — `uncapturederror` 이벤트가 실어 나르는 객체.
 *
 * 명세는 `message` 하나만 요구하지만 `kind`·`path`를 함께 둔다. 커맨드 스트림에서 온 오류는
 * "몇 번째 명령의 어느 필드"까지 알고 있고, 그걸 버리면 진단이 크게 나빠지기 때문이다.
 * 웹 코드가 종류를 볼 때는 `instanceof`(또는 `constructor.name`)를 쓴다 — 그래서 하위 클래스로 나눈다.
 */
class GPUError {
  /** @param {WGPUError} payload */
  constructor(payload) {
    this.message = payload.message;
    /** 이 구현이 붙이는 추가 정보 — 명세에는 없다. */
    this.kind = payload.kind;
    this.path = payload.path;
  }
}
class GPUValidationError extends GPUError {}
class GPUOutOfMemoryError extends GPUError {}
class GPUInternalError extends GPUError {}

/**
 * 커맨드 오류의 네 종류를 명세의 세 `GPUError` 하위 클래스로 접는다.
 *
 * `unsupported`가 `GPUValidationError`인 것은 `pushErrorScope('validation')`이 그것을 잡는 것과
 * 같은 이유다 — 브라우저에서 같은 코드는 validation으로 나거나 성공하므로, 앱의 분기가 맞아떨어진다.
 *
 * @param {WGPUError} payload
 * @returns {GPUError}
 */
function makeGPUError(payload) {
  switch (payload.kind) {
    case 'out-of-memory': return new GPUOutOfMemoryError(payload);
    case 'backend': return new GPUInternalError(payload);
    default: return new GPUValidationError(payload);
  }
}

/**
 * `createRenderPipelineAsync` 계열이 거부할 때 쓰는 오류 (명세 `GPUPipelineError`).
 *
 * 명세의 `reason`은 `'validation' | 'internal'` 둘뿐이라, 커맨드 오류의 네 종류를 그리로
 * 접는다 — 셰이더 번역·컴파일 실패(`backend`)가 `internal`이고 나머지는 `validation`이다.
 * `pushErrorScope` 필터의 대응 관계와 같은 규칙이다.
 *
 * @param {WGPUError} error
 * @returns {Error & {reason: string}}
 */
/**
 * 명세가 `OperationError`로 던지라고 정한 자리 (`getMappedRange`의 인자 검증 등).
 *
 * 이름을 맞추는 것이 요점이다 — 웹 코드가 `error.name`으로 갈라 잡는다.
 */
class OperationError extends Error {
  /** @param {string} message */
  constructor(message) {
    super(message);
    this.name = 'OperationError';
  }
}

function makePipelineError(/** @type {WGPUError} */ error) {
  const failure = /** @type {Error & {reason: string}} */ (
    new Error(`${error.path ? error.path + ' — ' : ''}${error.message}`)
  );
  failure.name = 'GPUPipelineError';
  failure.reason = error.kind === 'backend' ? 'internal' : 'validation';
  return failure;
}

// ---------------------------------------------------------------------------
// 커맨드 레코더
// ---------------------------------------------------------------------------

/**
 * 커맨드에 실을 값의 **기록 시점 스냅샷**.
 *
 * 브라우저 WebGPU는 호출 시점에 인자를 직렬화한다 — 호출 뒤에 디스크립터 객체를 재사용하거나
 * 리셋해도 이미 기록된 명령은 변하지 않는다. 이 shim은 flush까지 명령을 들고 있으므로, 참조를
 * 그대로 실으면 그 사이의 변이가 스트림에 새어 든다. three.js가 싱글턴 디스크립터를 인코딩 직후
 * `reset()`하는 패턴이 정확히 이 경우다 — `copySize`가 flush 전에 0이 되어 **폭 0짜리 복사가
 * 오류 없이** 나가고, 텍스처 업로드도 같은 식으로 조용히 사라진다.
 *
 * `ArrayBuffer`(전송 페이로드 — shim이 이미 복사해 만든다)와 TypedArray는 그대로 둔다.
 *
 * @param {any} value
 * @returns {any}
 */
function snapshotValue(value) {
  if (!value || typeof value !== 'object' || value instanceof ArrayBuffer || ArrayBuffer.isView(value)) {
    return value;
  }
  if (Array.isArray(value)) return value.map(snapshotValue);
  /** @type {Record<string, any>} */
  const out = {};
  for (const key in value) {
    const entry = /** @type {Record<string, any>} */ (value)[key];
    if (typeof entry !== 'function') out[key] = snapshotValue(entry);
  }
  return out;
}

class Recorder {
  constructor() {
    /** @type {GPUCommand[]} */
    this.pending = [];
    /** @type {((error: WGPUError, text: string) => void)[]} */
    this.errorHandlers = [];
    /**
     * 스코프에 안 잡힌 오류를 명세의 `uncapturederror` 경로로도 흘려보내는 훅.
     * 디바이스가 자기 자신을 꽂는다. 하나라도 받아 갔으면 `true`를 돌려준다.
     * @type {((error: WGPUError) => boolean) | null}
     */
    this.uncapturedDispatch = null;
    /**
     * `popErrorScope()`가 돌려준 Promise의 결과 함수들 — **pop한 순서 그대로**다.
     * 네이티브가 같은 순서로 `errorScopes` 배열을 돌려주므로 인덱스로 짝을 맞춘다.
     * @type {{resolve: (error: WGPUError | null) => void, reject: (reason: Error) => void}[]}
     */
    this.pendingErrorScopes = [];
  }

  /**
   * 새 핸들 id. **모듈 공용 카운터**에서 낸다 — 레코더마다 세면 디바이스가 둘일 때
   * 겹친다 (`allocateHandle` 주석 참고).
   * @returns {number}
   */
  allocate() {
    return allocateHandle();
  }

  /**
   * 명령을 **기록 시점 값으로 고정해** 쌓는다 (`snapshotValue` 참고) — 디바이스/큐 op은
   * 여기가 곧 호출 시점이라, 호출 뒤 디스크립터 재사용이 스트림을 오염시키지 못한다.
   * @param {GPUCommand} command
   * @returns {GPUCommand}
   */
  push(command) {
    const frozen = snapshotValue(command);
    this.pending.push(frozen);
    return frozen;
  }

  /**
   * 모아 둔 명령을 네이티브로 넘긴다. 실행할 것이 없으면 아무것도 하지 않는다.
   *
   * `present: false`는 **프레임 중간의 내부 제출**이라는 표시다 (`popErrorScope`·`mapAsync`가
   * 결과를 받으려고 미리 흘려보내는 배치). 네이티브는 이 배치를 커밋하되 드로어블 present와
   * 스왑체인 핸들 만료를 진짜 프레임 제출(`queue.submit`)까지 미룬다 — 안 그러면 획득해 둔
   * 캔버스 텍스처가 그리기도 전에 present되어 남은 패스가 통째로 거부된다.
   *
   * **프레임 루프 콜백 안에서는 present를 미룬다** (`frameTickDepth`) — 브라우저가 태스크
   * 끝에 present하는 것과 같은 자리다. 미룬 present는 틱이 끝날 때 `endFrameTick()`이 낸다.
   *
   * @param {boolean} [present] 이 배치가 프레임 제출인가 (기본 true)
   * @param {{presentOnly?: boolean}} [options] `presentOnly`는 틱 끝의 마무리 호출 —
   *   명령이 없어도 배치를 보내 **드로어블을 내보낸다.**
   * @returns {WGPUExecuteResult}
   */
  flush(present = true, options) {
    const presentOnly = !!(options && options.presentOnly);
    // 틱 안의 프레임 제출은 present를 틱 끝으로 미룬다. 내부 제출(present=false)은 그대로다.
    if (present && !presentOnly && frameTickDepth > 0) {
      framePresentDebt.add(this);
      present = false;
    }
    if (this.pending.length === 0 && !presentOnly) {
      this.settleErrorScopes([]);
      return { ok: true, commandCount: 0 };
    }
    const commands = this.pending;
    this.pending = [];
    /** @type {WGPUExecuteResult} */
    let result;
    try {
      result = /** @type {WGPUExecuteResult} */ (nativeModule().execute({ commands, present }) || {});
    } catch (error) {
      // 브리지 호출 자체가 실패해도 기다리던 Promise는 반드시 풀어 준다. 안 그러면 영원히
      // pending으로 남아 초기화 진단이 매달리고, 다음 pop이 stale resolver를 가져가
      // **인덱스가 어긋난다**.
      this.settleErrorScopes([]);
      this.report([{ kind: 'backend', message: `네이티브 실행 실패: ${(error && /** @type {Error} */ (error).message) || error}` }]);
      return { ok: false, commandCount: commands.length };
    }
    this.settleErrorScopes(result.errorScopes || []);
    if (result.canvases) {
      for (const canvasId in result.canvases) {
        const info = result.canvases[canvasId];
        if (info && typeof info.width === 'number') {
          canvasSizeCache.set(canvasId, { width: info.width, height: info.height });
          // 명세는 **리사이즈에서도** 현재 텍스처를 만료시킨다 — 크기가 달라진 드로어블을
          // 옛 텍스처로 계속 가리키면 다음 패스가 어긋난 크기로 그린다.
          const context = canvasContexts.get(canvasId);
          if (context && context._currentSize
              && (context._currentSize.width !== info.width
                  || context._currentSize.height !== info.height)) {
            context._expireCurrentTexture();
          }
        }
      }
    }
    // 명세의 "presentation"이 "Expire the current texture"를 부르는 자리다.
    if (present) {
      for (const context of canvasContexts.values()) context._expireCurrentTexture();
    }
    if (result.ok === false) this.report(result.errors || []);
    return result;
  }

  /**
   * `popErrorScope` 명령을 쌓고 결과 Promise를 돌려준다 — **flush하지 않는다.**
   *
   * 스코프 여러 개를 한 배치에 닫을 때 쓴다 (비동기 파이프라인 생성이 그렇다).
   * 네이티브가 pop 순서 그대로 결과를 돌려주므로, 부른 순서가 곧 짝짓는 순서다.
   *
   * @returns {Promise<WGPUError | null>}
   */
  recordPop() {
    this.push({ op: 'popErrorScope' });
    return /** @type {Promise<WGPUError | null>} */ (
      new Promise((resolve, reject) => {
        this.pendingErrorScopes.push({ resolve, reject });
      })
    );
  }

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
  settleErrorScopes(popped) {
    if (this.pendingErrorScopes.length === 0) return;
    const waiting = this.pendingErrorScopes;
    this.pendingErrorScopes = [];
    waiting.forEach((settle, index) => {
      const slot = popped[index];
      if (slot && /** @type {{rejected?: boolean}} */ (slot).rejected) {
        const error = new Error('popErrorScope: 열려 있는 오류 스코프가 없다 (push와 짝이 맞는지 확인)');
        error.name = 'OperationError';
        settle.reject(error);
        return;
      }
      settle.resolve(/** @type {WGPUError | null} */ (slot) || null);
    });
  }

  /**
   * 스코프에 안 잡힌 오류를 등록된 모든 통로로 보낸다.
   *
   * 통로는 둘이고 **함께** 받는다 — 이 구현의 `onError`(경로가 붙은 텍스트까지 준다)와
   * 명세의 `uncapturederror`(웹 코드가 아는 이름). 아무도 안 듣고 있을 때만 콘솔로 떨어뜨린다:
   * 조용히 사라지는 오류가 없어야 하고, 듣고 있는데 콘솔에도 찍히면 로그가 두 번 남는다.
   *
   * @param {WGPUError[]} errors
   */
  report(errors) {
    for (const error of errors) {
      const text = `[WebGPU:${error.kind}] ${error.path ? error.path + ' — ' : ''}${error.message}`;
      let delivered = false;
      for (const handler of this.errorHandlers) {
        handler(error, text);
        delivered = true;
      }
      if (this.uncapturedDispatch && this.uncapturedDispatch(error)) delivered = true;
      if (!delivered) console.error(text);
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
    /** `mappedAtCreation`의 초기 데이터인가 (unmap이 생성 명령을 기록해야 하는가). */
    this._mappedAtCreation = false;
    /** `mapAsync`가 아직 결과를 기다리는 중인가 (명세의 `"pending"` 상태). */
    this._mapPending = false;
    /**
     * `getMappedRange()`로 내준 구간들 — 겹침 검사와 `unmap()`의 되돌려 쓰기에 쓴다.
     * @type {{offset: number, length: number, view: ArrayBuffer | null}[]}
     */
    this._mappedRanges = [];
  }

  /**
   * 명세의 `GPUBufferMapState` — `'unmapped'` · `'pending'` · `'mapped'`.
   *
   * 매핑 중인 버퍼는 큐 작업에서 거부되므로, 재사용하려는 코드가 **묻지 않고도** 상태를
   * 알 수 있어야 한다. 없으면 `undefined`를 보고 "매핑 안 됐다"로 오해한다.
   *
   * @returns {'unmapped' | 'pending' | 'mapped'}
   */
  get mapState() {
    if (this._mapped) return 'mapped';
    return this._mapPending ? 'pending' : 'unmapped';
  }

  /** `mappedAtCreation: true`로 만든 버퍼의 초기 데이터 영역. */
  /**
   * 매핑된 구간을 `ArrayBuffer`로 얻는다 (`mappedAtCreation` 또는 `mapAsync` 이후).
   *
   * **JS에서는 `ArrayBuffer`가 다른 `ArrayBuffer`의 일부를 가리킬 수 없다.** 브라우저는
   * 매핑 메모리를 그대로 가리키는 뷰를 주지만, 여기서는 그럴 수 없어 구간을 복사해 주고
   * `unmap()`에서 **되돌려 쓴다.** 그래서 쓴 내용이 사라지지 않는다 — 단, 반환된 버퍼는
   * `unmap()` **전까지만** 의미가 있다 (브라우저에서 detach되는 것과 같은 시점이다).
   *
   * 전체 구간을 처음 요청하면 복사 없이 매핑 자체를 돌려준다 — `mappedAtCreation`으로
   * 큰 정점 버퍼를 채우는 흔한 경로에서 복사를 한 번 아낀다.
   *
   * 명세 규칙: `offset`은 8의 배수, `size`는 4의 배수, 구간끼리 **겹칠 수 없다**.
   *
   * @param {number} [offset] 바이트 오프셋 (8의 배수)
   * @param {number} [size] 바이트 수 (4의 배수). 생략하면 끝까지
   * @returns {ArrayBuffer}
   */
  getMappedRange(offset, size) {
    if (!this._mapped) {
      throw new Error('getMappedRange는 mappedAtCreation 또는 mapAsync 이후에만 쓸 수 있다');
    }
    const total = this._mapped.byteLength;
    const start = offset || 0;
    const length = size === undefined ? Math.max(0, total - start) : size;

    if (start % 8 !== 0) {
      throw new OperationError(`getMappedRange: offset은 8의 배수여야 한다 (받은 값 ${start})`);
    }
    // 정렬 검사는 **명시한 값에만** 적용한다. 브라우저의 매핑은 항상 4의 배수지만
    // 여기서는 매핑 크기가 곧 네이티브 버퍼 크기라 3바이트짜리도 정상이다 —
    // 생략했을 때(=매핑 전체)까지 4의 배수를 요구하면 그런 버퍼를 아예 못 읽는다.
    if (size !== undefined && length % 4 !== 0) {
      throw new OperationError(`getMappedRange: size는 4의 배수여야 한다 (받은 값 ${length})`);
    }
    if (start < 0 || start + length > total) {
      throw new OperationError(
        `getMappedRange 범위가 매핑을 넘는다 — offset ${start} + ${length}B > ${total}B`
      );
    }
    for (const range of this._mappedRanges) {
      if (start < range.offset + range.length && range.offset < start + length) {
        throw new OperationError(
          `getMappedRange 구간이 앞서 얻은 구간과 겹친다 (${range.offset}~${range.offset + range.length})`
        );
      }
    }

    // 전체를 처음 요청하면 매핑 자체를 준다 — 되돌려 쓸 것이 없다.
    if (start === 0 && length === total && this._mappedRanges.length === 0) {
      this._mappedRanges.push({ offset: 0, length, view: null });
      return this._mapped;
    }
    const view = this._mapped.slice(start, start + length);
    this._mappedRanges.push({ offset: start, length, view });
    return view;
  }

  /** 구간 사본에 쓴 내용을 매핑으로 되돌린다 (`unmap` 직전). */
  _flushMappedRanges() {
    if (!this._mapped) return;
    for (const range of this._mappedRanges) {
      if (!range.view) continue;
      new Uint8Array(this._mapped).set(new Uint8Array(range.view), range.offset);
    }
    this._mappedRanges = [];
  }

  /**
   * 매핑을 푼다.
   *
   * `mappedAtCreation`이면 여기서 실제 생성 명령(초기 데이터 포함)이 기록되고,
   * `mapAsync`로 매핑한 것이면 네이티브에 "이제 큐 작업에 써도 된다"를 알린다 —
   * 매핑 중인 버퍼는 명세대로 큐 작업에서 거부되므로 **읽고 나면 반드시 불러야 한다.**
   *
   * @returns {void}
   */
  unmap() {
    if (!this._mapped) return;
    // 구간 사본에 쓴 내용을 매핑으로 되돌린 뒤에 보낸다 — 안 하면 조용히 사라진다.
    this._flushMappedRanges();
    if (this._mappedAtCreation) {
      this._recorder.push({
        op: 'createBuffer',
        id: this.id,
        size: this.size,
        usage: this.usage,
        label: this.label,
        data: this._mapped,   // 이미 ArrayBuffer다
      });
      this._mappedAtCreation = false;
    } else {
      this._recorder.push({ op: 'unmapBuffer', buffer: this.id });
    }
    this._mapped = null;
  }

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
  async mapAsync(_mode, offset, size) {
    this._recorder.flush(false);
    this._mapPending = true;
    const result = await /** @type {Promise<WGPUReadBufferResult | undefined>} */ (
      new Promise((resolve) => {
        nativeModule().readBuffer({ buffer: this.id, offset: offset || 0, size }, resolve);
      })
    );
    this._mapPending = false;
    if (!result || result.ok === false) {
      this._recorder.report((result && result.errors) || []);
      throw new Error('버퍼 읽기 실패');
    }
    // 네이티브가 `Data`로 돌려주면 Lynx가 ArrayBuffer로 바꿔 준다 — 디코딩할 것이 없다.
    const mapped = result.data;
    this._mapped = mapped;
    this._mappedAtCreation = false;
    // 새 매핑이므로 앞선 매핑의 구간 기록은 버린다 (겹침 검사가 헛돌지 않게).
    this._mappedRanges = [];
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
    // 명세의 읽기 전용 속성들 — 웹 코드가 텍스처를 받아 스스로 판단할 때 읽는다
    // (three.js는 밉맵 경로에서 `textureBindingViewDimension`을 본다).
    this.depthOrArrayLayers = size ? size.depthOrArrayLayers || size[2] || 1 : 1;
    this.mipLevelCount = (descriptor && descriptor.mipLevelCount) || 1;
    this.sampleCount = (descriptor && descriptor.sampleCount) || 1;
    this.dimension = (descriptor && descriptor.dimension) || '2d';
    this.format = descriptor && descriptor.format;
    this.usage = (descriptor && descriptor.usage) || 0;
    /**
     * 이 텍스처를 바인딩할 때의 기본 뷰 차원. 명세는 생략을 허용하고, 그때는 `dimension`과
     * 레이어 수에서 정해진다 (2d + 레이어 2 이상이면 `2d-array`).
     * @type {string}
     */
    this.textureBindingViewDimension = (descriptor && descriptor.textureBindingViewDimension)
      || (this.dimension === '2d' && this.depthOrArrayLayers > 1 ? '2d-array' : this.dimension);
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
class GPUShaderModule extends GPUObjectBase {
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
  async getCompilationInfo() {
    // 아직 안 보낸 명령이 있으면 이 모듈이 네이티브에 없다 — 먼저 흘려보낸다 (프레임 중간 제출).
    this._recorder.flush(false);
    const result = nativeModule().shaderCompilationInfo({ module: this.id });
    if (!result || result.ok === false) {
      this._recorder.report((result && result.errors) || []);
      return { messages: [] };
    }
    return { messages: result.messages || [] };
  }
}
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

/**
 * `bundleEncoder.finish()`가 돌려주는 재사용 가능한 드로우 묶음.
 *
 * 기록한 명령이 **래퍼보다 오래 사는** 유일한 구조라, 자기가 쓰는 리소스 래퍼를 붙잡는다
 * (`_retained`). 그러지 않으면 초기화 함수가 번들만 반환하고 파이프라인·버퍼를 버렸을 때
 * GC가 `destroy`를 끼워 넣어 번들이 **조용히 안 그려진다** (`docs/JS-AUTHORING.md` §8).
 */
class GPURenderBundle extends GPUObjectBase {
  /**
   * @param {GPUDevice} device
   * @param {number} id
   * @param {string} [label]
   */
  constructor(device, id, label) {
    super(device, id, label);
    /** @type {object[]} 이 번들이 참조하는 리소스 래퍼 — 수명을 함께 묶는다. */
    this._retained = [];
  }
}

/** `device.createQuerySet()`이 돌려주는 쿼리 저장소. */
class GPUQuerySet extends GPUObjectBase {
  /**
   * @param {GPUDevice} device
   * @param {number} id
   * @param {GPUQuerySetDescriptor} descriptor
   */
  constructor(device, id, descriptor) {
    super(device, id, descriptor.label);
    this.type = descriptor.type;
    this.count = descriptor.count;
  }
}

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
    /**
     * 기록 중 만난 리소스 래퍼 — **번들 인코더만** 채운다.
     *
     * 번들은 기록한 명령이 래퍼보다 오래 사는 유일한 구조라서, 자동 해제(GC)가 붙은 엔진에서는
     * 초기화 함수가 번들만 반환하고 파이프라인·버퍼 래퍼를 버리면 번들이 조용히 안 그려진다.
     * 명세 모델에서 번들은 자기가 쓰는 객체를 **소유**하므로, 여기서도 같은 소유 관계를 만든다.
     * @type {object[] | null}
     */
    this._retained = null;
  }

  /**
   * @param {object} resource
   * @returns {void}
   */
  _retain(resource) {
    if (this._retained) this._retained.push(resource);
  }

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
  pushDebugGroup(groupLabel) {
    this._commands.push({ op: 'pushDebugGroup', groupLabel: String(groupLabel) });
  }

  /** @returns {void} */
  popDebugGroup() {
    this._commands.push({ op: 'popDebugGroup' });
  }

  /**
   * 한 지점에 표식을 남긴다 (Metal `insertDebugSignpost`) — 구간이 아니라 점 이벤트다.
   * @param {string} markerLabel
   * @returns {void}
   */
  insertDebugMarker(markerLabel) {
    this._commands.push({ op: 'insertDebugMarker', markerLabel: String(markerLabel) });
  }

  /**
   * @param {GPURenderPipeline | GPUComputePipeline} pipeline
   * @returns {void}
   */
  setPipeline(pipeline) {
    this._retain(pipeline);
    this._commands.push({ op: 'setPipeline', pipeline: pipeline.id });
  }

  /**
   * @param {number} index
   * @param {GPUBindGroup} bindGroup
   * @param {number[]} [dynamicOffsets]
   * @returns {void}
   */
  setBindGroup(index, bindGroup, dynamicOffsets) {
    this._retain(bindGroup);
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
    this._retain(buffer);
    this._commands.push({ op: 'setVertexBuffer', slot, buffer: buffer.id, offset: offset || 0 });
  }

  /**
   * @param {GPUBuffer} buffer
   * @param {'uint16' | 'uint32'} format
   * @param {number} [offset]
   * @returns {void}
   */
  setIndexBuffer(buffer, format, offset) {
    this._retain(buffer);
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
   * `firstInstance`가 0이 아니면 `indirect-first-instance` 기능이 필요하다. 이 구현은 그 기능을
   * 항상 보고하지만, 브라우저에서는 **요청하지 않으면 드로우가 통째로 no-op**이 된다.
   *
   * @param {GPUBuffer} indirectBuffer
   * @param {number} [indirectOffset]
   * @returns {void}
   */
  drawIndirect(indirectBuffer, indirectOffset) {
    this._retain(indirectBuffer);
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
   * `firstInstance`가 0이 아니면 `indirect-first-instance` 기능이 필요하다. 이 구현은 그 기능을
   * 항상 보고하지만, 브라우저에서는 **요청하지 않으면 드로우가 통째로 no-op**이 된다.
   *
   * @param {GPUBuffer} indirectBuffer
   * @param {number} [indirectOffset]
   * @returns {void}
   */
  drawIndexedIndirect(indirectBuffer, indirectOffset) {
    this._retain(indirectBuffer);
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
   * 이 드로우들이 통과시킨 샘플 수를 세기 시작한다.
   *
   * `beginRenderPass`에 `occlusionQuerySet`을 준 패스에서만 쓸 수 있고, 중첩할 수 없다.
   * 같은 인덱스를 한 패스에서 두 번 쓸 수 없고, 패스를 닫기 전에 `endOcclusionQuery`로
   * 반드시 닫아야 한다.
   *
   * @param {number} queryIndex
   * @returns {void}
   */
  beginOcclusionQuery(queryIndex) {
    this._commands.push({ op: 'beginOcclusionQuery', queryIndex });
  }

  /** @returns {void} */
  endOcclusionQuery() {
    this._commands.push({ op: 'endOcclusionQuery' });
  }

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
    /** @type {object[]} 기록 중 만난 래퍼 — 만든 번들이 이어받아 붙잡는다. */
    this._retained = [];
    this._finished = false;
    this._device = device;
    // **기록 시점 값으로 고정한다.** 이 디스크립터는 `finish()`까지 들고 있는데, 호출자가
    // 싱글턴을 넘기고 곧바로 `reset()`하는 패턴이 흔하다 (three.js의 `createBundleEncoder`가
    // 정확히 그렇다). 참조를 그대로 쥐면 `colorFormats`가 빈 채로 번들이 만들어져
    // "어태치먼트가 없다"로 거부된다 — 원인이 한참 뒤 `executeBundles`에서 드러난다.
    this._descriptor = snapshotValue(descriptor) || { colorFormats: [] };
  }

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
  finish(descriptor) {
    const recorder = this._device._recorder;
    const id = recorder.allocate();
    const label = (descriptor && descriptor.label) || this._descriptor.label;
    if (this._finished) {
      recorder.report([{
        kind: 'validation',
        message: 'GPURenderBundleEncoder.finish()는 한 번만 부를 수 있다 (이미 끝난 인코더)',
      }]);
      // 네이티브에 만들지 않으므로 이 핸들을 쓰면 "존재하지 않는다"로 거부된다 (= invalid).
      return new GPURenderBundle(this._device, id, label);
    }
    this._finished = true;
    recorder.push({
      op: 'createRenderBundle', id,
      commands: this._commands,
      colorFormats: this._descriptor.colorFormats || [],
      depthStencilFormat: this._descriptor.depthStencilFormat,
      sampleCount: this._descriptor.sampleCount,
      depthReadOnly: this._descriptor.depthReadOnly,
      stencilReadOnly: this._descriptor.stencilReadOnly,
      label,
    });
    const bundle = new GPURenderBundle(this._device, id, label);
    // 번들이 자기가 쓰는 리소스를 붙잡는다 — 명세의 `[[used_bind_groups]]`와 같은 소유 관계다.
    bundle._retained = this._retained;
    this._commands = [];
    this._retained = [];
    return bundle;
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
        // 호출자가 clearValue 객체를 프레임마다 재사용해도(three.js 패턴) 기록이 안 변하게.
        clearValue: snapshotValue(attachment.clearValue),
      })
    );
    /** @type {GPUCommand} */
    const command = { op: 'beginRenderPass', colorAttachments, label: descriptor.label };
    if (descriptor.depthStencilAttachment) {
      const depth = descriptor.depthStencilAttachment;
      command.depthStencilAttachment = { ...depth, view: depth.view.id };
    }
    // 쿼리는 패스를 열 때만 붙일 수 있다 (Metal도 WebGPU도 같은 제약).
    if (descriptor.occlusionQuerySet) command.occlusionQuerySet = descriptor.occlusionQuerySet.id;
    if (descriptor.timestampWrites) {
      command.timestampWrites = serializeTimestampWrites(descriptor.timestampWrites);
    }
    this._commands.push(command);
    return new GPURenderPassEncoder(this._commands);
  }

  /**
   * @param {{label?: string, timestampWrites?: GPUPassTimestampWrites}} [descriptor]
   * @returns {GPUComputePassEncoder}
   */
  beginComputePass(descriptor) {
    /** @type {GPUCommand} */
    const command = { op: 'beginComputePass', label: descriptor && descriptor.label };
    if (descriptor && descriptor.timestampWrites) {
      command.timestampWrites = serializeTimestampWrites(descriptor.timestampWrites);
    }
    this._commands.push(command);
    return new GPUComputePassEncoder(this._commands);
  }

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
  resolveQuerySet(querySet, firstQuery, queryCount, destination, destinationOffset) {
    this._commands.push({
      op: 'resolveQuerySet',
      querySet: querySet.id,
      firstQuery: firstQuery || 0,
      queryCount,
      destination: destination.id,
      destinationOffset: destinationOffset || 0,
    });
  }

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
  copyBufferToBuffer(source, destinationOrSourceOffset, sizeOrDestination, destinationOffset, size) {
    let sourceOffset = 0;
    /** @type {GPUBuffer} */
    let destination;
    /** @type {number | undefined} */
    let byteLength;

    if (destinationOrSourceOffset && /** @type {GPUBuffer} */ (destinationOrSourceOffset).id !== undefined) {
      // 짧은 형태 — (source, destination, size?)
      destination = /** @type {GPUBuffer} */ (destinationOrSourceOffset);
      byteLength = /** @type {number | undefined} */ (sizeOrDestination);
    } else {
      sourceOffset = /** @type {number} */ (destinationOrSourceOffset) || 0;
      destination = /** @type {GPUBuffer} */ (sizeOrDestination);
      byteLength = size;
    }
    if (byteLength === undefined) byteLength = Math.max(0, source.size - sourceOffset);

    this._commands.push({
      op: 'copyBufferToBuffer',
      source: source.id, sourceOffset,
      destination: destination.id, destinationOffset: destinationOffset || 0,
      size: byteLength,
    });
  }

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
  clearBuffer(buffer, offset, size) {
    /** @type {GPUCommand} */
    const command = { op: 'clearBuffer', buffer: buffer.id, offset: offset || 0 };
    if (size !== undefined) command.size = size;
    this._commands.push(command);
  }

  /**
   * @param {{texture: GPUTexture} & Record<string, any>} source
   * @param {{buffer: GPUBuffer} & Record<string, any>} destination
   * @param {GPUExtent3D} copySize
   * @returns {void}
   */
  copyTextureToBuffer(source, destination, copySize) {
    // 인코더 명령은 submit까지 대기한다 — 그 사이 호출자가 디스크립터를 재사용/리셋해도
    // 기록이 변하지 않도록 여기서 값으로 고정한다 (snapshotValue 참고, 아래 복사 계열 공통).
    this._commands.push({
      op: 'copyTextureToBuffer',
      source: snapshotValue({ ...source, texture: source.texture.id }),
      destination: snapshotValue({ ...destination, buffer: destination.buffer.id }),
      copySize: snapshotValue(copySize),
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
      source: snapshotValue({ ...source, buffer: source.buffer.id }),
      destination: snapshotValue({ ...destination, texture: destination.texture.id }),
      copySize: snapshotValue(copySize),
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
      source: snapshotValue({ ...source, texture: source.texture.id }),
      destination: snapshotValue({ ...destination, texture: destination.texture.id }),
      copySize: snapshotValue(copySize),
    });
  }

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
  pushDebugGroup(groupLabel) {
    this._commands.push({ op: 'pushDebugGroup', groupLabel: String(groupLabel) });
  }

  /** @returns {void} */
  popDebugGroup() {
    this._commands.push({ op: 'popDebugGroup' });
  }

  /**
   * 한 지점에 표식을 남긴다 (Metal `insertDebugSignpost`) — 구간이 아니라 점 이벤트다.
   * @param {string} markerLabel
   * @returns {void}
   */
  insertDebugMarker(markerLabel) {
    this._commands.push({ op: 'insertDebugMarker', markerLabel: String(markerLabel) });
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
   * 디코딩해 둔 이미지(`createImageBitmap`)를 텍스처로 올린다.
   *
   * 웹에서는 `<img>`·`<canvas>`·`VideoFrame`도 소스가 되지만 Lynx에는 그런 엘리먼트가
   * 없다 — `GPUImageBitmap`이 그 자리다. 픽셀은 네이티브에 남아 있으므로 **브리지를
   * 건너는 것은 핸들 하나뿐**이다 (`writeTexture`로 올리면 이미지 전체가 오간다).
   *
   * `source.flipY`는 **복사 시점**에 위아래를 뒤집는다 (`createImageBitmap`의 `flipY`는
   * 디코딩 시점이라 별개다 — 둘 다 켜면 두 번 뒤집혀 제자리로 돌아온다). 웹 라이브러리는
   * 이쪽을 쓴다: three.js의 `Texture.flipY`가 기본 `true`다.
   *
   * @param {{source: GPUImageBitmap, origin?: GPUOrigin3DDict, flipY?: boolean}} source
   * @param {{texture: GPUTexture, mipLevel?: number, origin?: GPUOrigin3DDict,
   *          premultipliedAlpha?: boolean, colorSpace?: string}} destination
   * @param {GPUExtent3D} [copySize] 생략하면 이미지 전체
   * @returns {void}
   */
  copyExternalImageToTexture(source, destination, copySize) {
    this._recorder.push({
      op: 'copyExternalImageToTexture',
      source: {
        source: source.source.id,
        origin: source.origin,
        flipY: !!source.flipY,
      },
      destination: {
        texture: destination.texture.id,
        mipLevel: destination.mipLevel || 0,
        origin: destination.origin,
      },
      copySize,
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
  /**
   * @param {GPUAdapter} adapter
   * @param {string[]} [requiredFeatures] `requestDevice()`에서 검증을 마친 기능 이름들
   */
  constructor(adapter, requiredFeatures) {
    this.adapter = adapter;
    this.limits = adapter.limits;
    /** 명세 `GPUDevice.adapterInfo` — 어댑터의 것을 그대로 본다. */
    this.adapterInfo = adapter.info;
    /**
     * 이 디바이스에 활성화된 기능 — 명세대로 **요청한 것만** 들어 있다 (어댑터가 지원해도
     * `requiredFeatures`로 요청하지 않았으면 `has()`는 false다). Three.js 등 웹 코드가
     * 어댑터에서 고른 기능을 그대로 요청하고 여기서 다시 확인하는 패턴을 쓴다.
     */
    this.features = makeFeatureSet(requiredFeatures || []);
    /**
     * 디바이스 유실 통지 (`device.lost.then(...)` 대응).
     *
     * 이 구현은 유실을 보고하지 않으므로 **영원히 pending**이다 — Metal 디바이스가 사라지는
     * 시나리오(eGPU 분리 등)가 iOS에는 없고, 프로세스가 죽는 경우는 JS도 함께 죽는다.
     * 속성 자체는 있어야 `WebGPUBackend.init()`류의 부트스트랩이 TypeError 없이 지나간다.
     * @type {Promise<GPUDeviceLostInfo>}
     */
    this.lost = new Promise(() => {});
    /**
     * 스코프에 안 잡힌 오류 (명세 `GPUDevice.onuncapturederror`).
     *
     * 웹 코드가 아는 이름이라 그대로 둔다 — Three.js가 여기에 대입해 `renderer.onError`로
     * 넘긴다. 받는 값은 `{type, error}`이고 `error`는 `GPUValidationError` 계열이다.
     * @type {((event: {type: string, error: GPUError}) => void) | null}
     */
    this.onuncapturederror = null;
    /** @type {((event: {type: string, error: GPUError}) => void)[]} */
    this._uncapturedListeners = [];
    this._recorder = new Recorder();
    // 전역 `createImageBitmap()`이 쓸 레코더. 디바이스 없이는 핸들을 발급할 수 없다.
    activeRecorder = this._recorder;
    this._recorder.uncapturedDispatch = (error) => this._dispatchUncaptured(error);
    this.queue = new GPUQueue(this);
  }

  /**
   * 커맨드 실행 오류 핸들러 (`{kind, message, path}`). 등록하지 않으면 console.error로 나간다.
   *
   * 명세의 `onuncapturederror`와 **함께** 동작한다 — 둘 다 등록하면 둘 다 받는다.
   * 이쪽은 경로가 붙은 완성된 텍스트까지 주므로 진단에는 더 편하다.
   *
   * @param {(error: WGPUError, text: string) => void} handler
   * @returns {void}
   */
  onError(handler) {
    this._recorder.errorHandlers.push(handler);
  }

  /**
   * `uncapturederror` 리스너 등록 (명세의 `EventTarget` 자리 — 이 이벤트만 받는다).
   * @param {string} type
   * @param {(event: {type: string, error: GPUError}) => void} listener
   * @returns {void}
   */
  addEventListener(type, listener) {
    if (type === 'uncapturederror') this._uncapturedListeners.push(listener);
  }

  /**
   * @param {string} type
   * @param {(event: {type: string, error: GPUError}) => void} listener
   * @returns {void}
   */
  removeEventListener(type, listener) {
    if (type !== 'uncapturederror') return;
    const index = this._uncapturedListeners.indexOf(listener);
    if (index >= 0) this._uncapturedListeners.splice(index, 1);
  }

  /**
   * 오류 하나를 `uncapturederror` 통로로 흘린다. 받아 간 곳이 하나라도 있으면 `true`.
   *
   * 리스너가 던져도 **나머지 리스너와 다음 오류는 계속 간다** — 하나의 실수가 전체 보고를
   * 삼키면, 정작 원인인 오류가 사라져 진단이 불가능해진다.
   *
   * @param {WGPUError} payload
   * @returns {boolean}
   */
  _dispatchUncaptured(payload) {
    const event = { type: 'uncapturederror', error: makeGPUError(payload) };
    const listeners = this._uncapturedListeners.slice();
    if (typeof this.onuncapturederror === 'function') listeners.push(this.onuncapturederror);
    for (const listener of listeners) {
      try {
        listener(event);
      } catch (error) {
        console.error(`uncapturederror 리스너가 던졌다: ${(error && /** @type {Error} */ (error).message) || error}`);
      }
    }
    return listeners.length > 0;
  }

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
  pushErrorScope(filter) {
    if (ERROR_FILTERS.indexOf(filter) === -1) {
      throw new TypeError(
        `pushErrorScope: filter는 ${ERROR_FILTERS.join(' / ')} 중 하나여야 한다 (받은 값 ${String(filter)})`
      );
    }
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
    const promise = this._recorder.recordPop();
    // 프레임 중간 제출 표시 — 획득해 둔 캔버스 텍스처를 present하지 않는다 (flush 참고).
    this._recorder.flush(false);
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
      buffer._mappedAtCreation = true;
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
  createRenderPipelineAsync(descriptor) {
    return this._createPipelineAsync(() => this.createRenderPipeline(descriptor));
  }

  /**
   * `createComputePipeline`의 비동기 판 (`createRenderPipelineAsync`와 같은 계약).
   * @param {Record<string, any>} descriptor
   * @returns {Promise<GPUComputePipeline>}
   */
  createComputePipelineAsync(descriptor) {
    return this._createPipelineAsync(() => this.createComputePipeline(descriptor));
  }

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
  async _createPipelineAsync(record) {
    this.pushErrorScope('validation');
    this.pushErrorScope('internal');
    const pipeline = record();
    // 안쪽(internal)부터 닫는다 — 네이티브가 pop 순서 그대로 결과를 돌려준다.
    const internalPromise = this._recorder.recordPop();
    const validationPromise = this._recorder.recordPop();
    this._recorder.flush(false);

    const [internalError, validationError] = await Promise.all([internalPromise, validationPromise]);
    const failure = internalError || validationError;
    if (failure) {
      // 쓸 수 없는 핸들을 남기지 않는다 — 네이티브에는 애초에 객체가 없다.
      pipeline.destroy();
      throw makePipelineError(failure);
    }
    return pipeline;
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
  createQuerySet(descriptor) {
    const id = this._recorder.allocate();
    this._recorder.push({ op: 'createQuerySet', id, ...descriptor });
    return new GPUQuerySet(this, id, descriptor);
  }

  /** 모든 GPU 객체를 버린다 (페이지 이탈 시 호출). @returns {void} */
  destroy() {
    this._recorder.pending = [];
    // 기다리던 popErrorScope Promise를 그냥 두면 영원히 안 풀린다 — null로 닫는다.
    // 지금은 `flush()`가 어느 경로로 끝나든(성공·실패 모두) 스코프를 풀어 주므로 여기까지
    // 오는 경우가 없다. 마지막 안전망으로만 남긴다.
    this._recorder.settleErrorScopes([]);
    canvasSizeCache.clear();
    nativeModule().reset();
  }
}

/**
 * 타임스탬프 쓰기 자리를 커맨드에 실을 모양으로 — 쿼리셋은 핸들로 바꾼다.
 * @param {GPUPassTimestampWrites} writes
 * @returns {Record<string, any>}
 */
function serializeTimestampWrites(writes) {
  return {
    querySet: writes.querySet.id,
    beginningOfPassWriteIndex: writes.beginningOfPassWriteIndex,
    endOfPassWriteIndex: writes.endOfPassWriteIndex,
  };
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
    /**
     * `getConfiguration()`이 돌려줄 마지막 설정 — 아직 없으면 `null` (명세와 같다).
     * @type {GPUCanvasConfiguration | null}
     */
    this._configuration = null;
    /**
     * 명세 `[[currentTexture]]` — 만료 전까지 `getCurrentTexture()`가 돌려주는 **같은 객체**.
     * @type {GPUTexture | null}
     */
    this._currentTexture = null;
    /** 그 텍스처를 받을 때의 캔버스 크기 — 리사이즈 만료를 가리는 데 쓴다. */
    this._currentSize = null;
  }

  /**
   * @param {GPUCanvasConfiguration} configuration
   * @returns {void}
   */
  configure(configuration) {
    // 명세: `configure()`는 "Expire the current texture"를 부른다 — 새 설정으로 다시 받는다.
    this._expireCurrentTexture();
    this._device = configuration.device;
    this.format = configuration.format || 'bgra8unorm';
    this._configuration = { ...configuration, format: this.format };
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
   * 설정을 푼다 — 다시 `configure()`하기 전까지 이 컨텍스트로는 그릴 수 없다.
   *
   * 포맷을 바꿔 재구성하는 코드(HDR 토글 등)가 밟는 자리다. 이후 `getCurrentTexture()`는
   * "configure()를 먼저"로 거부한다.
   *
   * **이미 화면에 나간 프레임을 지우지는 않는다.** 브라우저는 캔버스를 투명 검정으로
   * 비우지만, 여기서 그러려면 표면을 한 번 클리어해 present해야 한다 — 설정을 푸는 호출이
   * 프레임을 하나 소비하는 편이 더 놀랍다고 보고 하지 않는다 (`docs/WEBGPU-API.md` §2).
   *
   * @returns {void}
   */
  unconfigure() {
    this._device = null;
    this._configuration = null;
    this._expireCurrentTexture();
    // 크기 캐시도 버린다 — 다음 configure가 새 크기를 다시 읽게 한다.
    canvasSizeCache.delete(this.canvasId);
  }

  /**
   * 마지막으로 준 설정 (아직 없거나 `unconfigure()` 뒤면 `null`).
   * @returns {GPUCanvasConfiguration | null}
   */
  getConfiguration() {
    return this._configuration;
  }

  /**
   * 이번 프레임의 스왑체인 텍스처. 프레임이 끝나면 무효해진다 (브라우저와 같은 규칙).
   * @returns {GPUTexture}
   */
  getCurrentTexture() {
    if (!this._device) {
      const error = new Error('configure()를 먼저 호출해야 한다');
      error.name = 'InvalidStateError';   // 명세가 정한 이름 — 웹 코드가 이걸로 가른다
      throw error;
    }
    // 명세: `[[currentTexture]]`가 있으면 **그대로 돌려준다.** 만료(present·configure·
    // 리사이즈) 전까지는 한 프레임 안에서 몇 번을 불러도 같은 객체다.
    //
    // 호출마다 새 텍스처를 내면 웹 라이브러리가 깨진다 — 한 프레임에 패스를 여러 개 도는
    // 코드(three.js `PostProcessing`)가 뷰를 캐시해 두는데, 그 뷰가 가리키는 텍스처가
    // 매번 달라지기 때문이다.
    if (this._currentTexture) return this._currentTexture;

    const id = this._device._recorder.allocate();
    this._device._recorder.push({ op: 'getCurrentTexture', id, canvas: this.canvasId });
    const info = canvasSizeCache.get(this.canvasId) || this._fetchSize();
    this._currentTexture = new GPUTexture(this._device, id, {
      size: { width: info.width, height: info.height },
      format: this.format,
      frameScoped: true,   // 네이티브가 프레임 끝에 회수 — GC 자동 해제 대상이 아니다
    });
    this._currentSize = { width: info.width, height: info.height };
    return this._currentTexture;
  }

  /**
   * 명세의 "Expire the current texture" — `[[currentTexture]]`를 비운다.
   *
   * 명세가 부르는 자리: **present**, `configure()`, 캔버스 리사이즈. 다음 호출에서
   * 새 드로어블을 받는다.
   */
  _expireCurrentTexture() {
    this._currentTexture = null;
    this._currentSize = null;
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

/**
 * `GPUSupportedFeatures` 흉내 — 웹 코드가 쓰는 것은 `has()`와 순회뿐이다.
 * @param {string[]} names
 * @returns {{has: (name: string) => boolean, size: number, values: () => string[]}}
 */
function makeFeatureSet(names) {
  const list = names.slice();
  return {
    has: (name) => list.indexOf(name) >= 0,
    size: list.length,
    values: () => list.slice(),
  };
}

/**
 * 명세의 `GPUAdapterInfo` — 웹 코드가 GPU 종류로 분기할 때 읽는 표준 이름들.
 *
 * 값을 모르는 자리는 **빈 문자열**이다 (명세 규칙). 지어내면 그 문자열로 분기하는 코드가
 * 잘못된 우회로를 탄다.
 *
 * @param {Record<string, any> | undefined} raw
 * @returns {GPUAdapterInfoView}
 */
function makeAdapterInfo(raw) {
  const source = raw || {};
  return {
    vendor: source.vendor || '',
    architecture: source.architecture || '',
    device: source.device || '',
    description: source.description || '',
    isFallbackAdapter: !!source.isFallbackAdapter,
    subgroupMinSize: source.subgroupMinSize || 0,
    subgroupMaxSize: source.subgroupMaxSize || 0,
  };
}

class GPUAdapter {
  /** @param {WGPUAdapterInfo} info */
  constructor(info) {
    /**
     * 명세 `GPUAdapterInfo`. 아래 `name`·`backend`·`hasUnifiedMemory`는 **이 구현의 추가**로,
     * 명세 이름이 없던 시절부터 있던 것이라 그대로 둔다 (기존 코드가 쓴다).
     */
    this.info = makeAdapterInfo(info.info);
    this.name = info.name;
    this.backend = info.backend;
    this.limits = info.limits || {};
    this.hasUnifiedMemory = info.hasUnifiedMemory;
    /**
     * 기기마다 갈리는 기능 — 웹과 같은 분기(`adapter.features.has('timestamp-query')`)가
     * 동작하도록 `has`만 흉내 낸다 (엔진에 `Set`이 없을 수 있다).
     */
    this.features = makeFeatureSet(info.features || []);
  }

  /**
   * 어댑터가 지원하지 않는 기능을 요구하면 명세대로 **거부**한다 — 조용히 빼고 만들면
   * 이후 `createQuerySet` 같은 호출이 훨씬 먼 곳에서 터진다.
   * @param {{label?: string, requiredFeatures?: string[], requiredLimits?: Record<string, number>}} [descriptor]
   * @returns {Promise<GPUDevice>}
   */
  async requestDevice(descriptor) {
    const required = (descriptor && descriptor.requiredFeatures) || [];
    for (const name of required) {
      if (!this.features.has(name)) {
        throw new Error(
          `requestDevice: 이 어댑터는 '${name}' 기능을 지원하지 않는다 ` +
            `(지원 목록: ${this.features.values().join(', ') || '없음'})`
        );
      }
    }
    return new GPUDevice(this, required);
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
   *
   * 같은 `canvasId`에는 **같은 객체**를 돌려준다 — 브라우저의 `getContext('webgpu')`와 같다.
   * 새로 만들어 주면 한 핸들로 `configure()`하고 다른 핸들로 `unconfigure()`했을 때 설정
   * 상태가 갈라져, 그린다고 생각한 컨텍스트가 실은 설정되지 않은 쪽이 된다.
   *
   * @param {string} canvasId
   * @returns {GPUCanvasContext}
   */
  getCanvasContext(canvasId) {
    let context = canvasContexts.get(canvasId);
    if (!context) {
      context = new GPUCanvasContext(canvasId);
      canvasContexts.set(canvasId, context);
    }
    return context;
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
  // 네이티브 티커는 **하나뿐**이다 — 구독자를 세지 않으면 한 쪽이 멈출 때 다른 쪽까지
  // 같이 죽는다. `installAnimationFrame`의 rAF 펌프가 그 위에 올라가 있어서, 씬이
  // 잠깐 자기 루프를 돌렸다 끄면 rAF가 조용히 영영 멈춘다 (실제로 겪었다).
  frameLoopSubscribers += 1;
  const releaseTicker = () => {
    frameLoopSubscribers -= 1;
    if (frameLoopSubscribers <= 0) {
      frameLoopSubscribers = 0;
      module.stopFrameLoop();
    }
  };

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
    const listener = (frame) => runFrameTick(handler, frame || { timestamp: 0, delta: 16 });
    emitter.addListener('webgpu:frame', listener);
    module.startFrameLoop({ fps });
    let stopped = false;
    return () => {
      if (stopped) return;   // 두 번 불러도 남의 구독을 깎지 않는다
      stopped = true;
      emitter.removeListener('webgpu:frame', listener);
      releaseTicker();
    };
  }

  // GlobalEventEmitter가 없는 환경(테스트 러너 등)용 폴백.
  let previous = Date.now();
  const timer = setInterval(() => {
    const now = Date.now();
    runFrameTick(handler, { timestamp: now, delta: now - previous });
    previous = now;
  }, Math.max(1, Math.round(1000 / fps)));
  let stopped = false;
  return () => {
    if (stopped) return;
    stopped = true;
    clearInterval(timer);
    releaseTicker();
  };
}

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
export function installAnimationFrame(options) {
  const globals = /** @type {Record<string, any>} */ (globalThis);
  if (typeof globals.requestAnimationFrame === 'function') return () => {};

  /** @type {{id: number, callback: (time: number) => void}[]} */
  let scheduled = [];
  /** @type {(() => void) | null} */
  let stopLoop = null;
  let nextId = 1;

  function pump() {
    if (stopLoop) return;
    stopLoop = startFrameLoop(({ timestamp }) => {
      // 이번 틱의 콜백만 실행한다 — 콜백 안에서 예약한 것은 다음 틱이다 (브라우저와 같다).
      const due = scheduled;
      scheduled = [];
      for (const entry of due) entry.callback(timestamp);
      // 아무도 다음 프레임을 예약하지 않았으면 디스플레이 링크를 놓아 준다.
      if (scheduled.length === 0 && stopLoop) {
        stopLoop();
        stopLoop = null;
      }
    }, options);
  }

  globals.requestAnimationFrame = (/** @type {(time: number) => void} */ callback) => {
    const id = nextId++;
    scheduled.push({ id, callback });
    pump();
    return id;
  };
  globals.cancelAnimationFrame = (/** @type {number} */ id) => {
    scheduled = scheduled.filter((entry) => entry.id !== id);
  };

  return () => {
    if (stopLoop) {
      stopLoop();
      stopLoop = null;
    }
    scheduled = [];
    delete globals.requestAnimationFrame;
    delete globals.cancelAnimationFrame;
  };
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

/**
 * 디코딩이 끝난 이미지 — 명세 `ImageBitmap`의 자리다.
 *
 * 픽셀은 **네이티브에 남는다.** JS가 아는 것은 핸들과 크기뿐이라, 큰 이미지를 다뤄도
 * 브리지를 건너는 데이터가 없다.
 */
class GPUImageBitmap {
  /**
   * @param {number} id
   * @param {number} width
   * @param {number} height
   * @param {Recorder | null} recorder
   */
  constructor(id, width, height, recorder) {
    this.id = id;
    this.width = width;
    this.height = height;
    this._recorder = recorder;
    this._closed = false;
  }

  /** 네이티브 픽셀을 버린다 (명세 `ImageBitmap.close()`). */
  close() {
    if (this._closed) return;
    this._closed = true;
    if (this._recorder) this._recorder.push({ op: 'destroy', id: this.id });
  }
}

/**
 * 인코딩된 이미지(PNG·JPEG·HEIC …)를 풀어 텍스처로 올릴 준비를 한다 — 웹의
 * `createImageBitmap()` 자리다.
 *
 * 디코딩은 네이티브(ImageIO)가 한다. JS에서 PNG를 손으로 푸는 것보다 훨씬 빠르고,
 * HEIC처럼 JS 디코더가 없는 형식도 열린다.
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
 * @param {ArrayBuffer | ArrayBufferView | string} source 이미지 바이트, 또는 애셋 이름
 * @param {{flipY?: boolean, premultiplyAlpha?: 'none' | 'premultiply' | 'default',
 *          resizeWidth?: number, resizeHeight?: number}} [options]
 * @returns {Promise<GPUImageBitmap>}
 */
export async function createImageBitmap(source, options) {
  const settings = options || {};
  if (!activeRecorder) {
    throw new Error('createImageBitmap은 디바이스가 만들어진 뒤에만 쓸 수 있다 (requestDevice를 먼저 부를 것)');
  }
  // 핸들은 모듈 공용 카운터에서 낸다 — 어느 디바이스가 활성이든 겹치지 않는다.
  // 레코더는 나중에 `close()`가 destroy를 실어 보낼 스트림으로만 기억한다.
  const recorder = activeRecorder;
  const id = allocateHandle();
  /** @type {{id: number, data?: ArrayBuffer, name?: string, flipY: boolean,
   *   premultiplyAlpha: boolean, resizeWidth?: number, resizeHeight?: number}} */
  const params = {
    id,
    flipY: !!settings.flipY,
    premultiplyAlpha: settings.premultiplyAlpha === 'premultiply',
    resizeWidth: settings.resizeWidth,
    resizeHeight: settings.resizeHeight,
  };
  if (typeof source === 'string') params.name = source;
  else params.data = toArrayBuffer(source);

  const result = await /** @type {Promise<WGPUDecodeImageResult | undefined>} */ (
    new Promise((resolve) => {
      nativeModule().decodeImage(params, resolve);
    })
  );
  if (!result || result.ok === false) {
    const errors = (result && result.errors) || [];
    throw new Error(errors.length ? errors[0].message : '이미지를 디코딩하지 못했다');
  }
  return new GPUImageBitmap(id, result.width, result.height, recorder);
}

export {
  GPUImageBitmap,
  GPUBuffer, GPUTexture, GPUTextureView, GPUSampler, GPUDevice, GPUCanvasContext,
  GPURenderBundle, GPURenderBundleEncoder, GPUQuerySet,
  // `uncapturederror`가 실어 나르는 오류 — `instanceof`로 종류를 가를 때 쓴다.
  GPUError, GPUValidationError, GPUOutOfMemoryError, GPUInternalError,
};
export default gpu;

// ---------------------------------------------------------------------------
// PrimJS 전역 브리지 — 웹 라이브러리(Three.js 등) 이식용
// ---------------------------------------------------------------------------
// 웹 전역(`navigator.gpu`, `performance`, `GPUBufferUsage` …)을 기대하는 라이브러리를 위해
// **네임스페이스가 붙은** 전역을 얹는다. PrimJS + rspeedy 번들에서는 `globalThis.navigator = …`
// 같은 전역 대입이 bare 식별자(`navigator`) 해석에 반영되지 않으므로, 대입만으로는 이식이
// 안 된다 — 번들 설정(`lynx.config.ts`)의 `source.define`이 bare 식별자를 아래 이름으로
// 바꿔치기해야 완성된다. 전체 조리법은 docs/JS-AUTHORING.md §10.
//
// 이름을 lynx*로 좁힌 것은 진짜 전역이 있는 런타임(node 테스트 등)을 덮어쓰지 않기 위해서다.

globalThis.lynxNavigator = { gpu };
globalThis.lynxPerformance =
  typeof performance !== 'undefined' && performance ? performance : { now: () => Date.now() };
globalThis.lynxGPUBufferUsage = GPUBufferUsage;
globalThis.lynxGPUTextureUsage = GPUTextureUsage;
globalThis.lynxGPUShaderStage = GPUShaderStage;
globalThis.lynxGPUColorWrite = GPUColorWrite;
globalThis.lynxGPUMapMode = GPUMapMode;
