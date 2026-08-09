/**
 * Lynx-WebGPU — a WebGPU-shaped JS client.
 *
 * It keeps the same object graph as browser WebGPU on the JS side, but actual calls are **only recorded as
 * commands** and sent to native in one go at `queue.submit()`. Handles (ids) are issued by JS, so object
 * creation does not wait for a native round trip — the bridge crossings per frame are fixed at one.
 *
 * See docs/ARCHITECTURE.md §3 for the design and docs/WEBGPU-API.md for the supported surface.
 */

/* eslint-disable no-bitwise */

// ---------------------------------------------------------------------------
// WebGPU constants (these must match native's OptionSet values exactly)
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
// Shared types — the definitions here go straight into webgpu.d.ts via `npm run types`
// ---------------------------------------------------------------------------

/**
 * An error raised during command execution.
 * @typedef {WGPUErrorPayload} WGPUError
 */

/** @typedef {{width: number, height?: number, depthOrArrayLayers?: number}} GPUExtent3DDict */
/** @typedef {GPUExtent3DDict | number[]} GPUExtent3D */
/** @typedef {{x?: number, y?: number, z?: number}} GPUOrigin3DDict */
/** @typedef {{r: number, g: number, b: number, a: number}} GPUColorDict */
/** @typedef {GPUColorDict | number[]} GPUColor */

/** The byte sequence `writeBuffer`/`writeTexture` accepts. */
/** @typedef {ArrayBuffer | ArrayBufferView | number[]} GPUDataSource */

/**
 * One item in the command stream.
 *
 * The fields are read **by string key** by native's `WGPUCommandInterpreter`. A name mismatch is not
 * caught here, so when adding or changing an op be sure to fix both sides
 * (`.claude/skills/webgpu-command/SKILL.md`).
 * @typedef {Record<string, any>} GPUCommand
 */

// --- Descriptors -----------------------------------------------------------
// Things as deep and wide as a pipeline are left as Record<string, any>. Rather than rewriting the whole
// spec here, it is better to let native's descriptor decoder raise an error with the path attached.

/** @typedef {{size: number, usage: number, mappedAtCreation?: boolean, label?: string}} GPUBufferDescriptor */
/** @typedef {{size: GPUExtent3D, format: string, usage: number, dimension?: string, mipLevelCount?: number, sampleCount?: number, label?: string}} GPUTextureDescriptor */
/** @typedef {{code: string, language?: 'wgsl' | 'msl', label?: string}} GPUShaderModuleDescriptor */
/** @typedef {{entries: Record<string, any>[], label?: string}} GPUBindGroupLayoutDescriptor */
/** @typedef {{bindGroupLayouts: GPUPipelineLayoutSource[], label?: string}} GPUPipelineLayoutDescriptor */
/** @typedef {{layout: GPUBindGroupLayout, entries: {binding: number, resource: any}[], label?: string}} GPUBindGroupDescriptor */
/**
 * `toneMapping.mode: 'extended'` sends values above 1.0 out into the display's headroom (EDR).
 * `format` must then be `'rgba16float'`, and the shader must write **linear** values with no sRGB encoding.
 */
/** @typedef {{device: GPUDevice, format?: string, usage?: number, alphaMode?: 'opaque' | 'premultiplied', colorSpace?: 'srgb' | 'display-p3', toneMapping?: {mode: 'standard' | 'extended'}}} GPUCanvasConfiguration */

/**
 * The stencil behaviour of one face (front/back). The spec default is "do nothing" —
 * `compare: 'always'` plus `'keep'` for all three operations.
 */
/** @typedef {{compare?: GPUCompareFunction, failOp?: GPUStencilOperation, depthFailOp?: GPUStencilOperation, passOp?: GPUStencilOperation}} GPUStencilFaceState */
/** @typedef {'never' | 'less' | 'equal' | 'less-equal' | 'greater' | 'not-equal' | 'greater-equal' | 'always'} GPUCompareFunction */
/** @typedef {'keep' | 'zero' | 'replace' | 'invert' | 'increment-clamp' | 'decrement-clamp' | 'increment-wrap' | 'decrement-wrap'} GPUStencilOperation */

/**
 * `createRenderPipeline`'s `depthStencil`. Both masks default to `0xFFFFFFFF`, and the comparison
 * happens between `(reference & readMask)` and `(the stored value & readMask)`.
 */
/** @typedef {{format: string, depthWriteEnabled?: boolean, depthCompare?: GPUCompareFunction, depthBias?: number, depthBiasSlopeScale?: number, depthBiasClamp?: number, stencilFront?: GPUStencilFaceState, stencilBack?: GPUStencilFaceState, stencilReadMask?: number, stencilWriteMask?: number}} GPUDepthStencilState */

/** @typedef {{type: 'occlusion' | 'timestamp', count: number, label?: string}} GPUQuerySetDescriptor */

/** The spec's `GPUAdapterInfo`. */
/** @typedef {{vendor: string, architecture: string, device: string, description: string, isFallbackAdapter: boolean, subgroupMinSize: number, subgroupMaxSize: number}} GPUAdapterInfoView */

/** A single shader compilation diagnostic (the spec's `GPUCompilationMessage`). */
/** @typedef {{message: string, type: 'error' | 'warning' | 'info', lineNum: number, linePos: number, offset: number, length: number}} GPUCompilationMessage */

/** The value `device.lost` resolves to (in implementations that report loss) — this one stays pending forever. */
/** @typedef {{reason: 'unknown' | 'destroyed', message: string}} GPUDeviceLostInfo */

/**
 * Where to write timestamps at a pass boundary. Either index may be omitted.
 */
/** @typedef {{querySet: GPUQuerySet, beginningOfPassWriteIndex?: number, endOfPassWriteIndex?: number}} GPUPassTimestampWrites */

/**
 * `createRenderBundleEncoder`'s descriptor — **the shape of the pass this bundle will run in**.
 * A `null` in `colorFormats` means "that slot is empty", and trailing `null`s are ignored in comparison.
 * `depthReadOnly`/`stencilReadOnly` declare "this bundle does not use depth/stencil" —
 * to run inside a pass opened with the same names, the bundle must say `true` too.
 */
/** @typedef {{colorFormats: (string | null)[], depthStencilFormat?: string, sampleCount?: number, depthReadOnly?: boolean, stencilReadOnly?: boolean, label?: string}} GPURenderBundleEncoderDescriptor */

/** The layout `createPipelineLayout` accepts — only the id is needed. */
/** @typedef {{id: number}} GPUPipelineLayoutSource */

/** Internal to the `GPUTexture` constructor — a swapchain texture may have no usage/format. */
/** @typedef {{size?: GPUExtent3D, format?: string, usage?: number, dimension?: string, mipLevelCount?: number, sampleCount?: number, textureBindingViewDimension?: string, label?: string, frameScoped?: boolean}} GPUTextureInit */

// ---------------------------------------------------------------------------
// Binary utilities
// ---------------------------------------------------------------------------

/**
 * Turns a TypedArray / ArrayBuffer / number array into an `ArrayBuffer` to put on a command.
 *
 * **A view (TypedArray) must not be put on as is.** Lynx's value converter only recognizes a real
 * `ArrayBuffer` (`isArrayBuffer`) and treats a view as a plain object, turning it into `{"0":1,"1":2,…}` —
 * the kind of breakage that is silent. Byte sequences must **always go through here** before riding a command.
 *
 * A view covering the whole backing buffer passes through with no copy; a view covering only part of it
 * (or one with an offset/count given) is sliced to that range — because `view.buffer` is the whole buffer, not the view.
 *
 * @param {GPUDataSource} source
 * @param {number} [elementOffset] the start position in elements (bytes for a DataView)
 * @param {number} [elementCount] the number of elements. Omitted, it runs to the end
 * @returns {ArrayBuffer}
 */
function toArrayBuffer(source, elementOffset, elementCount) {
  // Always **copy at call time** — it is the WebGPU spec's contract ("the contents of data are
  // copied"). Commands sit in the queue until submit, so putting one on by reference makes the last
  // value overwrite them all when the caller reuses the same array (the common pattern of filling
  // several buffers from one uniform array). The copy cost is negligible next to a bridge crossing (docs/ARCHITECTURE.md §3).
  if (source instanceof ArrayBuffer) return source.slice(0);
  if (ArrayBuffer.isView(source)) {
    // A DataView has no BYTES_PER_ELEMENT — in that case the offset is read as bytes.
    const elementSize = /** @type {{BYTES_PER_ELEMENT?: number}} */ (source).BYTES_PER_ELEMENT || 1;
    const start = source.byteOffset + (elementOffset || 0) * elementSize;
    const length =
      elementCount === undefined ? source.byteLength - (elementOffset || 0) * elementSize
        : elementCount * elementSize;
    const backing = /** @type {ArrayBuffer} */ (source.buffer);
    return backing.slice(start, start + length);
  }
  if (Array.isArray(source)) return /** @type {ArrayBuffer} */ (new Uint8Array(source).buffer);
  throw new TypeError('data must be a TypedArray, an ArrayBuffer or a number array');
}

// ---------------------------------------------------------------------------
// The native contact point
// ---------------------------------------------------------------------------

function nativeModule() {
  const modules =
    typeof NativeModules !== 'undefined' ? NativeModules
      : typeof lynx !== 'undefined' ? lynx.NativeModules : undefined;
  if (!modules || !modules.WebGPU) {
    throw new Error(
      'NativeModules.WebGPU could not be found — check whether the host called LynxWebGPU.register(in:host:)'
    );
  }
  return modules.WebGPU;
}

// ---------------------------------------------------------------------------
// The canvas size cache
// ---------------------------------------------------------------------------

/**
 * canvasId → `{width, height}` (in pixels).
 *
 * `execute`'s response refreshes `canvases` **on every submission**, so reading the size inside a frame
 * causes no synchronous native round trip. A synchronous query (`canvasInfo`) happens exactly once,
 * when the cache is empty (= the first query right after `configure`).
 */
/**
 * The GPU object handle allocator — **the whole module shares one.**
 *
 * The native registry is one per `LynxWebGPUContext` and finds objects **by handle integer alone**
 * (there is no per-device compartment). So a per-device counter would make the second device start
 * issuing from 1 again and **silently overwrite** the first device's objects — drawing into someone
 * else's buffer with no error.
 *
 * Numbers are never given back. Even when `device.destroy()` empties the registry, not reusing them
 * keeps a JS object still holding an old number from later pointing at **someone else's slot**.
 */
/**
 * Whether we are currently **inside a frame loop callback** (a depth, counting nesting).
 *
 * The browser's present point is not `queue.submit()` but **the end of the task** — however many times
 * you submit within one frame, the canvas goes out once, after the callback ends. That is why web
 * libraries submit several times in one frame while reusing the drawable texture view throughout
 * (three.js's `PostProcessing` does exactly that: scene pass → bloom mip chain → output pass).
 *
 * Presenting on every submit would make the first submit send the drawable out and expire its view, so
 * the rest of the frame's passes are rejected wholesale as "no such handle". So **a flush inside a tick
 * defers the present**, and the present happens once when the tick ends (`endFrameTick`).
 */
let frameTickDepth = 0;

/**
 * How many subscribers the frame ticker currently has. There is only one native ticker, so it stops only
 * when the last subscriber lets go — otherwise one side's `stop()` turns off the other (the rAF pump especially).
 */
let frameLoopSubscribers = 0;

/**
 * The recorders that owe a present this tick. When the tick ends, only these present —
 * a tick with no GPU work never crosses the bridge.
 * @type {Set<Recorder>}
 */
const framePresentDebt = new Set();

/**
 * Wraps one frame loop callback. **The present goes out even if the callback throws** —
 * otherwise the screen stays frozen on that frame.
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

/** The end of a tick — the deferred presents all go out here. */
function endFrameTick() {
  if (framePresentDebt.size === 0) return;
  const owing = Array.from(framePresentDebt);
  framePresentDebt.clear();
  for (const recorder of owing) recorder.flush(true, { presentOnly: true });
}

let nextHandle = 1;

/** @returns {number} a fresh handle nobody is using */
function allocateHandle() {
  return nextHandle++;
}

/**
 * The recorder of the most recently created device — used only to decide **which stream** the global
 * `createImageBitmap()` puts `close()`'s `destroy` command on. It has nothing to do with handle numbers
 * (those come from `allocateHandle()` above).
 *
 * An image belongs to the context rather than a device, so whichever stream carries it reaches the same place.
 * @type {Recorder | null}
 */
let activeRecorder = null;

const canvasSizeCache = new Map();

/**
 * canvasId → `GPUCanvasContext`.
 *
 * This matches the browser always returning the same object from `canvas.getContext('webgpu')` — making
 * a new one each time would split the configuration state (`configure`/`unconfigure`) per handle.
 * @type {Map<string, GPUCanvasContext>}
 */
const canvasContexts = new Map();

/** The filter `pushErrorScope` accepts (the spec's `GPUErrorFilter` spellings as they are). */
const ERROR_FILTERS = ['validation', 'out-of-memory', 'internal'];

/**
 * The spec's `GPUError` hierarchy — the object the `uncapturederror` event carries.
 *
 * The spec only requires `message`, but `kind` and `path` are carried along. An error from the command
 * stream knows "which field of which command", and throwing that away makes diagnosis much worse.
 * Web code reads the kind with `instanceof` (or `constructor.name`) — hence the subclasses.
 */
class GPUError {
  /** @param {WGPUError} payload */
  constructor(payload) {
    this.message = payload.message;
    /** Extra information this implementation attaches — not in the spec. */
    this.kind = payload.kind;
    this.path = payload.path;
  }
}
class GPUValidationError extends GPUError {}
class GPUOutOfMemoryError extends GPUError {}
class GPUInternalError extends GPUError {}

/**
 * Folds the four command error kinds into the spec's three `GPUError` subclasses.
 *
 * `unsupported` being a `GPUValidationError` is for the same reason `pushErrorScope('validation')` catches
 * it — in a browser the same code either fails as validation or succeeds, so the app's branching lines up.
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
 * The error the `createRenderPipelineAsync` family rejects with (the spec's `GPUPipelineError`).
 *
 * The spec's `reason` is only `'validation' | 'internal'`, so the four command error kinds fold into those —
 * a shader translation/compilation failure (`backend`) is `internal` and the rest are `validation`.
 * The same rule as the `pushErrorScope` filter correspondence.
 *
 * @param {WGPUError} error
 * @returns {Error & {reason: string}}
 */
/**
 * The slot where the spec says to throw an `OperationError` (argument validation in `getMappedRange`, etc.).
 *
 * Matching the name is the point — web code branches on `error.name`.
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
// The command recorder
// ---------------------------------------------------------------------------

/**
 * A **record-time snapshot** of a value to put on a command.
 *
 * Browser WebGPU serializes arguments at call time — reusing or resetting a descriptor object after the
 * call does not change an already-recorded command. This shim holds commands until flush, so putting a
 * reference on lets mutations in between leak into the stream. three.js's pattern of `reset()`ing a
 * singleton descriptor right after encoding is exactly this case — `copySize` becomes 0 before the flush
 * so **a zero-width copy goes out with no error**, and texture uploads vanish the same way.
 *
 * `ArrayBuffer` (a transfer payload — the shim already made it as a copy) and TypedArrays are left alone.
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
     * A hook that also lets errors uncaught by a scope out through the spec's `uncapturederror` path.
     * The device plugs itself in. It returns `true` if at least one listener took it.
     * @type {((error: WGPUError) => boolean) | null}
     */
    this.uncapturedDispatch = null;
    /**
     * The result functions of the Promises `popErrorScope()` returned — **in the order they were popped**.
     * Native returns the `errorScopes` array in the same order, so they are paired by index.
     * @type {{resolve: (error: WGPUError | null) => void, reject: (reason: Error) => void}[]}
     */
    this.pendingErrorScopes = [];
  }

  /**
   * A new handle id. It comes from the **module-wide counter** — counting per recorder would collide
   * when there are two devices (see the `allocateHandle` comment).
   * @returns {number}
   */
  allocate() {
    return allocateHandle();
  }

  /**
   * Stacks a command **frozen at its record-time value** (see `snapshotValue`) — for device/queue ops
   * this is the call site itself, so reusing a descriptor after the call cannot pollute the stream.
   * @param {GPUCommand} command
   * @returns {GPUCommand}
   */
  push(command) {
    const frozen = snapshotValue(command);
    this.pending.push(frozen);
    return frozen;
  }

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
  flush(present = true, options) {
    const presentOnly = !!(options && options.presentOnly);
    // A frame submission inside a tick defers its present to the tick's end. An internal submission (present=false) is unchanged.
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
      // Even when the bridge call itself fails, waiting Promises must be resolved. Otherwise they stay
      // pending forever, initialization diagnostics hang, and the next pop takes a stale resolver so
      // **the indices go out of step**.
      this.settleErrorScopes([]);
      this.report([{ kind: 'backend', message: `native execution failed: ${(error && /** @type {Error} */ (error).message) || error}` }]);
      return { ok: false, commandCount: commands.length };
    }
    this.settleErrorScopes(result.errorScopes || []);
    if (result.canvases) {
      for (const canvasId in result.canvases) {
        const info = result.canvases[canvasId];
        if (info && typeof info.width === 'number') {
          canvasSizeCache.set(canvasId, { width: info.width, height: info.height });
          // The spec expires the current texture **on a resize too** — pointing at a drawable whose size
          // changed with the old texture makes the next pass draw at the wrong size.
          const context = canvasContexts.get(canvasId);
          if (context && context._currentSize
              && (context._currentSize.width !== info.width
                  || context._currentSize.height !== info.height)) {
            context._expireCurrentTexture();
          }
        }
      }
    }
    // This is where the spec's "presentation" calls "Expire the current texture".
    if (present) {
      for (const context of canvasContexts.values()) context._expireCurrentTexture();
    }
    if (result.ok === false) this.report(result.errors || []);
    return result;
  }

  /**
   * Stacks a `popErrorScope` command and returns the result Promise — **without flushing.**
   *
   * Used when closing several scopes in one batch (asynchronous pipeline creation does this).
   * Native returns the results in pop order, so the call order is the pairing order.
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
  settleErrorScopes(popped) {
    if (this.pendingErrorScopes.length === 0) return;
    const waiting = this.pendingErrorScopes;
    this.pendingErrorScopes = [];
    waiting.forEach((settle, index) => {
      const slot = popped[index];
      if (slot && /** @type {{rejected?: boolean}} */ (slot).rejected) {
        const error = new Error('popErrorScope: there is no open error scope (check that it pairs with a push)');
        error.name = 'OperationError';
        settle.reject(error);
        return;
      }
      settle.resolve(/** @type {WGPUError | null} */ (slot) || null);
    });
  }

  /**
   * Sends an error uncaught by a scope down every registered channel.
   *
   * There are two channels and they receive it **together** — this implementation's `onError` (which even
   * gives the path-tagged text) and the spec's `uncapturederror` (the name web code knows). It falls back
   * to the console only when nobody is listening: no error may vanish silently, and logging to the console while someone listens would double the log.
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
// Resource objects
// ---------------------------------------------------------------------------

/**
 * Automatic release tied to GC — only when the engine supports FinalizationRegistry.
 *
 * A handle is an integer, so JS GC knows nothing of the native object's lifetime. When a wrapper is
 * collected, a destroy command is slipped into the next submission so native releases too. Use commands
 * can only be recorded while the wrapper is alive, so a late-arriving destroy can never run before an earlier use.
 *
 * On an engine without support, such as PrimJS, it quietly turns off — **an explicit destroy() is still
 * the right answer**, and this device is a safety net that picks up what was missed (docs/JS-AUTHORING.md §8).
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
   * @param {number} id the handle JS issued
   * @param {string} [label]
   * @param {boolean} [frameScoped] true if it is a handle native reclaims at the end of the frame
   */
  constructor(device, id, label, frameScoped) {
    this._device = device;
    // Every creation path passes a GPUDevice, so in practice this is always a Recorder. The defensive branch stays.
    this._recorder = /** @type {Recorder} */ (device ? device._recorder : null);
    this.id = id;
    this.label = label || '';
    // Frame-scoped handles (a swapchain texture and its views) are reclaimed by native at frame end — excluded from registration.
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
    /** Whether this is `mappedAtCreation` initial data (whether unmap must record the creation command). */
    this._mappedAtCreation = false;
    /** Whether `mapAsync` is still waiting for a result (the spec's `"pending"` state). */
    this._mapPending = false;
    /**
     * The ranges handed out by `getMappedRange()` — used for the overlap check and `unmap()`'s write-back.
     * @type {{offset: number, length: number, view: ArrayBuffer | null}[]}
     */
    this._mappedRanges = [];
  }

  /**
   * The spec's `GPUBufferMapState` — `'unmapped'` · `'pending'` · `'mapped'`.
   *
   * A buffer being mapped is rejected by queue operations, so code that wants to reuse it must be able to
   * learn the state **without asking**. Without this it sees `undefined` and misreads it as "not mapped".
   *
   * @returns {'unmapped' | 'pending' | 'mapped'}
   */
  get mapState() {
    if (this._mapped) return 'mapped';
    return this._mapPending ? 'pending' : 'unmapped';
  }

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
  getMappedRange(offset, size) {
    if (!this._mapped) {
      throw new Error('getMappedRange can only be used after mappedAtCreation or mapAsync');
    }
    const total = this._mapped.byteLength;
    const start = offset || 0;
    const length = size === undefined ? Math.max(0, total - start) : size;

    if (start % 8 !== 0) {
      throw new OperationError(`getMappedRange: offset must be a multiple of 8 (got ${start})`);
    }
    // The alignment check applies **only to an explicit value**. A browser's mapping is always a multiple
    // of 4, but here the mapping size is the native buffer size, so even a 3-byte one is normal —
    // requiring a multiple of 4 when omitted (= the whole mapping) would make such a buffer unreadable.
    if (size !== undefined && length % 4 !== 0) {
      throw new OperationError(`getMappedRange: size must be a multiple of 4 (got ${length})`);
    }
    if (start < 0 || start + length > total) {
      throw new OperationError(
        `the getMappedRange range exceeds the mapping — offset ${start} + ${length}B > ${total}B`
      );
    }
    for (const range of this._mappedRanges) {
      if (start < range.offset + range.length && range.offset < start + length) {
        throw new OperationError(
          `the getMappedRange range overlaps a previously obtained one (${range.offset}~${range.offset + range.length})`
        );
      }
    }

    // Requesting the whole thing first hands over the mapping itself — there is nothing to write back.
    if (start === 0 && length === total && this._mappedRanges.length === 0) {
      this._mappedRanges.push({ offset: 0, length, view: null });
      return this._mapped;
    }
    const view = this._mapped.slice(start, start + length);
    this._mappedRanges.push({ offset: start, length, view });
    return view;
  }

  /** Writes what was written into the range copies back into the mapping (just before `unmap`). */
  _flushMappedRanges() {
    if (!this._mapped) return;
    for (const range of this._mappedRanges) {
      if (!range.view) continue;
      new Uint8Array(this._mapped).set(new Uint8Array(range.view), range.offset);
    }
    this._mappedRanges = [];
  }

  /**
   * Unmaps.
   *
   * For `mappedAtCreation`, this is where the actual creation command (initial data included) is recorded;
   * for something mapped by `mapAsync` it tells native "queue operations may use it now" —
   * a buffer being mapped is rejected by queue operations per the spec, so **this must be called once you have read.**
   *
   * @returns {void}
   */
  unmap() {
    if (!this._mapped) return;
    // The writes into the range copies are put back into the mapping before sending — otherwise they vanish silently.
    this._flushMappedRanges();
    if (this._mappedAtCreation) {
      this._recorder.push({
        op: 'createBuffer',
        id: this.id,
        size: this.size,
        usage: this.usage,
        label: this.label,
        data: this._mapped,   // already an ArrayBuffer
      });
      this._mappedAtCreation = false;
    } else {
      this._recorder.push({ op: 'unmapBuffer', buffer: this.id });
    }
    this._mapped = null;
  }

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
      throw new Error('buffer read failed');
    }
    // When native returns `Data`, Lynx converts it to an ArrayBuffer — there is nothing to decode.
    const mapped = result.data;
    this._mapped = mapped;
    this._mappedAtCreation = false;
    // It is a new mapping, so the previous mapping's range records are dropped (so the overlap check does not spin on stale data).
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
    // size accepts both a dict ({width,…}) and an array ([w,h,…]) — it is read by duck typing as is.
    /** @type {any} */
    const size = descriptor && descriptor.size;
    this.width = size ? size.width || size[0] : 0;
    this.height = size ? size.height || size[1] || 1 : 0;
    // The spec's read-only properties — web code reads them when it takes a texture and decides for itself
    // (three.js looks at `textureBindingViewDimension` on the mipmap path).
    this.depthOrArrayLayers = size ? size.depthOrArrayLayers || size[2] || 1 : 1;
    this.mipLevelCount = (descriptor && descriptor.mipLevelCount) || 1;
    this.sampleCount = (descriptor && descriptor.sampleCount) || 1;
    this.dimension = (descriptor && descriptor.dimension) || '2d';
    this.format = descriptor && descriptor.format;
    this.usage = (descriptor && descriptor.usage) || 0;
    /**
     * The default view dimension when binding this texture. The spec allows omitting it, in which case it
     * is derived from `dimension` and the layer count (2d with 2 or more layers gives `2d-array`).
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
    // The views of a swapchain texture are frame-scoped too — native reclaims them together at frame end.
    return new GPUTextureView(this._device, id, descriptor && descriptor.label, this._frameScoped);
  }
}

class GPUTextureView extends GPUObjectBase {}
class GPUSampler extends GPUObjectBase {}
class GPUShaderModule extends GPUObjectBase {
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
  async getCompilationInfo() {
    // If there are unsent commands, this module is not in native yet — flush first (a mid-frame submission).
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
   * Pulls out a bind group layout derived by a `layout: 'auto'` pipeline.
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
 * The reusable bundle of draws `bundleEncoder.finish()` returns.
 *
 * It is the only structure whose recorded commands **outlive the wrappers**, so it holds on to the resource
 * wrappers it uses (`_retained`). Otherwise, when an initialization function returns only the bundle and
 * drops the pipelines and buffers, GC slips in a `destroy` and the bundle **quietly draws nothing** (`docs/JS-AUTHORING.md` §8).
 */
class GPURenderBundle extends GPUObjectBase {
  /**
   * @param {GPUDevice} device
   * @param {number} id
   * @param {string} [label]
   */
  constructor(device, id, label) {
    super(device, id, label);
    /** @type {object[]} the resource wrappers this bundle references — their lifetimes are tied together. */
    this._retained = [];
  }
}

/** The query store `device.createQuerySet()` returns. */
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
// The command encoder
// ---------------------------------------------------------------------------

/** An encoder gathers its own commands separately, then they merge into the stream at `finish()` → `submit()`. */
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
     * The resource wrappers met while recording — **only the bundle encoder** fills this.
     *
     * A bundle is the only structure whose recorded commands outlive the wrappers, so on an engine with
     * automatic release (GC), an initialization function returning only the bundle and dropping the
     * pipeline and buffer wrappers makes the bundle quietly draw nothing. In the spec model a bundle
     * **owns** the objects it uses, so the same ownership is built here.
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
  pushDebugGroup(groupLabel) {
    this._commands.push({ op: 'pushDebugGroup', groupLabel: String(groupLabel) });
  }

  /** @returns {void} */
  popDebugGroup() {
    this._commands.push({ op: 'popDebugGroup' });
  }

  /**
   * Leaves a marker at a single point (Metal `insertDebugSignpost`) — a point event, not a range.
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
 * The commands a render pass and a render bundle can **both** use.
 *
 * This boundary is the spec's — a bundle cannot carry a viewport, scissor, blend constant, stencil
 * reference or a nested bundle. Keeping those on `GPURenderPassEncoder` alone means the bundle encoder
 * never has those methods, so code using them wrongly never reaches native.
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
  drawIndirect(indirectBuffer, indirectOffset) {
    this._retain(indirectBuffer);
    this._commands.push({
      op: 'drawIndirect', indirectBuffer: indirectBuffer.id, indirectOffset: indirectOffset || 0,
    });
  }

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
  drawIndexedIndirect(indirectBuffer, indirectOffset) {
    this._retain(indirectBuffer);
    this._commands.push({
      op: 'drawIndexedIndirect',
      indirectBuffer: indirectBuffer.id,
      indirectOffset: indirectOffset || 0,
    });
  }
}

/** Pass-only commands — these four and `executeBundles` cannot go in a bundle (per the spec). */
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
   * Begins counting the samples these draws let through.
   *
   * It can only be used in a pass given an `occlusionQuerySet` in `beginRenderPass`, and cannot nest.
   * The same index cannot be used twice in one pass, and it must be closed with `endOcclusionQuery`
   * before the pass is closed.
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
 * Records a bundle of draws to reuse across many frames (`device.createRenderBundleEncoder`).
 *
 * A bundle's benefit here differs from a browser's — a browser builds driver commands ahead of time,
 * whereas here the benefit is **that JS does not have to rebuild the same command array every frame**.
 * The command that runs a bundle is a single handle, and the replay is native's job.
 */
class GPURenderBundleEncoder extends GPURenderCommandsBase {
  /**
   * @param {GPUDevice} device
   * @param {GPURenderBundleEncoderDescriptor} descriptor
   */
  constructor(device, descriptor) {
    super([]);
    /** @type {GPUCommand[]} a bundle encoder gathers into its own array only (it does not mix with the pass stream). */
    this._commands = [];
    /** @type {object[]} the wrappers met while recording — the finished bundle inherits and holds them. */
    this._retained = [];
    this._finished = false;
    this._device = device;
    // **Frozen at its record-time value.** This descriptor is held until `finish()`, and the caller often
    // passes a singleton and `reset()`s it immediately (three.js's `createBundleEncoder` does exactly
    // that). Holding the reference would build the bundle with an empty `colorFormats` and get it rejected
    // as "there are no attachments" — with the cause surfacing much later, at `executeBundles`.
    this._descriptor = snapshotValue(descriptor) || { colorFormats: [] };
  }

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
  finish(descriptor) {
    const recorder = this._device._recorder;
    const id = recorder.allocate();
    const label = (descriptor && descriptor.label) || this._descriptor.label;
    if (this._finished) {
      recorder.report([{
        kind: 'validation',
        message: 'GPURenderBundleEncoder.finish() can only be called once (this encoder is already finished)',
      }]);
      // It is not created natively, so using this handle is rejected as "does not exist" (= invalid).
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
    // The bundle holds on to the resources it uses — the same ownership as the spec's `[[used_bind_groups]]`.
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
   * Dispatches by reading the workgroup counts from a GPU buffer.
   *
   * The buffer must contain 3 `u32`s (`x, y, z`). It must be created with `GPUBufferUsage.INDIRECT`,
   * and `indirectOffset` must be a multiple of 4.
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
        // So the record does not change even when the caller reuses the clearValue object every frame (the three.js pattern).
        clearValue: snapshotValue(attachment.clearValue),
      })
    );
    /** @type {GPUCommand} */
    const command = { op: 'beginRenderPass', colorAttachments, label: descriptor.label };
    if (descriptor.depthStencilAttachment) {
      const depth = descriptor.depthStencilAttachment;
      command.depthStencilAttachment = { ...depth, view: depth.view.id };
    }
    // A query can only be attached when opening a pass (the same constraint in Metal and WebGPU).
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
  copyBufferToBuffer(source, destinationOrSourceOffset, sizeOrDestination, destinationOffset, size) {
    let sourceOffset = 0;
    /** @type {GPUBuffer} */
    let destination;
    /** @type {number | undefined} */
    let byteLength;

    if (destinationOrSourceOffset && /** @type {GPUBuffer} */ (destinationOrSourceOffset).id !== undefined) {
      // The short form — (source, destination, size?)
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
    // Encoder commands wait until submit — so they are frozen by value here, so the record does not change
    // if the caller reuses or resets the descriptor in between (see snapshotValue; the same for the copy family below).
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
  pushDebugGroup(groupLabel) {
    this._commands.push({ op: 'pushDebugGroup', groupLabel: String(groupLabel) });
  }

  /** @returns {void} */
  popDebugGroup() {
    this._commands.push({ op: 'popDebugGroup' });
  }

  /**
   * Leaves a marker at a single point (Metal `insertDebugSignpost`) — a point event, not a range.
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
// Queue / device
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
   * @param {number} [dataOffset] the start position in elements
   * @param {number} [size] the number of elements
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
   * Merges the commands the encoders gathered into the stream and sends them to native **in one go**.
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
   * @param {string[]} [requiredFeatures] the feature names already validated in `requestDevice()`
   */
  constructor(adapter, requiredFeatures) {
    this.adapter = adapter;
    this.limits = adapter.limits;
    /** The spec's `GPUDevice.adapterInfo` — it looks straight at the adapter's. */
    this.adapterInfo = adapter.info;
    /**
     * The features enabled on this device — per the spec it holds **only what was requested** (even if the
     * adapter supports it, `has()` is false unless it was asked for via `requiredFeatures`). Web code such
     * as three.js uses the pattern of requesting exactly what it picked from the adapter and re-checking here.
     */
    this.features = makeFeatureSet(requiredFeatures || []);
    /**
     * Device loss notification (the counterpart of `device.lost.then(...)`).
     *
     * This implementation does not report loss, so it stays **pending forever** — the scenarios where a
     * Metal device disappears (an eGPU being unplugged, say) do not exist on iOS, and when the process
     * dies JS dies with it. The property has to exist for bootstraps like `WebGPUBackend.init()` to pass without a TypeError.
     * @type {Promise<GPUDeviceLostInfo>}
     */
    this.lost = new Promise(() => {});
    /**
     * Errors uncaught by a scope (the spec's `GPUDevice.onuncapturederror`).
     *
     * The name web code knows, so it is kept as is — three.js assigns to it and forwards to
     * `renderer.onError`. The value received is `{type, error}`, where `error` is of the `GPUValidationError` family.
     * @type {((event: {type: string, error: GPUError}) => void) | null}
     */
    this.onuncapturederror = null;
    /** @type {((event: {type: string, error: GPUError}) => void)[]} */
    this._uncapturedListeners = [];
    this._recorder = new Recorder();
    // The recorder the global `createImageBitmap()` will use. Handles cannot be issued without a device.
    activeRecorder = this._recorder;
    this._recorder.uncapturedDispatch = (error) => this._dispatchUncaptured(error);
    this.queue = new GPUQueue(this);
  }

  /**
   * The command execution error handler (`{kind, message, path}`). Unregistered, errors go to console.error.
   *
   * It works **together with** the spec's `onuncapturederror` — register both and both receive.
   * This one even gives the finished text with the path attached, which is handier for diagnosis.
   *
   * @param {(error: WGPUError, text: string) => void} handler
   * @returns {void}
   */
  onError(handler) {
    this._recorder.errorHandlers.push(handler);
  }

  /**
   * Registers an `uncapturederror` listener (the spec's `EventTarget` slot — this event only).
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
   * Sends one error down the `uncapturederror` channel. `true` if at least one place took it.
   *
   * **The remaining listeners and the next errors keep going** even if a listener throws — one mistake
   * swallowing the whole report would make the very error that caused it disappear, and diagnosis impossible.
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
        console.error(`an uncapturederror listener threw: ${(error && /** @type {Error} */ (error).message) || error}`);
      }
    }
    return listeners.length > 0;
  }

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
  pushErrorScope(filter) {
    if (ERROR_FILTERS.indexOf(filter) === -1) {
      throw new TypeError(
        `pushErrorScope: filter must be one of ${ERROR_FILTERS.join(' / ')} (got ${String(filter)})`
      );
    }
    this._recorder.push({ op: 'pushErrorScope', filter });
  }

  /**
   * Closes the innermost scope and returns **the first error caught there** (`null` if none).
   *
   * It **submits immediately**, like `mapAsync` — otherwise the Promise would not resolve until the next
   * `submit()`. So calling it inside a frame loop adds a round trip.
   * It is an API meant for initialization and diagnostic paths (`docs/JS-AUTHORING.md` §5).
   *
   * @returns {Promise<WGPUError | null>}
   */
  popErrorScope() {
    const promise = this._recorder.recordPop();
    // The mid-frame submission marker — an acquired canvas texture is not presented (see flush).
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
      // The creation command is recorded in unmap(), together with the initial data.
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
  createRenderPipelineAsync(descriptor) {
    return this._createPipelineAsync(() => this.createRenderPipeline(descriptor));
  }

  /**
   * The asynchronous form of `createComputePipeline` (the same contract as `createRenderPipelineAsync`).
   * @param {Record<string, any>} descriptor
   * @returns {Promise<GPUComputePipeline>}
   */
  createComputePipelineAsync(descriptor) {
    return this._createPipelineAsync(() => this.createComputePipeline(descriptor));
  }

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
  async _createPipelineAsync(record) {
    this.pushErrorScope('validation');
    this.pushErrorScope('internal');
    const pipeline = record();
    // Closed from the inside (internal) out — native returns the results in pop order.
    const internalPromise = this._recorder.recordPop();
    const validationPromise = this._recorder.recordPop();
    this._recorder.flush(false);

    const [internalError, validationError] = await Promise.all([internalPromise, validationPromise]);
    const failure = internalError || validationError;
    if (failure) {
      // No unusable handle is left behind — there is no object in native to begin with.
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
   * Begins recording a bundle of draws to reuse across many frames.
   *
   * `colorFormats` (and `depthStencilFormat`/`sampleCount` if present) are **the shape of the pass this
   * bundle will run in**. A mismatch with the actual pass raises an error at `executeBundles`.
   *
   * @param {GPURenderBundleEncoderDescriptor} descriptor
   * @returns {GPURenderBundleEncoder}
   */
  createRenderBundleEncoder(descriptor) {
    return new GPURenderBundleEncoder(this, descriptor);
  }

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
  createQuerySet(descriptor) {
    const id = this._recorder.allocate();
    this._recorder.push({ op: 'createQuerySet', id, ...descriptor });
    return new GPUQuerySet(this, id, descriptor);
  }

  /** Throws away every GPU object (called when leaving the page). @returns {void} */
  destroy() {
    this._recorder.pending = [];
    // Leaving waiting popErrorScope Promises alone means they never resolve — they are closed with null.
    // As things stand `flush()` resolves the scopes however it ends (success or failure), so nothing gets
    // this far. It stays only as a last safety net.
    this._recorder.settleErrorScopes([]);
    canvasSizeCache.clear();
    nativeModule().reset();
  }
}

/**
 * Turns timestamp write slots into the shape to put on a command — the query set becomes a handle.
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
 * @param {any} resource a GPUBuffer, GPUSampler, GPUTextureView or `{buffer, offset?, size?}`
 * @returns {Record<string, any>}
 */
function serializeBindingResource(resource) {
  if (resource instanceof GPUSampler) return { sampler: resource.id };
  if (resource instanceof GPUTextureView) return { textureView: resource.id };
  if (resource && resource.buffer) {
    return { buffer: resource.buffer.id, offset: resource.offset || 0, size: resource.size };
  }
  if (resource instanceof GPUBuffer) return { buffer: resource.id };
  throw new TypeError('a binding resource must be a GPUBuffer, a GPUSampler or a GPUTextureView');
}

// ---------------------------------------------------------------------------
// The canvas context
// ---------------------------------------------------------------------------

class GPUCanvasContext {
  /** @param {string} canvasId the value of `<webgpu-canvas canvas-id="…">` */
  constructor(canvasId) {
    this.canvasId = canvasId;
    /** @type {GPUDevice | null} */
    this._device = null;
    this.format = 'bgra8unorm';
    /**
     * The last configuration `getConfiguration()` will return — `null` until there is one (as in the spec).
     * @type {GPUCanvasConfiguration | null}
     */
    this._configuration = null;
    /**
     * The spec's `[[currentTexture]]` — **the same object** `getCurrentTexture()` returns until it expires.
     * @type {GPUTexture | null}
     */
    this._currentTexture = null;
    /** The canvas size at the time that texture was taken — used to detect resize expiry. */
    this._currentSize = null;
  }

  /**
   * @param {GPUCanvasConfiguration} configuration
   * @returns {void}
   */
  configure(configuration) {
    // Spec: `configure()` calls "Expire the current texture" — it is taken again under the new configuration.
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
    // The size is cached ahead of time — after this the submission response refreshes it, so in practice this is the only synchronous query.
    this._fetchSize();
  }

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
  unconfigure() {
    this._device = null;
    this._configuration = null;
    this._expireCurrentTexture();
    // The size cache is dropped too — so the next configure reads the new size again.
    canvasSizeCache.delete(this.canvasId);
  }

  /**
   * The last configuration given (`null` if there is none yet or after `unconfigure()`).
   * @returns {GPUCanvasConfiguration | null}
   */
  getConfiguration() {
    return this._configuration;
  }

  /**
   * This frame's swapchain texture. It becomes invalid once the frame ends (the same rule as a browser).
   * @returns {GPUTexture}
   */
  getCurrentTexture() {
    if (!this._device) {
      const error = new Error('configure() must be called first');
      error.name = 'InvalidStateError';   // the name the spec fixed — web code branches on it
      throw error;
    }
    // Spec: if there is a `[[currentTexture]]`, **return it as is.** Until it expires (present, configure
    // or resize) it is the same object however many times it is called within one frame.
    //
    // Handing out a new texture per call breaks web libraries — code that runs several passes in one
    // frame (three.js `PostProcessing`) caches the view, and the texture that view points at would differ
    // every time.
    if (this._currentTexture) return this._currentTexture;

    const id = this._device._recorder.allocate();
    this._device._recorder.push({ op: 'getCurrentTexture', id, canvas: this.canvasId });
    const info = canvasSizeCache.get(this.canvasId) || this._fetchSize();
    this._currentTexture = new GPUTexture(this._device, id, {
      size: { width: info.width, height: info.height },
      format: this.format,
      frameScoped: true,   // native reclaims it at frame end — not a target for automatic GC release
    });
    this._currentSize = { width: info.width, height: info.height };
    return this._currentTexture;
  }

  /**
   * The spec's "Expire the current texture" — it empties `[[currentTexture]]`.
   *
   * Where the spec calls it: **present**, `configure()`, and a canvas resize. The next call takes a new drawable.
   */
  _expireCurrentTexture() {
    this._currentTexture = null;
    this._currentSize = null;
  }

  /**
   * The canvas's current pixel size.
   *
   * It reads a cache refreshed by the submission (`submit`) response, so calling it inside a frame costs no
   * round trip. Right after a resize with no submission yet, it may be one frame stale — if you need
   * immediacy, use `<webgpu-canvas>`'s `bindcanvasresize` event.
   *
   * @returns {{width: number, height: number}}
   */
  getSize() {
    const cached = canvasSizeCache.get(this.canvasId);
    if (cached) return { width: cached.width, height: cached.height };
    return this._fetchSize();
  }

  /**
   * A synchronous native query — used only when the cache is empty.
   * @returns {{width: number, height: number}}
   */
  _fetchSize() {
    const info = nativeModule().canvasInfo({ canvas: this.canvasId }) || {};
    const size = { width: info.width || 0, height: info.height || 0 };
    // If the surface is not registered yet (size 0) it is not cached — the next query tries again.
    if (info.ok !== false && size.width > 0 && size.height > 0) {
      canvasSizeCache.set(this.canvasId, { width: size.width, height: size.height });
    }
    return size;
  }
}

// ---------------------------------------------------------------------------
// The entry point (the counterpart of navigator.gpu)
// ---------------------------------------------------------------------------

/**
 * A stand-in for `GPUSupportedFeatures` — all web code uses is `has()` and iteration.
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
 * The spec's `GPUAdapterInfo` — the standard names web code reads when branching on GPU kind.
 *
 * A slot whose value is unknown is **the empty string** (a spec rule). Inventing one sends code that
 * branches on that string down the wrong path.
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
     * The spec's `GPUAdapterInfo`. The `name`, `backend` and `hasUnifiedMemory` below are **this
     * implementation's additions**, present since before the spec had names, and kept as is (existing code uses them).
     * Keys outside the spec are **optional** — another runtime (Dawn, say) may not fill them, and they are
     * filled in harmlessly here (the grade table in `docs/COMMAND-STREAM.md` §5).
     */
    this.info = makeAdapterInfo(info.info);
    this.name = info.name || this.info.description || '';
    this.backend = info.backend || '';
    this.limits = info.limits || {};
    this.hasUnifiedMemory = !!info.hasUnifiedMemory;
    /**
     * Device-dependent features — only `has` is imitated so that the same branch as on the web
     * (`adapter.features.has('timestamp-query')`) works (the engine may not have `Set`).
     */
    this.features = makeFeatureSet(info.features || []);
  }

  /**
   * Requiring a feature the adapter does not support **rejects** per the spec — quietly dropping it and
   * creating anyway would blow up much further away, at a later call like `createQuerySet`.
   * @param {{label?: string, requiredFeatures?: string[], requiredLimits?: Record<string, number>}} [descriptor]
   * @returns {Promise<GPUDevice>}
   */
  async requestDevice(descriptor) {
    const required = (descriptor && descriptor.requiredFeatures) || [];
    for (const name of required) {
      if (!this.features.has(name)) {
        throw new Error(
          `requestDevice: this adapter does not support the '${name}' feature ` +
            `(supported: ${this.features.values().join(', ') || 'none'})`
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
   * The format that best fits a canvas surface. The same as `<webgpu-canvas>`'s default.
   * @returns {string}
   */
  getPreferredCanvasFormat() {
    return 'bgra8unorm';
  },

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
 * Calls the callback in step with the display refresh (the counterpart of `requestAnimationFrame`).
 *
 * A native CADisplayLink drives it, so frames are more even than with `setInterval`.
 * Calling the returned function stops the loop — it must be called when leaving the page.
 *
 * @param {(frame: {timestamp: number, delta: number}) => void} handler
 * @param {{fps?: number}} [options]
 * @returns {() => void} the stop function
 */
export function startFrameLoop(handler, options) {
  const fps = (options && options.fps) || 60;
  const module = nativeModule();
  // There is **only one** native ticker — without counting subscribers, one stopping kills the other with
  // it. `installAnimationFrame`'s rAF pump sits on top of this, so a scene briefly running its own loop
  // and turning it off silently stops rAF forever (experienced first hand).
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
    // The global itself may be absent — checked the same way as `nativeModule()`.
    // (getJSModule itself can throw too, so the try stays.)
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
      if (stopped) return;   // calling twice does not eat someone else's subscription
      stopped = true;
      emitter.removeListener('webgpu:frame', listener);
      releaseTicker();
    };
  }

  // A fallback for environments with no GlobalEventEmitter (a test runner and the like).
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
      // Only this tick's callbacks run — what a callback schedules belongs to the next tick (as in a browser).
      const due = scheduled;
      scheduled = [];
      for (const entry of due) entry.callback(timestamp);
      // If nobody scheduled the next frame, the display link is let go.
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
export async function loadAsset(name) {
  const result = await /** @type {Promise<WGPULoadAssetResult | undefined>} */ (
    new Promise((resolve) => {
      nativeModule().loadAsset({ name }, resolve);
    })
  );
  if (!result || result.ok === false) {
    const errors = (result && result.errors) || [];
    throw new Error(
      errors.length ? errors[0].message : `could not read the asset '${name}'`
    );
  }
  return result.data;
}

/**
 * A decoded image — the slot the spec's `ImageBitmap` takes.
 *
 * The pixels **stay native.** All JS knows is the handle and the size, so no data crosses the bridge even
 * for a large image.
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

  /** Throws away the native pixels (the spec's `ImageBitmap.close()`). */
  close() {
    if (this._closed) return;
    this._closed = true;
    if (this._recorder) this._recorder.push({ op: 'destroy', id: this.id });
  }
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
export async function createImageBitmap(source, options) {
  const settings = options || {};
  if (!activeRecorder) {
    throw new Error('createImageBitmap can only be used after a device has been created (call requestDevice first)');
  }
  // The handle comes from the module-wide counter — it never collides whichever device is active.
  // The recorder is only remembered as the stream `close()` will later put the destroy on.
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
    throw new Error(errors.length ? errors[0].message : 'could not decode the image');
  }
  return new GPUImageBitmap(id, result.width, result.height, recorder);
}

export {
  GPUImageBitmap,
  GPUBuffer, GPUTexture, GPUTextureView, GPUSampler, GPUDevice, GPUCanvasContext,
  GPURenderBundle, GPURenderBundleEncoder, GPUQuerySet,
  // The errors `uncapturederror` carries — used when telling the kinds apart with `instanceof`.
  GPUError, GPUValidationError, GPUOutOfMemoryError, GPUInternalError,
};
export default gpu;

// ---------------------------------------------------------------------------
// The PrimJS global bridge — for porting web libraries (three.js and the like)
// ---------------------------------------------------------------------------
// For libraries that expect web globals (`navigator.gpu`, `performance`, `GPUBufferUsage` …), **namespaced**
// globals are installed. In a PrimJS + rspeedy bundle, a global assignment like `globalThis.navigator = …`
// is not reflected in bare identifier (`navigator`) resolution, so assignment alone does not port anything —
// it is completed by the bundle configuration's (`lynx.config.ts`) `source.define` swapping bare identifiers
// for the names below. The full recipe is in docs/JS-AUTHORING.md §10.
//
// The names are narrowed to lynx* so as not to overwrite a runtime that has the real globals (a node test, say).

globalThis.lynxNavigator = { gpu };
globalThis.lynxPerformance =
  typeof performance !== 'undefined' && performance ? performance : { now: () => Date.now() };
globalThis.lynxGPUBufferUsage = GPUBufferUsage;
globalThis.lynxGPUTextureUsage = GPUTextureUsage;
globalThis.lynxGPUShaderStage = GPUShaderStage;
globalThis.lynxGPUColorWrite = GPUColorWrite;
globalThis.lynxGPUMapMode = GPUMapMode;
