/**
 * Declarations of the **host runtime surface** the shim expects.
 *
 * This file has two purposes:
 * 1. It pins the signatures of `NativeModules.WebGPU` so typos and missing arguments are caught at the
 *    bridge boundary. The declarations here **must match** `methodLookup` in
 *    `Sources/LynxWebGPUBridge/WebGPUNativeModule.swift` — fixing one side only breaks quietly at runtime.
 * 2. It does not turn on the DOM lib. The shim runs on PrimJS rather than a browser, so using a global
 *    not written here (`window`, `fetch`, `btoa`, …) is caught by type checking.
 */

/** An error raised during command execution. */
interface WGPUErrorPayload {
  kind: 'validation' | 'out-of-memory' | 'unsupported' | 'backend';
  message: string;
  path?: string;
  /**
   * The WGSL source line number (1-based), attached to shader errors only.
   * `getCompilationInfo()`'s `lineNum` uses this value as is — `docs/COMMAND-STREAM.md` §2-1.
   */
  line?: number;
}

/** What `execute` returns. */
interface WGPUExecuteResult {
  ok: boolean;
  commandCount?: number;
  errors?: WGPUErrorPayload[];
  /** The pixel size of the canvases this submission touched — the size cache is refreshed from this. */
  canvases?: Record<string, { width: number; height: number }>;
  /** The number of GPU objects alive natively — growing every frame means a missing destroy. */
  objects?: number;
  /**
   * The results of the scopes closed by `popErrorScope` in this batch — **in the order they were popped**.
   * The shim resolves Promises in that order. There are three kinds of slot (`docs/COMMAND-STREAM.md` §2):
   * `null` (clean) · an error object (caught) · `{rejected: true}` (unpaired with a `push`, so it **rejects**).
   */
  errorScopes?: (WGPUErrorPayload | { rejected: true } | null)[];
}

/** What `shaderCompilationInfo` returns. */
interface WGPUShaderCompilationInfo {
  ok?: boolean;
  messages?: {
    message: string;
    type: 'error' | 'warning' | 'info';
    lineNum: number;
    linePos: number;
    offset: number;
    length: number;
  }[];
  errors?: WGPUErrorPayload[];
}

/** What `adapterInfo` returns. */
interface WGPUAdapterInfo {
  ok?: boolean;
  /** An extension — the display name. Being outside the spec, a runtime may omit it (the shim fills in from `info.description`). */
  name?: string;
  /** An extension — the backend name (`'metal'` and the like). Being outside the spec, a runtime may omit it. */
  backend?: string;
  limits?: Record<string, number>;
  /** An extension — whether memory is unified. A runtime that does not know omits it (the shim reads that as false). */
  hasUnifiedMemory?: boolean;
  /** The spec's `GPUAdapterInfo` (vendor/architecture/device/description …). */
  info?: Record<string, any>;
  /** The spec spellings of device-dependent features (`'timestamp-query'` and the like). */
  features?: string[];
}

/** What `canvasInfo` returns. */
interface WGPUCanvasInfo {
  ok?: boolean;
  width?: number;
  height?: number;
  format?: string;
}

/**
 * The value the `readBuffer` callback receives.
 *
 * `data` is what native returned as `Data`, converted to an `ArrayBuffer` by Lynx
 * (not a base64 string — see `LynxWebGPUContext.readBuffer`).
 */
interface WGPUReadBufferResult {
  ok?: boolean;
  data: ArrayBuffer;
  byteLength?: number;
  errors?: WGPUErrorPayload[];
}

/**
 * The value the `loadAsset` callback receives.
 *
 * The same convention as `readBuffer` — `data` is what native returned as `Data`, converted to an
 * `ArrayBuffer` by Lynx.
 */
interface WGPULoadAssetResult {
  ok?: boolean;
  data: ArrayBuffer;
  byteLength?: number;
  errors?: WGPUErrorPayload[];
}

/** The value the `createImageBitmap` callback receives — the pixels stay native and only the size comes across. */
interface WGPUDecodeImageResult {
  ok?: boolean;
  width: number;
  height: number;
  errors?: WGPUErrorPayload[];
}

/** `NativeModules.WebGPU` — the entrance to the command stream. */
interface WebGPUNativeModule {
  /** `present: false` = an internal mid-frame submission — it defers the drawable present and handle expiry. */
  execute(payload: { commands: Record<string, any>[]; present?: boolean }): WGPUExecuteResult | undefined;
  adapterInfo(): WGPUAdapterInfo | undefined;
  shaderCompilationInfo(params: { module: number }): WGPUShaderCompilationInfo | undefined;
  canvasInfo(params: { canvas: string }): WGPUCanvasInfo | undefined;
  readBuffer(
    params: { buffer: number; offset: number; size?: number },
    callback: (result: WGPUReadBufferResult | undefined) => void
  ): void;
  loadAsset(
    params: { name: string },
    callback: (result: WGPULoadAssetResult | undefined) => void
  ): void;
  decodeImage(
    params: {
      id: number;
      data?: ArrayBuffer;
      name?: string;
      flipY?: boolean;
      premultiplyAlpha?: boolean;
      resizeWidth?: number;
      resizeHeight?: number;
    },
    callback: (result: WGPUDecodeImageResult | undefined) => void
  ): void;
  startFrameLoop(params: { fps: number }): unknown;
  stopFrameLoop(): unknown;
  reset(): unknown;
}

interface LynxNativeModuleRegistry {
  WebGPU?: WebGPUNativeModule;
}

/** The Lynx JS module used to receive the `webgpu:frame` global event. */
interface LynxGlobalEventEmitter {
  addListener(event: string, listener: (...args: any[]) => void): void;
  removeListener(event: string, listener: (...args: any[]) => void): void;
}

interface LynxGlobal {
  NativeModules?: LynxNativeModuleRegistry;
  getJSModule(name: 'GlobalEventEmitter'): LynxGlobalEventEmitter;
  getJSModule(name: string): unknown;
}

/** The case where Lynx puts it directly on the global. It may be absent, so check with `typeof` before use. */
declare const NativeModules: LynxNativeModuleRegistry | undefined;

/** The Lynx runtime namespace. It may be absent, so check with `typeof` before use. */
declare const lynx: LynxGlobal | undefined;

// --- The minimal globals PrimJS provides ---------------------------------
// The DOM lib is not on, so only what is written here can be used.

declare const console: {
  log(...args: any[]): void;
  warn(...args: any[]): void;
  error(...args: any[]): void;
};

declare function setInterval(handler: () => void, timeout?: number): number;
declare function clearInterval(handle: number): void;
declare function setTimeout(handler: () => void, timeout?: number): number;
declare function clearTimeout(handle: number): void;

// --- PrimJS global bridge ------------------------------------------------
// The lynx* globals the shim installs at module load time (docs/JS-AUTHORING.md §10).
// On the bundle side, `source.define` swaps the bare identifiers of web globals for these names.

/** Present in a browser/Node but not in PrimJS — the shim only checks it when picking a fallback. */
declare const performance: { now(): number } | undefined;

declare var lynxNavigator: unknown;
declare var lynxPerformance: unknown;
declare var lynxGPUBufferUsage: unknown;
declare var lynxGPUTextureUsage: unknown;
declare var lynxGPUShaderStage: unknown;
declare var lynxGPUColorWrite: unknown;
declare var lynxGPUMapMode: unknown;
