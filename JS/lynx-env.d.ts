/**
 * shim이 기대하는 **호스트 런타임 표면**의 선언.
 *
 * 이 파일의 목적은 두 가지다:
 * 1. `NativeModules.WebGPU`의 시그니처를 고정해 브리지 경계에서 오타·인자 누락을 잡는다.
 *    여기 선언은 `Sources/LynxWebGPUBridge/WebGPUNativeModule.swift`의 `methodLookup`과
 *    **짝이 맞아야 한다** — 한쪽만 고치면 런타임에 조용히 깨진다.
 * 2. DOM lib을 켜지 않는다. shim은 브라우저가 아니라 PrimJS 위에서 돌기 때문에,
 *    여기 적히지 않은 전역(`window`, `fetch`, `btoa` 등)을 쓰면 타입 검사에서 걸린다.
 */

/** 커맨드 실행 중 발생한 오류. */
interface WGPUErrorPayload {
  kind: 'validation' | 'out-of-memory' | 'unsupported' | 'backend';
  message: string;
  path?: string;
}

/** `execute`의 반환. */
interface WGPUExecuteResult {
  ok: boolean;
  commandCount?: number;
  errors?: WGPUErrorPayload[];
  /** 이번 제출이 건드린 캔버스의 픽셀 크기 — 크기 캐시가 이걸로 갱신된다. */
  canvases?: Record<string, { width: number; height: number }>;
  /** 네이티브에 살아 있는 GPU 객체 수 — 프레임마다 늘면 destroy 누락이다. */
  objects?: number;
  /**
   * 이번 배치에서 `popErrorScope`로 닫힌 스코프들의 결과 — **pop한 순서 그대로**다.
   * 오류가 없던 스코프는 `null`. shim이 이 순서로 Promise를 푼다.
   */
  errorScopes?: (WGPUErrorPayload | null)[];
}

/** `adapterInfo`의 반환. */
interface WGPUAdapterInfo {
  ok?: boolean;
  name: string;
  backend: string;
  limits?: Record<string, number>;
  hasUnifiedMemory?: boolean;
  /** 기기마다 갈리는 기능의 명세 철자 (`'timestamp-query'` 등). */
  features?: string[];
}

/** `canvasInfo`의 반환. */
interface WGPUCanvasInfo {
  ok?: boolean;
  width?: number;
  height?: number;
  format?: string;
}

/**
 * `readBuffer` 콜백이 받는 값.
 *
 * `data`는 네이티브가 `Data`로 돌려준 것을 Lynx가 `ArrayBuffer`로 바꿔 준 것이다
 * (base64 문자열이 아니다 — `LynxWebGPUContext.readBuffer` 참고).
 */
interface WGPUReadBufferResult {
  ok?: boolean;
  data: ArrayBuffer;
  byteLength?: number;
  errors?: WGPUErrorPayload[];
}

/**
 * `loadAsset` 콜백이 받는 값.
 *
 * `readBuffer`와 같은 규약이다 — `data`는 네이티브가 `Data`로 돌려준 것을 Lynx가
 * `ArrayBuffer`로 바꿔 준 것이다.
 */
interface WGPULoadAssetResult {
  ok?: boolean;
  data: ArrayBuffer;
  byteLength?: number;
  errors?: WGPUErrorPayload[];
}

/** `NativeModules.WebGPU` — 커맨드 스트림 입구. */
interface WebGPUNativeModule {
  execute(payload: { commands: Record<string, any>[] }): WGPUExecuteResult | undefined;
  adapterInfo(): WGPUAdapterInfo | undefined;
  canvasInfo(params: { canvas: string }): WGPUCanvasInfo | undefined;
  readBuffer(
    params: { buffer: number; offset: number; size?: number },
    callback: (result: WGPUReadBufferResult | undefined) => void
  ): void;
  loadAsset(
    params: { name: string },
    callback: (result: WGPULoadAssetResult | undefined) => void
  ): void;
  startFrameLoop(params: { fps: number }): unknown;
  stopFrameLoop(): unknown;
  reset(): unknown;
}

interface LynxNativeModuleRegistry {
  WebGPU?: WebGPUNativeModule;
}

/** `webgpu:frame` 전역 이벤트를 받는 데 쓰는 Lynx JS 모듈. */
interface LynxGlobalEventEmitter {
  addListener(event: string, listener: (...args: any[]) => void): void;
  removeListener(event: string, listener: (...args: any[]) => void): void;
}

interface LynxGlobal {
  NativeModules?: LynxNativeModuleRegistry;
  getJSModule(name: 'GlobalEventEmitter'): LynxGlobalEventEmitter;
  getJSModule(name: string): unknown;
}

/** Lynx가 전역에 직접 얹어 주는 경우. 없을 수 있으므로 `typeof`로 확인하고 쓴다. */
declare const NativeModules: LynxNativeModuleRegistry | undefined;

/** Lynx 런타임 네임스페이스. 없을 수 있으므로 `typeof`로 확인하고 쓴다. */
declare const lynx: LynxGlobal | undefined;

// --- PrimJS가 제공하는 최소 전역 ------------------------------------------
// DOM lib을 켜지 않으므로 여기 적힌 것만 쓸 수 있다.

declare const console: {
  log(...args: any[]): void;
  warn(...args: any[]): void;
  error(...args: any[]): void;
};

declare function setInterval(handler: () => void, timeout?: number): number;
declare function clearInterval(handle: number): void;
declare function setTimeout(handler: () => void, timeout?: number): number;
declare function clearTimeout(handle: number): void;
