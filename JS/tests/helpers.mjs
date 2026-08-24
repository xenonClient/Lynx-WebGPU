/**
 * Shared helpers for the JS shim tests — they swap NativeModules.WebGPU for a hand-built mock.
 *
 * The shim looks the native module up at each call (webgpu.js `nativeModule()`), so laying a fresh mock
 * per test lets call counts and payloads be observed independently.
 */
import gpu from '../webgpu.js';

export function installNativeMock(overrides = {}) {
  const state = {
    executeCalls: [],
    canvasInfoCalls: 0,
    readBufferCalls: [],
    resetCalls: 0,
    executeResult:
      overrides.executeResult ?? ((payload) => ({ ok: true, commandCount: payload.commands.length })),
    canvasInfoResult:
      overrides.canvasInfoResult ?? { ok: true, width: 300, height: 150, format: 'bgra8unorm' },
    loadAssetCalls: [],
    // Native puts a `Data` on and Lynx converts it to an ArrayBuffer — the mock gives an ArrayBuffer too.
    readBufferResult: overrides.readBufferResult ?? { ok: true, data: new ArrayBuffer(0) },
    loadAssetResult: overrides.loadAssetResult ?? { ok: true, data: new ArrayBuffer(0) },
    stopFrameLoopCalls: 0,
    frameHandledCalls: 0,
    decodeImageCalls: [],
    decodeImageResult: overrides.decodeImageResult ?? { ok: true, width: 4, height: 4 },
  };

  const resolve = (value, argument) => (typeof value === 'function' ? value(argument) : value);

  globalThis.NativeModules = {
    WebGPU: {
      execute(payload) {
        state.executeCalls.push(payload);
        return resolve(state.executeResult, payload);
      },
      adapterInfo() {
        return { ok: true, name: 'mock-gpu', backend: 'metal', limits: {}, hasUnifiedMemory: true };
      },
      canvasInfo(params) {
        state.canvasInfoCalls += 1;
        return resolve(state.canvasInfoResult, params);
      },
      readBuffer(params, callback) {
        state.readBufferCalls.push(params);
        callback(resolve(state.readBufferResult, params));
      },
      loadAsset(params, callback) {
        state.loadAssetCalls.push(params);
        callback(resolve(state.loadAssetResult, params));
      },
      decodeImage(params, callback) {
        state.decodeImageCalls.push(params);
        callback(resolve(state.decodeImageResult, params));
      },
      startFrameLoop() {
        return { ok: true };
      },
      stopFrameLoop() {
        state.stopFrameLoopCalls += 1;
        return { ok: true };
      },
      frameHandled() {
        state.frameHandledCalls += 1;
        return { ok: true };
      },
      reset() {
        state.resetCalls += 1;
        return { ok: true };
      },
    },
  };
  return state;
}

/**
 * The path `startFrameLoop` really takes on a device — Lynx's `GlobalEventEmitter`.
 *
 * Unlike the timer fallback it **can drive ticks synchronously**, which makes frame boundary verification
 * deterministic (a throwing callback is caught right there too).
 */
export function installFrameEmitter() {
  const listeners = new Map();
  globalThis.lynx = {
    getJSModule: () => ({
      addListener(event, listener) {
        listeners.set(event, [...(listeners.get(event) || []), listener]);
      },
      removeListener(event, listener) {
        listeners.set(event, (listeners.get(event) || []).filter((entry) => entry !== listener));
      },
    }),
  };
  return {
    /** Drives one tick now. If the handler throws, it comes straight back up here. */
    tick(frame = { timestamp: 0, delta: 16 }) {
      for (const listener of listeners.get('webgpu:frame') || []) listener(frame);
    },
    uninstall() {
      delete globalThis.lynx;
    },
  };
}

export async function makeDevice() {
  const adapter = await gpu.requestAdapter();
  return adapter.requestDevice();
}

/** The command array of the index-th execute call (negative counts from the end). */
export function commandsOf(state, index = -1) {
  const calls = state.executeCalls;
  const call = calls[index < 0 ? calls.length + index : index];
  return call ? call.commands : [];
}

/** A deterministic pseudo-random byte sequence — a failure has to be reproducible. */
export function randomBytes(length, seed = 1) {
  const bytes = new Uint8Array(length);
  let s = (seed >>> 0) || 1;
  for (let index = 0; index < length; index += 1) {
    s ^= s << 13;
    s >>>= 0;
    s ^= s >>> 17;
    s ^= s << 5;
    s >>>= 0;
    bytes[index] = s & 0xff;
  }
  return bytes;
}
