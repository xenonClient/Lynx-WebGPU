/**
 * JS shim 테스트 공용 헬퍼 — NativeModules.WebGPU 를 손으로 만든 목으로 바꿔치기한다.
 *
 * shim은 네이티브 모듈을 매 호출 시점에 찾으므로(webgpu.js `nativeModule()`),
 * 테스트마다 목을 새로 깔면 호출 횟수·페이로드를 독립적으로 관찰할 수 있다.
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
    // 네이티브는 `Data`를 싣고 Lynx가 ArrayBuffer로 바꿔 준다 — 목도 ArrayBuffer를 준다.
    readBufferResult: overrides.readBufferResult ?? { ok: true, data: new ArrayBuffer(0) },
    loadAssetResult: overrides.loadAssetResult ?? { ok: true, data: new ArrayBuffer(0) },
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
      startFrameLoop() {
        return { ok: true };
      },
      stopFrameLoop() {
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

export async function makeDevice() {
  const adapter = await gpu.requestAdapter();
  return adapter.requestDevice();
}

/** index번째 execute 호출의 커맨드 배열 (음수면 뒤에서부터). */
export function commandsOf(state, index = -1) {
  const calls = state.executeCalls;
  const call = calls[index < 0 ? calls.length + index : index];
  return call ? call.commands : [];
}

/** 결정적 의사 난수 바이트열 — 실패를 재현할 수 있어야 한다. */
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
