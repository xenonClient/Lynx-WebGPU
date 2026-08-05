/**
 * `installAnimationFrame` — 웹 라이브러리가 기대하는 rAF를 전역에 깐다.
 *
 * PrimJS에는 rAF가 없어서, 자체 루프를 도는 라이브러리(Three.js 등)가 **오류 없이 영구 정지**한다.
 * 여기서 보는 것은 세 가지다: 예약한 것만 이번 틱에 실행하는가(브라우저와 같은 경계), 아무도
 * 예약하지 않으면 디스플레이 링크를 놓는가(배터리), 이미 rAF가 있으면 안 덮는가.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { installNativeMock } from './helpers.mjs';
import { installAnimationFrame } from '../webgpu.js';

/** `startFrameLoop`가 붙잡는 `webgpu:frame` 이벤트를 손으로 밀어 주는 목. */
function installFrameEmitter() {
  const state = installNativeMock();
  /** @type {((frame: {timestamp: number, delta: number}) => void)[]} */
  const listeners = [];
  state.frameLoopRunning = 0;

  globalThis.lynx = {
    getJSModule: () => ({
      addListener: (_event, listener) => listeners.push(listener),
      removeListener: (_event, listener) => {
        const index = listeners.indexOf(listener);
        if (index >= 0) listeners.splice(index, 1);
      },
    }),
  };
  const module = globalThis.NativeModules.WebGPU;
  module.startFrameLoop = () => { state.frameLoopRunning += 1; return { ok: true }; };
  module.stopFrameLoop = () => { state.frameLoopRunning -= 1; return { ok: true }; };

  state.tick = (timestamp) => {
    for (const listener of listeners.slice()) listener({ timestamp, delta: 16 });
  };
  return state;
}

test('rAF 콜백이 프레임 틱에 실행된다', () => {
  const state = installFrameEmitter();
  const uninstall = installAnimationFrame();

  /** @type {number[]} */
  const times = [];
  globalThis.requestAnimationFrame((time) => times.push(time));
  assert.equal(state.frameLoopRunning, 1, '예약이 있으면 루프가 돈다');

  state.tick(100);
  assert.deepEqual(times, [100]);

  uninstall();
  delete globalThis.lynx;
});

test('콜백 안에서 예약한 것은 다음 틱이다 (브라우저와 같은 경계)', () => {
  const state = installFrameEmitter();
  const uninstall = installAnimationFrame();

  /** @type {number[]} */
  const order = [];
  globalThis.requestAnimationFrame((time) => {
    order.push(time);
    globalThis.requestAnimationFrame((next) => order.push(next));
  });

  state.tick(1);
  assert.deepEqual(order, [1], '재예약이 같은 틱에서 돌면 무한 루프가 된다');
  state.tick(2);
  assert.deepEqual(order, [1, 2]);

  uninstall();
  delete globalThis.lynx;
});

test('예약이 비면 디스플레이 링크를 놓는다', () => {
  const state = installFrameEmitter();
  const uninstall = installAnimationFrame();

  globalThis.requestAnimationFrame(() => {});
  assert.equal(state.frameLoopRunning, 1);

  state.tick(1);                       // 콜백이 재예약하지 않았다
  assert.equal(state.frameLoopRunning, 0, '루프가 계속 돌면 배터리를 먹는다');

  // 다시 예약하면 되살아난다.
  globalThis.requestAnimationFrame(() => {});
  assert.equal(state.frameLoopRunning, 1);

  uninstall();
  delete globalThis.lynx;
});

test('cancelAnimationFrame이 그 콜백만 뗀다', () => {
  const state = installFrameEmitter();
  const uninstall = installAnimationFrame();

  /** @type {string[]} */
  const seen = [];
  const id = globalThis.requestAnimationFrame(() => seen.push('취소될 것'));
  globalThis.requestAnimationFrame(() => seen.push('남을 것'));
  globalThis.cancelAnimationFrame(id);

  state.tick(1);
  assert.deepEqual(seen, ['남을 것']);

  uninstall();
  delete globalThis.lynx;
});

test('uninstall이 전역을 원래대로 되돌린다', () => {
  installFrameEmitter();
  const uninstall = installAnimationFrame();
  assert.equal(typeof globalThis.requestAnimationFrame, 'function');

  uninstall();
  assert.equal(globalThis.requestAnimationFrame, undefined);
  assert.equal(globalThis.cancelAnimationFrame, undefined);
  delete globalThis.lynx;
});

test('이미 rAF가 있으면 덮지 않는다', () => {
  installFrameEmitter();
  const original = (callback) => { callback(0); return 42; };
  globalThis.requestAnimationFrame = original;

  const uninstall = installAnimationFrame();
  assert.equal(globalThis.requestAnimationFrame, original, '브라우저·러너의 것을 뺏으면 안 된다');

  uninstall();
  delete globalThis.requestAnimationFrame;
  delete globalThis.lynx;
});
