/**
 * `installAnimationFrame` — installs the rAF web libraries expect onto the global.
 *
 * PrimJS has no rAF, so a library running its own loop (three.js and the like) **stalls forever with no error**.
 * Three things are checked here: does only what was scheduled run this tick (the same boundary as a
 * browser), is the display link let go when nobody schedules (battery), and is an existing rAF left alone.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { installNativeMock } from './helpers.mjs';
import { installAnimationFrame } from '../webgpu.js';

/** A mock that hand-pushes the `webgpu:frame` event `startFrameLoop` latches onto. */
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

test('an rAF callback runs on a frame tick', () => {
  const state = installFrameEmitter();
  const uninstall = installAnimationFrame();

  /** @type {number[]} */
  const times = [];
  globalThis.requestAnimationFrame((time) => times.push(time));
  assert.equal(state.frameLoopRunning, 1, 'the loop runs while something is scheduled');

  state.tick(100);
  assert.deepEqual(times, [100]);

  uninstall();
  delete globalThis.lynx;
});

test('what is scheduled inside a callback belongs to the next tick (the same boundary as a browser)', () => {
  const state = installFrameEmitter();
  const uninstall = installAnimationFrame();

  /** @type {number[]} */
  const order = [];
  globalThis.requestAnimationFrame((time) => {
    order.push(time);
    globalThis.requestAnimationFrame((next) => order.push(next));
  });

  state.tick(1);
  assert.deepEqual(order, [1], 'rescheduling running in the same tick would be an infinite loop');
  state.tick(2);
  assert.deepEqual(order, [1, 2]);

  uninstall();
  delete globalThis.lynx;
});

test('lets the display link go when nothing is scheduled', () => {
  const state = installFrameEmitter();
  const uninstall = installAnimationFrame();

  globalThis.requestAnimationFrame(() => {});
  assert.equal(state.frameLoopRunning, 1);

  state.tick(1);                       // the callback did not reschedule
  assert.equal(state.frameLoopRunning, 0, 'a loop that keeps running eats battery');

  // Scheduling again brings it back.
  globalThis.requestAnimationFrame(() => {});
  assert.equal(state.frameLoopRunning, 1);

  uninstall();
  delete globalThis.lynx;
});

test('cancelAnimationFrame removes only that callback', () => {
  const state = installFrameEmitter();
  const uninstall = installAnimationFrame();

  /** @type {string[]} */
  const seen = [];
  const id = globalThis.requestAnimationFrame(() => seen.push('to be cancelled'));
  globalThis.requestAnimationFrame(() => seen.push('to remain'));
  globalThis.cancelAnimationFrame(id);

  state.tick(1);
  assert.deepEqual(seen, ['to remain']);

  uninstall();
  delete globalThis.lynx;
});

test('uninstall restores the globals', () => {
  installFrameEmitter();
  const uninstall = installAnimationFrame();
  assert.equal(typeof globalThis.requestAnimationFrame, 'function');

  uninstall();
  assert.equal(globalThis.requestAnimationFrame, undefined);
  assert.equal(globalThis.cancelAnimationFrame, undefined);
  delete globalThis.lynx;
});

test('does not overwrite an existing rAF', () => {
  installFrameEmitter();
  const original = (callback) => { callback(0); return 42; };
  globalThis.requestAnimationFrame = original;

  const uninstall = installAnimationFrame();
  assert.equal(globalThis.requestAnimationFrame, original, 'a browser or runner must not have its own taken away');

  uninstall();
  delete globalThis.requestAnimationFrame;
  delete globalThis.lynx;
});
