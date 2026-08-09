/**
 * The frame boundary — the present happens at **the end of the frame loop callback**, not at `submit()`.
 *
 * That is what a browser does: however many times you submit within one frame, the canvas goes out once,
 * after the task ends. That is why web libraries submit several times in one frame while reusing the
 * drawable texture view throughout (three.js `PostProcessing`: scene pass → bloom mip chain →
 * output pass).
 *
 * Presenting on every submit would make the first submit send the drawable out and expire the view, so the
 * rest of the frame's passes are rejected wholesale as "no such handle" — the `threelab` demo scene broke that way.
 *
 * Ticks are driven **synchronously** through the same path as on a device (`GlobalEventEmitter`). With the
 * timer fallback, an error thrown by a callback leaks outside the timer and the test runner takes it first.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { installNativeMock, makeDevice, installFrameEmitter } from './helpers.mjs';
import gpu, { startFrameLoop } from '../webgpu.js';

/** Lays the frame emitter, runs the body, and always removes it. */
function withFrames(body) {
  return async () => {
    const emitter = installFrameEmitter();
    try {
      await body(emitter);
    } finally {
      emitter.uninstall();
    }
  };
}

/** Attaches a handler, drives one tick and cleans up. */
function oneTick(emitter, handler) {
  const stop = startFrameLoop(handler, { fps: 240 });
  try {
    emitter.tick();
  } finally {
    stop();
  }
}

/** One piece of GPU work inside a frame. */
function doSomeWork(device) {
  device.queue.writeBuffer(device.createBuffer({ size: 4, usage: 0x0008 }), 0, new Uint8Array(4));
  device.queue.submit([]);
}

test('a submit inside a tick defers the present', withFrames(async (emitter) => {
  const state = installNativeMock();
  const device = await makeDevice();

  oneTick(emitter, () => {
    doSomeWork(device);
    doSomeWork(device);
  });

  const presents = state.executeCalls.filter((call) => call.present !== false);
  assert.equal(presents.length, 1, `there must be one present per tick (${presents.length} went out)`);
  // The two earlier batches ride out with commands but do not present — the GPU work is not deferred.
  const deferred = state.executeCalls.filter((call) => call.present === false);
  assert.equal(deferred.length, 2);
  assert.ok(deferred.every((call) => call.commands.length > 0), 'the commands must go out as they come');
}));

test('the wrap-up present goes out even with no commands', withFrames(async (emitter) => {
  // Its purpose is to push the drawable out, so the last batch may be empty.
  const state = installNativeMock();
  const device = await makeDevice();

  oneTick(emitter, () => doSomeWork(device));

  const last = state.executeCalls[state.executeCalls.length - 1];
  assert.equal(last.present, true);
  assert.equal(last.commands.length, 0, 'the previous batch already took the commands');
}));

test('a tick with no GPU work never crosses the bridge', withFrames(async (emitter) => {
  const state = installNativeMock();
  await makeDevice();

  oneTick(emitter, () => {});

  assert.equal(state.executeCalls.length, 0);
}));

test('a submit outside a tick presents right there', withFrames(async () => {
  // Code that draws directly without a frame loop (a test harness, offscreen) must behave as before.
  const state = installNativeMock();
  const device = await makeDevice();

  doSomeWork(device);

  assert.equal(state.executeCalls.length, 1);
  assert.notEqual(state.executeCalls[0].present, false);
}));

test('an internal submission (popErrorScope) stays present=false inside a tick too', withFrames(async (emitter) => {
  const state = installNativeMock();
  const device = await makeDevice();

  oneTick(emitter, () => {
    device.pushErrorScope('validation');
    device.popErrorScope();
  });

  const internal = state.executeCalls.filter((call) => call.present === false);
  assert.ok(internal.length > 0, 'there is no internal submission');
}));

test('the present goes out even when the callback throws — the screen must not freeze on that frame', withFrames(async (emitter) => {
  const state = installNativeMock();
  const device = await makeDevice();

  const stop = startFrameLoop(() => {
    doSomeWork(device);
    throw new Error('blew up inside the frame');
  }, { fps: 240 });

  // The error is **not swallowed** — only the present is sent, and it is rethrown upward as is.
  assert.throws(() => emitter.tick(), /blew up inside the frame/);
  stop();

  const presents = state.executeCalls.filter((call) => call.present !== false);
  assert.equal(presents.length, 1, 'that frame must reach the screen even on an error');
}));

test('the frame boundary survives the tick after a throw', withFrames(async (emitter) => {
  // A leaking depth counter would stop every later frame from getting a present, freezing the screen.
  const state = installNativeMock();
  const device = await makeDevice();

  let shouldThrow = true;
  const stop = startFrameLoop(() => {
    doSomeWork(device);
    if (shouldThrow) {
      shouldThrow = false;
      throw new Error('it blows up once');
    }
  }, { fps: 240 });

  assert.throws(() => emitter.tick());
  emitter.tick();
  stop();

  const presents = state.executeCalls.filter((call) => call.present !== false);
  assert.equal(presents.length, 2, 'once per tick');
}));

test('one subscriber stopping keeps the other loop running', withFrames(async (emitter) => {
  // There is only one native ticker. Without counting, a briefly-run loop being `stop()`ped kills the rAF
  // pump (`installAnimationFrame`) with it and **the screen quietly freezes**.
  const state = installNativeMock();
  const device = await makeDevice();

  let longLived = 0;
  const stopLong = startFrameLoop(() => { longLived += 1; doSomeWork(device); });
  const stopShort = startFrameLoop(() => {});

  emitter.tick();
  stopShort();          // the side that runs briefly and turns off
  emitter.tick();
  emitter.tick();
  stopLong();

  assert.equal(longLived, 3, 'it was dragged to a halt by someone else\'s stop()');
  assert.equal(state.stopFrameLoopCalls, 1, 'native stops only when the last subscriber lets go');
}));

test('calling stop() twice does not eat someone else\'s subscription', withFrames(async (emitter) => {
  const state = installNativeMock();
  const device = await makeDevice();

  const stopA = startFrameLoop(() => doSomeWork(device));
  const stopB = startFrameLoop(() => {});
  stopB();
  stopB();              // the second time nothing must happen

  emitter.tick();
  assert.equal(state.stopFrameLoopCalls, 0, 'A is still there');
  stopA();
  assert.equal(state.stopFrameLoopCalls, 1);
}));

test('the next tick counts its debts afresh', withFrames(async (emitter) => {
  const state = installNativeMock();
  const device = await makeDevice();

  const stop = startFrameLoop(() => doSomeWork(device), { fps: 240 });
  emitter.tick();
  emitter.tick();
  stop();

  const presents = state.executeCalls.filter((call) => call.present !== false);
  assert.equal(presents.length, 2);
}));

// ---------------------------------------------------------------------------
// The spec's "Expire the current texture" (W3C WebGPU §canvas rendering)
//
//   getCurrentTexture(): if there is a `[[currentTexture]]`, **return it as is.**
//   Where expiry is called: presentation · configure() · a canvas resize.
// ---------------------------------------------------------------------------

test('getCurrentTexture gives the same texture within one frame', withFrames(async (emitter) => {
  // Handing out a new texture per call would make web code that caches the view see a different texture each pass.
  installNativeMock();
  const device = await makeDevice();
  const context = gpu.getCanvasContext('main');
  context.configure({ device, format: 'bgra8unorm' });

  let first;
  let second;
  oneTick(emitter, () => {
    first = context.getCurrentTexture();
    second = context.getCurrentTexture();
    device.queue.submit([]);
  });

  assert.equal(first, second, 'a different texture came out within the same frame');
  assert.equal(first.id, second.id);
}));

test('a present expires it and the next frame gets a new texture', withFrames(async (emitter) => {
  installNativeMock();
  const device = await makeDevice();
  const context = gpu.getCanvasContext('main');
  context.configure({ device, format: 'bgra8unorm' });

  const seen = [];
  const stop = startFrameLoop(() => {
    seen.push(context.getCurrentTexture().id);
    device.queue.submit([]);
  });
  emitter.tick();
  emitter.tick();
  stop();

  assert.equal(seen.length, 2);
  assert.notEqual(seen[0], seen[1], 'handing back the old texture after a present freezes the screen');
}));

test('configure() expires the current texture too', withFrames(async () => {
  installNativeMock();
  const device = await makeDevice();
  const context = gpu.getCanvasContext('main');
  context.configure({ device, format: 'bgra8unorm' });

  const before = context.getCurrentTexture();
  context.configure({ device, format: 'rgba8unorm' });
  const after = context.getCurrentTexture();

  assert.notEqual(before.id, after.id);
  assert.equal(after.format, 'rgba8unorm', 'it must be taken under the new configuration');
}));

test('getCurrentTexture before configuration is an InvalidStateError', withFrames(async () => {
  installNativeMock();
  await makeDevice();
  const context = gpu.getCanvasContext('unconfigured-probe');

  assert.throws(() => context.getCurrentTexture(), (error) => {
    assert.equal(error.name, 'InvalidStateError', 'the name the spec fixed');
    return true;
  });
}));

test('a canvas size change expires it', withFrames(async (emitter) => {
  // Pointing at a drawable whose size changed with the old texture makes the next pass draw at the wrong size.
  let size = { width: 300, height: 150 };
  const state = installNativeMock();
  state.executeResult = () => ({ ok: true, canvases: { main: size } });
  const device = await makeDevice();
  const context = gpu.getCanvasContext('main');
  context.configure({ device, format: 'bgra8unorm' });

  let first;
  let second;
  oneTick(emitter, () => {
    first = context.getCurrentTexture();
    size = { width: 400, height: 200 };   // the size changes in the next submission response
    device.queue.submit([]);
    second = context.getCurrentTexture();
  });

  assert.notEqual(first.id, second.id, 'the old texture was handed back after a resize');
}));
