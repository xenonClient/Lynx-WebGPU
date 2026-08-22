/**
 * The frame tick ack — rAF-style backpressure toward the native ticker.
 *
 * Native sends the next `webgpu:frame` only after the shim acknowledges the previous one
 * (`frameHandled`). Otherwise, whenever the JS thread falls behind the refresh rate, frame events
 * pile up in the JS message queue without bound and touch events queue behind them — input lag
 * grows to seconds. So the shim must ack **exactly once at the end of every tick**: after the
 * presents, on idle ticks too, and even when the frame callback throws.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { installNativeMock, makeDevice, installFrameEmitter } from './helpers.mjs';
import { startFrameLoop } from '../webgpu.js';

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

/** One piece of GPU work inside a frame. */
function doSomeWork(device) {
  device.queue.writeBuffer(device.createBuffer({ size: 4, usage: 0x0008 }), 0, new Uint8Array(4));
  device.queue.submit([]);
}

test('every tick acks exactly once — an idle tick too', withFrames(async (emitter) => {
  const state = installNativeMock();
  const stop = startFrameLoop(() => {});
  try {
    emitter.tick();
    assert.equal(state.frameHandledCalls, 1, 'an idle tick must still release the native gate');
    emitter.tick();
    assert.equal(state.frameHandledCalls, 2);
  } finally {
    stop();
  }
}));

test('the ack leaves after the present', withFrames(async (emitter) => {
  const state = installNativeMock();
  const device = await makeDevice();

  // Record the bridge call order — the ack marks "the frame is fully handled", so it must not
  // overtake the present flush.
  /** @type {string[]} */
  const sequence = [];
  const module = globalThis.NativeModules.WebGPU;
  const execute = module.execute.bind(module);
  module.execute = (payload) => {
    sequence.push(payload.present === false ? 'execute' : 'present');
    return execute(payload);
  };
  const frameHandled = module.frameHandled.bind(module);
  module.frameHandled = () => {
    sequence.push('ack');
    return frameHandled();
  };

  const stop = startFrameLoop(() => doSomeWork(device));
  try {
    sequence.length = 0;
    emitter.tick();
  } finally {
    stop();
  }

  assert.equal(state.frameHandledCalls, 1);
  assert.equal(sequence[sequence.length - 1], 'ack', `the ack must come last (order: ${sequence})`);
  assert.ok(sequence.includes('present'), `the tick must have presented (order: ${sequence})`);
}));

test('a throwing frame callback still acks', withFrames(async (emitter) => {
  const state = installNativeMock();
  const stop = startFrameLoop(() => {
    throw new Error('scene bug');
  });
  try {
    assert.throws(() => emitter.tick(), /scene bug/);
    assert.equal(state.frameHandledCalls, 1, 'the gate must be released even when the callback throws');
  } finally {
    stop();
  }
}));

test('an old native without frameHandled does not break the tick', withFrames(async (emitter) => {
  const state = installNativeMock();
  const device = await makeDevice();
  delete globalThis.NativeModules.WebGPU.frameHandled;

  const stop = startFrameLoop(() => doSomeWork(device));
  try {
    emitter.tick();   // must not throw
  } finally {
    stop();
  }
  const presents = state.executeCalls.filter((call) => call.present !== false);
  assert.equal(presents.length, 1, 'the frame itself must still go through');
}));

test('after the loop stops, ticks no longer ack', withFrames(async (emitter) => {
  const state = installNativeMock();
  const stop = startFrameLoop(() => {});
  emitter.tick();
  assert.equal(state.frameHandledCalls, 1);
  stop();
  // A tick that was already queued when the loop stopped finds no listener — and the shim must
  // not ack from anywhere else (the native side clears its own gate on stopFrameLoop).
  emitter.tick();
  assert.equal(state.frameHandledCalls, 1);
  assert.equal(state.stopFrameLoopCalls, 1);
}));
