/**
 * `device.onuncapturederror` — the spec channel for errors uncaught by a scope.
 *
 * Web code knows only this name (three.js assigns to it and forwards to `renderer.onError`). It must work
 * **together with** this implementation's `onError`, and an error a scope intercepted must not arrive here —
 * an error already claimed for handling being reported again by the global handler is a duplicate.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { installNativeMock, makeDevice } from './helpers.mjs';
import { GPUValidationError, GPUInternalError, GPUOutOfMemoryError } from '../webgpu.js';

/** A mock where submit returns fixed errors. */
function mockWithErrors(errors) {
  return installNativeMock({
    executeResult: (payload) => ({ ok: false, commandCount: payload.commands.length, errors }),
  });
}

test('an error uncaught by a scope arrives as an event', async () => {
  mockWithErrors([{ kind: 'validation', message: 'no such handle', path: 'commands[0].buffer' }]);
  const device = await makeDevice();

  /** @type {any[]} */
  const events = [];
  device.onuncapturederror = (event) => events.push(event);

  device.createBuffer({ size: 16, usage: 0x40 });
  device.queue.submit([]);

  assert.equal(events.length, 1);
  assert.equal(events[0].type, 'uncapturederror');
  assert.equal(events[0].error.message, 'no such handle');
  assert.ok(events[0].error instanceof GPUValidationError);
  // Not in the spec, but carried along because diagnosis needs it.
  assert.equal(events[0].error.path, 'commands[0].buffer');
});

test('the error kinds split into the spec GPUError subclasses', async () => {
  mockWithErrors([
    { kind: 'validation', message: 'v' },
    { kind: 'unsupported', message: 'u' },
    { kind: 'backend', message: 'b' },
    { kind: 'out-of-memory', message: 'o' },
  ]);
  const device = await makeDevice();

  /** @type {any[]} */
  const errors = [];
  device.addEventListener('uncapturederror', (event) => errors.push(event.error));

  device.createBuffer({ size: 16, usage: 0x40 });
  device.queue.submit([]);

  assert.ok(errors[0] instanceof GPUValidationError);
  // unsupported folds into validation too — the same rule as pushErrorScope('validation') catching it.
  assert.ok(errors[1] instanceof GPUValidationError);
  assert.ok(errors[2] instanceof GPUInternalError, 'backend is internal');
  assert.ok(errors[3] instanceof GPUOutOfMemoryError);
});

test('it works together with onError (register both and both receive)', async () => {
  mockWithErrors([{ kind: 'validation', message: 'both' }]);
  const device = await makeDevice();

  /** @type {string[]} */
  const seen = [];
  device.onError((_error, text) => seen.push(`onError:${text}`));
  device.onuncapturederror = (event) => seen.push(`uncaptured:${event.error.message}`);

  device.createBuffer({ size: 16, usage: 0x40 });
  device.queue.submit([]);

  assert.equal(seen.length, 2);
  assert.match(seen[0], /onError:\[WebGPU:validation\] both/);
  assert.equal(seen[1], 'uncaptured:both');
});

test('a listener removed with removeEventListener receives no more', async () => {
  mockWithErrors([{ kind: 'validation', message: 'x' }]);
  const device = await makeDevice();

  let count = 0;
  const listener = () => { count += 1; };
  device.addEventListener('uncapturederror', listener);
  device.createBuffer({ size: 16, usage: 0x40 });
  device.queue.submit([]);
  assert.equal(count, 1);

  device.removeEventListener('uncapturederror', listener);
  device.createBuffer({ size: 16, usage: 0x40 });
  device.queue.submit([]);
  assert.equal(count, 1, 'nothing may arrive after removal');
});

test('the remaining listeners and the next errors keep going even when a listener throws', async () => {
  mockWithErrors([{ kind: 'validation', message: 'first' }, { kind: 'validation', message: 'second' }]);
  const device = await makeDevice();

  /** @type {string[]} */
  const seen = [];
  device.addEventListener('uncapturederror', () => { throw new Error('a listener bug'); });
  device.addEventListener('uncapturederror', (event) => seen.push(event.error.message));

  device.createBuffer({ size: 16, usage: 0x40 });
  device.queue.submit([]);

  assert.deepEqual(seen, ['first', 'second'], 'one mistake must not swallow the whole report');
});

test('an error a scope intercepted does not arrive at uncapturederror', async () => {
  // Native drops an error caught by a scope from errors and returns it only through errorScopes — the mock imitates that contract.
  installNativeMock({
    executeResult: (payload) => ({
      ok: true,
      commandCount: payload.commands.length,
      errorScopes: [{ kind: 'validation', message: 'intercepted' }],
    }),
  });
  const device = await makeDevice();

  let count = 0;
  device.onuncapturederror = () => { count += 1; };

  device.pushErrorScope('validation');
  const captured = await device.popErrorScope();

  assert.equal(captured.message, 'intercepted');
  assert.equal(count, 0, 'the global reporting an error already claimed for handling is a duplicate');
});
