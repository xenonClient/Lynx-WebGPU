/**
 * The handle space — **the whole module shares one.**
 *
 * The native registry is one per context and finds objects by handle integer alone. A per-device counter
 * would make the second device start issuing from 1 again and **silently overwrite the first device's
 * objects** — the kind of bug that raises no error, draws into someone else's buffer, and only shows on screen.
 *
 * Multiple devices are not a common pattern, but our own code already has one (the `spec` demo scene
 * creates a second device to verify `requiredFeatures`).
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { installNativeMock, makeDevice } from './helpers.mjs';
import gpu, { createImageBitmap } from '../webgpu.js';

test('handles do not collide even with two devices', async () => {
  installNativeMock();
  const adapter = await gpu.requestAdapter();
  const deviceA = await adapter.requestDevice();
  const deviceB = await adapter.requestDevice();

  const first = deviceA.createBuffer({ size: 16, usage: 0x0004 });
  const second = deviceB.createBuffer({ size: 16, usage: 0x0004 });
  const third = deviceA.createBuffer({ size: 16, usage: 0x0004 });

  const handles = new Set([first.id, second.id, third.id]);
  assert.equal(handles.size, 3, `the handles collided: ${first.id}, ${second.id}, ${third.id}`);
});

test('the different kinds within one device do not collide either', async () => {
  installNativeMock();
  const device = await makeDevice();

  const handles = [
    device.createBuffer({ size: 16, usage: 0x0004 }).id,
    device.createTexture({ size: [4, 4], format: 'rgba8unorm', usage: 0x04 }).id,
    device.createSampler().id,
    device.createShaderModule({ code: '@fragment fn fs() {}' }).id,
  ];
  assert.equal(new Set(handles).size, handles.length, `collided handles: ${handles.join(', ')}`);
});

test('a createImageBitmap handle is unique regardless of the active device', async () => {
  // This is the spot originally flagged — with the global `createImageBitmap` using the "most recently
  // created device"'s counter, it could collide with the first device's object numbers.
  installNativeMock({ decodeImageResult: { ok: true, width: 4, height: 4 } });
  const adapter = await gpu.requestAdapter();
  const deviceA = await adapter.requestDevice();
  const early = deviceA.createBuffer({ size: 16, usage: 0x0004 });

  await adapter.requestDevice();   // the second device becomes active

  const bitmap = await createImageBitmap(new ArrayBuffer(8));
  const later = deviceA.createBuffer({ size: 16, usage: 0x0004 });

  assert.notEqual(bitmap.id, early.id, 'the image overwrites the first device\'s buffer');
  assert.notEqual(bitmap.id, later.id);
  assert.notEqual(early.id, later.id);
});

test('numbers are not reused even after destroy', async () => {
  // Reusing them would make a JS object still holding a destroyed object's number point at **someone else's slot**.
  installNativeMock();
  const device = await makeDevice();

  const first = device.createBuffer({ size: 16, usage: 0x0004 });
  first.destroy();
  const second = device.createBuffer({ size: 16, usage: 0x0004 });

  assert.notEqual(second.id, first.id);
});
