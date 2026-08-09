/**
 * `GPUDevice`'s spec surface — `features` / `lost`.
 *
 * three.js's `WebGPUBackend.init()` requests the features it picked from the adapter via `requiredFeatures`,
 * reads `device.features` back, and immediately attaches `device.lost.then(...)`. Without these two
 * properties the renderer bootstrap dies with a TypeError, so the properties themselves are guaranteed regardless of support decisions.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import gpu from '../webgpu.js';
import { installNativeMock, makeDevice } from './helpers.mjs';

/** A mock where the adapter reports a feature list. */
function mockWithFeatures(features) {
  const state = installNativeMock();
  const base = globalThis.NativeModules.WebGPU.adapterInfo;
  globalThis.NativeModules.WebGPU.adapterInfo = () => ({ ...base(), features });
  return state;
}

test('only the requested features land in device.features', async () => {
  mockWithFeatures(['timestamp-query', 'shader-f16']);
  const adapter = await gpu.requestAdapter();

  const device = await adapter.requestDevice({ requiredFeatures: ['timestamp-query'] });

  assert.equal(device.features.has('timestamp-query'), true);
  assert.equal(device.features.has('shader-f16'), false, 'an unrequested feature must be absent');
  assert.equal(device.features.size, 1);
});

test('with no request, device.features is empty but present', async () => {
  installNativeMock();
  const device = await makeDevice();

  assert.equal(device.features.has('timestamp-query'), false);
  assert.equal(device.features.size, 0);
});

test('requiring a feature the adapter does not support is rejected', async () => {
  mockWithFeatures(['timestamp-query']);
  const adapter = await gpu.requestAdapter();

  await assert.rejects(
    adapter.requestDevice({ requiredFeatures: ['shader-f16'] }),
    /shader-f16/,
    'it has to name what is missing'
  );
});

test('adapter.info and device.adapterInfo have the spec shape', async () => {
  const state = installNativeMock();
  const base = globalThis.NativeModules.WebGPU.adapterInfo;
  globalThis.NativeModules.WebGPU.adapterInfo = () => ({
    ...base(),
    info: { vendor: 'apple', architecture: 'apple-8', description: 'Apple M3' },
  });

  const adapter = await gpu.requestAdapter();
  const device = await adapter.requestDevice();

  assert.equal(adapter.info.vendor, 'apple');
  assert.equal(adapter.info.architecture, 'apple-8');
  assert.equal(adapter.info.description, 'Apple M3');
  // An unknown slot is the empty string — inventing one sends code that branches on it down the wrong path.
  assert.equal(adapter.info.device, '');
  assert.equal(adapter.info.isFallbackAdapter, false);
  assert.equal(adapter.info.subgroupMinSize, 0);
  // The spec exposes the same thing on the device.
  assert.equal(device.adapterInfo, adapter.info);
  assert.ok(state);
});

test('with no info from native everything is empty (no undefined leaks)', async () => {
  installNativeMock();
  const adapter = await gpu.requestAdapter();

  assert.equal(adapter.info.vendor, '');
  assert.equal(adapter.info.architecture, '');
  assert.equal(adapter.info.isFallbackAdapter, false);
});

test('device.lost is a Promise you can attach then to', async () => {
  installNativeMock();
  const device = await makeDevice();

  assert.ok(device.lost instanceof Promise, 'lost must be a Promise');
  // It must stay pending forever — an implementation that does not report loss resolving it would run
  // web code's loss handler (reinitialization and the like) on a perfectly healthy device.
  const settled = await Promise.race([
    device.lost.then(() => 'settled'),
    new Promise((resolve) => setTimeout(() => resolve('pending'), 20)),
  ]);
  assert.equal(settled, 'pending');
});
