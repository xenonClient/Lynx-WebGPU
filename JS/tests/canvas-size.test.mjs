/**
 * The canvas size cache — verifying the "one bridge crossing per frame" contract.
 *
 * It asserts that `getCurrentTexture()`/`getSize()` do not call a synchronous `canvasInfo` every frame but
 * read a cache refreshed by `execute`'s `canvases` response.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import gpu from '../webgpu.js';
import { installNativeMock, makeDevice } from './helpers.mjs';

function mockWithCanvas(canvasId, size) {
  return installNativeMock({
    canvasInfoResult: () => ({ ok: true, width: size.width, height: size.height, format: 'bgra8unorm' }),
    executeResult: (payload) => ({
      ok: true,
      commandCount: payload.commands.length,
      canvases: { [canvasId]: { width: size.width, height: size.height } },
    }),
  });
}

test('in a normal frame loop the synchronous canvasInfo happens once, at configure', async () => {
  const size = { width: 300, height: 150 };
  const state = mockWithCanvas('steady', size);
  const device = await makeDevice();
  const context = gpu.getCanvasContext('steady');
  context.configure({ device, format: 'bgra8unorm' });
  assert.equal(state.canvasInfoCalls, 1, 'configure seeds the cache');

  for (let frame = 0; frame < 5; frame += 1) {
    const texture = context.getCurrentTexture();
    texture.createView();
    context.getSize();
    device.queue.submit([]);
  }

  assert.equal(state.canvasInfoCalls, 1, 'there must be no extra synchronous calls inside a frame');
  // configure only records and merges into the first submit, so execute runs once per frame.
  assert.equal(state.executeCalls.length, 5, 'one execute per frame');
});

test('the canvases in the execute response refresh the cache (a resize lands)', async () => {
  const size = { width: 320, height: 240 };
  const state = mockWithCanvas('resize', size);
  const device = await makeDevice();
  const context = gpu.getCanvasContext('resize');
  context.configure({ device, format: 'bgra8unorm' });

  context.getCurrentTexture();
  device.queue.submit([]);
  assert.deepEqual(context.getSize(), { width: 320, height: 240 });

  // The layout changed on the native side — the next submission response carries the new size.
  size.width = 640;
  size.height = 480;
  context.getCurrentTexture();
  device.queue.submit([]);

  assert.deepEqual(context.getSize(), { width: 640, height: 480 });
  assert.equal(state.canvasInfoCalls, 1, 'landing a resize needs no synchronous call');
});

test('with an empty cache getSize queries synchronously once and caches', async () => {
  const state = mockWithCanvas('fallback', { width: 100, height: 50 });
  await makeDevice();
  const context = gpu.getCanvasContext('fallback');

  assert.deepEqual(context.getSize(), { width: 100, height: 50 });
  assert.equal(state.canvasInfoCalls, 1);
  context.getSize();
  assert.equal(state.canvasInfoCalls, 1, 'from the second on it reads the cache');
});

test('before the surface is registered (size 0) nothing is cached and the next query retries', async () => {
  const state = installNativeMock({
    canvasInfoResult: { ok: false, errors: [{ kind: 'validation', message: 'none' }] },
  });
  await makeDevice();
  const context = gpu.getCanvasContext('unregistered');

  assert.deepEqual(context.getSize(), { width: 0, height: 0 });
  context.getSize();
  assert.equal(state.canvasInfoCalls, 2, 'a failed query is not cached');
});

test('the texture size from getCurrentTexture comes from the cache', async () => {
  mockWithCanvas('dims', { width: 128, height: 64 });
  const device = await makeDevice();
  const context = gpu.getCanvasContext('dims');
  context.configure({ device, format: 'bgra8unorm' });

  const texture = context.getCurrentTexture();
  assert.equal(texture.width, 128);
  assert.equal(texture.height, 64);
});

test('device.destroy empties the cache', async () => {
  const state = mockWithCanvas('destroyed', { width: 10, height: 10 });
  const device = await makeDevice();
  const context = gpu.getCanvasContext('destroyed');
  context.configure({ device, format: 'bgra8unorm' });
  assert.equal(state.canvasInfoCalls, 1);

  device.destroy();
  assert.equal(state.resetCalls, 1);
  context.getSize();
  assert.equal(state.canvasInfoCalls, 2, 'the cache was emptied, so it queries synchronously again');
});

test('after unconfigure nothing can be drawn and getConfiguration is null', async () => {
  installNativeMock();
  const device = await makeDevice();
  const context = gpu.getCanvasContext('main');

  assert.equal(context.getConfiguration(), null, 'it is null before configuration');

  context.configure({ device, format: 'rgba8unorm' });
  const configuration = context.getConfiguration();
  assert.equal(configuration.format, 'rgba8unorm');
  assert.equal(configuration.device, device);

  context.unconfigure();
  assert.equal(context.getConfiguration(), null);
  assert.throws(() => context.getCurrentTexture(), /configure/, 'nothing can be drawn once unconfigured');

  // Configuring again brings it back — this is the path for reconfiguring with a different format.
  context.configure({ device, format: 'rgba16float' });
  assert.equal(context.getConfiguration().format, 'rgba16float');
  assert.doesNotThrow(() => context.getCurrentTexture());
});

test('the same canvasId gives the same context', async () => {
  installNativeMock();
  const device = await makeDevice();

  const first = gpu.getCanvasContext('shared');
  const second = gpu.getCanvasContext('shared');
  assert.equal(first, second, 'the same as a browser getContext');

  // Making a new one each time would split things here — configuring on one handle and drawing on
  // another would raise "call configure() first", or conversely unconfiguring would not take.
  first.configure({ device, format: 'rgba8unorm' });
  assert.equal(second.getConfiguration().format, 'rgba8unorm');
  second.unconfigure();
  assert.equal(first.getConfiguration(), null);

  assert.notEqual(gpu.getCanvasContext('other'), first, 'a different canvas is a different object');
});
