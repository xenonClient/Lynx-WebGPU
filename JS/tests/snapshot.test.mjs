/**
 * Record-time snapshots — reusing a descriptor after the call cannot pollute the stream.
 *
 * Browser WebGPU serializes arguments at call time. three.js relies on that contract and reset()s a
 * singleton descriptor right after encoding, so with the shim holding a reference copySize becomes 0
 * before the flush and **a zero-width copy goes out with no error**
 * (a real incident where render target readbacks all came out as (0,0,0)).
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { installNativeMock, makeDevice, commandsOf } from './helpers.mjs';

test('encoder commands: resetting copySize and origin after the call leaves the record unchanged', async () => {
  const state = installNativeMock();
  const device = await makeDevice();

  const texture = device.createTexture({
    size: { width: 8, height: 8 }, format: 'rgba8unorm', usage: 0x07,
  });
  const buffer = device.createBuffer({ size: 256, usage: 0x09 });

  // three.js style — a singleton descriptor, reset right after encoding.
  const source = { texture, origin: { x: 4, y: 4, z: 0 } };
  const destination = { buffer, bytesPerRow: 256 };
  const copySize = { width: 1, height: 1, depthOrArrayLayers: 1 };

  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer(source, destination, copySize);
  copySize.width = 0;
  copySize.height = 0;
  source.origin.x = 0;
  source.origin.y = 0;

  device.queue.submit([encoder.finish()]);

  const command = commandsOf(state).find((entry) => entry.op === 'copyTextureToBuffer');
  assert.equal(command.copySize.width, 1, 'a reset width leaking into the stream sends out a zero-width copy');
  assert.equal(command.copySize.height, 1);
  assert.equal(command.source.origin.x, 4);
  assert.equal(command.source.origin.y, 4);
});

test('beginRenderPass: changing clearValue after the call leaves the record unchanged', async () => {
  const state = installNativeMock();
  const device = await makeDevice();

  const texture = device.createTexture({
    size: { width: 8, height: 8 }, format: 'rgba8unorm', usage: 0x10,
  });
  const view = texture.createView();

  const clearValue = { r: 1, g: 0, b: 0, a: 1 };
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    colorAttachments: [{ view, loadOp: 'clear', storeOp: 'store', clearValue }],
  });
  clearValue.r = 0;
  pass.end();
  device.queue.submit([encoder.finish()]);

  const command = commandsOf(state).find((entry) => entry.op === 'beginRenderPass');
  assert.equal(command.colorAttachments[0].clearValue.r, 1);
});

test('queue commands: resetting writeTexture\'s origin and size after the call leaves the record unchanged', async () => {
  const state = installNativeMock();
  const device = await makeDevice();

  const texture = device.createTexture({
    size: { width: 4, height: 4 }, format: 'rgba8unorm', usage: 0x06,
  });
  const origin = { x: 2, y: 2 };
  const size = { width: 2, height: 2 };
  device.queue.writeTexture(
    { texture, origin }, new Uint8Array(16), { bytesPerRow: 8 }, size
  );
  origin.x = 0;
  size.width = 0;
  device.queue.submit([]);

  const command = commandsOf(state).find((entry) => entry.op === 'writeTexture');
  assert.equal(command.origin.x, 2);
  assert.equal(command.size.width, 2);
});

test('the bundle encoder: resetting the descriptor after the call keeps the attachments', async () => {
  const state = installNativeMock();
  const device = await makeDevice();

  // three.js's createBundleEncoder has exactly this shape — it passes a singleton and resets right away.
  const descriptor = { label: 'bundle', colorFormats: ['bgra8unorm'], depthStencilFormat: 'depth32float' };
  const encoder = device.createRenderBundleEncoder(descriptor);
  descriptor.colorFormats = [];
  descriptor.depthStencilFormat = undefined;

  encoder.finish();
  device.queue.submit([]);

  const create = commandsOf(state).find((command) => command.op === 'createRenderBundle');
  assert.deepEqual(
    create.colorFormats, ['bgra8unorm'],
    'a leaking reset builds a bundle with no attachments, and the cause only surfaces at executeBundles'
  );
  assert.equal(create.depthStencilFormat, 'depth32float');
});
