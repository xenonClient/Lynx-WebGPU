/**
 * `createImageBitmap()` + `queue.copyExternalImageToTexture()` — an external image into a texture.
 *
 * On the web the browser does the decoding. Here native (ImageIO) does, and **the pixels stay native** —
 * whether the handle is all that crosses the bridge is this file's core contract.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { installNativeMock, makeDevice, commandsOf } from './helpers.mjs';
import { createImageBitmap } from '../webgpu.js';

test('hands the image bytes to native and receives the size', async () => {
  const state = installNativeMock({ decodeImageResult: { ok: true, width: 64, height: 32 } });
  await makeDevice();

  const bitmap = await createImageBitmap(new Uint8Array([1, 2, 3, 4]).buffer);

  assert.equal(bitmap.width, 64);
  assert.equal(bitmap.height, 32);
  assert.equal(state.decodeImageCalls.length, 1);
  assert.ok(state.decodeImageCalls[0].data instanceof ArrayBuffer);
  assert.equal(typeof state.decodeImageCalls[0].id, 'number', 'the handle is issued by JS');
});

test('a string goes across as an asset name', async () => {
  const state = installNativeMock();
  await makeDevice();

  await createImageBitmap('photo.jpg');

  assert.equal(state.decodeImageCalls[0].name, 'photo.jpg');
  assert.equal(state.decodeImageCalls[0].data, undefined, 'given a name, no bytes are put on');
});

test('the options are moved across under the native names', async () => {
  const state = installNativeMock();
  await makeDevice();

  await createImageBitmap('a.png', {
    flipY: true, premultiplyAlpha: 'premultiply', resizeWidth: 8, resizeHeight: 4,
  });

  const params = state.decodeImageCalls[0];
  assert.equal(params.flipY, true);
  assert.equal(params.premultiplyAlpha, true, "only 'premultiply' is true");
  assert.equal(params.resizeWidth, 8);
  assert.equal(params.resizeHeight, 4);
});

test("premultiplyAlpha of 'none' does not multiply", async () => {
  const state = installNativeMock();
  await makeDevice();

  await createImageBitmap('a.png', { premultiplyAlpha: 'none' });

  assert.equal(state.decodeImageCalls[0].premultiplyAlpha, false);
});

test('a decode failure throws — it does not quietly hand back an empty image', async () => {
  installNativeMock({
    decodeImageResult: { ok: false, errors: [{ kind: 'validation', message: 'corrupt PNG' }] },
  });
  await makeDevice();

  await assert.rejects(() => createImageBitmap(new ArrayBuffer(4)), /corrupt PNG/);
});

test('calling it with no device gives a clear error', async () => {
  // The handle is issued by the device's recorder — without one, no number can be made.
  installNativeMock();
  const { createImageBitmap: fresh } = await import(`../webgpu.js?fresh=${Date.now()}`);
  await assert.rejects(() => fresh(new ArrayBuffer(4)), /requestDevice/);
});

test('copyExternalImageToTexture puts only the handle on', async () => {
  const state = installNativeMock({ decodeImageResult: { ok: true, width: 16, height: 16 } });
  const device = await makeDevice();
  const bitmap = await createImageBitmap('a.png');
  const texture = device.createTexture({
    size: [16, 16], format: 'rgba8unorm', usage: 0x04 | 0x02,
  });

  device.queue.copyExternalImageToTexture(
    { source: bitmap }, { texture }, [16, 16]
  );
  device.queue.submit([]);

  const copy = commandsOf(state).find((command) => command.op === 'copyExternalImageToTexture');
  assert.ok(copy, 'the command was not recorded');
  assert.equal(copy.source.source, bitmap.id);
  assert.equal(copy.destination.texture, texture.id);
  assert.deepEqual(copy.copySize, [16, 16]);
  assert.equal(copy.data, undefined, 'pixels must not cross the bridge');
});

test('copy-time flipY rides along — this is what three.js uses', async () => {
  // `Texture.flipY` defaults to true, so without sending it a web library's textures flip silently.
  const state = installNativeMock({ decodeImageResult: { ok: true, width: 8, height: 8 } });
  const device = await makeDevice();
  const bitmap = await createImageBitmap('a.png');
  const texture = device.createTexture({ size: [8, 8], format: 'rgba8unorm', usage: 0x06 });

  device.queue.copyExternalImageToTexture({ source: bitmap, flipY: true }, { texture });
  device.queue.copyExternalImageToTexture({ source: bitmap }, { texture });
  device.queue.submit([]);

  const copies = commandsOf(state).filter((c) => c.op === 'copyExternalImageToTexture');
  assert.equal(copies[0].source.flipY, true);
  assert.equal(copies[1].source.flipY, false, 'omitted, it does not flip');
});

test('a partial copy puts its origin and mipLevel on', async () => {
  const state = installNativeMock({ decodeImageResult: { ok: true, width: 16, height: 16 } });
  const device = await makeDevice();
  const bitmap = await createImageBitmap('a.png');
  const texture = device.createTexture({ size: [8, 8], format: 'rgba8unorm', usage: 0x06 });

  device.queue.copyExternalImageToTexture(
    { source: bitmap, origin: { x: 4, y: 4 } },
    { texture, mipLevel: 1, origin: { x: 2, y: 0 } },
    [4, 4]
  );
  device.queue.submit([]);

  const copy = commandsOf(state).find((command) => command.op === 'copyExternalImageToTexture');
  assert.deepEqual(copy.source.origin, { x: 4, y: 4 });
  assert.equal(copy.destination.mipLevel, 1);
  assert.deepEqual(copy.destination.origin, { x: 2, y: 0 });
});

test('close() throws away the native pixels — twice still means once', async () => {
  const state = installNativeMock();
  const device = await makeDevice();
  const bitmap = await createImageBitmap('a.png');

  bitmap.close();
  bitmap.close();
  device.queue.submit([]);

  const destroys = commandsOf(state).filter(
    (command) => command.op === 'destroy' && command.id === bitmap.id
  );
  assert.equal(destroys.length, 1);
});
