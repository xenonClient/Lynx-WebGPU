/**
 * The **read-only properties** the spec fixed — web code reads them when it takes an object and decides for itself.
 *
 * Without them `undefined` goes out, and the other side misreads it as "the value is 0 / it was not set" and branches wrongly.
 * (three.js reads `texture.textureBindingViewDimension` on the mipmap path to build a view.)
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { installNativeMock, makeDevice } from './helpers.mjs';

test('GPUTexture has the spec read-only properties', async () => {
  installNativeMock();
  const device = await makeDevice();

  const texture = device.createTexture({
    size: { width: 64, height: 32, depthOrArrayLayers: 6 },
    format: 'rgba8unorm',
    usage: 0x04 | 0x10,
    mipLevelCount: 4,
    sampleCount: 1,
    dimension: '2d',
  });

  assert.equal(texture.width, 64);
  assert.equal(texture.height, 32);
  assert.equal(texture.depthOrArrayLayers, 6);
  assert.equal(texture.mipLevelCount, 4);
  assert.equal(texture.sampleCount, 1);
  assert.equal(texture.dimension, '2d');
  assert.equal(texture.format, 'rgba8unorm');
  assert.equal(texture.usage, 0x14);
});

test('the spec defaults are filled in (for what was omitted)', async () => {
  installNativeMock();
  const device = await makeDevice();

  const texture = device.createTexture({
    size: { width: 8, height: 8 }, format: 'rgba8unorm', usage: 0x10,
  });

  assert.equal(texture.depthOrArrayLayers, 1);
  assert.equal(texture.mipLevelCount, 1);
  assert.equal(texture.sampleCount, 1);
  assert.equal(texture.dimension, '2d', 'the spec default dimension is 2d');
});

test('textureBindingViewDimension is derived from the layer count', async () => {
  installNativeMock();
  const device = await makeDevice();

  const flat = device.createTexture({
    size: { width: 8, height: 8 }, format: 'rgba8unorm', usage: 0x04,
  });
  assert.equal(flat.textureBindingViewDimension, '2d');

  const layered = device.createTexture({
    size: { width: 8, height: 8, depthOrArrayLayers: 4 }, format: 'rgba8unorm', usage: 0x04,
  });
  assert.equal(layered.textureBindingViewDimension, '2d-array', 'with layers it is an array view');

  // Stated explicitly, it is used as is (for cases the layer count alone cannot decide, like a cube map).
  const cube = device.createTexture({
    size: { width: 8, height: 8, depthOrArrayLayers: 6 },
    format: 'rgba8unorm', usage: 0x04, textureBindingViewDimension: 'cube',
  });
  assert.equal(cube.textureBindingViewDimension, 'cube');
});

test('GPUBuffer.mapState follows the three states', async () => {
  const state = installNativeMock({ readBufferResult: { ok: true, data: new ArrayBuffer(16) } });
  const device = await makeDevice();

  const buffer = device.createBuffer({ size: 16, usage: 0x0001 /* MAP_READ */ });
  assert.equal(buffer.mapState, 'unmapped');

  const pending = buffer.mapAsync();
  assert.equal(buffer.mapState, 'pending', 'it is pending while waiting for a result');

  await pending;
  assert.equal(buffer.mapState, 'mapped');

  buffer.unmap();
  assert.equal(buffer.mapState, 'unmapped');
  assert.ok(state.readBufferCalls.length > 0);
});

test('a mappedAtCreation buffer has mapState mapped too', async () => {
  installNativeMock();
  const device = await makeDevice();

  const buffer = device.createBuffer({ size: 16, usage: 0x0020, mappedAtCreation: true });
  assert.equal(buffer.mapState, 'mapped');
  new Uint8Array(buffer.getMappedRange())[0] = 7;
  buffer.unmap();
  assert.equal(buffer.mapState, 'unmapped');
});
