/**
 * 명세가 정한 **읽기 전용 속성**들 — 웹 코드가 객체를 받아 스스로 판단할 때 읽는다.
 *
 * 없으면 `undefined`가 나가고, 그쪽은 "값이 0이다/설정 안 됐다"로 오해해 잘못된 분기를 탄다.
 * (three.js는 밉맵 경로에서 `texture.textureBindingViewDimension`을 읽어 뷰를 만든다.)
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { installNativeMock, makeDevice } from './helpers.mjs';

test('GPUTexture가 명세의 읽기 전용 속성을 갖는다', async () => {
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

test('명세 기본값이 채워진다 (생략한 것들)', async () => {
  installNativeMock();
  const device = await makeDevice();

  const texture = device.createTexture({
    size: { width: 8, height: 8 }, format: 'rgba8unorm', usage: 0x10,
  });

  assert.equal(texture.depthOrArrayLayers, 1);
  assert.equal(texture.mipLevelCount, 1);
  assert.equal(texture.sampleCount, 1);
  assert.equal(texture.dimension, '2d', '명세 기본 dimension은 2d다');
});

test('textureBindingViewDimension은 레이어 수에서 정해진다', async () => {
  installNativeMock();
  const device = await makeDevice();

  const flat = device.createTexture({
    size: { width: 8, height: 8 }, format: 'rgba8unorm', usage: 0x04,
  });
  assert.equal(flat.textureBindingViewDimension, '2d');

  const layered = device.createTexture({
    size: { width: 8, height: 8, depthOrArrayLayers: 4 }, format: 'rgba8unorm', usage: 0x04,
  });
  assert.equal(layered.textureBindingViewDimension, '2d-array', '레이어가 있으면 배열 뷰다');

  // 명시하면 그대로 쓴다 (큐브맵처럼 레이어 수만으로는 못 정하는 경우).
  const cube = device.createTexture({
    size: { width: 8, height: 8, depthOrArrayLayers: 6 },
    format: 'rgba8unorm', usage: 0x04, textureBindingViewDimension: 'cube',
  });
  assert.equal(cube.textureBindingViewDimension, 'cube');
});

test('GPUBuffer.mapState가 세 상태를 그대로 따라간다', async () => {
  const state = installNativeMock({ readBufferResult: { ok: true, data: new ArrayBuffer(16) } });
  const device = await makeDevice();

  const buffer = device.createBuffer({ size: 16, usage: 0x0001 /* MAP_READ */ });
  assert.equal(buffer.mapState, 'unmapped');

  const pending = buffer.mapAsync();
  assert.equal(buffer.mapState, 'pending', '결과를 기다리는 동안은 pending이다');

  await pending;
  assert.equal(buffer.mapState, 'mapped');

  buffer.unmap();
  assert.equal(buffer.mapState, 'unmapped');
  assert.ok(state.readBufferCalls.length > 0);
});

test('mappedAtCreation 버퍼도 mapState가 mapped다', async () => {
  installNativeMock();
  const device = await makeDevice();

  const buffer = device.createBuffer({ size: 16, usage: 0x0020, mappedAtCreation: true });
  assert.equal(buffer.mapState, 'mapped');
  new Uint8Array(buffer.getMappedRange())[0] = 7;
  buffer.unmap();
  assert.equal(buffer.mapState, 'unmapped');
});
