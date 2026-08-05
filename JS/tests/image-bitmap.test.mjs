/**
 * `createImageBitmap()` + `queue.copyExternalImageToTexture()` — 외부 이미지를 텍스처로.
 *
 * 웹에서는 브라우저가 디코딩을 맡는다. 여기서는 네이티브(ImageIO)가 하고 **픽셀은
 * 네이티브에 남는다** — 브리지를 건너는 것이 핸들뿐인지가 이 파일의 핵심 계약이다.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { installNativeMock, makeDevice, commandsOf } from './helpers.mjs';
import { createImageBitmap } from '../webgpu.js';

test('이미지 바이트를 네이티브에 넘기고 크기를 받는다', async () => {
  const state = installNativeMock({ decodeImageResult: { ok: true, width: 64, height: 32 } });
  await makeDevice();

  const bitmap = await createImageBitmap(new Uint8Array([1, 2, 3, 4]).buffer);

  assert.equal(bitmap.width, 64);
  assert.equal(bitmap.height, 32);
  assert.equal(state.decodeImageCalls.length, 1);
  assert.ok(state.decodeImageCalls[0].data instanceof ArrayBuffer);
  assert.equal(typeof state.decodeImageCalls[0].id, 'number', '핸들은 JS가 발급한다');
});

test('문자열은 애셋 이름으로 넘어간다', async () => {
  const state = installNativeMock();
  await makeDevice();

  await createImageBitmap('photo.jpg');

  assert.equal(state.decodeImageCalls[0].name, 'photo.jpg');
  assert.equal(state.decodeImageCalls[0].data, undefined, '이름을 줬으면 바이트를 싣지 않는다');
});

test('옵션이 네이티브 이름으로 옮겨진다', async () => {
  const state = installNativeMock();
  await makeDevice();

  await createImageBitmap('a.png', {
    flipY: true, premultiplyAlpha: 'premultiply', resizeWidth: 8, resizeHeight: 4,
  });

  const params = state.decodeImageCalls[0];
  assert.equal(params.flipY, true);
  assert.equal(params.premultiplyAlpha, true, "'premultiply'만 true다");
  assert.equal(params.resizeWidth, 8);
  assert.equal(params.resizeHeight, 4);
});

test("premultiplyAlpha가 'none'이면 곱하지 않는다", async () => {
  const state = installNativeMock();
  await makeDevice();

  await createImageBitmap('a.png', { premultiplyAlpha: 'none' });

  assert.equal(state.decodeImageCalls[0].premultiplyAlpha, false);
});

test('디코딩 실패는 던진다 — 조용히 빈 이미지를 주지 않는다', async () => {
  installNativeMock({
    decodeImageResult: { ok: false, errors: [{ kind: 'validation', message: '손상된 PNG' }] },
  });
  await makeDevice();

  await assert.rejects(() => createImageBitmap(new ArrayBuffer(4)), /손상된 PNG/);
});

test('디바이스 없이 부르면 분명한 오류를 낸다', async () => {
  // 핸들은 디바이스의 레코더가 발급한다 — 없으면 번호를 만들 수 없다.
  installNativeMock();
  const { createImageBitmap: fresh } = await import(`../webgpu.js?fresh=${Date.now()}`);
  await assert.rejects(() => fresh(new ArrayBuffer(4)), /requestDevice/);
});

test('copyExternalImageToTexture는 핸들만 싣는다', async () => {
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
  assert.ok(copy, '명령이 기록되지 않았다');
  assert.equal(copy.source.source, bitmap.id);
  assert.equal(copy.destination.texture, texture.id);
  assert.deepEqual(copy.copySize, [16, 16]);
  assert.equal(copy.data, undefined, '픽셀이 브리지를 건너면 안 된다');
});

test('부분 복사의 origin과 mipLevel이 실린다', async () => {
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

test('close()는 네이티브 픽셀을 버린다 — 두 번 불러도 한 번만', async () => {
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
