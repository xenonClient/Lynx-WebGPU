/**
 * 기록 시점 스냅샷 — 호출 뒤 디스크립터 재사용이 스트림을 오염시키지 못한다.
 *
 * 브라우저 WebGPU는 호출 시점에 인자를 직렬화한다. three.js는 그 계약에 기대
 * 싱글턴 디스크립터를 인코딩 직후 reset()하는데, shim이 참조를 들고 있으면
 * copySize가 flush 전에 0이 되어 **폭 0짜리 복사가 오류 없이** 나간다
 * (실제로 렌더 타깃 리드백이 전부 (0,0,0)으로 나온 사고다).
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { installNativeMock, makeDevice, commandsOf } from './helpers.mjs';

test('인코더 명령: 호출 뒤 copySize·origin을 리셋해도 기록은 그대로다', async () => {
  const state = installNativeMock();
  const device = await makeDevice();

  const texture = device.createTexture({
    size: { width: 8, height: 8 }, format: 'rgba8unorm', usage: 0x07,
  });
  const buffer = device.createBuffer({ size: 256, usage: 0x09 });

  // three.js 스타일 — 싱글턴 디스크립터를 쓰고 인코딩 직후 리셋한다.
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
  assert.equal(command.copySize.width, 1, '리셋된 width가 스트림에 새면 폭 0짜리 복사가 나간다');
  assert.equal(command.copySize.height, 1);
  assert.equal(command.source.origin.x, 4);
  assert.equal(command.source.origin.y, 4);
});

test('beginRenderPass: 호출 뒤 clearValue를 바꿔도 기록은 그대로다', async () => {
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

test('큐 명령: 호출 뒤 writeTexture의 origin·size를 리셋해도 기록은 그대로다', async () => {
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
