/**
 * 디버그 마커 — 명세의 `GPUDebugCommandsMixin`.
 *
 * 커맨드 인코더와 패스·번들 인코더가 **함께** 가져야 한다. 클래스 계층이 갈려 있어
 * (커맨드 인코더는 패스 인코더를 상속하지 않는다) 한쪽만 넣기 쉬운 자리다.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { installNativeMock, makeDevice, commandsOf } from './helpers.mjs';

test('커맨드 인코더에 마커를 기록한다', async () => {
  const state = installNativeMock();
  const device = await makeDevice();

  const encoder = device.createCommandEncoder();
  encoder.pushDebugGroup('업로드');
  encoder.insertDebugMarker('표식');
  encoder.popDebugGroup();
  device.queue.submit([encoder.finish()]);

  const ops = commandsOf(state).map((command) => command.op);
  assert.deepEqual(ops, ['pushDebugGroup', 'insertDebugMarker', 'popDebugGroup']);
  const [push, marker] = commandsOf(state);
  assert.equal(push.groupLabel, '업로드');
  assert.equal(marker.markerLabel, '표식');
});

test('렌더 패스 인코더에도 있다', async () => {
  const state = installNativeMock();
  const device = await makeDevice();
  const texture = device.createTexture({
    size: { width: 4, height: 4 }, format: 'rgba8unorm', usage: 0x10,
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    colorAttachments: [{ view: texture.createView(), loadOp: 'clear', storeOp: 'store' }],
  });
  pass.pushDebugGroup('메인 패스');
  pass.popDebugGroup();
  pass.end();
  device.queue.submit([encoder.finish()]);

  const ops = commandsOf(state).map((command) => command.op);
  assert.ok(ops.indexOf('pushDebugGroup') > ops.indexOf('beginRenderPass'), '패스 안에 들어가야 한다');
});

test('컴퓨트 패스와 번들 인코더에도 있다', async () => {
  installNativeMock();
  const device = await makeDevice();

  const encoder = device.createCommandEncoder();
  const compute = encoder.beginComputePass();
  assert.equal(typeof compute.pushDebugGroup, 'function');
  assert.equal(typeof compute.insertDebugMarker, 'function');
  compute.end();

  const bundle = device.createRenderBundleEncoder({ colorFormats: ['rgba8unorm'] });
  assert.equal(typeof bundle.pushDebugGroup, 'function');
  assert.equal(typeof bundle.popDebugGroup, 'function');
});

test('번들에 기록한 마커는 번들 명령에 실린다', async () => {
  const state = installNativeMock();
  const device = await makeDevice();

  const bundleEncoder = device.createRenderBundleEncoder({ colorFormats: ['rgba8unorm'] });
  bundleEncoder.pushDebugGroup('번들 구간');
  bundleEncoder.popDebugGroup();
  bundleEncoder.finish();
  device.queue.submit([]);

  const create = commandsOf(state).find((command) => command.op === 'createRenderBundle');
  const ops = create.commands.map((command) => command.op);
  assert.deepEqual(ops, ['pushDebugGroup', 'popDebugGroup']);
});
