/**
 * flush의 `present` 표시 — 어떤 배치가 프레임 제출이고 어떤 배치가 내부 제출인가.
 *
 * `popErrorScope`·`mapAsync`는 결과를 받으려고 프레임 중간에 제출한다. 이 배치가 프레임
 * 제출로 취급되면, 획득해 둔 캔버스 텍스처가 그리기도 전에 present되고 핸들이 만료되어
 * 뒤따르는 출력 패스가 통째로 거부된다 (Three.js 지연 파이프라인 생성에서 실제로 난 사고).
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { installNativeMock, makeDevice } from './helpers.mjs';

test('queue.submit은 프레임 제출(present: true)로 나간다', async () => {
  const state = installNativeMock();
  const device = await makeDevice();

  device.createBuffer({ size: 16, usage: 0x40 });
  device.queue.submit([]);

  assert.equal(state.executeCalls.length, 1);
  assert.equal(state.executeCalls[0].present, true);
});

test('popErrorScope의 즉시 제출은 내부 제출(present: false)이다', async () => {
  const state = installNativeMock();
  const device = await makeDevice();

  device.pushErrorScope('validation');
  await device.popErrorScope();

  assert.equal(state.executeCalls.length, 1);
  assert.equal(state.executeCalls[0].present, false);

  // 뒤따르는 진짜 프레임 제출은 여전히 present: true다.
  device.createBuffer({ size: 16, usage: 0x40 });
  device.queue.submit([]);
  assert.equal(state.executeCalls[1].present, true);
});

test('mapAsync의 즉시 제출도 내부 제출이다', async () => {
  const state = installNativeMock({ readBufferResult: { ok: true, data: new ArrayBuffer(16) } });
  const device = await makeDevice();

  const buffer = device.createBuffer({ size: 16, usage: 0x0001 /* MAP_READ */ });
  await buffer.mapAsync();

  assert.equal(state.executeCalls.length, 1);
  assert.equal(state.executeCalls[0].present, false);
});
