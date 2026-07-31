/**
 * 캔버스 크기 캐시 — "프레임당 브리지 왕복 1회" 계약 검증.
 *
 * `getCurrentTexture()`/`getSize()`가 매 프레임 동기 `canvasInfo`를 부르지 않고,
 * `execute` 응답의 `canvases`로 갱신되는 캐시를 읽는지 단언한다.
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

test('정상 프레임 루프에서 동기 canvasInfo는 configure 때 1회뿐이다', async () => {
  const size = { width: 300, height: 150 };
  const state = mockWithCanvas('steady', size);
  const device = await makeDevice();
  const context = gpu.getCanvasContext('steady');
  context.configure({ device, format: 'bgra8unorm' });
  assert.equal(state.canvasInfoCalls, 1, 'configure가 캐시를 씨딩한다');

  for (let frame = 0; frame < 5; frame += 1) {
    const texture = context.getCurrentTexture();
    texture.createView();
    context.getSize();
    device.queue.submit([]);
  }

  assert.equal(state.canvasInfoCalls, 1, '프레임 안에서는 추가 동기 호출이 없어야 한다');
  // configure는 기록만 하고 첫 submit에 합쳐지므로 execute는 프레임 수만큼이다.
  assert.equal(state.executeCalls.length, 5, '프레임당 execute 1회');
});

test('execute 응답의 canvases가 캐시를 갱신한다 (리사이즈 반영)', async () => {
  const size = { width: 320, height: 240 };
  const state = mockWithCanvas('resize', size);
  const device = await makeDevice();
  const context = gpu.getCanvasContext('resize');
  context.configure({ device, format: 'bgra8unorm' });

  context.getCurrentTexture();
  device.queue.submit([]);
  assert.deepEqual(context.getSize(), { width: 320, height: 240 });

  // 네이티브 쪽에서 레이아웃이 바뀌었다 — 다음 제출 응답에 새 크기가 실린다.
  size.width = 640;
  size.height = 480;
  context.getCurrentTexture();
  device.queue.submit([]);

  assert.deepEqual(context.getSize(), { width: 640, height: 480 });
  assert.equal(state.canvasInfoCalls, 1, '리사이즈 반영에 동기 호출이 필요 없다');
});

test('캐시가 비어 있으면 getSize가 1회 동기 조회 후 캐시한다', async () => {
  const state = mockWithCanvas('fallback', { width: 100, height: 50 });
  await makeDevice();
  const context = gpu.getCanvasContext('fallback');

  assert.deepEqual(context.getSize(), { width: 100, height: 50 });
  assert.equal(state.canvasInfoCalls, 1);
  context.getSize();
  assert.equal(state.canvasInfoCalls, 1, '두 번째부터는 캐시를 읽는다');
});

test('표면 등록 전(크기 0)은 캐시하지 않고 다음 조회가 다시 시도한다', async () => {
  const state = installNativeMock({
    canvasInfoResult: { ok: false, errors: [{ kind: 'validation', message: '없음' }] },
  });
  await makeDevice();
  const context = gpu.getCanvasContext('unregistered');

  assert.deepEqual(context.getSize(), { width: 0, height: 0 });
  context.getSize();
  assert.equal(state.canvasInfoCalls, 2, '실패한 조회는 캐시되지 않는다');
});

test('getCurrentTexture의 텍스처 크기가 캐시에서 온다', async () => {
  mockWithCanvas('dims', { width: 128, height: 64 });
  const device = await makeDevice();
  const context = gpu.getCanvasContext('dims');
  context.configure({ device, format: 'bgra8unorm' });

  const texture = context.getCurrentTexture();
  assert.equal(texture.width, 128);
  assert.equal(texture.height, 64);
});

test('device.destroy가 캐시를 비운다', async () => {
  const state = mockWithCanvas('destroyed', { width: 10, height: 10 });
  const device = await makeDevice();
  const context = gpu.getCanvasContext('destroyed');
  context.configure({ device, format: 'bgra8unorm' });
  assert.equal(state.canvasInfoCalls, 1);

  device.destroy();
  assert.equal(state.resetCalls, 1);
  context.getSize();
  assert.equal(state.canvasInfoCalls, 2, '캐시가 비워졌으므로 다시 동기 조회한다');
});
