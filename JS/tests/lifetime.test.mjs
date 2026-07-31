/**
 * 리소스 수명 — GC 자동 해제(FinalizationRegistry)와 objects 카운트 전달.
 *
 * GC 테스트는 `node --expose-gc`가 필요하다 (`npm test`가 켜 준다). 엔진이 조건을
 * 만족하지 않으면 건너뛴다 — 자동 해제 자체가 "지원되면 켜지는 안전망"이기 때문이다.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import gpu, { GPUBufferUsage } from '../webgpu.js';
import { installNativeMock, makeDevice } from './helpers.mjs';

const canAutoRelease =
  typeof globalThis.gc === 'function' && typeof FinalizationRegistry === 'function';

/** gc → 파이널라이저 태스크 양보 → 제출을 반복하며 조건을 기다린다. */
async function collectUntil(device, predicate, rounds = 50) {
  for (let round = 0; round < rounds; round += 1) {
    globalThis.gc();
    await new Promise((resolve) => setTimeout(resolve, 5));
    device.queue.submit([]);
    if (predicate()) return true;
  }
  return predicate();
}

function destroyOps(state) {
  return state.executeCalls.flatMap((call) => call.commands.filter((c) => c.op === 'destroy'));
}

test('GC로 사라진 래퍼가 destroy 명령을 만든다', { skip: !canAutoRelease }, async () => {
  const state = installNativeMock();
  const device = await makeDevice();

  let droppedIds = [];
  (function allocateAndDrop() {
    for (let index = 0; index < 50; index += 1) {
      droppedIds.push(device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM }).id);
    }
  })();
  device.queue.submit([]); // 생성 명령을 먼저 내보낸다

  const sawDestroy = await collectUntil(device, () =>
    destroyOps(state).some((c) => droppedIds.includes(c.id))
  );
  assert.ok(sawDestroy, '도달 불가가 된 버퍼의 destroy가 제출에 실려야 한다');
});

test('명시적 destroy 후에는 GC가 중복 destroy를 만들지 않는다', { skip: !canAutoRelease }, async () => {
  const state = installNativeMock();
  const device = await makeDevice();

  let id = 0;
  (function allocateDestroyAndDrop() {
    const buffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM });
    id = buffer.id;
    buffer.destroy();
  })();
  device.queue.submit([]);

  await collectUntil(device, () => false, 10); // gc를 여러 번 돌려도
  const count = destroyOps(state).filter((c) => c.id === id).length;
  assert.equal(count, 1, 'destroy는 명시적 1회여야 한다 (unregister 확인)');
});

test('스왑체인 텍스처와 그 뷰는 GC 자동 해제 대상이 아니다', { skip: !canAutoRelease }, async () => {
  const state = installNativeMock({
    executeResult: (payload) => ({
      ok: true,
      commandCount: payload.commands.length,
      canvases: { fs: { width: 32, height: 32 } },
    }),
  });
  const device = await makeDevice();
  const context = gpu.getCanvasContext('fs');
  context.configure({ device, format: 'bgra8unorm' });

  let frameIds = [];
  (function frameAndDrop() {
    const texture = context.getCurrentTexture();
    const view = texture.createView();
    frameIds = [texture.id, view.id];
  })();
  device.queue.submit([]);

  await collectUntil(device, () => false, 10);
  const leaked = destroyOps(state).filter((c) => frameIds.includes(c.id));
  assert.deepEqual(leaked, [], '프레임 스코프 핸들은 네이티브가 회수한다 — destroy를 보내면 안 된다');
});

test('submit 결과에 네이티브 live 객체 수(objects)가 그대로 온다', async () => {
  installNativeMock({ executeResult: { ok: true, commandCount: 1, objects: 7 } });
  const device = await makeDevice();
  device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM });

  const result = device.queue.submit([]);
  assert.equal(result.objects, 7);
});
