/**
 * Resource lifetime — automatic release via GC (FinalizationRegistry) and the objects count coming through.
 *
 * The GC tests need `node --expose-gc` (`npm test` turns it on). If the engine does not meet the
 * conditions they are skipped — automatic release is itself "a safety net that turns on where supported".
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import gpu, { GPUBufferUsage } from '../webgpu.js';
import { installNativeMock, makeDevice } from './helpers.mjs';

const canAutoRelease =
  typeof globalThis.gc === 'function' && typeof FinalizationRegistry === 'function';

/** Repeats gc → yielding to the finalizer task → submitting, waiting for a condition. */
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

test('a wrapper lost to GC produces a destroy command', { skip: !canAutoRelease }, async () => {
  const state = installNativeMock();
  const device = await makeDevice();

  let droppedIds = [];
  (function allocateAndDrop() {
    for (let index = 0; index < 50; index += 1) {
      droppedIds.push(device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM }).id);
    }
  })();
  device.queue.submit([]); // send the creation command out first

  const sawDestroy = await collectUntil(device, () =>
    destroyOps(state).some((c) => droppedIds.includes(c.id))
  );
  assert.ok(sawDestroy, 'the destroy of an unreachable buffer must ride out on a submission');
});

test('after an explicit destroy, GC does not produce a duplicate', { skip: !canAutoRelease }, async () => {
  const state = installNativeMock();
  const device = await makeDevice();

  let id = 0;
  (function allocateDestroyAndDrop() {
    const buffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM });
    id = buffer.id;
    buffer.destroy();
  })();
  device.queue.submit([]);

  await collectUntil(device, () => false, 10); // even running gc many times
  const count = destroyOps(state).filter((c) => c.id === id).length;
  assert.equal(count, 1, 'destroy must be the one explicit call (unregister confirmed)');
});

test('a swapchain texture and its views are not targets for automatic GC release', { skip: !canAutoRelease }, async () => {
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
  assert.deepEqual(leaked, [], 'native reclaims frame-scoped handles — no destroy may be sent');
});

test('the native live object count (objects) comes through on the submit result', async () => {
  installNativeMock({ executeResult: { ok: true, commandCount: 1, objects: 7 } });
  const device = await makeDevice();
  device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM });

  const result = device.queue.submit([]);
  assert.equal(result.objects, 7);
});
