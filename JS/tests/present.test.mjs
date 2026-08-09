/**
 * flush's `present` marker — which batches are frame submissions and which are internal ones.
 *
 * `popErrorScope` and `mapAsync` submit mid-frame to get a result. If such a batch were treated as a frame
 * submission, an acquired canvas texture would be presented before it is drawn and its handle expired, so
 * the following output pass would be rejected wholesale (a real incident with three.js's deferred pipeline creation).
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { installNativeMock, makeDevice } from './helpers.mjs';

test('queue.submit goes out as a frame submission (present: true)', async () => {
  const state = installNativeMock();
  const device = await makeDevice();

  device.createBuffer({ size: 16, usage: 0x40 });
  device.queue.submit([]);

  assert.equal(state.executeCalls.length, 1);
  assert.equal(state.executeCalls[0].present, true);
});

test('popErrorScope\'s immediate submission is an internal one (present: false)', async () => {
  const state = installNativeMock();
  const device = await makeDevice();

  device.pushErrorScope('validation');
  await device.popErrorScope();

  assert.equal(state.executeCalls.length, 1);
  assert.equal(state.executeCalls[0].present, false);

  // The real frame submission that follows is still present: true.
  device.createBuffer({ size: 16, usage: 0x40 });
  device.queue.submit([]);
  assert.equal(state.executeCalls[1].present, true);
});

test('mapAsync\'s immediate submission is internal too', async () => {
  const state = installNativeMock({ readBufferResult: { ok: true, data: new ArrayBuffer(16) } });
  const device = await makeDevice();

  const buffer = device.createBuffer({ size: 16, usage: 0x0001 /* MAP_READ */ });
  await buffer.mapAsync();

  assert.equal(state.executeCalls.length, 1);
  assert.equal(state.executeCalls[0].present, false);
});
