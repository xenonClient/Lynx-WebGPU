/**
 * The command payloads of a query set.
 *
 * A query can only be attached **when opening a pass**, so whether the descriptor is properly turned into a
 * handle matters especially — an object riding on as is gets turned into something strange by Lynx's value converter and breaks silently.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { installNativeMock, makeDevice, commandsOf } from './helpers.mjs';

async function setUp() {
  const state = installNativeMock();
  const device = await makeDevice();
  const view = device
    .createTexture({ size: { width: 1, height: 1 }, format: 'rgba8unorm', usage: 0x10 })
    .createView();
  return { state, device, view, encoder: device.createCommandEncoder() };
}

const findOp = (state, op) => commandsOf(state).find((command) => command.op === op);

test('createQuerySet puts the kind and count on', async () => {
  const { state, device } = await setUp();
  const querySet = device.createQuerySet({ type: 'occlusion', count: 4, label: 'visible' });
  device.queue.submit([]);

  const command = findOp(state, 'createQuerySet');
  assert.equal(command.id, querySet.id);
  assert.equal(command.type, 'occlusion');
  assert.equal(command.count, 4);
  assert.equal(command.label, 'visible');
  // The properties web code reads must be there as well.
  assert.equal(querySet.type, 'occlusion');
  assert.equal(querySet.count, 4);
});

test('occlusionQuerySet becomes a handle and rides on beginRenderPass', async () => {
  const { state, device, view, encoder } = await setUp();
  const querySet = device.createQuerySet({ type: 'occlusion', count: 2 });
  const pass = encoder.beginRenderPass({ colorAttachments: [{ view }], occlusionQuerySet: querySet });
  pass.beginOcclusionQuery(1);
  pass.draw(3);
  pass.endOcclusionQuery();
  pass.end();
  device.queue.submit([encoder.finish()]);

  assert.equal(findOp(state, 'beginRenderPass').occlusionQuerySet, querySet.id);
  assert.equal(findOp(state, 'beginOcclusionQuery').queryIndex, 1);
  assert.ok(findOp(state, 'endOcclusionQuery'));
});

test('the query sets in timestampWrites become handles too', async () => {
  const { state, device, view, encoder } = await setUp();
  const querySet = device.createQuerySet({ type: 'timestamp', count: 2 });
  encoder
    .beginRenderPass({
      colorAttachments: [{ view }],
      timestampWrites: { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 },
    })
    .end();
  encoder.beginComputePass({ timestampWrites: { querySet, endOfPassWriteIndex: 1 } }).end();
  device.queue.submit([encoder.finish()]);

  const render = findOp(state, 'beginRenderPass').timestampWrites;
  assert.equal(render.querySet, querySet.id, 'it must be a handle, not an object');
  assert.equal(render.beginningOfPassWriteIndex, 0);
  assert.equal(render.endOfPassWriteIndex, 1);

  const compute = findOp(state, 'beginComputePass').timestampWrites;
  assert.equal(compute.querySet, querySet.id);
  assert.equal(compute.beginningOfPassWriteIndex, undefined, 'an omitted slot is left empty');
});

test('resolveQuerySet puts the range and destination on', async () => {
  const { state, device, encoder } = await setUp();
  const querySet = device.createQuerySet({ type: 'occlusion', count: 4 });
  const destination = device.createBuffer({ size: 512, usage: 0x0200 });
  encoder.resolveQuerySet(querySet, 1, 2, destination, 256);
  device.queue.submit([encoder.finish()]);

  const command = findOp(state, 'resolveQuerySet');
  assert.equal(command.querySet, querySet.id);
  assert.equal(command.firstQuery, 1);
  assert.equal(command.queryCount, 2);
  assert.equal(command.destination, destination.id);
  assert.equal(command.destinationOffset, 256);
});

test('adapter.features reports features through has', async () => {
  installNativeMock();
  globalThis.NativeModules.WebGPU.adapterInfo = () => ({
    ok: true, name: 'mock-gpu', backend: 'metal', limits: {}, features: ['timestamp-query'],
  });
  const gpu = (await import('../webgpu.js')).default;
  const adapter = await gpu.requestAdapter();

  assert.equal(adapter.features.has('timestamp-query'), true);
  assert.equal(adapter.features.has('texture-compression-astc'), false);
});

test('the bridge crossings stay at one per frame even with queries', async () => {
  const { state, device, view, encoder } = await setUp();
  const querySet = device.createQuerySet({ type: 'occlusion', count: 1 });
  const destination = device.createBuffer({ size: 256 + 8, usage: 0x0200 });
  const pass = encoder.beginRenderPass({ colorAttachments: [{ view }], occlusionQuerySet: querySet });
  pass.beginOcclusionQuery(0);
  pass.draw(3);
  pass.endOcclusionQuery();
  pass.end();
  encoder.resolveQuerySet(querySet, 0, 1, destination, 256);
  device.queue.submit([encoder.finish()]);

  assert.equal(state.executeCalls.length, 1);
});
