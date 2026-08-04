/**
 * 쿼리셋의 커맨드 페이로드.
 *
 * 쿼리는 **패스를 열 때만** 붙일 수 있어서, 디스크립터가 핸들로 제대로 바뀌는지가 특히 중요하다 —
 * 객체가 그대로 실리면 Lynx 값 변환기가 이상한 것으로 만들고 오류 없이 조용히 깨진다.
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

test('createQuerySet이 종류와 개수를 싣는다', async () => {
  const { state, device } = await setUp();
  const querySet = device.createQuerySet({ type: 'occlusion', count: 4, label: 'visible' });
  device.queue.submit([]);

  const command = findOp(state, 'createQuerySet');
  assert.equal(command.id, querySet.id);
  assert.equal(command.type, 'occlusion');
  assert.equal(command.count, 4);
  assert.equal(command.label, 'visible');
  // 웹 코드가 읽는 프로퍼티도 그대로 있어야 한다.
  assert.equal(querySet.type, 'occlusion');
  assert.equal(querySet.count, 4);
});

test('occlusionQuerySet이 핸들로 바뀌어 beginRenderPass에 실린다', async () => {
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

test('timestampWrites의 쿼리셋도 핸들로 바뀐다', async () => {
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
  assert.equal(render.querySet, querySet.id, '객체가 아니라 핸들이어야 한다');
  assert.equal(render.beginningOfPassWriteIndex, 0);
  assert.equal(render.endOfPassWriteIndex, 1);

  const compute = findOp(state, 'beginComputePass').timestampWrites;
  assert.equal(compute.querySet, querySet.id);
  assert.equal(compute.beginningOfPassWriteIndex, undefined, '생략한 자리는 비워 둔다');
});

test('resolveQuerySet이 구간과 목적지를 싣는다', async () => {
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

test('adapter.features가 has로 기능을 알려 준다', async () => {
  installNativeMock();
  globalThis.NativeModules.WebGPU.adapterInfo = () => ({
    ok: true, name: 'mock-gpu', backend: 'metal', limits: {}, features: ['timestamp-query'],
  });
  const gpu = (await import('../webgpu.js')).default;
  const adapter = await gpu.requestAdapter();

  assert.equal(adapter.features.has('timestamp-query'), true);
  assert.equal(adapter.features.has('texture-compression-astc'), false);
});

test('쿼리를 써도 프레임당 브리지 왕복은 1회다', async () => {
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
