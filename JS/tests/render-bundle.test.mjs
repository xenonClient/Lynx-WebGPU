/**
 * 렌더 번들의 커맨드 페이로드와 인코더 경계.
 *
 * 번들에 담을 수 없는 명령(뷰포트·시저·블렌드 상수·스텐실 참조·중첩 번들)은 **번들 인코더에
 * 메서드 자체가 없어야 한다** — 네이티브까지 가서 거부당하는 것보다 여기서 막히는 편이 낫다.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { installNativeMock, makeDevice, commandsOf } from './helpers.mjs';

async function setUp() {
  const state = installNativeMock();
  const device = await makeDevice();
  return { state, device };
}

function bundleCommand(state) {
  return commandsOf(state).find((command) => command.op === 'createRenderBundle');
}

test('finish가 모아 둔 명령과 패스 모양을 함께 싣는다', async () => {
  const { state, device } = await setUp();
  const encoder = device.createRenderBundleEncoder({
    colorFormats: ['rgba8unorm'],
    depthStencilFormat: 'depth24plus',
    sampleCount: 4,
    label: 'scenery',
  });
  encoder.draw(3);
  const bundle = encoder.finish();
  device.queue.submit([]);

  const command = bundleCommand(state);
  assert.ok(command, 'createRenderBundle 명령이 있어야 한다');
  assert.equal(command.id, bundle.id);
  assert.deepEqual(command.colorFormats, ['rgba8unorm']);
  assert.equal(command.depthStencilFormat, 'depth24plus');
  assert.equal(command.sampleCount, 4);
  assert.equal(command.label, 'scenery');
  assert.deepEqual(
    command.commands.map((entry) => entry.op),
    ['draw'],
    '번들 안 명령이 그대로 실린다'
  );
});

test('번들 인코더에는 패스 전용 명령이 아예 없다', async () => {
  const { device } = await setUp();
  const encoder = device.createRenderBundleEncoder({ colorFormats: ['rgba8unorm'] });

  for (const method of ['setViewport', 'setScissorRect', 'setBlendConstant', 'setStencilReference',
    'executeBundles', 'end']) {
    assert.equal(
      typeof (/** @type {any} */ (encoder))[method], 'undefined',
      `번들 인코더에 ${method}가 있으면 안 된다`
    );
  }
  // 담을 수 있는 것들은 그대로 있어야 한다.
  for (const method of ['setPipeline', 'setBindGroup', 'setVertexBuffer', 'setIndexBuffer',
    'draw', 'drawIndexed', 'drawIndirect', 'drawIndexedIndirect']) {
    assert.equal(typeof (/** @type {any} */ (encoder))[method], 'function', method);
  }
});

test('번들 명령은 패스 스트림과 섞이지 않는다', async () => {
  const { state, device } = await setUp();
  const commandEncoder = device.createCommandEncoder();
  const view = device.createTexture({ size: { width: 1, height: 1 }, format: 'rgba8unorm', usage: 0x10 })
    .createView();
  const pass = commandEncoder.beginRenderPass({ colorAttachments: [{ view }] });

  const bundleEncoder = device.createRenderBundleEncoder({ colorFormats: ['rgba8unorm'] });
  bundleEncoder.draw(6);
  const bundle = bundleEncoder.finish();

  pass.draw(3);
  pass.executeBundles([bundle]);
  pass.end();
  device.queue.submit([commandEncoder.finish()]);

  const stream = commandsOf(state);
  const execute = stream.find((command) => command.op === 'executeBundles');
  assert.deepEqual(execute.bundles, [bundle.id]);
  // 번들 안의 draw(6)는 패스 스트림이 아니라 createRenderBundle 안에만 있어야 한다.
  const passDraws = stream.filter((command) => command.op === 'draw');
  assert.deepEqual(passDraws.map((command) => command.vertexCount), [3]);
  assert.deepEqual(bundleCommand(state).commands.map((entry) => entry.vertexCount), [6]);
});

test('createRenderBundle이 executeBundles보다 먼저 스트림에 온다', async () => {
  const { state, device } = await setUp();
  const commandEncoder = device.createCommandEncoder();
  const view = device.createTexture({ size: { width: 1, height: 1 }, format: 'rgba8unorm', usage: 0x10 })
    .createView();
  const pass = commandEncoder.beginRenderPass({ colorAttachments: [{ view }] });

  // 패스를 이미 연 뒤에 번들을 만든다 — 그래도 네이티브는 번들을 먼저 봐야 한다.
  const bundleEncoder = device.createRenderBundleEncoder({ colorFormats: ['rgba8unorm'] });
  bundleEncoder.draw(3);
  pass.executeBundles([bundleEncoder.finish()]);
  pass.end();
  device.queue.submit([commandEncoder.finish()]);

  const ops = commandsOf(state).map((command) => command.op);
  assert.ok(
    ops.indexOf('createRenderBundle') < ops.indexOf('executeBundles'),
    `번들이 먼저 등록돼야 한다: ${ops.join(', ')}`
  );
});

test('finish는 명령을 비워 두 번째 번들에 새지 않는다', async () => {
  const { state, device } = await setUp();
  const encoder = device.createRenderBundleEncoder({ colorFormats: ['rgba8unorm'] });
  encoder.draw(3);
  encoder.finish();
  encoder.draw(6);
  encoder.finish();
  device.queue.submit([]);

  const bundles = commandsOf(state).filter((command) => command.op === 'createRenderBundle');
  assert.deepEqual(bundles[0].commands.map((entry) => entry.vertexCount), [3]);
  assert.deepEqual(bundles[1].commands.map((entry) => entry.vertexCount), [6]);
});

test('번들을 써도 프레임당 브리지 왕복은 1회다', async () => {
  const { state, device } = await setUp();
  const bundleEncoder = device.createRenderBundleEncoder({ colorFormats: ['rgba8unorm'] });
  bundleEncoder.draw(3);
  const bundle = bundleEncoder.finish();

  const commandEncoder = device.createCommandEncoder();
  const view = device.createTexture({ size: { width: 1, height: 1 }, format: 'rgba8unorm', usage: 0x10 })
    .createView();
  const pass = commandEncoder.beginRenderPass({ colorAttachments: [{ view }] });
  pass.executeBundles([bundle]);
  pass.end();
  device.queue.submit([commandEncoder.finish()]);

  assert.equal(state.executeCalls.length, 1);
});
