/**
 * A render bundle's command payload and the encoder boundary.
 *
 * The commands that cannot go in a bundle (viewport, scissor, blend constant, stencil reference, a nested
 * bundle) **must have no method on the bundle encoder at all** — better blocked here than rejected all the way at native.
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

test('finish puts the gathered commands and the pass shape on together', async () => {
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
  assert.ok(command, 'there must be a createRenderBundle command');
  assert.equal(command.id, bundle.id);
  assert.deepEqual(command.colorFormats, ['rgba8unorm']);
  assert.equal(command.depthStencilFormat, 'depth24plus');
  assert.equal(command.sampleCount, 4);
  assert.equal(command.label, 'scenery');
  assert.deepEqual(
    command.commands.map((entry) => entry.op),
    ['draw'],
    'the commands inside the bundle ride on as they are'
  );
});

test('the bundle encoder has no pass-only commands at all', async () => {
  const { device } = await setUp();
  const encoder = device.createRenderBundleEncoder({ colorFormats: ['rgba8unorm'] });

  for (const method of ['setViewport', 'setScissorRect', 'setBlendConstant', 'setStencilReference',
    'executeBundles', 'end']) {
    assert.equal(
      typeof (/** @type {any} */ (encoder))[method], 'undefined',
      `the bundle encoder must not have ${method}`
    );
  }
  // What can go in must still be there.
  for (const method of ['setPipeline', 'setBindGroup', 'setVertexBuffer', 'setIndexBuffer',
    'draw', 'drawIndexed', 'drawIndirect', 'drawIndexedIndirect']) {
    assert.equal(typeof (/** @type {any} */ (encoder))[method], 'function', method);
  }
});

test('bundle commands do not mix with the pass stream', async () => {
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
  // The draw(6) inside the bundle must be inside createRenderBundle only, not the pass stream.
  const passDraws = stream.filter((command) => command.op === 'draw');
  assert.deepEqual(passDraws.map((command) => command.vertexCount), [3]);
  assert.deepEqual(bundleCommand(state).commands.map((entry) => entry.vertexCount), [6]);
});

test('createRenderBundle comes into the stream before executeBundles', async () => {
  const { state, device } = await setUp();
  const commandEncoder = device.createCommandEncoder();
  const view = device.createTexture({ size: { width: 1, height: 1 }, format: 'rgba8unorm', usage: 0x10 })
    .createView();
  const pass = commandEncoder.beginRenderPass({ colorAttachments: [{ view }] });

  // The bundle is built after the pass is already open — native must still see the bundle first.
  const bundleEncoder = device.createRenderBundleEncoder({ colorFormats: ['rgba8unorm'] });
  bundleEncoder.draw(3);
  pass.executeBundles([bundleEncoder.finish()]);
  pass.end();
  device.queue.submit([commandEncoder.finish()]);

  const ops = commandsOf(state).map((command) => command.op);
  assert.ok(
    ops.indexOf('createRenderBundle') < ops.indexOf('executeBundles'),
    `the bundle has to be registered first: ${ops.join(', ')}`
  );
});

test('calling finish twice is rejected and returns an invalid bundle', async () => {
  const { state, device } = await setUp();
  /** @type {{kind: string, message: string}[]} */
  const reported = [];
  device.onError((error) => reported.push(error));

  const encoder = device.createRenderBundleEncoder({ colorFormats: ['rgba8unorm'] });
  encoder.draw(3);
  const first = encoder.finish();
  const second = encoder.finish();
  device.queue.submit([]);

  // Quietly handing back an empty bundle would make executeBundles draw nothing with no error, and the cause unfindable.
  assert.equal(reported.length, 1, `the second finish is an error: ${JSON.stringify(reported)}`);
  assert.equal(reported[0].kind, 'validation');
  assert.notEqual(second.id, first.id);

  const bundles = commandsOf(state).filter((command) => command.op === 'createRenderBundle');
  assert.equal(bundles.length, 1, 'an invalid bundle is not created natively');
  assert.deepEqual(bundles[0].commands.map((entry) => entry.vertexCount), [3]);
});

test('a bundle holds on to the resource wrappers it uses', async () => {
  const { device } = await setUp();
  const buffer = device.createBuffer({ size: 16, usage: 0x20 });
  const bindGroup = { id: 77 };

  const encoder = device.createRenderBundleEncoder({ colorFormats: ['rgba8unorm'] });
  encoder.setBindGroup(0, /** @type {any} */ (bindGroup));
  encoder.setVertexBuffer(0, buffer);
  encoder.draw(3);
  const bundle = encoder.finish();

  // It is the only structure whose recorded commands outlive the wrappers — without holding them, GC slips
  // in a destroy and the bundle quietly draws nothing.
  assert.ok(bundle._retained.includes(buffer), 'it must hold the vertex buffer');
  assert.ok(bundle._retained.includes(bindGroup), 'it must hold the bind group');
  assert.deepEqual(
    device.createRenderBundleEncoder({ colorFormats: ['rgba8unorm'] }).finish()._retained,
    [],
    'it must not leak into the next encoder'
  );
});

test('the bridge crossings stay at one per frame even with a bundle', async () => {
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
