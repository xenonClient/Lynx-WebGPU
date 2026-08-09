/**
 * Debug markers — the spec's `GPUDebugCommandsMixin`.
 *
 * The command encoder and the pass/bundle encoders must have them **together**. The class hierarchies are
 * split (a command encoder does not inherit from a pass encoder), so it is an easy place to add to one side only.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { installNativeMock, makeDevice, commandsOf } from './helpers.mjs';

test('records a marker on the command encoder', async () => {
  const state = installNativeMock();
  const device = await makeDevice();

  const encoder = device.createCommandEncoder();
  encoder.pushDebugGroup('upload');
  encoder.insertDebugMarker('marker');
  encoder.popDebugGroup();
  device.queue.submit([encoder.finish()]);

  const ops = commandsOf(state).map((command) => command.op);
  assert.deepEqual(ops, ['pushDebugGroup', 'insertDebugMarker', 'popDebugGroup']);
  const [push, marker] = commandsOf(state);
  assert.equal(push.groupLabel, 'upload');
  assert.equal(marker.markerLabel, 'marker');
});

test('the render pass encoder has them too', async () => {
  const state = installNativeMock();
  const device = await makeDevice();
  const texture = device.createTexture({
    size: { width: 4, height: 4 }, format: 'rgba8unorm', usage: 0x10,
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    colorAttachments: [{ view: texture.createView(), loadOp: 'clear', storeOp: 'store' }],
  });
  pass.pushDebugGroup('main pass');
  pass.popDebugGroup();
  pass.end();
  device.queue.submit([encoder.finish()]);

  const ops = commandsOf(state).map((command) => command.op);
  assert.ok(ops.indexOf('pushDebugGroup') > ops.indexOf('beginRenderPass'), 'it has to go inside the pass');
});

test('the compute pass and bundle encoders have them too', async () => {
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

test('a marker recorded into a bundle rides on the bundle commands', async () => {
  const state = installNativeMock();
  const device = await makeDevice();

  const bundleEncoder = device.createRenderBundleEncoder({ colorFormats: ['rgba8unorm'] });
  bundleEncoder.pushDebugGroup('bundle range');
  bundleEncoder.popDebugGroup();
  bundleEncoder.finish();
  device.queue.submit([]);

  const create = commandsOf(state).find((command) => command.op === 'createRenderBundle');
  const ops = create.commands.map((command) => command.op);
  assert.deepEqual(ops, ['pushDebugGroup', 'popDebugGroup']);
});
