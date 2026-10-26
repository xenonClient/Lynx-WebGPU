/**
 * Spec fidelity the review found silently off — each case pins the browser's behaviour.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import gpu, { GPUBufferUsage, OperationError, GPUSupportedFeatures } from '../webgpu.js';
import { installNativeMock, makeDevice, commandsOf } from './helpers.mjs';

/** The bytes of the first command of `op` in the last batch. */
function bytesOf(state, op) {
  const command = commandsOf(state).find((entry) => entry.op === op);
  assert.ok(command, `no ${op} command`);
  assert.ok(command.data instanceof ArrayBuffer, `${op}.data must be an ArrayBuffer`);
  return Array.from(new Uint8Array(command.data));
}

test('writeBuffer honours dataOffset/size in bytes for an ArrayBuffer and in elements for an array', async () => {
  const state = installNativeMock();
  const device = await makeDevice();
  const buffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST });
  const source = new Uint8Array([0, 1, 2, 3, 4, 5, 6, 7]).buffer;

  device.queue.writeBuffer(buffer, 0, source, 2, 4);
  device.queue.submit([]);
  assert.deepEqual(bytesOf(state, 'writeBuffer'), [2, 3, 4, 5]);

  device.queue.writeBuffer(buffer, 0, source, 6);
  device.queue.submit([]);
  assert.deepEqual(bytesOf(state, 'writeBuffer'), [6, 7]);

  device.queue.writeBuffer(buffer, 0, [9, 8, 7, 6, 5], 1, 3);
  device.queue.submit([]);
  assert.deepEqual(bytesOf(state, 'writeBuffer'), [8, 7, 6]);
});

test('writeTexture carries dataLayout.offset on the wire', async () => {
  const state = installNativeMock();
  const device = await makeDevice();
  const texture = device.createTexture({ size: [2, 2], format: 'rgba8unorm', usage: 2 });
  device.queue.writeTexture({ texture }, new Uint8Array(32), { offset: 16, bytesPerRow: 8 }, [2, 2]);
  device.queue.submit([]);
  const command = commandsOf(state).find((entry) => entry.op === 'writeTexture');
  assert.equal(command.offset, 16);
});

test('setVertexBuffer/setIndexBuffer send the size argument only when given', async () => {
  const state = installNativeMock();
  const device = await makeDevice();
  const buffer = device.createBuffer({ size: 64, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.INDEX });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginRenderPass({ colorAttachments: [] });
  pass.setVertexBuffer(0, buffer);
  pass.setVertexBuffer(1, buffer, 16, 32);
  pass.setIndexBuffer(buffer, 'uint16', 4, 20);
  pass.end();
  device.queue.submit([encoder.finish()]);
  const commands = commandsOf(state);
  const [plain, sized] = commands.filter((entry) => entry.op === 'setVertexBuffer');
  assert.equal('size' in plain, false);
  assert.equal(sized.offset, 16);
  assert.equal(sized.size, 32);
  const index = commands.find((entry) => entry.op === 'setIndexBuffer');
  assert.equal(index.offset, 4);
  assert.equal(index.size, 20);
});

test('the setBindGroup(Uint32Array, start, length) overload sends only the window', async () => {
  const state = installNativeMock();
  const device = await makeDevice();
  const layout = device.createBindGroupLayout({ entries: [] });
  const group = device.createBindGroup({ layout, entries: [] });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setBindGroup(0, group, new Uint32Array([256, 512, 768, 1024]), 1, 2);
  pass.setBindGroup(1, group, [0, 256]);
  pass.end();
  device.queue.submit([encoder.finish()]);
  const [windowed, plain] = commandsOf(state).filter((entry) => entry.op === 'setBindGroup');
  assert.deepEqual(windowed.dynamicOffsets, [512, 768]);
  assert.deepEqual(plain.dynamicOffsets, [0, 256]);
});

test('features is setlike — iteration, spread, forEach and keys work', async () => {
  installNativeMock();
  const adapter = await gpu.requestAdapter();
  const device = await adapter.requestDevice();
  assert.ok(device.features instanceof GPUSupportedFeatures);
  assert.equal(device.features.size, 0);
  assert.deepEqual([...adapter.features], Array.from(adapter.features.keys()));
  let visited = 0;
  adapter.features.forEach(() => { visited += 1; });
  assert.equal(visited, adapter.features.size);
  for (const name of adapter.features) assert.equal(typeof name, 'string');
});

test('requestDevice rejects a requiredLimits entry the adapter does not report', async () => {
  installNativeMock();
  const adapter = await gpu.requestAdapter();
  await assert.rejects(
    adapter.requestDevice({ requiredLimits: { maxTextureDimension2D: 1 << 20 } }),
    (error) => error instanceof OperationError
  );
  const device = await adapter.requestDevice({ requiredLimits: {} });
  assert.equal(device.limits, adapter.limits);
});

test('destroying one device settles its lost promise and leaves the other device alone', async () => {
  const state = installNativeMock();
  const adapter = await gpu.requestAdapter();
  const first = await adapter.requestDevice();
  const second = await adapter.requestDevice();

  first.destroy();
  const info = await first.lost;
  assert.equal(info.reason, 'destroyed');
  assert.equal(state.resetCalls, 0, 'the native registry is shared — it must not be reset while a device lives');

  second.destroy();
  assert.equal(state.resetCalls, 1);
  first.destroy();   // idempotent
  assert.equal(state.resetCalls, 1);
});
