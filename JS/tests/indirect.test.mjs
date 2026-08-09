/**
 * The command payloads of indirect draw/dispatch.
 *
 * Field names are not caught by type checking — JS puts them on as `Record<string, any>` and native reads
 * them by string key, so a mismatched name compiles on both sides. This is where that mismatch gets caught
 * (`.claude/skills/webgpu-command/SKILL.md`).
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { GPUBufferUsage } from '../webgpu.js';
import { installNativeMock, makeDevice, commandsOf } from './helpers.mjs';

/** A device equipped with one argument buffer and one encoder. */
async function setUp() {
  const state = installNativeMock();
  const device = await makeDevice();
  const indirectBuffer = device.createBuffer({
    size: 20,
    usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
  });
  return { state, device, indirectBuffer, encoder: device.createCommandEncoder() };
}

/** The minimal render pass an encoder can actually use. */
function beginRenderPass(encoder, device) {
  const view = device.createTexture({ size: { width: 1, height: 1 }, format: 'rgba8unorm', usage: 0x10 })
    .createView();
  return encoder.beginRenderPass({ colorAttachments: [{ view }] });
}

test('drawIndirect puts the handle and offset on', async () => {
  const { state, device, indirectBuffer, encoder } = await setUp();
  const pass = beginRenderPass(encoder, device);
  pass.drawIndirect(indirectBuffer, 16);
  pass.end();
  device.queue.submit([encoder.finish()]);

  const command = commandsOf(state).find((entry) => entry.op === 'drawIndirect');
  assert.ok(command, 'the drawIndirect command must be in the stream');
  assert.equal(command.indirectBuffer, indirectBuffer.id);
  assert.equal(command.indirectOffset, 16);
});

test('drawIndexedIndirect puts the handle and offset on', async () => {
  const { state, device, indirectBuffer, encoder } = await setUp();
  const pass = beginRenderPass(encoder, device);
  pass.drawIndexedIndirect(indirectBuffer, 4);
  pass.end();
  device.queue.submit([encoder.finish()]);

  const command = commandsOf(state).find((entry) => entry.op === 'drawIndexedIndirect');
  assert.ok(command);
  assert.equal(command.indirectBuffer, indirectBuffer.id);
  assert.equal(command.indirectOffset, 4);
});

test('dispatchWorkgroupsIndirect puts the handle and offset on', async () => {
  const { state, device, indirectBuffer, encoder } = await setUp();
  const pass = encoder.beginComputePass();
  pass.dispatchWorkgroupsIndirect(indirectBuffer);
  pass.end();
  device.queue.submit([encoder.finish()]);

  const command = commandsOf(state).find((entry) => entry.op === 'dispatchWorkgroupsIndirect');
  assert.ok(command);
  assert.equal(command.indirectBuffer, indirectBuffer.id);
  assert.equal(command.indirectOffset, 0, 'omitted, it is 0');
});

test('the bridge crossings stay at one per frame even with indirect draws', async () => {
  const { state, device, indirectBuffer, encoder } = await setUp();
  const pass = beginRenderPass(encoder, device);
  pass.drawIndirect(indirectBuffer, 0);
  pass.drawIndexedIndirect(indirectBuffer, 0);
  pass.end();
  device.queue.submit([encoder.finish()]);

  assert.equal(state.executeCalls.length, 1, 'it must only record and go across in one go at submit');
});
