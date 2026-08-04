/**
 * 간접 드로우/디스패치의 커맨드 페이로드.
 *
 * 필드 이름은 타입 검사가 잡아 주지 않는다 — JS는 `Record<string, any>`로 싣고 네이티브는
 * 문자열 키로 읽으므로, 이름이 어긋나도 양쪽 다 컴파일된다. 여기가 그 어긋남을 잡는 자리다
 * (`.claude/skills/webgpu-command/SKILL.md`).
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { GPUBufferUsage } from '../webgpu.js';
import { installNativeMock, makeDevice, commandsOf } from './helpers.mjs';

/** 인자 버퍼 하나와 인코더 하나를 갖춘 디바이스. */
async function setUp() {
  const state = installNativeMock();
  const device = await makeDevice();
  const indirectBuffer = device.createBuffer({
    size: 20,
    usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
  });
  return { state, device, indirectBuffer, encoder: device.createCommandEncoder() };
}

/** 인코더가 실제로 쓸 수 있는 최소 렌더 패스. */
function beginRenderPass(encoder, device) {
  const view = device.createTexture({ size: { width: 1, height: 1 }, format: 'rgba8unorm', usage: 0x10 })
    .createView();
  return encoder.beginRenderPass({ colorAttachments: [{ view }] });
}

test('drawIndirect가 핸들과 오프셋을 싣는다', async () => {
  const { state, device, indirectBuffer, encoder } = await setUp();
  const pass = beginRenderPass(encoder, device);
  pass.drawIndirect(indirectBuffer, 16);
  pass.end();
  device.queue.submit([encoder.finish()]);

  const command = commandsOf(state).find((entry) => entry.op === 'drawIndirect');
  assert.ok(command, 'drawIndirect 명령이 스트림에 있어야 한다');
  assert.equal(command.indirectBuffer, indirectBuffer.id);
  assert.equal(command.indirectOffset, 16);
});

test('drawIndexedIndirect가 핸들과 오프셋을 싣는다', async () => {
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

test('dispatchWorkgroupsIndirect가 핸들과 오프셋을 싣는다', async () => {
  const { state, device, indirectBuffer, encoder } = await setUp();
  const pass = encoder.beginComputePass();
  pass.dispatchWorkgroupsIndirect(indirectBuffer);
  pass.end();
  device.queue.submit([encoder.finish()]);

  const command = commandsOf(state).find((entry) => entry.op === 'dispatchWorkgroupsIndirect');
  assert.ok(command);
  assert.equal(command.indirectBuffer, indirectBuffer.id);
  assert.equal(command.indirectOffset, 0, '생략하면 0');
});

test('간접 드로우를 써도 프레임당 브리지 왕복은 1회다', async () => {
  const { state, device, indirectBuffer, encoder } = await setUp();
  const pass = beginRenderPass(encoder, device);
  pass.drawIndirect(indirectBuffer, 0);
  pass.drawIndexedIndirect(indirectBuffer, 0);
  pass.end();
  device.queue.submit([encoder.finish()]);

  assert.equal(state.executeCalls.length, 1, '기록만 하고 submit에서 한 번에 넘어가야 한다');
});
