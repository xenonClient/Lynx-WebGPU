/**
 * `copyBufferToBuffer` 오버로드와 `clearBuffer` — 명세가 정한 호출 형태들.
 *
 * 명세는 짧은 형태 `(source, destination, size?)`를 함께 정의하고 `size`를 생략할 수 있게 한다.
 * 5-인자 형태만 받으면 짧게 부른 코드가 **오류 없이** 엉뚱한 인자로 기록된다 (버퍼가 정수
 * 자리에 들어가 NaN 오프셋이 나가는 식) — 그래서 형태 구분이 값 검증만큼 중요하다.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { installNativeMock, makeDevice, commandsOf } from './helpers.mjs';

function makeBuffers(device) {
  return {
    source: device.createBuffer({ size: 64, usage: 0x0004 /* COPY_SRC */ }),
    destination: device.createBuffer({ size: 64, usage: 0x0008 /* COPY_DST */ }),
  };
}

function lastCopy(state) {
  return commandsOf(state).filter((command) => command.op === 'copyBufferToBuffer').pop();
}

test('짧은 형태 (source, destination) — 원본 전체를 복사한다', async () => {
  const state = installNativeMock();
  const device = await makeDevice();
  const { source, destination } = makeBuffers(device);

  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, destination);
  device.queue.submit([encoder.finish()]);

  const command = lastCopy(state);
  assert.equal(command.source, source.id);
  assert.equal(command.destination, destination.id);
  assert.equal(command.sourceOffset, 0);
  assert.equal(command.destinationOffset, 0);
  assert.equal(command.size, 64, 'size를 생략하면 원본의 남은 바이트 전부다');
});

test('짧은 형태 + size', async () => {
  const state = installNativeMock();
  const device = await makeDevice();
  const { source, destination } = makeBuffers(device);

  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, destination, 16);
  device.queue.submit([encoder.finish()]);

  const command = lastCopy(state);
  assert.equal(command.size, 16);
  assert.equal(command.destination, destination.id, '두 번째 인자가 버퍼면 목적지다');
});

test('긴 형태 (source, srcOffset, destination, dstOffset, size)', async () => {
  const state = installNativeMock();
  const device = await makeDevice();
  const { source, destination } = makeBuffers(device);

  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, 16, destination, 32, 8);
  device.queue.submit([encoder.finish()]);

  const command = lastCopy(state);
  assert.equal(command.sourceOffset, 16);
  assert.equal(command.destinationOffset, 32);
  assert.equal(command.size, 8);
});

test('긴 형태에서 size를 생략하면 원본의 남은 바이트다', async () => {
  const state = installNativeMock();
  const device = await makeDevice();
  const { source, destination } = makeBuffers(device);

  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, 16, destination, 0);
  device.queue.submit([encoder.finish()]);

  assert.equal(lastCopy(state).size, 48, '64 - 16');
});

test('clearBuffer는 offset·size를 그대로 싣고, size 생략은 네이티브가 채운다', async () => {
  const state = installNativeMock();
  const device = await makeDevice();
  const buffer = device.createBuffer({ size: 64, usage: 0x0008 });

  const encoder = device.createCommandEncoder();
  encoder.clearBuffer(buffer, 16, 32);
  encoder.clearBuffer(buffer);
  device.queue.submit([encoder.finish()]);

  const commands = commandsOf(state).filter((command) => command.op === 'clearBuffer');
  assert.equal(commands[0].offset, 16);
  assert.equal(commands[0].size, 32);
  assert.equal(commands[1].offset, 0);
  assert.equal(
    'size' in commands[1], false,
    'size를 안 실어야 네이티브가 "끝까지"로 해석한다 — 0을 실으면 아무것도 안 지운다'
  );
});
