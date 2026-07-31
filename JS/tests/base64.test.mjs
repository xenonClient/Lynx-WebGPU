/**
 * base64 코덱 검증 — shim의 인코더/디코더를 Node의 Buffer 구현과 대조한다.
 *
 * 코덱은 내부 함수이므로 공개 API를 통해 검증한다:
 * 인코딩은 `queue.writeBuffer`가 커맨드에 싣는 `data`, 디코딩은 `buffer.mapAsync`의 반환값.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Buffer } from 'node:buffer';
import { GPUBufferUsage, GPUMapMode } from '../webgpu.js';
import { installNativeMock, makeDevice, commandsOf, randomBytes } from './helpers.mjs';

/** 청크 경계(4096 문자 = 3072바이트)와 패딩 조합을 모두 밟는 길이들. */
const LENGTHS = [0, 1, 2, 3, 4, 5, 255, 256, 3071, 3072, 3073, 100_000];

async function encodedByShim(bytes) {
  const state = installNativeMock();
  const device = await makeDevice();
  const buffer = device.createBuffer({ size: Math.max(bytes.length, 4), usage: GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(buffer, 0, bytes);
  device.queue.submit([]);
  return commandsOf(state).find((command) => command.op === 'writeBuffer').data;
}

async function decodedByShim(base64, size) {
  installNativeMock({ readBufferResult: { ok: true, data: base64, byteLength: size } });
  const device = await makeDevice();
  const buffer = device.createBuffer({ size: Math.max(size, 4), usage: GPUBufferUsage.MAP_READ });
  return new Uint8Array(await buffer.mapAsync(GPUMapMode.READ));
}

test('인코딩이 Node Buffer 구현과 일치한다 (패딩·청크 경계 포함)', async () => {
  for (const length of LENGTHS) {
    const bytes = randomBytes(length, length + 1);
    assert.equal(
      await encodedByShim(bytes),
      Buffer.from(bytes).toString('base64'),
      `length=${length}`
    );
  }
});

test('디코딩이 Node Buffer 구현과 일치한다 (라운드트립)', async () => {
  for (const length of LENGTHS) {
    const bytes = randomBytes(length, length + 7);
    const decoded = await decodedByShim(Buffer.from(bytes).toString('base64'), length);
    assert.deepEqual(decoded, bytes, `length=${length}`);
  }
});

test('디코더는 개행·공백을 건너뛴다', async () => {
  const bytes = randomBytes(300, 3);
  const wrapped = Buffer.from(bytes)
    .toString('base64')
    .replace(/(.{60})/g, '$1\r\n')
    .concat('\n');
  assert.deepEqual(await decodedByShim(wrapped, 300), bytes);
});

test('toBase64가 TypedArray의 byteOffset과 요소 오프셋·개수를 반영한다', async () => {
  const backing = new ArrayBuffer(64);
  const full = new Float32Array(backing);
  full.forEach((_, index) => {
    full[index] = index + 0.5;
  });
  const view = new Float32Array(backing, 8, 10); // byteOffset 8부터 10개 요소

  const state = installNativeMock();
  const device = await makeDevice();
  const buffer = device.createBuffer({ size: 64, usage: GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(buffer, 0, view, 2, 4); // 요소 2부터 4개
  device.queue.submit([]);

  const command = commandsOf(state).find((c) => c.op === 'writeBuffer');
  const expected = Buffer.from(backing, 8 + 2 * 4, 4 * 4).toString('base64');
  assert.equal(command.data, expected);
});

test('ArrayBuffer와 숫자 배열도 인코딩된다', async () => {
  const bytes = randomBytes(10, 11);
  const expected = Buffer.from(bytes).toString('base64');

  const state = installNativeMock();
  const device = await makeDevice();
  const buffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(buffer, 0, bytes.buffer.slice(0, 10));
  device.queue.writeBuffer(buffer, 0, Array.from(bytes));
  device.queue.submit([]);

  const writes = commandsOf(state).filter((c) => c.op === 'writeBuffer');
  assert.equal(writes[0].data, expected);
  assert.equal(writes[1].data, expected);
});

test('mappedAtCreation → unmap이 초기 데이터를 인코딩해 createBuffer에 싣는다', async () => {
  const bytes = randomBytes(32, 5);
  const state = installNativeMock();
  const device = await makeDevice();
  const buffer = device.createBuffer({ size: 32, usage: GPUBufferUsage.VERTEX, mappedAtCreation: true });
  new Uint8Array(buffer.getMappedRange()).set(bytes);
  buffer.unmap();
  device.queue.submit([]);

  const command = commandsOf(state).find((c) => c.op === 'createBuffer');
  assert.equal(command.data, Buffer.from(bytes).toString('base64'));
});
