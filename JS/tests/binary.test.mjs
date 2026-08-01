/**
 * 바이너리 경로 검증 — 커맨드에 실리는 `data`가 **진짜 `ArrayBuffer`인지**, 그리고 그 안의
 * 바이트가 요청한 구간과 정확히 같은지 본다.
 *
 * 타입이 핵심이다. Lynx의 값 변환기는 `ArrayBuffer`만 알아보고 뷰(TypedArray)는 평범한
 * 객체로 취급해 `{"0":1,…}` 로 만들어 버리므로, 뷰가 새어 나가면 **오류 없이 조용히**
 * 깨진다. 그래서 매번 `instanceof ArrayBuffer`를 함께 단언한다.
 *
 * 코덱이 내부 함수이므로 공개 API로 확인한다:
 * 올리는 쪽은 `queue.writeBuffer`가 싣는 `data`, 내리는 쪽은 `buffer.mapAsync`의 반환값.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { GPUBufferUsage, GPUMapMode } from '../webgpu.js';
import { installNativeMock, makeDevice, commandsOf, randomBytes } from './helpers.mjs';

const LENGTHS = [0, 1, 2, 3, 4, 5, 255, 256, 3071, 3072, 3073, 100_000];

/** 커맨드에 실린 `data`를 꺼내면서 타입까지 확인한다. */
function payloadOf(state, op) {
  const command = commandsOf(state).find((entry) => entry.op === op);
  assert.ok(command, `${op} 명령이 없다`);
  assert.ok(
    command.data instanceof ArrayBuffer,
    `${op}의 data가 ArrayBuffer가 아니다 (${Object.prototype.toString.call(command.data)}) — ` +
      'TypedArray가 그대로 새면 Lynx가 객체로 바꿔 조용히 깨진다'
  );
  return new Uint8Array(command.data);
}

async function uploadedByShim(bytes) {
  const state = installNativeMock();
  const device = await makeDevice();
  const buffer = device.createBuffer({ size: Math.max(bytes.length, 4), usage: GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(buffer, 0, bytes);
  device.queue.submit([]);
  return payloadOf(state, 'writeBuffer');
}

test('writeBuffer가 ArrayBuffer를 그대로 싣는다 (길이 전 범위)', async () => {
  for (const length of LENGTHS) {
    const bytes = randomBytes(length, length + 1);
    assert.deepEqual(await uploadedByShim(bytes), bytes, `length=${length}`);
  }
});

test('mapAsync는 네이티브가 준 ArrayBuffer를 디코딩 없이 그대로 돌려준다', async () => {
  for (const length of LENGTHS) {
    const bytes = randomBytes(length, length + 7);
    // 네이티브(`LynxWebGPUContext.readBuffer`)는 Data를 싣고, Lynx가 ArrayBuffer로 바꿔 준다.
    const backing = bytes.buffer.slice(0);
    installNativeMock({ readBufferResult: { ok: true, data: backing, byteLength: length } });
    const device = await makeDevice();
    const buffer = device.createBuffer({ size: Math.max(length, 4), usage: GPUBufferUsage.MAP_READ });

    const mapped = await buffer.mapAsync(GPUMapMode.READ);
    assert.ok(mapped instanceof ArrayBuffer, `length=${length}: ArrayBuffer가 아니다`);
    assert.deepEqual(new Uint8Array(mapped), bytes, `length=${length}`);
    // 사본을 뜨지 않고 넘겨받은 것을 그대로 쓴다.
    assert.equal(buffer.getMappedRange(), mapped);
  }
});

test('TypedArray의 byteOffset과 요소 오프셋·개수를 반영해 잘라 낸다', async () => {
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

  const expected = new Uint8Array(backing, 8 + 2 * 4, 4 * 4);
  assert.deepEqual(payloadOf(state, 'writeBuffer'), new Uint8Array(expected));
});

test('버퍼 전체를 덮는 뷰는 잘라 내지 않고 백킹 버퍼를 그대로 넘긴다', async () => {
  const bytes = randomBytes(32, 9);
  const state = installNativeMock();
  const device = await makeDevice();
  const buffer = device.createBuffer({ size: 32, usage: GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(buffer, 0, bytes);
  device.queue.submit([]);

  const command = commandsOf(state).find((entry) => entry.op === 'writeBuffer');
  assert.equal(command.data, bytes.buffer, '불필요한 복사가 생겼다');
});

test('ArrayBuffer와 숫자 배열도 ArrayBuffer로 실린다', async () => {
  const bytes = randomBytes(10, 11);

  const state = installNativeMock();
  const device = await makeDevice();
  const buffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(buffer, 0, bytes.buffer.slice(0, 10));
  device.queue.writeBuffer(buffer, 0, Array.from(bytes));
  device.queue.submit([]);

  const writes = commandsOf(state).filter((entry) => entry.op === 'writeBuffer');
  for (const write of writes) {
    assert.ok(write.data instanceof ArrayBuffer);
    assert.deepEqual(new Uint8Array(write.data), bytes);
  }
});

test('writeTexture도 ArrayBuffer를 싣는다', async () => {
  const texels = randomBytes(64, 21);
  const state = installNativeMock();
  const device = await makeDevice();
  const texture = device.createTexture({
    size: { width: 4, height: 4 },
    format: 'rgba8unorm',
    usage: GPUBufferUsage.COPY_DST,
  });
  device.queue.writeTexture({ texture }, texels, { bytesPerRow: 16 }, { width: 4, height: 4 });
  device.queue.submit([]);

  assert.deepEqual(payloadOf(state, 'writeTexture'), texels);
});

test('mappedAtCreation → unmap이 초기 데이터를 ArrayBuffer로 createBuffer에 싣는다', async () => {
  const bytes = randomBytes(32, 5);
  const state = installNativeMock();
  const device = await makeDevice();
  const buffer = device.createBuffer({ size: 32, usage: GPUBufferUsage.VERTEX, mappedAtCreation: true });
  new Uint8Array(buffer.getMappedRange()).set(bytes);
  buffer.unmap();
  device.queue.submit([]);

  assert.deepEqual(payloadOf(state, 'createBuffer'), bytes);
});
