/**
 * `getMappedRange(offset, size)` — 부분 매핑.
 *
 * 예전에는 인자를 **조용히 무시하고** 항상 전체를 돌려줬다. 부분을 기대한 코드는 오프셋 없이
 * 읽어 잘못된 값을 쓰게 되고, 그게 화면에 나타날 때쯤엔 원인을 찾기 어렵다.
 *
 * JS에서는 `ArrayBuffer`가 다른 `ArrayBuffer`의 일부를 가리킬 수 없어 구간을 복사해 주고
 * `unmap()`에서 되돌려 쓴다 — **쓴 내용이 사라지지 않는지**가 여기서 가장 중요한 계약이다.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { installNativeMock, makeDevice, commandsOf } from './helpers.mjs';

test('오프셋과 크기를 존중해 그 구간만 준다', async () => {
  const backing = new Uint8Array(32);
  backing.forEach((_, index) => { backing[index] = index; });
  installNativeMock({ readBufferResult: { ok: true, data: backing.buffer } });
  const device = await makeDevice();
  const buffer = device.createBuffer({ size: 32, usage: 0x0001 });
  await buffer.mapAsync();

  const middle = new Uint8Array(buffer.getMappedRange(8, 8));

  assert.equal(middle.length, 8);
  assert.deepEqual(Array.from(middle), [8, 9, 10, 11, 12, 13, 14, 15]);
});

test('구간에 쓴 내용이 unmap에서 되돌아간다', async () => {
  const state = installNativeMock();
  const device = await makeDevice();
  const buffer = device.createBuffer({ size: 32, usage: 0x0020, mappedAtCreation: true });

  // 뒤쪽 8바이트만 얻어서 채운다 — 사본이라 되돌려 쓰지 않으면 조용히 사라진다.
  new Uint8Array(buffer.getMappedRange(24, 8)).fill(0xab);
  buffer.unmap();
  device.queue.submit([]);

  const create = commandsOf(state).find((command) => command.op === 'createBuffer');
  const bytes = new Uint8Array(create.data);
  assert.equal(bytes.length, 32);
  assert.deepEqual(Array.from(bytes.slice(24)), Array(8).fill(0xab), '쓴 내용이 사라졌다');
  assert.deepEqual(Array.from(bytes.slice(0, 8)), Array(8).fill(0), '건드리지 않은 곳은 그대로');
});

test('전체를 처음 요청하면 복사 없이 매핑 자체를 준다', async () => {
  const backing = new ArrayBuffer(16);
  installNativeMock({ readBufferResult: { ok: true, data: backing } });
  const device = await makeDevice();
  const buffer = device.createBuffer({ size: 16, usage: 0x0001 });
  const mapped = await buffer.mapAsync();

  assert.equal(buffer.getMappedRange(), mapped, '큰 버퍼에서 복사를 한 번 아낀다');
});

test('명세 정렬 규칙을 지킨다 (offset 8의 배수, 명시한 size 4의 배수)', async () => {
  installNativeMock();
  const device = await makeDevice();
  const buffer = device.createBuffer({ size: 32, usage: 0x0020, mappedAtCreation: true });

  assert.throws(() => buffer.getMappedRange(4), (error) => {
    assert.equal(error.name, 'OperationError');
    assert.match(error.message, /8의 배수/);
    return true;
  });
  assert.throws(() => buffer.getMappedRange(0, 6), /4의 배수/);
});

test('겹치는 구간은 거부한다', async () => {
  installNativeMock();
  const device = await makeDevice();
  const buffer = device.createBuffer({ size: 32, usage: 0x0020, mappedAtCreation: true });

  buffer.getMappedRange(0, 16);
  assert.throws(() => buffer.getMappedRange(8, 16), /겹친다/);
  // 겹치지 않으면 된다.
  assert.doesNotThrow(() => buffer.getMappedRange(16, 16));
});

test('범위를 넘으면 거부한다', async () => {
  installNativeMock();
  const device = await makeDevice();
  const buffer = device.createBuffer({ size: 16, usage: 0x0020, mappedAtCreation: true });

  assert.throws(() => buffer.getMappedRange(8, 16), /넘는다/);
});

test('4의 배수가 아닌 매핑도 전체는 읽을 수 있다', async () => {
  // 매핑 크기가 곧 네이티브 버퍼 크기라 3바이트짜리도 정상이다 —
  // 생략한 size까지 4의 배수를 요구하면 그런 버퍼를 아예 못 읽는다.
  const backing = new Uint8Array([1, 2, 3]).buffer;
  installNativeMock({ readBufferResult: { ok: true, data: backing } });
  const device = await makeDevice();
  const buffer = device.createBuffer({ size: 3, usage: 0x0001 });
  await buffer.mapAsync();

  assert.equal(buffer.getMappedRange().byteLength, 3);
});
