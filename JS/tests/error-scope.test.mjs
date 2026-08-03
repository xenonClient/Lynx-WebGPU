/**
 * `pushErrorScope` / `popErrorScope` — Promise 계약과 브리지 왕복 횟수.
 *
 * 스코프를 여는 것은 **기록만** 하므로 프레임당 왕복 1회 계약을 깨지 않아야 한다.
 * 닫는 쪽은 결과를 기다려야 하므로 `mapAsync`처럼 즉시 제출한다 — 그 차이를 여기서 못 박는다.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { installNativeMock, makeDevice, commandsOf } from './helpers.mjs';

/** 스코프 pop마다 정해진 결과를 돌려주는 목. */
function mockWithScopes(results) {
  let cursor = 0;
  return installNativeMock({
    executeResult: (payload) => {
      const pops = payload.commands.filter((command) => command.op === 'popErrorScope').length;
      const errorScopes = results.slice(cursor, cursor + pops);
      cursor += pops;
      return { ok: true, commandCount: payload.commands.length, errorScopes };
    },
  });
}

test('popErrorScope가 잡힌 오류를 Promise로 돌려준다', async () => {
  const captured = { kind: 'validation', message: '없는 핸들' };
  const state = mockWithScopes([captured]);
  const device = await makeDevice();

  device.pushErrorScope('validation');
  device.createBuffer({ size: 16, usage: 0x40 });
  const error = await device.popErrorScope();

  assert.deepEqual(error, captured);
  const commands = commandsOf(state);
  assert.equal(commands[0].op, 'pushErrorScope');
  assert.equal(commands[0].filter, 'validation');
  assert.equal(commands[commands.length - 1].op, 'popErrorScope');
});

test('오류가 없으면 null로 풀린다', async () => {
  mockWithScopes([null]);
  const device = await makeDevice();

  device.pushErrorScope('validation');
  assert.equal(await device.popErrorScope(), null);
});

test('중첩 스코프는 pop한 순서대로 짝지어진다', async () => {
  const inner = { kind: 'validation', message: '안쪽' };
  const outer = { kind: 'out-of-memory', message: '바깥' };
  // 두 pop이 같은 배치에 들어가지 않도록(각 pop이 즉시 제출한다) 순서만 확인한다.
  mockWithScopes([inner, outer]);
  const device = await makeDevice();

  device.pushErrorScope('out-of-memory');
  device.pushErrorScope('validation');
  const first = await device.popErrorScope();
  const second = await device.popErrorScope();

  assert.deepEqual(first, inner, '먼저 닫은 쪽이 먼저 풀린다');
  assert.deepEqual(second, outer);
});

test('스코프를 열어 두어도 프레임당 왕복은 1회로 유지된다', async () => {
  const state = mockWithScopes([]);
  const device = await makeDevice();

  const buffer = device.createBuffer({ size: 16, usage: 0x48 });

  // 스코프를 연 채로 다섯 프레임을 돌린다 — push는 기록만 해야 한다.
  device.pushErrorScope('validation');
  for (let frame = 0; frame < 5; frame += 1) {
    device.queue.writeBuffer(buffer, 0, new Float32Array([frame]));
    device.queue.submit([]);
  }

  assert.equal(state.executeCalls.length, 5, 'pushErrorScope가 추가 제출을 만들면 안 된다');
  const first = commandsOf(state, 0);
  assert.ok(
    first.some((command) => command.op === 'pushErrorScope'),
    '스코프는 첫 프레임 제출에 실려 나간다'
  );
});

test('popErrorScope는 submit 없이도 풀린다 (스스로 제출한다)', async () => {
  const state = mockWithScopes([null]);
  const device = await makeDevice();

  device.pushErrorScope('validation');
  // submit을 부르지 않는다 — 그래도 Promise가 풀려야 한다. 안 그러면 초기화 진단이 매달린다.
  await device.popErrorScope();

  assert.equal(state.executeCalls.length, 1, 'pop이 스스로 제출한다');
});

test('device.destroy가 기다리던 스코프를 null로 닫는다', async () => {
  installNativeMock({ executeResult: () => ({ ok: true }) });
  const device = await makeDevice();

  device.pushErrorScope('validation');
  // 응답에 errorScopes가 없으므로 pop만으로는 안 풀린다 — destroy가 마무리해야 한다.
  const pending = device.popErrorScope();
  device.destroy();

  assert.equal(await pending, null, '풀리지 않는 Promise를 남기면 안 된다');
});
