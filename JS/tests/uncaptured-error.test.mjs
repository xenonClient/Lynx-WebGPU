/**
 * `device.onuncapturederror` — 스코프에 안 잡힌 오류의 명세 통로.
 *
 * 웹 코드는 이 이름만 안다 (Three.js가 여기 대입해 `renderer.onError`로 넘긴다). 이 구현의
 * `onError`와 **함께** 동작해야 하고, 스코프가 가로챈 오류는 여기로 오면 안 된다 —
 * 이미 처리하기로 한 오류를 전역 핸들러가 다시 보고하면 중복이다.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { installNativeMock, makeDevice } from './helpers.mjs';
import { GPUValidationError, GPUInternalError, GPUOutOfMemoryError } from '../webgpu.js';

/** submit이 정해진 오류를 돌려주는 목. */
function mockWithErrors(errors) {
  return installNativeMock({
    executeResult: (payload) => ({ ok: false, commandCount: payload.commands.length, errors }),
  });
}

test('스코프에 안 잡힌 오류가 이벤트로 온다', async () => {
  mockWithErrors([{ kind: 'validation', message: '없는 핸들', path: 'commands[0].buffer' }]);
  const device = await makeDevice();

  /** @type {any[]} */
  const events = [];
  device.onuncapturederror = (event) => events.push(event);

  device.createBuffer({ size: 16, usage: 0x40 });
  device.queue.submit([]);

  assert.equal(events.length, 1);
  assert.equal(events[0].type, 'uncapturederror');
  assert.equal(events[0].error.message, '없는 핸들');
  assert.ok(events[0].error instanceof GPUValidationError);
  // 명세에 없지만 진단에 필요해 함께 싣는다.
  assert.equal(events[0].error.path, 'commands[0].buffer');
});

test('오류 종류가 명세의 GPUError 하위 클래스로 갈린다', async () => {
  mockWithErrors([
    { kind: 'validation', message: 'v' },
    { kind: 'unsupported', message: 'u' },
    { kind: 'backend', message: 'b' },
    { kind: 'out-of-memory', message: 'o' },
  ]);
  const device = await makeDevice();

  /** @type {any[]} */
  const errors = [];
  device.addEventListener('uncapturederror', (event) => errors.push(event.error));

  device.createBuffer({ size: 16, usage: 0x40 });
  device.queue.submit([]);

  assert.ok(errors[0] instanceof GPUValidationError);
  // unsupported도 validation으로 접힌다 — pushErrorScope('validation')이 잡는 것과 같은 규칙.
  assert.ok(errors[1] instanceof GPUValidationError);
  assert.ok(errors[2] instanceof GPUInternalError, 'backend는 internal이다');
  assert.ok(errors[3] instanceof GPUOutOfMemoryError);
});

test('onError와 함께 동작한다 (둘 다 등록하면 둘 다 받는다)', async () => {
  mockWithErrors([{ kind: 'validation', message: '둘 다' }]);
  const device = await makeDevice();

  /** @type {string[]} */
  const seen = [];
  device.onError((_error, text) => seen.push(`onError:${text}`));
  device.onuncapturederror = (event) => seen.push(`uncaptured:${event.error.message}`);

  device.createBuffer({ size: 16, usage: 0x40 });
  device.queue.submit([]);

  assert.equal(seen.length, 2);
  assert.match(seen[0], /onError:\[WebGPU:validation\] 둘 다/);
  assert.equal(seen[1], 'uncaptured:둘 다');
});

test('removeEventListener로 뗀 리스너는 더 받지 않는다', async () => {
  mockWithErrors([{ kind: 'validation', message: 'x' }]);
  const device = await makeDevice();

  let count = 0;
  const listener = () => { count += 1; };
  device.addEventListener('uncapturederror', listener);
  device.createBuffer({ size: 16, usage: 0x40 });
  device.queue.submit([]);
  assert.equal(count, 1);

  device.removeEventListener('uncapturederror', listener);
  device.createBuffer({ size: 16, usage: 0x40 });
  device.queue.submit([]);
  assert.equal(count, 1, '뗀 뒤에는 오지 않아야 한다');
});

test('리스너가 던져도 나머지 리스너와 다음 오류는 계속 간다', async () => {
  mockWithErrors([{ kind: 'validation', message: '첫째' }, { kind: 'validation', message: '둘째' }]);
  const device = await makeDevice();

  /** @type {string[]} */
  const seen = [];
  device.addEventListener('uncapturederror', () => { throw new Error('리스너 버그'); });
  device.addEventListener('uncapturederror', (event) => seen.push(event.error.message));

  device.createBuffer({ size: 16, usage: 0x40 });
  device.queue.submit([]);

  assert.deepEqual(seen, ['첫째', '둘째'], '하나의 실수가 전체 보고를 삼키면 안 된다');
});

test('스코프가 가로챈 오류는 uncapturederror로 오지 않는다', async () => {
  // 네이티브가 스코프에 잡힌 오류를 errors에서 빼고 errorScopes로만 돌려준다 — 그 계약을 목으로 흉내 낸다.
  installNativeMock({
    executeResult: (payload) => ({
      ok: true,
      commandCount: payload.commands.length,
      errorScopes: [{ kind: 'validation', message: '가로챔' }],
    }),
  });
  const device = await makeDevice();

  let count = 0;
  device.onuncapturederror = () => { count += 1; };

  device.pushErrorScope('validation');
  const captured = await device.popErrorScope();

  assert.equal(captured.message, '가로챔');
  assert.equal(count, 0, '이미 처리하기로 한 오류를 전역이 다시 보고하면 중복이다');
});
