/**
 * `GPUShaderModule.getCompilationInfo()` — 셰이더 진단을 구조화해 받는 유일한 통로.
 *
 * 명세에서 셰이더 모듈은 **컴파일에 실패해도 만들어진다.** 그래서 `createShaderModule()`이
 * 돌아온 뒤에도 확인할 값이 있다.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { installNativeMock, makeDevice } from './helpers.mjs';

/** shaderCompilationInfo 응답을 정해 주는 목. */
function mockWithInfo(result) {
  const state = installNativeMock();
  state.compilationCalls = [];
  globalThis.NativeModules.WebGPU.shaderCompilationInfo = (params) => {
    state.compilationCalls.push(params);
    return result;
  };
  return state;
}

test('진단을 messages 배열로 돌려준다', async () => {
  const state = mockWithInfo({
    ok: true,
    messages: [{
      message: 'WGSL 파싱 실패 (line 3): …',
      type: 'error', lineNum: 3, linePos: 0, offset: 0, length: 0,
    }],
  });
  const device = await makeDevice();
  const module = device.createShaderModule({ code: 'fn broken( {' });

  const info = await module.getCompilationInfo();

  assert.equal(info.messages.length, 1);
  assert.equal(info.messages[0].type, 'error');
  assert.equal(info.messages[0].lineNum, 3, '편집기가 그 줄로 점프하려면 숫자여야 한다');
  assert.deepEqual(state.compilationCalls, [{ module: module.id }]);
});

test('먼저 flush해서 모듈이 네이티브에 있게 한다', async () => {
  const state = mockWithInfo({ ok: true, messages: [] });
  const device = await makeDevice();
  const module = device.createShaderModule({ code: '@fragment fn fs() {}' });

  // 아직 submit하지 않았다 — createShaderModule이 pending에만 있다.
  await module.getCompilationInfo();

  assert.equal(state.executeCalls.length, 1, '모듈을 먼저 보내지 않으면 네이티브가 못 찾는다');
  assert.equal(state.executeCalls[0].present, false, '프레임 중간 제출이어야 한다');
  assert.ok(state.executeCalls[0].commands.some((c) => c.op === 'createShaderModule'));
});

test('정상 셰이더는 빈 배열이다', async () => {
  mockWithInfo({ ok: true, messages: [] });
  const device = await makeDevice();
  const module = device.createShaderModule({ code: '@fragment fn fs() {}' });

  assert.deepEqual((await module.getCompilationInfo()).messages, []);
});

test('네이티브가 실패하면 오류로 보고하고 빈 배열을 준다', async () => {
  mockWithInfo({ ok: false, errors: [{ kind: 'validation', message: '없는 모듈' }] });
  const device = await makeDevice();
  const module = device.createShaderModule({ code: '@fragment fn fs() {}' });

  /** @type {string[]} */
  const seen = [];
  device.onError((_error, text) => seen.push(text));

  const info = await module.getCompilationInfo();
  assert.deepEqual(info.messages, [], '던지지 않고 빈 배열이다 — 진단 API가 앱을 깨면 안 된다');
  assert.equal(seen.length, 1);
  assert.match(seen[0], /없는 모듈/);
});
