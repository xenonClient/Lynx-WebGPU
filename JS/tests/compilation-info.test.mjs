/**
 * `GPUShaderModule.getCompilationInfo()` — the only channel for structured shader diagnostics.
 *
 * In the spec a shader module **is created even when compilation fails.** So there is something to check
 * even after `createShaderModule()` has returned.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { installNativeMock, makeDevice } from './helpers.mjs';

/** A mock that fixes the shaderCompilationInfo response. */
function mockWithInfo(result) {
  const state = installNativeMock();
  state.compilationCalls = [];
  globalThis.NativeModules.WebGPU.shaderCompilationInfo = (params) => {
    state.compilationCalls.push(params);
    return result;
  };
  return state;
}

test('returns the diagnostics as a messages array', async () => {
  const state = mockWithInfo({
    ok: true,
    messages: [{
      message: 'WGSL parsing failed (line 3): …',
      type: 'error', lineNum: 3, linePos: 0, offset: 0, length: 0,
    }],
  });
  const device = await makeDevice();
  const module = device.createShaderModule({ code: 'fn broken( {' });

  const info = await module.getCompilationInfo();

  assert.equal(info.messages.length, 1);
  assert.equal(info.messages[0].type, 'error');
  assert.equal(info.messages[0].lineNum, 3, 'it has to be a number for an editor to jump to that line');
  assert.deepEqual(state.compilationCalls, [{ module: module.id }]);
});

test('flushes first so the module exists natively', async () => {
  const state = mockWithInfo({ ok: true, messages: [] });
  const device = await makeDevice();
  const module = device.createShaderModule({ code: '@fragment fn fs() {}' });

  // Nothing has been submitted yet — createShaderModule is only in pending.
  await module.getCompilationInfo();

  assert.equal(state.executeCalls.length, 1, 'without sending the module first, native cannot find it');
  assert.equal(state.executeCalls[0].present, false, 'it must be a mid-frame submission');
  assert.ok(state.executeCalls[0].commands.some((c) => c.op === 'createShaderModule'));
});

test('a healthy shader gives an empty array', async () => {
  mockWithInfo({ ok: true, messages: [] });
  const device = await makeDevice();
  const module = device.createShaderModule({ code: '@fragment fn fs() {}' });

  assert.deepEqual((await module.getCompilationInfo()).messages, []);
});

test('reports an error and gives an empty array when native fails', async () => {
  mockWithInfo({ ok: false, errors: [{ kind: 'validation', message: 'no such module' }] });
  const device = await makeDevice();
  const module = device.createShaderModule({ code: '@fragment fn fs() {}' });

  /** @type {string[]} */
  const seen = [];
  device.onError((_error, text) => seen.push(text));

  const info = await module.getCompilationInfo();
  assert.deepEqual(info.messages, [], 'an empty array rather than a throw — a diagnostic API must not break the app');
  assert.equal(seen.length, 1);
  assert.match(seen[0], /no such module/);
});
