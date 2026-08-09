/**
 * `createRenderPipelineAsync` / `createComputePipelineAsync` — they report failure **right there**.
 *
 * The synchronous forms only record a command, so a failure arrives late, in the next submit's error array.
 * The asynchronous forms wrap creation in two layers of error scope (validation + internal), submit
 * immediately, and resolve the Promise from the result. Two layers because a pipeline fails in two ways —
 * a descriptor problem is validation, and a shader translation/compilation failure is backend (= the
 * internal filter). Striking one layer only resolves the other half as success and **you hold an unusable pipeline.**
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { installNativeMock, makeDevice, commandsOf } from './helpers.mjs';

/** A mock that returns popErrorScope responses in order. */
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

function renderDescriptor(device) {
  const module = device.createShaderModule({ code: '@vertex fn vs() -> @builtin(position) vec4f { return vec4f(0); }' });
  return {
    layout: 'auto',
    vertex: { module, entryPoint: 'vs' },
    fragment: { module, entryPoint: 'fs', targets: [{ format: 'bgra8unorm' }] },
  };
}

test('resolves to a pipeline on success, closing both scopes in one batch', async () => {
  const state = mockWithScopes([null, null]);
  const device = await makeDevice();

  const pipeline = await device.createRenderPipelineAsync(renderDescriptor(device));

  assert.ok(pipeline.id > 0, 'a handle must be issued');
  const commands = commandsOf(state);
  const ops = commands.map((command) => command.op);
  assert.deepEqual(
    ops.filter((op) => op.indexOf('ErrorScope') >= 0),
    ['pushErrorScope', 'pushErrorScope', 'popErrorScope', 'popErrorScope'],
    'it must be wrapped in two layers'
  );
  assert.deepEqual(
    commands.filter((command) => command.op === 'pushErrorScope').map((command) => command.filter),
    ['validation', 'internal'],
    'internal has to be the inner one to take backend errors first'
  );
  assert.equal(state.executeCalls.length, 1, 'both pops ride in one batch, so there must be one crossing');
  assert.equal(state.executeCalls[0].present, false, 'being a mid-frame submission, it does not present');
});

test('a descriptor error (validation) is rejected as a GPUPipelineError', async () => {
  // The case where the inner (internal) scope is empty and the outer (validation) one caught it.
  mockWithScopes([null, { kind: 'validation', message: 'unknown format', path: 'commands[1].fragment' }]);
  const device = await makeDevice();

  await assert.rejects(
    device.createRenderPipelineAsync(renderDescriptor(device)),
    (error) => {
      assert.equal(error.name, 'GPUPipelineError');
      assert.equal(error.reason, 'validation');
      assert.match(error.message, /unknown format/);
      assert.match(error.message, /commands\[1\]\.fragment/, 'the path has to come along for diagnosis');
      return true;
    }
  );
});

test('a shader compilation failure (backend) is rejected too — the internal scope catches it', async () => {
  mockWithScopes([{ kind: 'backend', message: 'MSL compilation failed' }, null]);
  const device = await makeDevice();

  await assert.rejects(
    device.createRenderPipelineAsync(renderDescriptor(device)),
    (error) => {
      assert.equal(error.reason, 'internal', "backend folds into the spec's internal");
      assert.match(error.message, /MSL compilation failed/);
      return true;
    }
  );
});

test('a failed pipeline leaves no handle behind', async () => {
  const state = mockWithScopes([null, { kind: 'validation', message: 'rejected' }]);
  const device = await makeDevice();

  await assert.rejects(device.createRenderPipelineAsync(renderDescriptor(device)));

  // A destroy must ride on the next submission after the rejection (no garbage left in the registry).
  device.queue.submit([]);
  const ops = commandsOf(state).map((command) => command.op);
  assert.ok(ops.indexOf('destroy') >= 0, `there is no destroy: ${ops.join(' ')}`);
});

test('a compute pipeline has the same contract', async () => {
  const state = mockWithScopes([null, null]);
  const device = await makeDevice();
  const module = device.createShaderModule({ code: '@compute @workgroup_size(1) fn main() {}' });

  const pipeline = await device.createComputePipelineAsync({
    layout: 'auto', compute: { module, entryPoint: 'main' },
  });

  assert.ok(pipeline.id > 0);
  const ops = commandsOf(state).map((command) => command.op);
  assert.ok(ops.indexOf('createComputePipeline') >= 0);
});
