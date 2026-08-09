/**
 * `pushErrorScope` / `popErrorScope` — the Promise contract and the bridge crossing count.
 *
 * Opening a scope **only records**, so it must not break the one-crossing-per-frame contract.
 * Closing has to wait for a result, so it submits immediately like `mapAsync` — that difference is pinned here.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { installNativeMock, makeDevice, commandsOf } from './helpers.mjs';

/** A mock that returns a fixed result per scope pop. */
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

test('popErrorScope returns the caught error as a Promise', async () => {
  const captured = { kind: 'validation', message: 'no such handle' };
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

test('resolves to null when there is no error', async () => {
  mockWithScopes([null]);
  const device = await makeDevice();

  device.pushErrorScope('validation');
  assert.equal(await device.popErrorScope(), null);
});

test('nested scopes pair up in pop order', async () => {
  const inner = { kind: 'validation', message: 'inner' };
  const outer = { kind: 'out-of-memory', message: 'outer' };
  // Only the order is checked, so the two pops do not land in the same batch (each pop submits immediately).
  mockWithScopes([inner, outer]);
  const device = await makeDevice();

  device.pushErrorScope('out-of-memory');
  device.pushErrorScope('validation');
  const first = await device.popErrorScope();
  const second = await device.popErrorScope();

  assert.deepEqual(first, inner, 'the one closed first resolves first');
  assert.deepEqual(second, outer);
});

test('the crossings stay at one per frame even with a scope left open', async () => {
  const state = mockWithScopes([]);
  const device = await makeDevice();

  const buffer = device.createBuffer({ size: 16, usage: 0x48 });

  // Run five frames with a scope open — push must only record.
  device.pushErrorScope('validation');
  for (let frame = 0; frame < 5; frame += 1) {
    device.queue.writeBuffer(buffer, 0, new Float32Array([frame]));
    device.queue.submit([]);
  }

  assert.equal(state.executeCalls.length, 5, 'pushErrorScope must not create an extra submission');
  const first = commandsOf(state, 0);
  assert.ok(
    first.some((command) => command.op === 'pushErrorScope'),
    'the scope rides out on the first frame submission'
  );
});

test('popErrorScope resolves without a submit (it submits itself)', async () => {
  const state = mockWithScopes([null]);
  const device = await makeDevice();

  device.pushErrorScope('validation');
  // submit is never called — the Promise must resolve anyway. Otherwise initialization diagnostics hang.
  await device.popErrorScope();

  assert.equal(state.executeCalls.length, 1, 'pop submits by itself');
});

test('an unpaired pop rejects with an OperationError', async () => {
  // Native marks an unpaired slot as `{rejected: true}` and sends it.
  mockWithScopes([{ rejected: true }]);
  const device = await makeDevice();

  // The spec produces no error in this case and only rejects the Promise. An app has to be able to tell
  // "the scope was clean (null)" apart from "the push/pop were unpaired".
  await assert.rejects(() => device.popErrorScope(), { name: 'OperationError' });
});

test('an unknown filter is a synchronous TypeError', async () => {
  const state = installNativeMock();
  const device = await makeDevice();

  // It must fail at the same place as in a browser (the WebIDL enum conversion). Letting it through shifts
  // the native scope stack and a later pop takes an outer scope.
  assert.throws(() => device.pushErrorScope('Validation'), TypeError);
  assert.throws(() => device.pushErrorScope('oom'), TypeError);
  assert.equal(
    commandsOf(state).filter((command) => command.op === 'pushErrorScope').length,
    0,
    'a rejected push must not ride out as a command'
  );

  for (const filter of ['validation', 'out-of-memory', 'internal']) {
    device.pushErrorScope(filter);
  }
});

test('a waiting scope resolves even when the native call fails', async () => {
  installNativeMock({
    executeResult: () => {
      throw new Error('the bridge died');
    },
  });
  const device = await makeDevice();

  device.pushErrorScope('validation');
  // Without resolving, initialization diagnostics hang and the next pop takes a stale resolver, throwing the indices off.
  assert.equal(await device.popErrorScope(), null);
});

test('device.destroy closes waiting scopes with null', async () => {
  installNativeMock();
  const device = await makeDevice();

  // The waiting state is built directly rather than through flush — only whether destroy finishes it matters.
  const pending = new Promise((resolve, reject) => {
    device._recorder.pendingErrorScopes.push({ resolve, reject });
  });
  device.destroy();

  // Without resolution it would wait forever, so it is raced — a hang must show as a failure too.
  const settled = await Promise.race([
    pending,
    new Promise((resolve) => setTimeout(() => resolve('hung'), 50)),
  ]);
  assert.equal(settled, null, 'no unresolved Promise may be left behind');
});
