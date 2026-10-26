/**
 * `mapAsync(mode)` — `GPUMapMode.WRITE` and the mode/usage rules.
 *
 * `mapAsync` used to ignore its mode and always read, and `unmap()` sent only "unmapped" — so with
 * `GPUMapMode.WRITE` **everything written into the mapping vanished silently.** The mapping is a JS copy
 * (an `ArrayBuffer` cannot point into native memory), so a write mapping has to carry its bytes back at
 * `unmap()`: `{op: 'unmapBuffer', buffer, offset, data}`.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { GPUBufferUsage, GPUMapMode } from '../webgpu.js';
import { installNativeMock, makeDevice, commandsOf } from './helpers.mjs';

/** Native hands back the requested range filled with `fill` (the buffer's "current contents"). */
function contentsOf(fill, bufferSize = 32) {
  return (params) => ({
    ok: true,
    data: new Uint8Array(params.size ?? bufferSize - params.offset).fill(fill).buffer,
  });
}

function unmapCommand(state) {
  return commandsOf(state).find((command) => command.op === 'unmapBuffer');
}

test('a WRITE mapping sends what was written back with unmap', async () => {
  const state = installNativeMock({ readBufferResult: contentsOf(0x11) });
  const device = await makeDevice();
  const buffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC });

  await buffer.mapAsync(GPUMapMode.WRITE, 0, 16);
  // A range copy — it only survives through unmap's write-back.
  new Uint8Array(buffer.getMappedRange(8, 8)).fill(0xab);
  buffer.unmap();
  device.queue.submit([]);

  const unmap = unmapCommand(state);
  assert.ok(unmap, 'no unmapBuffer was recorded');
  assert.equal(unmap.buffer, buffer.id);
  assert.equal(unmap.offset, 0);
  const bytes = new Uint8Array(unmap.data);
  assert.equal(bytes.length, 16);
  assert.deepEqual(Array.from(bytes.slice(0, 8)), Array(8).fill(0x11), 'untouched bytes keep the current contents');
  assert.deepEqual(Array.from(bytes.slice(8)), Array(8).fill(0xab), 'what was written vanished');
  assert.equal(buffer.mapState, 'unmapped');
});

test('writing straight into the ArrayBuffer mapAsync resolved with is sent too', async () => {
  const state = installNativeMock({ readBufferResult: contentsOf(0, 8) });
  const device = await makeDevice();
  const buffer = device.createBuffer({ size: 8, usage: GPUBufferUsage.MAP_WRITE });

  const mapped = await buffer.mapAsync(GPUMapMode.WRITE);
  new Float32Array(mapped).set([1.5, -2]);
  buffer.unmap();
  device.queue.submit([]);

  assert.deepEqual(Array.from(new Float32Array(unmapCommand(state).data)), [1.5, -2]);
});

test('a partial WRITE mapping is addressed in buffer offsets and written back at its offset', async () => {
  const state = installNativeMock({ readBufferResult: contentsOf(0) });
  const device = await makeDevice();
  const buffer = device.createBuffer({ size: 32, usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC });

  await buffer.mapAsync(GPUMapMode.WRITE, 16, 16);
  assert.deepEqual(state.readBufferCalls.at(-1), { buffer: buffer.id, offset: 16, size: 16 });

  // Outside the mapped range — the old mapping-relative reading would have accepted this.
  assert.throws(() => buffer.getMappedRange(0, 16), (error) => {
    assert.equal(error.name, 'OperationError');
    assert.match(error.message, /exceeds the mapping/);
    return true;
  });
  new Uint8Array(buffer.getMappedRange(16, 16)).fill(7);
  buffer.unmap();
  device.queue.submit([]);

  const unmap = unmapCommand(state);
  assert.equal(unmap.offset, 16);
  assert.deepEqual(Array.from(new Uint8Array(unmap.data)), Array(16).fill(7));
});

test('a READ mapping sends no data with unmap', async () => {
  const state = installNativeMock({ readBufferResult: contentsOf(3) });
  const device = await makeDevice();
  const buffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

  await buffer.mapAsync(GPUMapMode.READ);
  new Uint8Array(buffer.getMappedRange()).fill(9);   // writes into a read mapping go nowhere (spec)
  buffer.unmap();
  device.queue.submit([]);

  assert.deepEqual(unmapCommand(state), { op: 'unmapBuffer', buffer: buffer.id });
});

test('mapAsync rejects a mode its usage does not allow, without calling native', async () => {
  const state = installNativeMock({ readBufferResult: contentsOf(0) });
  const device = await makeDevice();
  const readable = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const writable = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC });

  const cases = [
    [readable, [GPUMapMode.WRITE], /MAP_WRITE/],
    [writable, [GPUMapMode.READ], /cannot map a MAP_WRITE buffer/],
    [writable, [], /cannot map a MAP_WRITE buffer/],   // an omitted mode is READ
    [writable, [GPUMapMode.READ | GPUMapMode.WRITE], /exactly one/],
    [readable, [0], /exactly one/],
  ];
  for (const [buffer, args, message] of cases) {
    await assert.rejects(buffer.mapAsync(...args), (error) => {
      assert.equal(error.name, 'OperationError');
      assert.match(error.message, message);
      return true;
    }, `mapAsync(${args.join(', ')}) passed`);
    assert.equal(buffer.mapState, 'unmapped', 'a rejected mapAsync leaves the buffer unmapped');
  }
  assert.equal(state.readBufferCalls.length, 0, 'a rejected mapAsync must not reach native');
});

test('mapAsync(WRITE) keeps the spec alignment and range rules', async () => {
  const state = installNativeMock({ readBufferResult: contentsOf(0) });
  const device = await makeDevice();
  const buffer = device.createBuffer({ size: 32, usage: GPUBufferUsage.MAP_WRITE });

  await assert.rejects(buffer.mapAsync(GPUMapMode.WRITE, 4, 8), /multiple of 8/);
  await assert.rejects(buffer.mapAsync(GPUMapMode.WRITE, 0, 6), /multiple of 4/);
  await assert.rejects(buffer.mapAsync(GPUMapMode.WRITE, 24, 16), /exceeds the buffer size/);
  assert.equal(state.readBufferCalls.length, 0);

  await buffer.mapAsync(GPUMapMode.WRITE, 8);   // the size defaults to the rest of the buffer
  assert.deepEqual(state.readBufferCalls.at(-1), { buffer: buffer.id, offset: 8, size: undefined });
});

test('the mode of the previous mapping does not leak into the next one', async () => {
  const state = installNativeMock({ readBufferResult: contentsOf(0, 16) });
  const device = await makeDevice();
  const buffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_WRITE, mappedAtCreation: true });

  buffer.unmap();   // the mappedAtCreation bytes go back as the first unmapBuffer
  await buffer.mapAsync(GPUMapMode.WRITE);   // flushes the creation and the write-back in their own batch
  buffer.unmap();
  device.queue.submit([]);

  const commands = state.executeCalls.flatMap((call) => call.commands);
  const unmaps = commands.filter((command) => command.op === 'unmapBuffer');
  assert.equal(unmaps.length, 2);
  assert.equal(commands.find((command) => command.op === 'createBuffer').mappedAtCreation, true);
  for (const unmap of unmaps) {
    assert.ok(unmap.data instanceof ArrayBuffer);
    assert.equal(unmap.data.byteLength, 16);
    assert.equal(unmap.offset, 0);
  }
});
